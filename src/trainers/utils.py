import os
from abc import ABC
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import transform
import seaborn as sns
import torch
import wandb
from lightning import LightningDataModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from rasterio.warp import Resampling, reproject
from torchgeo.datasets.utils import percentile_normalization
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    JaccardIndex,
    Precision,
    Recall,
)
from torchmetrics.classification import (
    ConfusionMatrix as TorchMetricsConfusionMatrix,
)
from tqdm import tqdm
import gc


def compute_class_weights(freq_0=80, freq_1=20):
    """
    Compute class weights for imbalanced binary classification.
    Args:
        freq_0: frequency of class 0 samples
        freq_1: frequency of class 1 samples
    Returns:
        Normalized tensor of class weights
    """
    total = freq_0 + freq_1

    # Compute inverse weights
    weight_0 = 1 / (freq_0 / total)  # 1.11
    weight_1 = 1 / (freq_1 / total)  # 10.0

    # Create tensor of weights
    class_weights = torch.tensor([weight_0, weight_1])

    # Normalize weights (optional but recommended)
    class_weights = class_weights / class_weights.sum() * 2  # multiply by num_classes
    return class_weights


def compute_final_metrics(
    reference_change_map: Path,
    prediction_change_map: Path,
    num_classes: int,
    class_names: list[str] = None,
    averaging: str = "macro",
    ignore_index: int = 255,
):
    """Compute final metrics from the reference and prediction change maps.

    Args:
        reference_change_map (Path): Reference change map path.
        prediction_change_map (Path): Prediction change map path.
        num_classes (int): Number of classes in the segmentation task.
        class_names (list[str], optional): List of class names for plotting. Defaults to None.
        averaging (str, optional): Defines the reduction that is applied over labels. Defaults to "macro".
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric calculation. Defaults to 255.

    Returns:
        _type_: _description_
    """
    if class_names is not None:
        assert len(class_names) == num_classes, "Number of class names must match num_classes"
    
    task = "multiclass" if num_classes > 1 else "binary"

    multiclass_metric_collection = MetricCollection(
        [
            Accuracy(
                task=task,
                num_classes=num_classes,
                multidim_average="global",
                average=averaging,
                ignore_index=ignore_index,
            ),
            JaccardIndex(
                task=task,
                num_classes=num_classes,
                average=averaging,
                ignore_index=ignore_index,
            ),
            Precision(
                task=task,
                num_classes=num_classes,
                average=averaging,
                ignore_index=ignore_index,
            ),
            Recall(
                task=task,
                num_classes=num_classes,
                average=averaging,
                ignore_index=ignore_index,
            ),
            F1Score(
                task=task,
                num_classes=num_classes,
                average=averaging,
                ignore_index=ignore_index,
            ),
        ]
    )

    metric_collection_each_label = MetricCollection(
        {
            "ConfusionMatrix": TorchMetricsConfusionMatrix(
                normalize="true",
                task=task,
                num_classes=num_classes,
                ignore_index=ignore_index,
            ),
            "Accuracy": Accuracy(
                task=task,
                num_classes=num_classes,
                average="none",
                ignore_index=ignore_index,
            ),
            "JaccardIndex": JaccardIndex(
                task=task,
                num_classes=num_classes,
                average="none",
                ignore_index=ignore_index,
            ),
            "Precision": Precision(
                task=task,
                num_classes=num_classes,
                average="none",
                ignore_index=ignore_index,
            ),
            "Recall": Recall(
                task=task,
                num_classes=num_classes,
                average="none",
                ignore_index=ignore_index,
            ),
            "F1Score": F1Score(
                task=task,
                num_classes=num_classes,
                average="none",
                ignore_index=ignore_index,
            ),
        },
    )

    # Load mask and prediction
    with (
        rasterio.open(reference_change_map) as mask_src,
        rasterio.open(prediction_change_map) as pred_src,
    ):
        # Calculate intersecting bounding box
        bounds_mask = mask_src.bounds
        bounds_pred = pred_src.bounds
        intersection = rasterio.coords.BoundingBox(
            left=max(bounds_mask.left, bounds_pred.left),
            bottom=max(bounds_mask.bottom, bounds_pred.bottom),
            right=min(bounds_mask.right, bounds_pred.right),
            top=min(bounds_mask.top, bounds_pred.top),
        )

        # Validate the intersection
        if (
            intersection.left >= intersection.right
            or intersection.bottom >= intersection.top
        ):
            raise ValueError("No overlapping area between rasters.")

        # Calculate the window bounds for the intersection
        mask_window = mask_src.window(*intersection)
        
        # Read mask data
        mask_tensor = torch.from_numpy(mask_src.read(1, window=mask_window))
        
        # Check if resolutions match
        if mask_src.res == pred_src.res:
            # Same resolution - can read directly
            pred_window = pred_src.window(*intersection)
            pred_tensor = torch.from_numpy(pred_src.read(1, window=pred_window))
        else:
            print("Different resolutions - need to resample prediction to match mask")

            mask_transform = mask_src.window_transform(mask_window)
            pred_resampled = np.empty(mask_tensor.shape, dtype=pred_src.dtypes[0])
            
            reproject(
                source=rasterio.band(pred_src, 1),
                destination=pred_resampled,
                src_transform=pred_src.transform,
                src_crs=pred_src.crs,
                dst_transform=mask_transform,
                dst_crs=mask_src.crs,
                resampling=Resampling.nearest,
            )
            
            pred_tensor = torch.from_numpy(pred_resampled)
            # Explicitly delete the numpy array to free memory
            del pred_resampled
        
        # Convert multiclass mask to binary for change detection
        if num_classes == 2:  # Binary change detection
            # Create binary mask: 0 stays 0, 1-8 become 1, 255 stays 255
            # Use clone() but immediately delete the original to save memory
            binary_mask = mask_tensor.clone()
            del mask_tensor  # Free original immediately
            binary_mask[(binary_mask >= 1) & (binary_mask <= 8)] = 1
            mask_tensor = binary_mask
            
            binary_pred = pred_tensor.clone()
            del pred_tensor  # Free original immediately
            binary_pred[(binary_pred >= 1) & (binary_pred <= 8)] = 1
            pred_tensor = binary_pred
            
            # Clean up temporary variables
            del binary_mask, binary_pred

    # Flatten tensor and remove NaN values to save memory
    mask_tensor = mask_tensor.flatten()
    pred_tensor = pred_tensor.flatten()
    valid_mask = (mask_tensor != ignore_index) & (pred_tensor != ignore_index)
    mask_tensor = mask_tensor[valid_mask]
    pred_tensor = pred_tensor[valid_mask]
    # Explicitly delete the valid_mask to free memory
    del valid_mask

    # Process metrics in chunks to avoid memory overflow
    chunk_size = 50_000_000  # 50M pixels at a time
    total_pixels = mask_tensor.shape[0]
    
    # Process in chunks
    for start_idx in range(0, total_pixels, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pixels)
        
        # Extract chunk
        mask_chunk = mask_tensor[start_idx:end_idx]
        pred_chunk = pred_tensor[start_idx:end_idx]
        
        # Update metrics with this chunk
        multiclass_metric_collection.update(pred_chunk, mask_chunk)
        metric_collection_each_label.update(pred_chunk, mask_chunk)
        
        # Clear chunk references
        del mask_chunk, pred_chunk
        
        # Force garbage collection every 5 chunks
        if (start_idx // chunk_size) % 5 == 0:
            gc.collect()
    
    # Compute final metrics
    multiclass_metrics = multiclass_metric_collection.compute()
    metrics_each_label = metric_collection_each_label.compute()

    # Clear tensor references to free memory
    del pred_tensor, mask_tensor
    gc.collect()
    
    figure_collection = []

    # Plot Confusion Matrix
    fig1, ax = plt.subplots(figsize=(8, 8), dpi=100)
    metric_collection_each_label['ConfusionMatrix'].plot(ax=ax, labels=class_names)
    ax.set_title("Confusion Matrix")
    figure_collection.append(fig1)
    plt.close(fig1)

    # Plot Accuracy
    fig2, ax = plt.subplots(figsize=(15, 8), dpi=100)
    metric_collection_each_label['Accuracy'].plot(ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    class_names = labels if class_names is None else class_names
    ax.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax.set_title("Accuracy")
    fig2.tight_layout()
    figure_collection.append(fig2)
    plt.close(fig2)

    # Plot Jaccard Index
    fig3, ax = plt.subplots(figsize=(15, 8), dpi=100)
    metric_collection_each_label['JaccardIndex'].plot(ax=ax)
    ax.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax.set_title("Jaccard Index")
    fig3.tight_layout()
    figure_collection.append(fig3)
    plt.close(fig3)

    # Plot Precision
    fig4, ax = plt.subplots(figsize=(15, 8), dpi=100)
    metric_collection_each_label['Precision'].plot(ax=ax)
    ax.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax.set_title("Precision")
    fig4.tight_layout()
    figure_collection.append(fig4)
    plt.close(fig4)

    # Plot Recall
    fig5, ax = plt.subplots(figsize=(15, 8), dpi=100)
    metric_collection_each_label['Recall'].plot(ax=ax)
    ax.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax.set_title("Recall")
    fig5.tight_layout()
    figure_collection.append(fig5)
    plt.close(fig5)

    # Plot F1 Score
    fig6, ax = plt.subplots(figsize=(15, 8), dpi=100)
    metric_collection_each_label['F1Score'].plot(ax=ax)
    ax.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax.set_title("F1 Score")
    fig6.tight_layout()
    figure_collection.append(fig6)
    plt.close(fig6)

    # Reset metric collections to free their internal state
    multiclass_metric_collection.reset()
    metric_collection_each_label.reset()

    return multiclass_metrics, metrics_each_label, figure_collection


def calculate_class_frequencies(
    datamodule: LightningDataModule, num_classes: int
) -> np.ndarray:
    """
    Efficiently calculate class frequencies from a datamodule's training set.

    Args:
        datamodule: Lightning datamodule containing train_dataloader
        num_classes: Number of classes to count

    Returns:
        class_pixel_counts: Array of pixel counts per class
    """
    # Initialize counts array on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.int64, device=device)

    print("Calculating class frequencies from training set...")
    for batch in datamodule.train_dataloader():
        # Move masks to same device as counts
        masks = batch["mask"].to(device)

        # Use torch.bincount which is faster than numpy unique
        # Flatten masks and count occurrences of each class
        counts = torch.bincount(masks.flatten(), minlength=num_classes)
        class_pixel_counts += counts

    # Move final counts to CPU and convert to numpy
    class_pixel_counts = class_pixel_counts.cpu().numpy()
    print(f"Total pixel counts per class: {class_pixel_counts}")

    return class_pixel_counts


def compute_class_weights_multiclass(
    class_frequencies: np.ndarray[float | int],
) -> torch.tensor:
    """
    Compute class weights for imbalanced multi-class classification using inverse frequency.

    Args:
        class_frequencies: A list, tuple, or array containing the frequency (count or proportion)
                           of samples for each class. The order should correspond to class indices
                           (e.g., index 0 for class 0).
    Returns:
        Normalized tensor of class weights
    """
    num_classes = len(class_frequencies)
    total = class_frequencies.sum()

    # Compute inverse weights: total / frequency
    weights = total / class_frequencies

    # Create tensor of weights
    class_weights = torch.from_numpy(weights).float()

    # Normalize weights so they sum to num_classes (average weight is 1)
    class_weights = (class_weights / class_weights.sum()) * num_classes

    print(f"Computed multi-class weights: {class_weights.numpy()}")
    return class_weights


def find_optimal_learning_rate(
    model,
    datamodule,
    min_epochs,
    gpu_id,
    num_training=250,
    min_lr=1e-7,
    max_lr=1,
    early_stop_threshold=4,
    max_epochs=300,
    original_format=False,
):
    """
    Find optimal learning rate using Lightning's learning rate finder.

    Args:
        model: Lightning model to tune
        datamodule: Lightning datamodule
        min_epochs: Minimum number of epochs to train
        gpu_id: GPU device ID to use
        num_training: Number of training steps for lr finder
        min_lr: Minimum learning rate to test
        max_lr: Maximum learning rate to test
        early_stop_threshold: Early stopping threshold
        max_epochs: Maximum number of epochs
        original_format: Whether to return learning rate in original format (default: False)

    Returns:
        suggested_lr: Suggested learning rate in scientific notation
        fig: Learning rate finder plot
    """
    trainer_lr_finder = Trainer(
        accelerator="gpu",
        devices=[gpu_id],
        min_epochs=min_epochs,
        max_epochs=max_epochs,
    )

    tuner = Tuner(trainer_lr_finder)
    lr_finder = tuner.lr_find(
        model,
        datamodule=datamodule,
        num_training=num_training,
        min_lr=min_lr,
        max_lr=max_lr,
        early_stop_threshold=early_stop_threshold,
    )

    # Plot results
    fig = lr_finder.plot(suggest=True)
    plt.show()

    if original_format:
        suggested_lr = lr_finder.suggestion()
    else:
        # Format suggested learning rate in scientific notation
        suggested_lr = f"{lr_finder.suggestion():.0e}".replace("e-0", "e-").replace(
            "e+0", "e+"
        )

    print("Suggested learning rate: ", suggested_lr)

    return suggested_lr, fig


def load_config_from_yaml(config_path: Path) -> dict:
    """Load yaml config file and return as a dictionary.

    Args:
        yaml_path (Path): Config file path

    Returns:
        dict: Dictionary with config parameters
    """
    import yaml

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        config[key] = value["value"]

    return config


def setup_training(
    experiment_name: str,
    experiment_dir: str,
    min_epochs: int,
    max_epochs: int,
    gpu_id: int,
    patience: int,
    wandb_project: str,
    logging: str = "remote",
    wandb_key: str = None,
    monitor_metric: str = "val_jaccard",
    save_model_locally: bool = True,
    log_model: str = "all",
) -> Trainer:
    """
    Set up PyTorch Lightning training configuration including callbacks and logging.
    Args:
        experiment_name: Name of the experiment
        experiment_dir: Directory to save checkpoints
        min_epochs: Minimum training epochs
        max_epochs: Maximum training epochs
        gpu_id: GPU device ID to use
        patience: Early stopping patience
        wandb_project: W&B project name
        logging: Either 'local' or 'remote' for W&B logging
        wandb_key: W&B API key (required if logging='remote')
        monitor_metric: Metric to monitor for checkpoints and early stopping (default: "val_jaccard")
        save_model_locally: Whether to save model checkpoints locally (default: True)
        log_model: Log checkpoints created by ModelCheckpoint as W&B artifacts

    Returns:
        trainer: Configured PyTorch Lightning trainer
    """
    # Define the filename format using the experiment_name
    # Example: multiclass-change-resnet50-epoch=19-step=20840-val_jaccard=0.1850.ckpt
    checkpoint_filename_format = (
        f"{experiment_name}" + "-{epoch:02d}-{step}-{" + monitor_metric + ":.4f}"
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        dirpath=experiment_dir,
        filename=checkpoint_filename_format,
        save_top_k=1 if save_model_locally else 0,
        save_last=save_model_locally,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        mode="max",
        min_delta=0.00,
        patience=patience,
        verbose=True,
    )

    # Set up W&B logging
    if logging == "local":
        os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
        wandb.login()
        wb_logger = WandbLogger(
            name=experiment_name,
            save_dir="logs_dir",
            project=wandb_project,
            log_model=log_model,
        )
    elif logging == "remote":
        if wandb_key is None:
            raise ValueError("wandb_key is required for remote logging")
        wandb.login(key=wandb_key)
        wb_logger = WandbLogger(
            name=experiment_name,
            project=wandb_project,
            log_model=log_model,
        )
    else:
        raise ValueError("logging must be either 'local' or 'remote'")
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Initialize trainer
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, RichProgressBar()],
        logger=[wb_logger],
        default_root_dir=experiment_dir,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=[gpu_id] if accelerator == "gpu" else "auto",
    )

    return trainer


def predict_large_image(
    model,
    image1_path,
    image2_path,
    output_path,
    tile_size=512,
    overlap=0,
    probability_threshold=0.5,
    save_probability_map=True,
):
    """
    Predict on large images by processing them in tiles and optionally save probability map.

    Args:
        model: The trained model
        image1_path: Path to the first (before) image
        image2_path: Path to the second (after) image
        output_path: Path to save the prediction
        tile_size: Size of tiles to process
        overlap: Overlap between adjacent tiles to avoid edge artifacts
        probability_threshold: Threshold to convert probabilities to binary predictions (default: 0.5)
        save_probability_map: Whether to save the probability map (default: True)

    Returns:
        The prediction array and probability map
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Open the images
    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        # Check that the images have the same dimensions and CRS
        if src1.shape != src2.shape or src1.crs != src2.crs:
            print(src1.shape, src2.shape)
            print(src1.crs, src2.crs)
            raise ValueError("Input images must have the same dimensions and CRS")

        # Get metadata for binary prediction output
        binary_meta = src1.meta.copy()
        binary_meta.update(
            {
                "count": 1,
                "dtype": "uint8",
                "driver": "COG",
                "compress": "LZW",
                "predictor": 1,  # Predictor 1 for uint8 data
            }
        )

        # Get metadata for probability map output
        prob_meta = src1.meta.copy()
        prob_meta.update(
            {
                "count": 1,
                "dtype": "float32",  # Use float32 for probabilities
                "driver": "COG",
                "compress": "ZSTD",
                "predictor": 3,  # Predictor 3 for floating point
            }
        )

        # Create output arrays
        height, width = src1.shape
        prediction = np.zeros((height, width), dtype=np.uint8)
        probability_map = np.zeros((height, width), dtype=np.float32)

        # Calculate effective tile size with overlap
        effective_size = tile_size - 2 * overlap

        # Calculate number of tiles in each dimension
        n_tiles_h = int(np.ceil(height / effective_size))
        n_tiles_w = int(np.ceil(width / effective_size))
        total_tiles = n_tiles_h * n_tiles_w

        print(f"Processing {total_tiles} tiles ({n_tiles_h}x{n_tiles_w}) with tile_size={tile_size}, overlap={overlap}")

        # Calculate normalization statistics for both images
        def get_image_stats_uint8_normalized(src):
            band_means = []
            band_stds = []
            for band in range(1, src.count + 1):
                stats = src.statistics(band, approx=False)
                if src.profile["dtype"] == "uint8":
                    band_means.append(stats.mean / 255.0)
                    band_stds.append(stats.std / 255.0)
                else:
                    raise ValueError("Only uint8 images are supported")
            return torch.tensor(band_means), torch.tensor(band_stds)

        # Get normalization statistics
        means1, stds1 = get_image_stats_uint8_normalized(src1)
        means2, stds2 = get_image_stats_uint8_normalized(src2)
        means = torch.cat([means1, means2], dim=0)
        stds = torch.cat([stds1, stds2], dim=0)

        # Process each tile
        tile_count = 0
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                tile_count += 1
                if tile_count % 50 == 0 or tile_count == 1:
                    print(f"Processing tile {tile_count}/{total_tiles} ({tile_count/total_tiles*100:.1f}%)")

                # Calculate tile coordinates
                y_start = i * effective_size
                x_start = j * effective_size
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)

                # Adjust start coordinates to maintain tile size
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                # Read tile data
                img1_tile = src1.read(window=((y_start, y_end), (x_start, x_end)))
                img2_tile = src2.read(window=((y_start, y_end), (x_start, x_end)))

                # Convert to PyTorch tensors and normalize
                img1_tensor = (
                    torch.from_numpy(img1_tile).float() / 255.0
                )  # Normalize to [0,1]
                img2_tensor = torch.from_numpy(img2_tile).float() / 255.0

                # Stack images along channel dimension
                input_tensor = torch.cat([img1_tensor, img2_tensor], dim=0)

                # Apply normalization with calculated statistics
                for t, m, s in zip(input_tensor, means, stds):
                    t.sub_(m).div_(s)

                # Add batch dimension and move to device
                input_tensor = input_tensor.unsqueeze(0).to(device)

                # Make prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    # Probability of change class (still on GPU)
                    change_prob_gpu = probabilities[0, 1]
                    # Perform thresholding on GPU
                    pred_gpu = (change_prob_gpu > probability_threshold).to(torch.uint8)
                    # Move final prediction to CPU/NumPy if needed
                    pred = pred_gpu.cpu().numpy()

                # Calculate the effective area to update in the output
                if overlap > 0 and i < n_tiles_h - 1 and j < n_tiles_w - 1:
                    # For tiles that are not at the edges, only update the central region (if overlap exists)
                    effective_y_start = y_start + overlap
                    effective_x_start = x_start + overlap
                    effective_y_end = y_end - overlap
                    effective_x_end = x_end - overlap

                    # Extract the corresponding region from the prediction
                    y_offset = overlap
                    x_offset = overlap
                    y_size = effective_y_end - effective_y_start
                    x_size = effective_x_end - effective_x_start

                    effective_pred = pred[
                        y_offset : y_offset + y_size, x_offset : x_offset + x_size
                    ]
                    # Slice the GPU tensor and then move to CPU
                    effective_change_prob = (
                        change_prob_gpu[
                            y_offset : y_offset + y_size, x_offset : x_offset + x_size
                        ]
                        .cpu()
                        .numpy()
                    )

                    # Update the prediction and probability map
                    prediction[
                        effective_y_start:effective_y_end,
                        effective_x_start:effective_x_end,
                    ] = effective_pred
                    if save_probability_map:  # Check if saving prob map
                        probability_map[
                            effective_y_start:effective_y_end,
                            effective_x_start:effective_x_end,
                        ] = effective_change_prob
                else:
                    # For edge tiles OR if overlap is 0, update the entire tile region
                    # Calculate the actual height and width of the current prediction tile
                    tile_h, tile_w = pred.shape
                    # Ensure the target slice matches the prediction tile dimensions
                    target_y_end = y_start + tile_h
                    target_x_end = x_start + tile_w
                    prediction[y_start:target_y_end, x_start:target_x_end] = pred
                    if save_probability_map:  # Check if saving prob map
                        # Assign the entire probability tensor (moved to CPU) to the correct global slice
                        probability_map[y_start:target_y_end, x_start:target_x_end] = (
                            change_prob_gpu.cpu().numpy()
                        )

        print(f"Completed processing all {total_tiles} tiles!")

        # Write the binary prediction to disk
        with rasterio.open(str(output_path), "w", **binary_meta) as dst:
            dst.write(prediction, 1)

        print(f"Binary prediction saved to {output_path}")

        # Write the probability map to disk if requested
        if save_probability_map:
            prob_output_path = str(output_path).replace(".tif", "_probability.tif")
            with rasterio.open(prob_output_path, "w", **prob_meta) as dst:
                dst.write(probability_map, 1)

            print(f"Probability map saved to {prob_output_path}")

        return prediction, probability_map


def read_intersecting_area_of_two_rasters(raster1_path: Path, raster2_path: Path):
    """
    Read the intersecting area of two rasters and return as numpy arrays.
    Assumes that both rasters are single-band and have aligning pixels.

    Args:
        raster1_path (Path): Path to the first raster
        raster2_path (Path): Path to the second raster

    Returns:
        raster1_array: Numpy array of the intersecting area from raster 1
        raster2_array: Numpy array of the intersecting area from raster 2
        output_meta: Metadata for the intersecting area
    """
    # Load both rasters
    with (
        rasterio.open(raster1_path) as raster1_src,
        rasterio.open(raster2_path) as raster2_src,
    ):
        # Ensure both rasters have the same resolution and CRS
        #assert raster1_src.res == raster2_src.res
        assert raster1_src.crs == raster2_src.crs

        # Calculate intersecting bounding box
        bounds_raster1 = raster1_src.bounds
        bounds_raster2 = raster2_src.bounds
        intersection = rasterio.coords.BoundingBox(
            left=max(bounds_raster1.left, bounds_raster2.left),
            bottom=max(bounds_raster1.bottom, bounds_raster2.bottom),
            right=min(bounds_raster1.right, bounds_raster2.right),
            top=min(bounds_raster1.top, bounds_raster2.top),
        )

        # Validate the intersection
        if (
            intersection.left >= intersection.right
            or intersection.bottom >= intersection.top
        ):
            raise ValueError("No overlapping area between rasters.")

        # Read intersecting rasters
        raster1_window = raster1_src.window(*intersection)
        raster1_array = raster1_src.read(1, window=raster1_window)
        # Check if resolutions match
        if raster1_src.res == raster2_src.res:
            # Same resolution - can read directly
            raster2_window = raster2_src.window(*intersection)
            raster2_array = raster2_src.read(1, window=raster2_window)
        else:
            print("Different resolutions - need to resample raster 2 to match raster 1")

            mask_transform = raster1_src.window_transform(raster1_window)
            raster2_array = np.empty(raster1_array.shape, dtype=raster2_src.dtypes[0])
            
            reproject(
                source=rasterio.band(raster2_src, 1),
                destination=raster2_array,
                src_transform=raster2_src.transform,
                src_crs=raster2_src.crs,
                dst_transform=mask_transform,
                dst_crs=raster1_src.crs,
                resampling=Resampling.nearest,
            )

        # Ensure both arrays have the same shape
        assert (
            raster1_array.shape == raster2_array.shape
        ), "Something went wrong. Overlapping area between raster1 and raster2 has different shape"

        # Set output map parameters
        new_transform = transform(raster1_window, raster1_src.transform)
        output_meta = raster1_src.meta
        output_meta.update(
            {
                "transform": new_transform,
                "height": raster1_array.shape[0],
                "width": raster1_array.shape[1],
                "bounds": intersection,
            }
        )

    return raster1_array, raster2_array, output_meta


def visualize_prediction(image1_path, image2_path, prediction_path, bbox=None):
    """
    Visualize the prediction results for a specific area.

    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        prediction_path: Path to the prediction result
        bbox: Bounding box to visualize (minx, miny, maxx, maxy) or None for a random area
    """
    with rasterio.open(image1_path) as src1, rasterio.open(
        image2_path
    ) as src2, rasterio.open(prediction_path) as src_pred:

        if bbox is None:
            # Select a random area
            window_size = 512
            max_row = src1.height - window_size
            max_col = src1.width - window_size
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)
            window = ((row, row + window_size), (col, col + window_size))
        else:
            # Convert bbox to pixel coordinates
            window = rasterio.windows.from_bounds(*bbox, src1.transform)

        # Read data
        img1 = src1.read(window=window)
        img2 = src2.read(window=window)
        pred = src_pred.read(1, window=window)

        # Normalize images for visualization
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

        img1 = percentile_normalization(img1)
        img2 = percentile_normalization(img2)

        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot images and prediction
        axs[0].imshow(img1)
        axs[0].set_title("Image 2003")
        axs[0].axis("off")

        axs[1].imshow(img2)
        axs[1].set_title("Image 2013")
        axs[1].axis("off")

        axs[2].imshow(pred, cmap="Reds")
        axs[2].set_title("Predicted Changes")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

        return fig


def visualize_probability_map(
    image1_path, image2_path, prob_map_path, sample_size=1024
):
    """
    Visualize the probability map for a random area.
    """
    with rasterio.open(image1_path) as src1, rasterio.open(
        image2_path
    ) as src2, rasterio.open(prob_map_path) as src_prob:

        # Get image dimensions
        height, width = src1.shape

        # Select a random area
        max_row = height - sample_size
        max_col = width - sample_size
        row = np.random.randint(0, max_row)
        col = np.random.randint(0, max_col)
        window = ((row, row + sample_size), (col, col + sample_size))

        # Read data
        img1 = src1.read(window=window)
        img2 = src2.read(window=window)
        prob = src_prob.read(1, window=window)

        # Normalize images for visualization
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

        img1 = percentile_normalization(img1)
        img2 = percentile_normalization(img2)

        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot images and probability map
        axs[0].imshow(img1)
        axs[0].set_title("Image 2003")
        axs[0].axis("off")

        axs[1].imshow(img2)
        axs[1].set_title("Image 2013")
        axs[1].axis("off")

        # Use a colormap that highlights small probabilities
        im = axs[2].imshow(
            prob, cmap="hot", vmin=0, vmax=0.1
        )  # Adjust vmax to highlight small probabilities
        axs[2].set_title("Change Probability")
        axs[2].axis("off")

        # Add colorbar
        cbar = fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        cbar.set_label("Probability of Change")

        plt.tight_layout()
        plt.show()

        # Print statistics
        print("Probability statistics:")
        print(f"Min: {prob.min():.6f}")
        print(f"Max: {prob.max():.6f}")
        print(f"Mean: {prob.mean():.6f}")
        print(f"Std: {prob.std():.6f}")
        print(f"Pixels with prob > 0.01: {np.sum(prob > 0.01)}")
        print(f"Pixels with prob > 0.05: {np.sum(prob > 0.05)}")
        print(f"Pixels with prob > 0.1: {np.sum(prob > 0.1)}")

        return fig


def compute_metrics_from_checkpoint(
    checkpoint_path: str | Path, 
    datamodule: LightningDataModule,
    task_class: ABC,
    device: torch.device | None = None,
) -> dict:
    """
    Compute confusion matrix and metrics from a checkpoint file using torchmetrics.

    Args:
        checkpoint_path (str or Path): Path to the model checkpoint
        datamodule (LightningDataModule): Lightning datamodule containing test_dataloader
        task_class (type): The task class to load the checkpoint with
        device (torch.device, optional): Device to run model on. Defaults to CUDA if available.

    Returns:
        dict: Dictionary containing:
            - confusion_matrix (np.array)
            - per_class_metrics (dict): Contains precision, recall, f1, support per class
            - macro_metrics (dict): Contains precision, recall, f1, jaccard
            - accuracy (float): Overall accuracy (micro-averaged)
            - class_names (list): Names of classes (excluding ignored class)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    task = task_class.load_from_checkpoint(checkpoint_path, map_location=device)
    model = task.model.to(device).eval()

    # Get hparams from the loaded task
    num_classes = task.hparams.get("num_classes")
    ignore_index = task.hparams.get("ignore_index", None)

    tm_task_type = task.hparams.get("task", "multiclass")
    if num_classes == 2 and tm_task_type == "binary":  # Specific binary case
        pass

    if num_classes is None:
        raise ValueError("num_classes not found in model hyperparameters.")

    metrics_to_compute = {
        "accuracy": Accuracy(
            task=tm_task_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="micro",
        ),
        "precision_per_class": Precision(
            task=tm_task_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="none",
        ),
        "recall_per_class": Recall(
            task=tm_task_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="none",
        ),
        "f1_per_class": F1Score(
            task=tm_task_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="none",
        ),
        "precision_macro": Precision(
            task=tm_task_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="macro",
        ),
        "recall_macro": Recall(
            task=tm_task_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="macro",
        ),
        "f1_macro": F1Score(
            task=tm_task_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="macro",
        ),
        "jaccard_macro": JaccardIndex(
            task=tm_task_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="macro",
        ),
        "confusion_matrix": TorchMetricsConfusionMatrix(
            task=tm_task_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
            normalize=None,
        ),
    }
    metrics_collection = MetricCollection(metrics_to_compute).to(device)

    for batch in tqdm(datamodule.test_dataloader(), desc="Evaluating"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)  # Ensure masks are on the same device

        with torch.inference_mode():
            logits = model(images)  # Assuming output is [B, C, H, W]
            # For segmentation, preds are typically [B, H, W] after argmax
            # If model outputs [B, 1, H, W] for binary, sigmoid + threshold might be needed before argmax or direct use.
            # Assuming standard multiclass output needing argmax:
            preds = logits.argmax(dim=1)

        metrics_collection.update(preds, masks)

    final_computed_metrics = metrics_collection.compute()

    # Process results
    cm_tensor = final_computed_metrics["confusion_matrix"]
    # If ignore_index was set, the CM from torchmetrics will be num_classes x num_classes
    # and computations for ignore_index are handled (e.g. it won't contribute to TP of other classes).
    # Support for each class will be the sum of its row in the CM.
    cm_numpy = cm_tensor.cpu().numpy()
    support_per_class_numpy = cm_numpy.sum(axis=1)

    # Prepare class names, excluding ignored class for the final list
    all_class_indices = list(range(num_classes))
    if ignore_index is not None:
        class_indices_to_report = [i for i in all_class_indices if i != ignore_index]
    else:
        class_indices_to_report = all_class_indices

    target_names = [f"Class {i}" for i in class_indices_to_report]

    per_class_results = {}
    # Torchmetric 'none' average returns a tensor of size num_classes.
    # We iterate through class_indices_to_report to build the dict.
    # The index into the metric tensor should be the original class index.
    prec_pc = final_computed_metrics["precision_per_class"].cpu().numpy()
    rec_pc = final_computed_metrics["recall_per_class"].cpu().numpy()
    f1_pc = final_computed_metrics["f1_per_class"].cpu().numpy()

    for i, original_class_idx in enumerate(class_indices_to_report):
        class_name = target_names[i]  # This uses the filtered list
        per_class_results[class_name] = {
            "precision": prec_pc[original_class_idx],
            "recall": rec_pc[original_class_idx],
            "f1": f1_pc[original_class_idx],
            "support": support_per_class_numpy[original_class_idx],
        }

    results = {
        "confusion_matrix": cm_numpy,  # Full CM including ignored class (handled by torchmetrics)
        "per_class_metrics": per_class_results,
        "macro_metrics": {
            "precision": final_computed_metrics["precision_macro"].item(),
            "recall": final_computed_metrics["recall_macro"].item(),
            "f1": final_computed_metrics["f1_macro"].item(),
            "jaccard": final_computed_metrics["jaccard_macro"].item(),
        },
        "accuracy": final_computed_metrics[
            "accuracy"
        ].item(),  # Micro-averaged overall accuracy
        "class_names": target_names,
    }

    # Print results
    print(f"\nOverall Accuracy (Micro Avg): {results['accuracy']:.4f}")
    print("\n--- Macro Average Metrics ---")
    print(f"Precision (Macro): {results['macro_metrics']['precision']:.4f}")
    print(f"Recall (Macro):    {results['macro_metrics']['recall']:.4f}")
    print(f"F1 Score (Macro):  {results['macro_metrics']['f1']:.4f}")
    print(f"Jaccard (mIoU):    {results['macro_metrics']['jaccard']:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=results["class_names"],
        yticklabels=results["class_names"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Print per-class metrics
    print("\n--- Per-Class Metrics ---")
    print(
        f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
    )
    print("-" * 55)
    for class_name, metrics in results["per_class_metrics"].items():
        print(
            f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
            f"{metrics['f1']:<10.4f} {metrics['support']:<10}"
        )
    print("-" * 55)
    print(f"\nOverall Accuracy (Micro Avg): {results['accuracy']:.4f}")

    return results


def predict_large_image_multiclass(
    model,
    image1_path,
    image2_path,
    output_path,
    tile_size=512,
    overlap=0,
    save_probability_map=True,
):
    """
    Predict on large images using a multiclass model by processing them in tiles
    and optionally save the probability map of the predicted class.

    Args:
        model: The trained multiclass model.
        image1_path: Path to the first (before) image.
        image2_path: Path to the second (after) image.
        output_path: Path to save the prediction (class indices).
        tile_size: Size of tiles to process.
        overlap: Overlap between adjacent tiles to avoid edge artifacts.
        save_probability_map: Whether to save the probability map (confidence of predicted class) (default: True).

    Returns:
        The prediction array (class indices) and probability map (confidence).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)  # Ensure model is in eval mode and on the correct device

    output_path = Path(output_path)  # Ensure output_path is a Path object

    # Open the images
    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        # Check that the images have the same dimensions and CRS
        if src1.shape != src2.shape or src1.crs != src2.crs:
            print(f"Shape mismatch: {src1.shape} vs {src2.shape}")
            print(f"CRS mismatch: {src1.crs} vs {src2.crs}")
            raise ValueError("Input images must have the same dimensions and CRS")

        # Get metadata for prediction output (class indices)
        prediction_meta = src1.meta.copy()
        prediction_meta.update(
            {
                "count": 1,
                "dtype": "uint8",  # Assuming number of classes < 256
                "driver": "COG",
                "compress": "LZW",
                "predictor": 1,  # Predictor 1 for uint8 data
            }
        )

        # Get metadata for probability map output (confidence of predicted class)
        prob_meta = src1.meta.copy()
        prob_meta.update(
            {
                "count": 1,
                "dtype": "float32",  # Use float32 for probabilities
                "driver": "COG",
                "compress": "ZSTD",
                "predictor": 3,  # Predictor 3 for floating point
            }
        )

        # Create output arrays
        height, width = src1.shape
        prediction = np.zeros((height, width), dtype=np.uint8)
        # Initialize probability map with a fill value indicating no prediction yet, e.g., -1 or NaN
        # Using NaN requires float, which is fine since prob_meta uses float32
        probability_map = np.full((height, width), np.nan, dtype=np.float32)

        # Calculate effective tile size with overlap
        # Ensure overlap is not too large
        if overlap >= tile_size // 2:
            raise ValueError(
                f"Overlap ({overlap}) must be less than half the tile size ({tile_size // 2})"
            )
        effective_size = tile_size - overlap  # Adjusted calculation for clarity

        # Calculate number of tiles needed to cover the image
        # Stride is the distance between the start of adjacent tiles
        stride = tile_size - overlap
        n_tiles_h = int(np.ceil(height / stride))
        n_tiles_w = int(np.ceil(width / stride))

        print(
            f"Image size: {height}x{width}, Tile size: {tile_size}x{tile_size}, Overlap: {overlap}, Stride: {stride}"
        )
        total_tiles = n_tiles_h * n_tiles_w
        print(
            f"Processing {total_tiles} tiles ({n_tiles_h}x{n_tiles_w}) with tile_size={tile_size}, overlap={overlap}"
        )

        # Calculate normalization statistics for both images
        def get_image_stats_uint8_normalized(src):
            # ... existing code ...
            # Ensure this function handles potential non-uint8 inputs if needed, or keeps the check
            band_means = []
            band_stds = []
            num_bands = src.count
            print(
                f"Calculating stats for {num_bands} bands, dtype: {src.profile['dtype']}"
            )
            for band in range(1, num_bands + 1):
                # Use approx=True for faster stats on large images if needed
                stats = src.statistics(band, approx=False)
                if src.profile["dtype"] == "uint8":
                    # Handle potential None values in stats if bands are empty/masked
                    mean = stats.mean if stats.mean is not None else 0.0
                    std = stats.std if stats.std is not None else 1.0
                    band_means.append(mean / 255.0)
                    # Avoid division by zero if std is 0
                    band_stds.append(std / 255.0 if std > 0 else 1.0 / 255.0)
                # Add handling for other dtypes if necessary
                # elif src.profile['dtype'] == 'uint16':
                #     mean = stats.mean if stats.mean is not None else 0.0
                #     std = stats.std if stats.std is not None else 1.0
                #     band_means.append(mean / 65535.0)
                #     band_stds.append(std / 65535.0 if std > 0 else 1.0 / 65535.0)
                else:
                    # Fallback or raise error for unsupported types
                    raise ValueError(
                        f"Unsupported image dtype for normalization: {src.profile['dtype']}"
                    )
            return torch.tensor(band_means, dtype=torch.float32), torch.tensor(
                band_stds, dtype=torch.float32
            )

        # Get normalization statistics
        means1, stds1 = get_image_stats_uint8_normalized(src1)
        means2, stds2 = get_image_stats_uint8_normalized(src2)
        # Ensure means/stds are on the correct device early
        means = torch.cat([means1, means2], dim=0).to(device)
        stds = torch.cat([stds1, stds2], dim=0).to(device)
        print(f"Normalization means: {means.cpu().numpy()}")
        print(f"Normalization stds: {stds.cpu().numpy()}")

        tile_count = 0
        # Process each tile
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                tile_count += 1
                if tile_count % 50 == 0 or tile_count == 1:
                    print(f"Processing tile {tile_count}/{total_tiles} ({tile_count/total_tiles*100:.1f}%)")
                
                # Calculate tile coordinates based on stride
                y_start = i * stride
                x_start = j * stride

                # Define the window for reading data (up to tile_size)
                read_y_end = min(y_start + tile_size, height)
                read_x_end = min(x_start + tile_size, width)

                # Adjust start coordinates if near the edge to maintain tile_size for the model
                read_y_start = max(0, read_y_end - tile_size)
                read_x_start = max(0, read_x_end - tile_size)

                # Define the actual window dimensions read
                window_h = read_y_end - read_y_start
                window_w = read_x_end - read_x_start

                # Read tile data
                # Use boundless=True if window might go out of bounds (shouldn't with current logic)
                # Use fill_value=0 for areas outside the image bounds if boundless=True
                img1_tile = src1.read(
                    window=((read_y_start, read_y_end), (read_x_start, read_x_end))
                )  # , boundless=True, fill_value=0)
                img2_tile = src2.read(
                    window=((read_y_start, read_y_end), (read_x_start, read_x_end))
                )  # , boundless=True, fill_value=0)

                # Check if tiles have expected shape (Channels, Height, Width)
                if img1_tile.shape[1:] != (window_h, window_w) or img2_tile.shape[
                    1:
                ] != (window_h, window_w):
                    print(
                        f"Warning: Tile read shape mismatch. Expected ({window_h}, {window_w}), got {img1_tile.shape[1:]}, {img2_tile.shape[1:]}"
                    )
                    # Handle potential issues, e.g., skip tile or pad
                    # For now, we assume rasterio handles padding or edge cases correctly

                # Convert to PyTorch tensors and normalize
                # Ensure correct dtype and normalization factor based on input image dtype
                norm_factor = (
                    255.0 if src1.profile["dtype"] == "uint8" else 65535.0
                )  # Adjust if needed
                img1_tensor = torch.from_numpy(img1_tile).float() / norm_factor
                img2_tensor = torch.from_numpy(img2_tile).float() / norm_factor

                # Stack images along channel dimension
                input_tensor = torch.cat([img1_tensor, img2_tensor], dim=0).to(device)

                # Apply normalization (ensure broadcasting works correctly)
                # Reshape means and stds for broadcasting: (C, 1, 1)
                norm_means = means.view(-1, 1, 1)
                norm_stds = stds.view(-1, 1, 1)
                input_tensor = (input_tensor - norm_means) / norm_stds

                # Add batch dimension
                input_tensor = input_tensor.unsqueeze(0)

                # Pad if the tile is smaller than tile_size (at edges)
                pad_h = tile_size - input_tensor.shape[2]
                pad_w = tile_size - input_tensor.shape[3]
                if pad_h > 0 or pad_w > 0:
                    # Pad tuple format: (pad_left, pad_right, pad_top, pad_bottom)
                    padding = (0, pad_w, 0, pad_h)
                    input_tensor = torch.nn.functional.pad(
                        input_tensor, padding, mode="constant", value=0
                    )

                # Make prediction
                with torch.no_grad():
                    output = model(input_tensor)  # Output shape: (1, NumClasses, H, W)
                    probabilities = torch.softmax(output, dim=1)

                    # Get the probability of the predicted class (confidence)
                    max_prob_gpu, pred_gpu = torch.max(
                        probabilities, dim=1
                    )  # Shape: (1, H, W)

                    # Remove batch dimension
                    max_prob_gpu = max_prob_gpu.squeeze(0)  # Shape: (H, W)
                    pred_gpu = pred_gpu.squeeze(0).to(torch.uint8)  # Shape: (H, W)

                # Crop the prediction and probability map if padding was added
                if pad_h > 0 or pad_w > 0:
                    pred_gpu = pred_gpu[:window_h, :window_w]
                    max_prob_gpu = max_prob_gpu[:window_h, :window_w]

                # --- Overlap handling ---
                # Define the area within the *full output arrays* (prediction, probability_map)
                # that this tile's prediction corresponds to.
                write_y_start = read_y_start
                write_x_start = read_x_start
                write_y_end = read_y_end
                write_x_end = read_x_end

                # Define the area within the *current tile's prediction* (pred_gpu, max_prob_gpu)
                # that needs to be written to the output arrays.
                tile_read_y_start = 0
                tile_read_x_start = 0
                tile_read_y_end = window_h
                tile_read_x_end = window_w

                # Adjust write/read areas to handle overlap - only write the non-overlapped central part,
                # except for edge tiles.
                if overlap > 0:
                    # Adjust top edge unless it's the first row of tiles
                    if i > 0:
                        write_y_start += overlap // 2
                        tile_read_y_start += overlap // 2
                    # Adjust left edge unless it's the first column of tiles
                    if j > 0:
                        write_x_start += overlap // 2
                        tile_read_x_start += overlap // 2
                    # Adjust bottom edge unless it's the last row of tiles
                    if i < n_tiles_h - 1:
                        write_y_end -= (
                            overlap + 1
                        ) // 2  # Use ceil division for odd overlaps
                        tile_read_y_end -= (overlap + 1) // 2
                    # Adjust right edge unless it's the last column of tiles
                    if j < n_tiles_w - 1:
                        write_x_end -= (overlap + 1) // 2
                        tile_read_x_end -= (overlap + 1) // 2

                # Ensure indices are valid
                write_y_start = max(0, write_y_start)
                write_x_start = max(0, write_x_start)
                write_y_end = min(height, write_y_end)
                write_x_end = min(width, write_x_end)

                tile_read_y_start = max(0, tile_read_y_start)
                tile_read_x_start = max(0, tile_read_x_start)
                tile_read_y_end = min(window_h, tile_read_y_end)
                tile_read_x_end = min(window_w, tile_read_x_end)

                # Extract the relevant part of the tile prediction
                pred_tile_section = pred_gpu[
                    tile_read_y_start:tile_read_y_end, tile_read_x_start:tile_read_x_end
                ]
                prob_tile_section = max_prob_gpu[
                    tile_read_y_start:tile_read_y_end, tile_read_x_start:tile_read_x_end
                ]

                # Move data to CPU just before assignment
                pred_to_write = pred_tile_section.cpu().numpy()
                prob_to_write = prob_tile_section.cpu().numpy()

                # Update the prediction and probability map
                # Check if the shapes match before assignment
                target_shape = (
                    write_y_end - write_y_start,
                    write_x_end - write_x_start,
                )
                if pred_to_write.shape == target_shape:
                    prediction[write_y_start:write_y_end, write_x_start:write_x_end] = (
                        pred_to_write
                    )
                    if save_probability_map:
                        probability_map[
                            write_y_start:write_y_end, write_x_start:write_x_end
                        ] = prob_to_write
                else:
                    print(
                        f"Warning: Shape mismatch during write. Target: {target_shape}, Source: {pred_to_write.shape}. Skipping write for this section of tile {i},{j}."
                    )
                    print(
                        f"Write coords: y={write_y_start}:{write_y_end}, x={write_x_start}:{write_x_end}"
                    )
                    print(
                        f"Tile read coords: y={tile_read_y_start}:{tile_read_y_end}, x={tile_read_x_start}:{tile_read_x_end}"
                    )

        print(f"Completed processing all {total_tiles} tiles!")

        # Write the prediction (class indices) to disk
        print(f"Writing prediction to {output_path}...")
        with rasterio.open(output_path, "w", **prediction_meta) as dst:
            dst.write(prediction, 1)

        print(f"Multiclass prediction saved to {output_path}")

        # Write the probability map (confidence) to disk if requested
        if save_probability_map:
            # Handle potential NaN values if desired (e.g., replace with 0 or a specific nodata value)
            # probability_map[np.isnan(probability_map)] = 0 # Example: replace NaN with 0

            # Update prob_meta nodata if necessary
            # prob_meta['nodata'] = -1 # Or some other suitable value if replacing NaN
            # probability_map[np.isnan(probability_map)] = prob_meta['nodata']

            prob_output_path = output_path.with_name(
                f"{output_path.stem}_probability{output_path.suffix}"
            )
            print(f"Writing probability map to {prob_output_path}...")
            with rasterio.open(prob_output_path, "w", **prob_meta) as dst:
                dst.write(probability_map, 1)

            print(f"Probability map saved to {prob_output_path}")

        # Return prediction and probability map (or just prediction if map not saved)
        returned_prob_map = probability_map if save_probability_map else None
        return prediction, returned_prob_map


def get_change_array(
    from_array: np.ndarray,
    to_array: np.ndarray,
    in_nodata: int = 0,
    out_nodata: int = 255,
) -> int:
    """Assigns a transition ID based on the change from one class to another.
    Only works with target classes v3.

    Target classes:
    0: No change
    1: Mature_tree_density_loss
    2: Old_growth_density_loss
    3: Forest_Setback_YoungLoss
    4: Forest_Stage_Progression
    5: Forest_Density_Gain
    6: Early_Forest_Establishment
    7: Clearcut_Loss
    8: Other Transition

    Args:
        from_array (int): Class from which the transition is made.
        to_array (int): Class to which the transition is made.
        in_nodata (int, optional): Value to consider as no data in the input arrays. Defaults to 0.
        out_nodata (int, optional): Value to assign as no data in the output change map. Defaults to 255.

    Returns:
        int: Change map filled with Transition IDs corresponding to the change.
    """

    # Define density levels for easier checking
    high_density = np.array([9, 12, 14, 16, 18])  # DG80-100
    low_density = np.array([8, 11, 13, 15, 17])  # DG0-70
    young_growth = np.array([6, 7])
    pole_wood = np.array([8, 9, 10])
    mature_forest = np.array([11, 12, 13, 14])
    old_forest = np.array([15, 16, 17, 18])
    non_forest_shrubland = np.array([20, 21, 22])
    all_forest_stages = [young_growth, pole_wood, mature_forest, old_forest]

    from_class_forest_stage = np.zeros(from_array.shape, dtype=np.uint8)
    from_class_forest_stage = np.where(np.isin(from_array, young_growth), 1, from_class_forest_stage)
    from_class_forest_stage = np.where(np.isin(from_array, pole_wood), 2, from_class_forest_stage)
    from_class_forest_stage = np.where(np.isin(from_array, mature_forest), 3, from_class_forest_stage)
    from_class_forest_stage = np.where(np.isin(from_array, old_forest), 4, from_class_forest_stage)

    to_class_forest_stage = np.zeros(to_array.shape, dtype=np.uint8)
    to_class_forest_stage = np.where(np.isin(to_array, young_growth), 1, to_class_forest_stage)
    to_class_forest_stage = np.where(np.isin(to_array, pole_wood), 2, to_class_forest_stage)
    to_class_forest_stage = np.where(np.isin(to_array, mature_forest), 3, to_class_forest_stage)
    to_class_forest_stage = np.where(np.isin(to_array, old_forest), 4, to_class_forest_stage)

    # --- 0. All transitions ---
    change_map = np.where((from_array != to_array), 8, 0)

    # --- 1. Clearcutting ---
    change_map = np.where((from_array != 19) & (to_array == 19), 7, change_map)  # 19: Schlagflächen

    # --- 2. Clearcut Regeneration ---
    change_map = np.where((from_array == 19) & (to_class_forest_stage == 1), 6, change_map)

    # --- 3. Major Stage Changes (Development & Setback) ---
    # Development
    change_map = np.where(
        ((from_class_forest_stage == 1) & (to_class_forest_stage == 2))
        | ((from_class_forest_stage == 2) & (to_class_forest_stage == 3))
        | ((from_class_forest_stage == 3) & (to_class_forest_stage == 4))
        | ((from_class_forest_stage == 2) & (to_class_forest_stage == 4)),
        4,
        change_map,
    )
    # Setback (Prioritized over density/type change if stage changes)
    change_map = np.where(
        (
            np.isin(from_array, np.concatenate([pole_wood, mature_forest, old_forest]))
            & (to_class_forest_stage == 1)
        )
        | ((from_class_forest_stage == 4) & (to_class_forest_stage == 3))
        | ((from_class_forest_stage == 4) & (to_class_forest_stage == 2))
        | ((from_class_forest_stage == 3) & (to_class_forest_stage == 2)),
        3,
        change_map,
    )
    from_class_forest_stage = None
    to_class_forest_stage = None

    # --- 4. Density Changes within Same Stage ---
    # Check for Density Gain (Low -> High), fill others with 8: Other Transition
    density_gain = np.where(
        np.isin(from_array, low_density) & np.isin(to_array, high_density),
        5,
        change_map,
    )
    # Check only if the major stage is NOT changing
    density_loss_classes = [3, 3, 1, 2]
    for density_loss_class, stage in zip(density_loss_classes, all_forest_stages):
        # Check for Density Loss (High -> Low), fill others with density_gain
        density_loss_or_gain = np.where(
            np.isin(from_array, high_density) & np.isin(to_array, low_density),
            density_loss_class,
            density_gain,
        )
        # Apply density loss/gain/other transition only if both classes are in the same stage
        change_map = np.where(
            np.isin(from_array, stage) & np.isin(to_array, stage),
            density_loss_or_gain,
            change_map,
        )
    density_gain = None
    density_loss_or_gain = None

    # --- 5. Non-Forest / Shrubland to Forest (if not caught above) ---
    change_map = np.where(
        np.isin(from_array, non_forest_shrubland)
        & np.isin(to_array, np.concatenate(all_forest_stages)),
        6,
        change_map,
    )

    # --- 6. Handle No Data ---
    change_map = np.where(
        (from_array == in_nodata) | (to_array == in_nodata), out_nodata, change_map
    )

    return change_map
    