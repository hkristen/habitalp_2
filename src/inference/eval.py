import gc
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window
from torchgeo.datasets import BoundingBox
from tqdm import tqdm

import wandb

BASE_FOLDER = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_FOLDER))

from src.data.datamodules.unet_change import SegmentationDataModule
from src.trainers.unet_segmentation.train import get_datamodule
from src.trainers.unet_segmentation.unet_segmentation import (
    MultiClassSemanticSegmentationTask,
)
from src.trainers.utils import (
    compute_final_metrics,
    get_change_array,
    load_config_from_yaml,
    read_intersecting_area_of_two_rasters,
)


def get_blend_mask(patch_size: int, overlap: int) -> np.ndarray:
    """Create cosine-weighted blend mask for overlapping patches.

    Generates a 2D weight mask where center regions have weight 1.0 and edges
    taper to 0.0 using a cosine ramp. This produces smooth transitions when
    blending overlapping patches.

    Args:
        patch_size: Size of the square patch in pixels.
        overlap: Width of overlap region on each side in pixels.

    Returns:
        2D numpy array of shape (patch_size, patch_size) with blend weights.
    """
    # Create vertical ramp
    y_pos = np.arange(patch_size)
    y = np.ones_like(y_pos, dtype=np.float32)
    if overlap > 0:
        # Cosine ramp: 0.0 at edge -> 1.0 at overlap distance
        ramp = np.cos(np.pi * (y_pos[:overlap] + 1) / (overlap + 1)) / 2 + 0.5
        y[:overlap] = ramp[::-1]  # Top edge
        y[-overlap:] = ramp  # Bottom edge

    # Create horizontal ramp
    x_pos = np.arange(patch_size)
    x = np.ones_like(x_pos, dtype=np.float32)
    if overlap > 0:
        ramp = np.cos(np.pi * (x_pos[:overlap] + 1) / (overlap + 1)) / 2 + 0.5
        x[:overlap] = ramp[::-1]  # Left edge
        x[-overlap:] = ramp  # Right edge

    # 2D mask = outer product
    mask = y[:, None] * x[None, :]

    # Small epsilon to avoid division by zero
    mask += 1e-6

    return mask


def weighted_merge(
    patch_files: list[str],
    output_path: str,
    patch_size: int,
    overlap: int,
    chunk_size: int = 4096,
) -> None:
    """Merge overlapping prediction patches with cosine-weighted blending.

    Implements memory-efficient merging optimized for large-scale predictions:
    - Spatial indexing for O(n) instead of O(n²) patch lookups
    - On-demand patch reading to minimize memory usage
    - Chunked output processing to handle arbitrarily large outputs
    - Cosine-weighted blending at overlaps for seamless mosaicking

    Args:
        patch_files: List of paths to patch GeoTIFF files containing logits.
        output_path: Path where merged output GeoTIFF will be saved.
        patch_size: Size of each square patch in pixels.
        overlap: Width of overlap region on each patch side in pixels.
        chunk_size: Size of output chunks for memory-efficient processing.
    """
    print(f"Weighted merge with {len(patch_files)} patches...")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Overlap: {overlap}px per side")

    blend_mask = get_blend_mask(patch_size, overlap)

    with rasterio.open(patch_files[0]) as src:
        meta = src.meta.copy()

    print("  Reading patch metadata...")
    patch_info = []
    num_classes = None
    for pfile in tqdm(patch_files, desc="Reading metadata"):
        with rasterio.open(pfile) as src:
            bounds = src.bounds
            transform_matrix = src.transform
            if num_classes is None:
                num_classes = src.count
                print(f"  Detected {num_classes} classes")
            patch_info.append({
                'path': pfile,
                'bounds': bounds,
                'transform': transform_matrix
            })

    all_bounds = [p['bounds'] for p in patch_info]
    out_minx = min(b.left for b in all_bounds)
    out_miny = min(b.bottom for b in all_bounds)
    out_maxx = max(b.right for b in all_bounds)
    out_maxy = max(b.top for b in all_bounds)

    res_x = meta['transform'][0]
    res_y = -meta['transform'][4]
    out_width = int((out_maxx - out_minx) / res_x)
    out_height = int((out_maxy - out_miny) / res_y)

    out_transform = rasterio.transform.from_origin(out_minx, out_maxy, res_x, res_y)

    print(f"  Output size: {out_width}x{out_height}")

    meta.update({
        'count': 1,
        'dtype': rasterio.uint8,
        'height': out_height,
        'width': out_width,
        'transform': out_transform,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'nodata': 0,
    })

    print("  Building spatial index...")
    patch_grid = defaultdict(list)
    grid_size = chunk_size * 2

    for idx, pinfo in enumerate(patch_info):
        pbounds = pinfo['bounds']
        grid_col_start = int((pbounds.left - out_minx) / res_x / grid_size)
        grid_col_end = int((pbounds.right - out_minx) / res_x / grid_size)
        grid_row_start = int((out_maxy - pbounds.top) / res_y / grid_size)
        grid_row_end = int((out_maxy - pbounds.bottom) / res_y / grid_size)

        for gr in range(grid_row_start, grid_row_end + 1):
            for gcol in range(grid_col_start, grid_col_end + 1):
                patch_grid[(gr, gcol)].append(idx)

    print(f"  Indexed {len(patch_info)} patches into {len(patch_grid)} grid cells")
    with rasterio.open(output_path, 'w', **meta) as dst:
        n_chunks_y = int(np.ceil(out_height / chunk_size))
        n_chunks_x = int(np.ceil(out_width / chunk_size))
        total_chunks = n_chunks_y * n_chunks_x

        print(f"  Processing {total_chunks} output chunks ({n_chunks_y}x{n_chunks_x})...")

        for chunk_y in tqdm(range(n_chunks_y), desc="Merging chunks"):
            for chunk_x in range(n_chunks_x):
                col_off = chunk_x * chunk_size
                row_off = chunk_y * chunk_size
                width = min(chunk_size, out_width - col_off)
                height = min(chunk_size, out_height - row_off)

                chunk_window = Window(col_off, row_off, width, height)
                chunk_bounds = rasterio.windows.bounds(chunk_window, out_transform)

                accumulated_pred_chunk = np.zeros((num_classes, height, width), dtype=np.float32)
                accumulated_weight_chunk = np.zeros((height, width), dtype=np.float32)

                grid_col = int(col_off / grid_size)
                grid_row = int(row_off / grid_size)
                relevant_patch_indices = set()
                for gr in range(max(0, grid_row - 1), grid_row + 2):
                    for gcol in range(max(0, grid_col - 1), grid_col + 2):
                        relevant_patch_indices.update(patch_grid.get((gr, gcol), []))

                for pidx in relevant_patch_indices:
                    pinfo = patch_info[pidx]
                    pbounds = pinfo['bounds']

                    if (pbounds.right <= chunk_bounds[0] or pbounds.left >= chunk_bounds[2] or
                        pbounds.top <= chunk_bounds[1] or pbounds.bottom >= chunk_bounds[3]):
                        continue

                    with rasterio.open(pinfo['path']) as src:
                        patch_data = src.read()

                    patch_col_start = int((pbounds.left - out_minx) / res_x)
                    patch_row_start = int((out_maxy - pbounds.top) / res_y)

                    overlap_col_start = max(0, patch_col_start - col_off)
                    overlap_row_start = max(0, patch_row_start - row_off)
                    overlap_col_end = min(width, patch_col_start + patch_size - col_off)
                    overlap_row_end = min(height, patch_row_start + patch_size - row_off)

                    patch_col_start_local = max(0, col_off - patch_col_start)
                    patch_row_start_local = max(0, row_off - patch_row_start)
                    patch_col_end_local = patch_col_start_local + (overlap_col_end - overlap_col_start)
                    patch_row_end_local = patch_row_start_local + (overlap_row_end - overlap_row_start)

                    patch_region = patch_data[
                        :,
                        patch_row_start_local:patch_row_end_local,
                        patch_col_start_local:patch_col_end_local
                    ]
                    weight_region = blend_mask[
                        patch_row_start_local:patch_row_end_local,
                        patch_col_start_local:patch_col_end_local
                    ]

                    accumulated_pred_chunk[
                        :,
                        overlap_row_start:overlap_row_end,
                        overlap_col_start:overlap_col_end
                    ] += patch_region.astype(np.float32) * weight_region[None, :, :]

                    accumulated_weight_chunk[
                        overlap_row_start:overlap_row_end,
                        overlap_col_start:overlap_col_end
                    ] += weight_region

                blended_logits_chunk = accumulated_pred_chunk / (accumulated_weight_chunk[None, :, :] + 1e-8)
                final_pred_chunk = np.argmax(blended_logits_chunk, axis=0).astype(np.uint8)
                dst.write(final_pred_chunk, 1, window=chunk_window)

                # Force garbage collection every 5 chunks to prevent heap fragmentation
                chunk_num = chunk_y * n_chunks_x + chunk_x
                if chunk_num % 5 == 0 and chunk_num > 0:
                    gc.collect()

    print("  Weighted merge complete")


def infer_on_whole_image(
    datamodule: SegmentationDataModule,
    task,
    experiment_dir: str,
    patch_size: int | None = 256,
    overlap: int = 64,
    delta: int = 0,
    predict_on_test_ds: bool = False,
    roi: BoundingBox | None = None,
    output_filename: str = "prediction",
    inference_batch_size: int = 1,
    chunk_size: int = 4096,
    compress_output: bool = True,
):
    """Perform inference on entire image using trained model and save predictions.

    Generates predictions for the full image by processing it in patches with
    weighted blending at overlaps. Patch files are compressed to save disk space,
    merged into an uncompressed tiled COG for efficiency, then optionally compressed.

    Args:
        datamodule: Datamodule providing data access and preprocessing.
        task: Trained model task (UNet or terratorch) for inference.
        experiment_dir: Directory to save predictions and temporary files.
        patch_size: Size of square patches for inference. If None, uses datamodule default.
        overlap: Overlap width in pixels on each patch side for blending.
        delta: Pixels to crop from patch edges to remove unreliable predictions.
        predict_on_test_ds: If True, predict on test dataset instead of prediction dataset.
        roi: Optional bounding box to limit predictions to region of interest.
        output_filename: Base name for output prediction file (without extension).
        inference_batch_size: Number of patches to process in parallel on GPU.
        chunk_size: Size of chunks for memory-efficient output merging.
        compress_output: If True, compress final output with LZW after merge (default: True).
    """

    if not patch_size:
        patch_size = datamodule.patch_size[0]
    stride = patch_size - (overlap * 2)

    datamodule.setup(stage="predict_on_test_ds" if predict_on_test_ds else "predict", roi=roi, stride=stride)
    ds_pred = datamodule.predict_dataloader()
    res = datamodule.predict_dataset.res

    print("Create Predictions ...")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Overlap: {overlap}px per side")
    print(f"  Delta cropping: {delta}px")
    print(f"  Inference batch size: {inference_batch_size}")

    # Create unique temp folder to avoid conflicts with parallel runs
    unique_id = str(uuid.uuid4())[:8]
    pred_folder = Path(experiment_dir) / f"tmp_patches_{unique_id}"
    Path(pred_folder).mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_terratorch_model = (
        hasattr(task, '__class__') and
        'terratorch' in task.__class__.__module__
    )

    total_patches = len(ds_pred)
    print(f"Processing {total_patches} patches with batch size {inference_batch_size}...")

    batch_samples = []
    batch_metadata = []
    patches_processed = 0

    for it, sample in tqdm(enumerate(ds_pred), total=total_patches, desc="Inference"):
        sample["image"] = sample["image"].to(device)
        x = sample["image"]
        y = sample["mask"].float().unsqueeze(1)

        transformed = datamodule.val_aug(x, y)
        sample["image"] = transformed[0]
        sample["mask"] = transformed[1].squeeze(1).long()

        batch_samples.append(sample["image"])
        batch_metadata.append({
            'crs': sample["crs"][0],
            'bbox': sample["bounds"][0],
            'mask': sample["mask"],
            'x': x,
            'index': it
        })

        if len(batch_samples) >= inference_batch_size or it == total_patches - 1:
            batch_images = torch.cat(batch_samples, dim=0)

            if is_terratorch_model:
                device_model = next(task.parameters()).device
                filtered_sample = {"image": batch_images.to(device_model)}
                y_hat_batch = task.predict_step(filtered_sample, 0)

                if isinstance(y_hat_batch, tuple) and len(y_hat_batch) > 0 and isinstance(y_hat_batch[0], tuple):
                    y_hat_batch = y_hat_batch[0][0]

                num_classes = datamodule.n_classes + 1
                y_hat_batch_onehot = torch.nn.functional.one_hot(
                    y_hat_batch.long(), num_classes=num_classes
                )
                y_hat_batch = y_hat_batch_onehot.permute(0, 3, 1, 2).float()
            else:
                y_hat_batch = task.model(batch_images)

            for i, meta in enumerate(batch_metadata):
                logits = y_hat_batch[i].detach().cpu().numpy()
                num_classes = logits.shape[0]

                if delta > 0:
                    logits_cropped = logits[:, delta:-delta, delta:-delta]
                    cropped_size = patch_size - 2 * delta
                    out_transform = rasterio.transform.from_origin(
                        meta['bbox'].minx + delta * res[0],
                        meta['bbox'].maxy - delta * res[1],
                        res[0], res[1]
                    )
                else:
                    logits_cropped = logits
                    cropped_size = patch_size
                    out_transform = rasterio.transform.from_origin(
                        meta['bbox'].minx, meta['bbox'].maxy, res[0], res[1]
                    )

                with rasterio.open(
                    pred_folder / f"{meta['index']}.tif",
                    "w",
                    driver="GTiff",
                    transform=out_transform,
                    crs=meta['crs'],
                    count=num_classes,
                    width=cropped_size,
                    height=cropped_size,
                    dtype=rasterio.float32,
                    compress="lzw",
                    tiled=True,
                    predictor=2,
                ) as dst:
                    dst.write(logits_cropped)

                patches_processed += 1

            batch_samples = []
            batch_metadata = []

            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Merging predictions with weighted blending...")
    print("=" * 80)
    files = glob(f"{pred_folder}/*.tif")

    effective_patch_size = patch_size - 2 * delta

    # Merge to uncompressed tiled COG first (fast constant-time writes)
    uncompressed_output = f"{experiment_dir}/{output_filename}_uncompressed.tif"
    final_output = f"{experiment_dir}/{output_filename}.tif"

    weighted_merge(
        patch_files=files,
        output_path=uncompressed_output,
        patch_size=effective_patch_size,
        overlap=overlap,
        chunk_size=chunk_size,
    )

    # Optional: Compress the merged output with gdal_translate
    if compress_output:
        print("\n" + "=" * 80)
        print("Compressing final output with LZW...")
        print("=" * 80)
        try:
            subprocess.run(
                [
                    "gdal_translate",
                    "-co", "COMPRESS=LZW",
                    "-co", "TILED=YES",
                    "-co", "BLOCKXSIZE=256",
                    "-co", "BLOCKYSIZE=256",
                    "-co", "PREDICTOR=2",
                    "-co", "BIGTIFF=YES",
                    uncompressed_output,
                    final_output
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Compressed output saved to: {final_output}")
            Path(uncompressed_output).unlink()
            print("Removed uncompressed intermediate file")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Compression failed: {e.stderr}")
            print(f"Keeping uncompressed output: {uncompressed_output}")
            Path(uncompressed_output).rename(final_output)
        except FileNotFoundError:
            print("Warning: gdal_translate not found. Keeping uncompressed output.")
            Path(uncompressed_output).rename(final_output)
    else:
        Path(uncompressed_output).rename(final_output)
        print(f"Uncompressed output saved to: {final_output}")

    print("Removing temporary patch files ...")
    shutil.rmtree(pred_folder)


def generate_change_map(
    mask_path: Path, prediction_path: Path, output_path: Path, n_batches: int = 20
):
    """Generate land cover change map by comparing reference mask and prediction.

    Processes rasters in batches to manage memory usage for large files.

    Args:
        mask_path: Path to reference mask GeoTIFF file.
        prediction_path: Path to prediction GeoTIFF file.
        output_path: Path where change map GeoTIFF will be saved.
        n_batches: Number of horizontal batches to split processing into.
    """
    mask, prediction, output_meta = read_intersecting_area_of_two_rasters(
        mask_path, prediction_path
    )
    in_nodata = output_meta.get("nodata", 0)
    output_meta.update(
        {
            "compress": "lzw",
            "nodata": 255,
            "tiled": True,
            "predictor": 2,
            "bigtiff": True,
        }
    )

    height, width = mask.shape
    chunk_size = height // n_batches

    with rasterio.open(output_path, "w", **output_meta) as dst:
        for i in range(n_batches):
            # Generate batches
            row_start = i * chunk_size
            row_end = (i + 1) * chunk_size if i < n_batches - 1 else height
            mask_batch = mask[row_start:row_end]
            pred_batch = prediction[row_start:row_end]

            # Generate change map
            change_chunk = get_change_array(mask_batch, pred_batch, in_nodata=in_nodata, out_nodata=output_meta["nodata"])

            # Save change map chunk
            window = rasterio.windows.Window(0, row_start, width, row_end - row_start)
            dst.write(change_chunk, 1, window=window)


def main():
    """Execute full inference pipeline: predict, generate change map, compute metrics.

    Loads model and configuration, runs inference on entire image, generates
    change map, and logs results to Weights & Biases.
    """
    experiment_name = "best_parameters_3"
    experiment_dir = BASE_FOLDER / "models" / experiment_name

    # Load config
    config = load_config_from_yaml(experiment_dir / "config.yaml")

    # Load task
    with open(experiment_dir / "model_paths.yaml", "r") as file:
        model_paths = yaml.safe_load(file)
    model_path = experiment_dir / model_paths["best_model_path"]
    wandb_project = model_paths["wandb_project"]
    wandb_run_id = model_paths["wandb_run_id"]

    task = MultiClassSemanticSegmentationTask.load_from_checkpoint(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task.to(device).eval()

    # Get datamodule
    datamodule = get_datamodule(
        config=config
    )

    # Start inference
    infer_on_whole_image(
        datamodule=datamodule,
        task=task,
        experiment_dir=experiment_dir,
        overlap=64,
        predict_on_test_ds=False,
        output_filename="prediction",
    )

    # Get Change Map
    print("Generating Change Map ...")
    generate_change_map(
        mask_path=datamodule.mask_path,
        prediction_path=experiment_dir / "prediction.tif",
        output_path=experiment_dir / "change_map.tif",
    )

    # Compute and log change metrics
    print("Computing final metrics ...")
    class_names = [
        "No change",
        "Mature Tree Density Loss",
        "Old Growth Density Loss",
        "Forest Setback YoungLoss",
        "Forest Stage Progression",
        "Forest Density Gain",
        "Early Forest Establishment",
        "Clearcut Loss",
        "Other Transition"]
    num_classes = len(class_names)

    multiclass_metrics, metrics_each_label, figure_collection = compute_final_metrics(
        reference_change_map=Path(config["data_folder"]) / "habitalp_change/habitalp_change_2013_2020.tif",
        prediction_change_map=experiment_dir / "change_map.tif",
        num_classes=num_classes, # 9 classes to take no change and index into account
        class_names=class_names,
    )
    with wandb.init(project=wandb_project, id=wandb_run_id, resume="must") as run:
        metric_name = "change_metrics"
        # log Multiclass metrics
        run.summary[metric_name] = multiclass_metrics

        # Log metrics for each class
        metrics_each_label.pop("ConfusionMatrix")
        tensor_stack = torch.stack(list(metrics_each_label.values())).T
        column_lists = [row.tolist() for row in tensor_stack]
        change_metrics_each_class_table = wandb.Table(
            columns=list(metrics_each_label.keys()),
            data=column_lists,
        )
        run.log({f"{metric_name}/table_each_class": change_metrics_each_class_table})

        # Log figures
        run.log({f"{metric_name}/confusion_matrix": wandb.Image(figure_collection[0])})
        run.log({f"{metric_name}/accuracy_each_class": wandb.Image(figure_collection[1])})
        run.log({f"{metric_name}/jaccard_each_class": wandb.Image(figure_collection[2])})
        run.log({f"{metric_name}/precision_each_class": wandb.Image(figure_collection[3])})
        run.log({f"{metric_name}/recall_each_class": wandb.Image(figure_collection[4])})
        run.log({f"{metric_name}/f1score_each_class": wandb.Image(figure_collection[5])})

if __name__ == '__main__':
    main()
