from pathlib import Path

import geopandas as gpd
import pandas as pd
import kornia.augmentation as K
import rasterio
from shapely.geometry import box
import torch
from torch import Generator, Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import BoundingBox, RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, PreChippedGeoSampler
from typing import Tuple

from ...data.datasets import MultiModalDataset
from ...data.datasets.utils import random_grid_cell_assignment
from ...data.samplers import RandomGeoSamplerWithinArea


class SegmentationDataModule(GeoDataModule):
    """Manage data for semantic segmentation tasks.
    - Load raster images and masks inside a RasterDataset.
    - Preprocess data on creation of a dataset.
    - Apply augmentations (flip, resize) after transfer of batch to device.
    - Create Train/Val/Test Datasets with randomly assigned grid cells.

    Args:
        GeoDataModule (GeoDataModule): Base Class
    """

    def __init__(
        self,
        image_paths: dict[Path],
        mask_path: str,
        n_classes: int,
        roi_shape_path: str | None = None,
        batch_size: int = 64,
        patch_size: tuple[int, int] = (256, 256),
        num_workers: int = 6,
        res: float | None = None,
        train_batches_per_epoch: int = 512,
        val_batches_per_epoch: int = 32,
        class_names: list[str] | None = None,
        target_class_definition_path: Path | None = None,
        image_paths_pred: dict[Path] | None = None,
        prediction_mask_path: str | None = None,
    ):
        """Initialize the data module.

        Args:
            image_paths (dict): Paths to input images
            mask_path (str): Path to label mask image
            n_classes (int): Number of target classes without the nodata class
            roi_shape_path (str): Path to ROI geopackage file. If None, full dataset extent is used. Defaults to None.
            batch_size (int, optional): Number of samples per batch. Defaults to 64.
            patch_size (tuple[int, int], optional): Size of image patches to sample (height, width). Defaults to (256, 256).
            num_workers (int, optional): Number of parallel workers for data loading. Defaults to 6.
            res (float, optional): Resolution of the dataset in units of CRS (defaults to the resolution of the first file found). Defaults to None.
            train_batches_per_epoch (int, optional): Number of training batches per epoch. Defaults to 512.
            val_batches_per_epoch (int, optional): Number of validation batches per epoch. Defaults to 32.
            class_names (list[str], optional): List with class names without the nodata class. Used for plot labels. Defaults to None.
            target_class_definition_path (Path, optional): Path to CSV file with target class definitions. Defaults to None.
            image_paths_pred (dict, optional): Paths to input images for prediction. Defaults to None.
            prediction_mask_path (str, optional): Path to the label mask image of the prediction dataset. Defaults to None.
        """
        super().__init__(dataset_class=RasterDataset)

        self.image_paths = image_paths
        self.mask_path = mask_path
        self.roi_shape_path = roi_shape_path
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.res = res
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch
        self.class_names = class_names
        self.image_paths_pred = image_paths_pred
        self.prediction_mask_path = prediction_mask_path

        # Data loaders
        self.predict_batch_size = 1  # For inference, we use batch size of 1

        def get_image_stats(
            image_paths: dict,
        ) -> Tuple[Tensor, Tensor]:
            """Calculate per-band mean and standard deviation statistics for single- or multi-band images.

            Args:
                image_paths: Paths to the raster input image file in uint8 format.

            Returns:
                Tuple containing:
                    - Tensor of normalized mean values for each band
                    - Tensor of normalized standard deviation values for each band
            """
            band_means = []
            band_stds = []

            # Iterate over every input file
            for dataset, path in image_paths.items():
                with rasterio.open(path) as src:
                    stats_all = src.stats(approx=False)
                    for band in range(src.count):
                        # Skip R and G bands for CIR datasets
                        if dataset == "cir" and band > 0:
                            break
                        stats = stats_all[band]
                        band_means.append(stats.mean)
                        band_stds.append(stats.std)

            return (torch.tensor(band_means), torch.tensor(band_stds))

        self.mean, self.std = get_image_stats(self.image_paths)

        self.train_aug_all_before = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomResizedCrop(
                size=self.patch_size, scale=(0.8, 1.0), ratio=(1, 1), p=1.0
            ),
            data_keys=["image", "mask"],
        )

        self.train_aug_img = K.AugmentationSequential(
            K.ColorJiggle(brightness=0.3, contrast=0.5, p=0.8),
            data_keys=["image"],
        )

        self.train_aug_rgb = K.AugmentationSequential(
            K.RandomHue(hue=(-0.05, 0.05), p=0.5),
            K.RandomSaturation(saturation=(0.7, 1.3), p=0.5),
            data_keys=["image"],
        )

        self.train_aug_all_after = K.AugmentationSequential(
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.1),
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image", "mask"],
        )

        self.val_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image", "mask"],
            same_on_batch=True,
        )
        self.test_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image", "mask"],
            same_on_batch=True,
        )

        # Read class names if target class definition path is provided
        if target_class_definition_path is not None:
            self.class_names = pd.read_csv(target_class_definition_path)[
                "Zielklasse 2024"
            ].to_list()

        self.in_channels = (
            len(image_paths.keys())
            + (2 if "rgb" in image_paths else 0)
            + (3 if "spot-4" in image_paths else 0)
        )  # add bands because rgb and spot contain more channels

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply augmentations after batch is transferred to device.

        Args:
            batch: Dictionary containing image and mask data
            dataloader_idx: Index of the dataloader

        Returns:
            Batch with augmentations applied
        """

        if self.trainer.training:
            # Apply training augmentations to both images and masks
            x = batch["image"]  # [B, 6, H, W]
            y = batch["mask"].float().unsqueeze(1)  # Add channel dim [B, 1, H, W]

            # Apply same geometric transforms to both input and mask
            transformed = self.train_aug_all_before(x, y)
            # Apply intensity transforms to image bands only (Have to be rescaled to [0,1] first)
            transformed[0][:, self.dataset.is_image_band, :, :] = self.train_aug_img(
                transformed[0][:, self.dataset.is_image_band, :, :] / 255.0
            )
            # Apply RGB specific augmentations
            if "rgb" in self.image_paths:
                transformed[0][:, [0, 1, 2], :, :] = self.train_aug_rgb(
                    transformed[0][:, [0, 1, 2], :, :]
                )
            transformed[0][:, self.dataset.is_image_band, :, :] *= 255.0
            # Normalize
            transformed = self.train_aug_all_after(transformed[0], transformed[1])

            batch["image"] = transformed[0]  # First output is transformed image
            batch["mask"] = (
                transformed[1].squeeze(1).long()
            )  # Second output is transformed mask
        else:
            # For validation/test, apply normalization to both
            x = batch["image"]  # [B, 6, H, W]
            y = batch["mask"].float().unsqueeze(1)  # Add channel dim [B, 1, H, W]

            transformed = self.val_aug(x, y)
            batch["image"] = transformed[0]
            batch["mask"] = transformed[1].squeeze(1).long()

        return batch

    def setup(
        self,
        stage: str = "fit",
        roi: BoundingBox | None = None,
        stride: int | None = None,
    ):
        """Set up datasets and samplers.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate", "test", "predict_on_test_ds"] and self.train_dataset is None:
            self.dataset = MultiModalDataset(
                image_paths=self.image_paths,
                mask_path=self.mask_path,
                res=self.res,
            )

            # Create Train/Val/Test Datasets with randomly assigned grid cells
            if self.roi_shape_path is None:
                self.roi_shape = box(
                    minx=self.dataset.bounds.minx,
                    miny=self.dataset.bounds.miny,
                    maxx=self.dataset.bounds.maxx,
                    maxy=self.dataset.bounds.maxy,
                )
            else:
                gdf = gpd.read_file(self.roi_shape_path)
                self.roi_shape = gdf.geometry.unary_union
            generator = Generator().manual_seed(0)
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = random_grid_cell_assignment(
                self.dataset,
                [0.7, 0.15, 0.15],
                roi_shape=self.roi_shape,
                grid_size=12,
                generator=generator,
            )  # here is the random grid cell split
            print(
                f"Assigned {self.train_dataset.__len__()}/{self.val_dataset.__len__()}/{self.test_dataset.__len__()} cells to train/val/test datasets."
            )
        if stage in ["fit"]:
            self.train_sampler = RandomGeoSamplerWithinArea(
                self.train_dataset,
                size=self.patch_size,
                roi_shape=self.roi_shape,
                length=self.train_batches_per_epoch * self.batch_size,
                roi=None if not roi else roi,
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = RandomGeoSamplerWithinArea(
                self.val_dataset,
                size=self.patch_size,
                roi_shape=self.roi_shape,
                length=self.val_batches_per_epoch * self.batch_size,
                roi=None if not roi else roi,
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset,
                size=self.patch_size,
                stride=self.patch_size,
                roi=None if not roi else roi,
            )
        if stage in ["predict"]:
            if self.image_paths_pred is None:
                raise ValueError(
                    "Argument 'image_paths_pred' must be provided for prediction stage."
                )
            if stride is None:
                raise ValueError(
                    "Argument 'stride' must be provided for prediction stage."
                )

            self.predict_dataset = MultiModalDataset(
                image_paths=self.image_paths_pred,
                mask_path=self.mask_path,
                res=self.res,
            )

            if not hasattr(self, "roi_shape"):
                self.roi_shape = None

            if self.roi_shape_path is not None and self.roi_shape is None:
                gdf = gpd.read_file(self.roi_shape_path)
                self.roi_shape = gdf.geometry.unary_union

            if roi is None and self.roi_shape is not None:
                bounds = self.roi_shape.bounds
                roi = BoundingBox(
                    minx=bounds[0],
                    miny=bounds[1],
                    maxx=bounds[2],
                    maxy=bounds[3],
                    mint=self.predict_dataset.bounds.mint,
                    maxt=self.predict_dataset.bounds.maxt
                )
                print(f"Using ROI from roi_shape_path: {bounds}")
            elif roi is None:
                roi = self.predict_dataset.bounds
                print("Using full dataset bounds for prediction")

            self.predict_sampler = GridGeoSampler(
                self.predict_dataset,
                size=self.patch_size,
                stride=stride,
                roi=roi,
            )
        if stage in ["predict_on_test_ds"]:
            if stride is None:
                raise ValueError(
                    "Argument 'stride' must be provided for prediction stage."
                )

            self.predict_dataset = self.test_dataset

            self.predict_sampler = GridGeoSampler(
                self.predict_dataset,
                size=self.patch_size,
                stride=stride,
                roi=self.predict_dataset.bounds if not roi else roi,
            )

    def cell_dataloader(self):
        """Create dataloaders for whole datasets without batching.

        Returns:
            DataLoaders for train/val/test data
        """
        train = DataLoader(
            self.train_dataset,
            sampler=PreChippedGeoSampler(
                self.train_dataset,
            ),
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )
        val = DataLoader(
            self.val_dataset,
            sampler=PreChippedGeoSampler(
                self.val_dataset,
            ),
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )
        test = DataLoader(
            self.test_dataset,
            sampler=PreChippedGeoSampler(
                self.test_dataset,
            ),
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

        return (train, val, test)

    def visualize_samples(
        self,
        num_samples: int = 4,
        split="train",
        bright: float = 1.0,
        cols: int = 2,
        chnls: list[int] = [0, 1, 2],
    ):
        """Visualize samples from the datamodule.

        Args:
            num_samples (int, optional): Number of samples to visualize. Defaults to 4.
            split (str, optional): Which dataset to sample from. Defaults to "train".
            bright (float, optional): Brightness adjustment. Defaults to 1.0.
            cols (int, optional): Number of cols (each including a image and mask pair). Defaults to 2.
            chnls (list[int], optional): Channels to visualize. Defaults to [0, 1, 2].
        """
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        from torchgeo.datasets import unbind_samples

        # Make sure the datamodule is set up
        if not hasattr(self, "train_dataset"):
            self.setup()

        # Get the appropriate dataloader
        if split == "train":
            dataloader = self.train_dataloader()
        else:
            dataloader = self.val_dataloader()

        # Get a batch of data
        batch = next(iter(dataloader))

        # Get the samples and the number of items in the batch
        samples = unbind_samples(batch.copy())

        # if batch contains images and masks, the number of images will be doubled
        n = (
            2 * len(samples)
            if ("image" in batch) and ("mask" in batch)
            else len(samples)
        )

        # calculate the number of rows in the grid
        rows = num_samples // cols + (1 if n % cols != 0 else 0)

        # Define colormap for labels
        if self.n_classes == 1:
            cmap = ListedColormap(["white", "blue"])
        else:
            cmap = "hsv"

        # Create a figure with subplots
        _, axs = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 4))

        # Plot each sample
        for it in range(num_samples):
            arr = (
                torch.clamp(samples[it]["image"] / 255.0 * bright, min=0, max=1)
                .squeeze(0)
                .numpy()
            )
            rgb = arr.transpose(1, 2, 0)[:, :, chnls]
            row = it // cols
            col = it % cols

            ax_img = axs[row, col * 2]
            ax_img.imshow(rgb)
            ax_img.axis("off")

            ax_msk = axs[row, col * 2 + 1]
            ax_msk.imshow(
                samples[it]["mask"].squeeze().numpy(),
                cmap=cmap,
                vmin=0,
                vmax=self.n_classes,
            )
            ax_msk.axis("off")
