import rasterio
from typing import Optional, Tuple
from torch import Tensor
import torch
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.samplers import RandomGeoSampler
from ...data.samplers.single_samplers import RandomGeoSamplerIntersectingPolygons
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from ...data.datasets import BinaryChangeDetectionDataset, MultiClassChangeDetectionDataset
import kornia.augmentation as K
import geopandas as gpd
from torchgeo.datasets.utils import BoundingBox
import matplotlib.pyplot as plt
import warnings

class BinaryChangeDetectionDataModule(NonGeoDataModule):
    """NonGeoDataModule for binary change detection between pairs of satellite images.

    This module handles loading and preprocessing pairs of satellite images and their corresponding
    change masks for training change detection models. It supports:
    - Loading train/val/test splits based on ROI (region of interest) files
    - Patch-based sampling from the input images
    - Data augmentation using Kornia transforms
    - Normalization based on per-band statistics
    """

    def __init__(
        self,
        image1_path: str,
        image2_path: str,
        mask_path: str,
        train_roi_path: str,
        val_roi_path: str,
        test_roi_path: str,
        label_poly_path: str,
        num_classes: int = 2,
        patch_size: tuple[int, int] = (256, 256),
        batch_size: int = 32,
        num_workers: int = 4,
        samples_per_epoch: int | None = None,
    ) -> None:
        """Initialize the data module.

        Args:
            image1_path: Path to the first (before) image
            image2_path: Path to the second (after) image
            mask_path: Path to the change mask
            train_roi_path: Path to training region of interest geopackage
            val_roi_path: Path to validation region of interest geopackage
            test_roi_path: Path to test region of interest geopackage
            label_poly_path: Path to the geopackage/shapefile containing label polygons for sampling.
            patch_size: Size of image patches to sample (height, width)
            batch_size: Number of samples per batch
            num_workers: Number of parallel workers for data loading
            samples_per_epoch: Number of patches to sample per epoch (optional)
        """
        super().__init__(BinaryChangeDetectionDataset, batch_size=batch_size, num_workers=num_workers)
        
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.mask_path = mask_path
        self.train_roi_path = train_roi_path
        self.val_roi_path = val_roi_path
        self.test_roi_path = test_roi_path
        self.label_poly_path = label_poly_path
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_epoch = samples_per_epoch

        def get_image_stats_uint8_normalized(image_path: str) -> Tuple[Tensor, Tensor]:
            """Calculate per-band mean and standard deviation statistics for a multi-band image and normalize it to [0,1] range.
            
            Args:
                image_path: Path to the raster input image file in uint8 format.
                
            Returns:
                Tuple containing:
                    - Tensor of normalized mean values for each band
                    - Tensor of normalized standard deviation values for each band
                    
            The statistics are normalized by dividing by 255.0 to get values in [0,1] range.
            """
            with rasterio.open(image_path) as src:
                band_means = []
                band_stds = []
                for band in range(1, src.count + 1):
                    stats = src.statistics(band, approx=False)
                    if src.profile['dtype'] == 'uint8':
                        band_means.append(stats.mean / 255.0)
                        band_stds.append(stats.std / 255.0)
                    else:
                        raise ValueError("Only uint8 images are supported")
                    
            return (
                torch.tensor(band_means),
                torch.tensor(band_stds)
            )
        
        #calculate and store means and stds for the two images in single tensors with 6 channels
        means1, stds1 = get_image_stats_uint8_normalized(self.image1_path)
        means2, stds2 = get_image_stats_uint8_normalized(self.image2_path)
        self.means = torch.cat([means1, means2], dim=0)
        self.stds = torch.cat([stds1, stds2], dim=0)
    
        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.means, std=self.stds),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomResizedCrop(
                size=self.patch_size, scale=(0.8, 1.0), ratio=(1, 1), p=1.0
            ),
            data_keys=['image', 'mask'],
            )
        
        self.val_aug = K.AugmentationSequential(
            K.Normalize(mean=self.means, std=self.stds),
            data_keys=['image', 'mask'],
            same_on_batch=True,
        )
        self.test_aug = K.AugmentationSequential(
            K.Normalize(mean=self.means, std=self.stds),
            data_keys=['image', 'mask'],
            same_on_batch=True,
        )

        # Initialize label_polygons attribute, will be populated in setup
        self.label_polygons = None

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

            # Apply same transform to both input and mask
            transformed = self.train_aug(x, y)
            batch["image"] = transformed[0]  # First output is transformed image
            batch["mask"] = transformed[1].squeeze(1).long()  # Second output is transformed mask
        else:
            # For validation/test, apply normalization to both
            x = batch["image"]  # [B, 6, H, W]
            y = batch["mask"].float().unsqueeze(1)  # Add channel dim [B, 1, H, W]
            
            transformed = self.val_aug(x, y)
            batch["image"] = transformed[0]
            batch["mask"] = transformed[1].squeeze(1).long()

        return batch
    

    def mask_to_long(self, sample: dict) -> dict:
        """Convert mask to PyTorch long tensor for loss functions.
        
        Args:
            sample: Dictionary containing image and mask data
            
        Returns:
            Sample with mask converted to long tensor
        """
        return {**sample, "mask": sample["mask"].long()}
    


    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation and testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """

        train_transforms = Compose([self.mask_to_long])
        test_transforms = Compose([self.mask_to_long])

        print("Setting up datasets...")
        if stage == 'fit' or stage is None:
            # Create datasets with their respective ROIs from geopackage files
            self.train_dataset = BinaryChangeDetectionDataset(
                image1_path=self.image1_path,
                image2_path=self.image2_path,
                mask_path=self.mask_path,
                transforms=train_transforms
            )

            self.val_dataset = BinaryChangeDetectionDataset(
                image1_path=self.image1_path,
                image2_path=self.image2_path,
                mask_path=self.mask_path,
                transforms=test_transforms
            )

            self.test_dataset = BinaryChangeDetectionDataset(
                image1_path=self.image1_path,
                image2_path=self.image2_path,
                mask_path=self.mask_path,
                transforms=test_transforms
            )


            # Load ROIs from geopackage files and convert to BoundingBox objects

            def bounds_to_bbox(bounds):
                return BoundingBox(
                    minx=bounds[0], maxx=bounds[2],
                    miny=bounds[1], maxy=bounds[3],
                    mint=0, maxt=1
                )
            
            rois = {
                'train': self.train_roi_path,
                'val': self.val_roi_path, 
                'test': self.test_roi_path
            }
            
            for split, path in rois.items():
                bounds = gpd.read_file(path).total_bounds
                setattr(self, f'{split}_roi', bounds_to_bbox(bounds))

            # Load label polygons for sampling
            try:
                label_gdf = gpd.read_file(self.label_poly_path)

                # Use the CRS from one of the initialized datasets
                dataset_crs = self.train_dataset.crs
                if label_gdf.crs != dataset_crs:
                    print(f"Reprojecting label polygons from {label_gdf.crs} to {dataset_crs}...")
                    label_gdf = label_gdf.to_crs(dataset_crs)

                self.label_polygons = [
                    geom for geom in label_gdf.geometry.tolist() if geom.is_valid and not geom.is_empty
                ]
                print(f"Loaded {len(self.label_polygons)} valid label polygons.")

                if not self.label_polygons:
                    warnings.warn("No valid label polygons were loaded. Sampler might not yield any samples.")

            except Exception as e:
                print(f"Error loading or processing label polygons: {e}")
                self.label_polygons = []  # Set to empty list to avoid downstream errors
                warnings.warn("Failed to load label polygons. Proceeding without polygon intersection sampling.")

            print("Datasets created successfully")


    def train_dataloader(self):
        """Create the training data loader.
        
        Returns:
            DataLoader for training data
        """        
        if self.label_polygons is None:
            raise RuntimeError("Label polygons have not been loaded. Ensure setup() was called.")
        
        if not self.label_polygons:
            warnings.warn("Training sampler created with no label polygons. May yield no samples.")
            # Fall back to regular RandomGeoSampler if no polygons are available
            sampler = RandomGeoSampler(
                dataset=self.train_dataset,
                size=self.patch_size,
                roi=self.train_roi,
                length=self.samples_per_epoch
            )
        else:
            sampler = RandomGeoSamplerIntersectingPolygons(
                dataset=self.train_dataset,
                label_polygons=self.label_polygons,
                size=self.patch_size,
                roi=self.train_roi,
                length=self.samples_per_epoch
            )
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=False,
        )

    def val_dataloader(self):
        """Create the validation data loader.
        
        Returns:
            DataLoader for validation data
        """        
        val_length = self.samples_per_epoch // 5 if self.samples_per_epoch is not None else 1000
        if val_length == 0 and self.samples_per_epoch is not None:
            val_length = 1  # Ensure at least 1 sample if samples_per_epoch is small
        
        if self.label_polygons is None:
            raise RuntimeError("Label polygons have not been loaded. Ensure setup() was called.")
        
        if not self.label_polygons:
            warnings.warn("Validation sampler created with no label polygons. May yield no samples.")
            # Fall back to regular RandomGeoSampler if no polygons are available
            sampler = RandomGeoSampler(
                dataset=self.val_dataset,
                size=self.patch_size,
                roi=self.val_roi,
                length=val_length
            )
        else:
            sampler = RandomGeoSamplerIntersectingPolygons(
                dataset=self.val_dataset,
                label_polygons=self.label_polygons,
                size=self.patch_size,
                roi=self.val_roi,
                length=val_length
            )
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
            pin_memory=False,
        )


    def test_dataloader(self):
        """Create the test data loader.
        
        Returns:
            DataLoader for test data
        """
        print("Creating test dataloader")
        
        test_length = self.samples_per_epoch // 5 if self.samples_per_epoch is not None else 1000
        if test_length == 0 and self.samples_per_epoch is not None:
            test_length = 1  # Ensure at least 1 sample if samples_per_epoch is small
        
        if self.label_polygons is None:
            raise RuntimeError("Label polygons have not been loaded. Ensure setup() was called.")
        
        if not self.label_polygons:
            warnings.warn("Test sampler created with no label polygons. May yield no samples.")
            # Fall back to regular RandomGeoSampler if no polygons are available
            sampler = RandomGeoSampler(
                dataset=self.test_dataset,
                size=self.patch_size,
                roi=self.test_roi,
                length=test_length
            )
        else:
            sampler = RandomGeoSamplerIntersectingPolygons(
                dataset=self.test_dataset,
                label_polygons=self.label_polygons,
                size=self.patch_size,
                roi=self.test_roi,
                length=test_length
            )
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False
        )


    def plot_sample(self, sample, title=None):
        """Plot a single sample."""
        import matplotlib.pyplot as plt
        from torchgeo.datasets.utils import percentile_normalization
        import numpy as np
        
        # Create figure with three subplots, ensuring equal sizes
        fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3, figure=fig)
        gs.update(wspace=0.05)  # Reduce spacing between subplots
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # Prepare images for visualization
        def prepare_image(img_tensor):
            # Convert to dense tensor if sparse
            if img_tensor.is_sparse:
                img_tensor = img_tensor.to_dense()
            img = img_tensor.permute(1, 2, 0).numpy()
            img = percentile_normalization(img)
            return np.clip(img, 0, 1)
        
        # Plot images
        image1 = prepare_image(sample['image1'])
        image2 = prepare_image(sample['image2'])
        
        # Create mask visualization
        mask = sample['mask'].numpy()
        mask_rgb = np.zeros((*mask.shape, 3))
        mask_rgb[mask == 1] = [1, 0, 0]  # Red for changes
        
        # Plot with equal aspect ratios
        ax1.imshow(image1, aspect='equal')
        ax1.axis('off')
        ax1.set_title('Before (t1)')
        
        ax2.imshow(image2, aspect='equal')
        ax2.axis('off')
        ax2.set_title('After (t2)')
        
        ax3.imshow(mask_rgb, aspect='equal')
        ax3.axis('off')
        ax3.set_title('Changes')
        
        if title:
            plt.suptitle(title)
        
        return fig
    
    def visualize_samples(self, num_samples=4, split='train'):
        """Visualize samples from the datamodule.
        
        Args:
            datamodule: The ChangeDetectionDataModule instance
            num_samples: Number of samples to visualize
            split: Either 'train' or 'val'
        """
        
        # Make sure the datamodule is set up
        if not hasattr(self, 'train_dataset'):
            self.setup()
        
        # Get the appropriate dataloader
        if split == 'train':
            dataloader = self.train_dataloader()
            roi = self.train_roi
            title_prefix = "Training"
        else:
            dataloader = self.val_dataloader()
            roi = self.val_roi
            title_prefix = "Validation"
        
        # Get a batch of data
        batch = next(iter(dataloader))
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 7))
        
        # Plot each sample
        for i in range(min(num_samples, len(batch['image']))):
            # Extract individual sample
            sample = {
                'image1': batch['image'][i, :3],  # First 3 channels
                'image2': batch['image'][i, 3:],  # Last 3 channels
                'mask': batch['mask'][i]
            }
            
            # Plot the sample
            self.plot_sample(sample, title=f"{title_prefix} Sample {i+1}")
        
        plt.tight_layout()

class MultiClassChangeDetectionDataModule(BinaryChangeDetectionDataModule):
    """NonGeoDataModule for multi-class change detection between pairs of satellite images.

    This module handles loading and preprocessing pairs of satellite images and their corresponding
    change masks for training change detection models. It supports:
    - Loading train/val/test splits based on ROI (region of interest) files
    - Patch-based sampling from the input images
    - Data augmentation using Kornia transforms
    - Normalization based on per-band statistics
    """

    def __init__(
        self,
        image1_path: str,
        image2_path: str,
        mask_path: str,
        train_roi_path: str,
        val_roi_path: str,
        test_roi_path: str,
        label_poly_path: str,
        num_classes: int,
        patch_size: tuple[int, int] = (256, 256),
        batch_size: int = 32,
        num_workers: int = 4,
        samples_per_epoch: int | None = None,
    ) -> None:
        """Initialize the data module.

        Args:
            image1_path: Path to the first (before) image
            image2_path: Path to the second (after) image
            mask_path: Path to the change mask
            train_roi_path: Path to training region of interest geopackage
            val_roi_path: Path to validation region of interest geopackage
            test_roi_path: Path to test region of interest geopackage
            patch_size: Size of image patches to sample (height, width)
            batch_size: Number of samples per batch
            num_workers: Number of parallel workers for data loading
            samples_per_epoch: Number of patches to sample per epoch (optional)
            label_poly_path: Path to the geopackage/shapefile containing label polygons.
            num_classes: Total number of classes in the dataset (including background).
        """
        super().__init__(
            image1_path=image1_path,
            image2_path=image2_path,
            mask_path=mask_path,
            train_roi_path=train_roi_path,
            val_roi_path=val_roi_path,
            test_roi_path=test_roi_path,
            label_poly_path=label_poly_path,
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_epoch=samples_per_epoch
        )

        # Store num_classes
        self.num_classes = num_classes

    def plot_sample(self, sample, title=None):
        """Plot a single sample."""
        import matplotlib.pyplot as plt
        from torchgeo.datasets.utils import percentile_normalization
        import numpy as np
        
        # Create figure with three subplots, ensuring equal sizes
        fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3, figure=fig)
        gs.update(wspace=0.05)  # Reduce spacing between subplots
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # Prepare images for visualization
        def prepare_image(img_tensor):
            # Convert to dense tensor if sparse
            if img_tensor.is_sparse:
                img_tensor = img_tensor.to_dense()
            img = img_tensor.permute(1, 2, 0).numpy()
            img = percentile_normalization(img)
            return np.clip(img, 0, 1)
        
        # Plot images
        image1 = prepare_image(sample['image1'])
        image2 = prepare_image(sample['image2'])
        

        # Plot multi-class mask using a colormap
        # Determine number of classes from unique values in mask
        mask = sample['mask'].numpy()
        print(f"Unique mask values in this sample: {np.unique(mask)}") # <-- Add this line
        vmin = 0
        vmax = self.num_classes - 1 # Max possible class index
        
        cmap = plt.cm.get_cmap('viridis', self.num_classes)  # Use detected number of classes

        # Plot with equal aspect ratios
        ax1.imshow(image1, aspect='equal')
        ax1.axis('off')
        ax1.set_title('Before (t1)')
        
        ax2.imshow(image2, aspect='equal')
        ax2.axis('off')
        ax2.set_title('After (t2)')
        
        ax3.imshow(mask, cmap=cmap)
        ax3.axis('off')
        ax3.set_title('Changes')
        
        if title:
            plt.suptitle(title)
        
        return fig
    

    


    