from typing import Any, Sequence, Callable
import torch
from torch import Tensor
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import percentile_normalization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class BinaryChangeDetectionDataset(RasterDataset):
    """Dataset for detecting binary changes between pairs of GeoTIFF images.

    A dataset that loads pairs of GeoTIFF images and their corresponding change mask.
    The images are concatenated along the channel dimension to create a 6-channel input.
    
    Attributes:
        is_image: Flag indicating if this is an image dataset
        image1_path: Path to the first (before) image
        image2_path: Path to the second (after) image 
        mask_path: Path to the change mask
    """
    
    is_image: bool = False
    
    def __init__(
        self,
        image1_path: str,
        image2_path: str,
        mask_path: str,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            image1_path: Path to the first (before) image
            image2_path: Path to the second (after) image
            mask_path: Path to the change mask
            bands: Names of bands to load
            transforms: Callable to transform the data samples
            cache: Whether to cache the index
        """
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.mask_path = mask_path
        
        super().__init__(
            paths=[image1_path, image2_path, mask_path],
            bands=bands,
            transforms=transforms,
            cache=cache
        )


    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Get a data sample for a given bounding box.
        
        Args:
            query: Bounding box defining the region to load
            
        Returns:
            Dictionary containing:
                - image: Concatenated image pair tensor [6, H, W]
                - mask: Binary change mask tensor [H, W]
                
        Raises:
            IndexError: If query bounds are not found in the dataset
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(f'query: {query} not found in index with bounds: {self.bounds}')

        # Load both images and mask
        image1 = self._merge_files([self.image1_path], query, self.band_indexes)  # [3, H, W]
        image2 = self._merge_files([self.image2_path], query, self.band_indexes)  # [3, H, W]
        mask = self._merge_files([self.mask_path], query)  # [1, H, W]

        # Concatenate images along channel dimension to get 6 channels
        image = torch.cat([image1, image2], dim=0)  # [6, H, W]
        image = image.float() / 255.0

       # Remove channel dimension, keep original integer class labels
        mask = mask.squeeze(0)  # [H, W]

        sample = {
            'image': image,  # [6, H, W]
            'mask': mask,    # [H, W]
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
    

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Creates a figure with two subplots showing the before and after images.
        The after image includes an overlay of the change mask in green.

        Args:
            sample: A sample returned by __getitem__ containing:
                   - image: Concatenated image tensor [6, H, W]
                   - mask: Binary change mask tensor [H, W]
            show_titles: Flag indicating whether to show titles above each panel
            suptitle: Optional suptitle to use for figure

        Returns:
            A matplotlib Figure with the rendered sample
        """
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

        # Normalize and prepare images ensuring RGB order
        def prepare_image(img_tensor: torch.Tensor) -> np.ndarray:
            """Prepare image for visualization with correct RGB ordering.
            
            Args:
                img_tensor: Input image tensor in [C, H, W] format
                
            Returns:
                Normalized numpy array in [H, W, C] format with values in [0,1]
            """
            # img_tensor is in (C,H,W) format with R=0, G=1, B=2
            img = img_tensor.permute(1, 2, 0).numpy()  # to (H,W,C)
            # Normalize each channel independently
            img = percentile_normalization(img, axis=(0, 1))
            # Ensure values are in [0,1] for proper display
            return np.clip(img, 0, 1)

        split_images = int(sample['image'].shape[0] / 2)

        image1 = prepare_image(sample['image'][:split_images])
        image2 = prepare_image(sample['image'][split_images:])

        # Create mask overlay
        mask = sample['mask'].numpy()
        mask_rgba = np.zeros((*mask.shape, 4))
        mask_rgba[mask == 1] = [0, 1, 0, 0.6]  # Green with 60% opacity where mask == 1

        # Plot before image
        axs[0].imshow(image1)
        axs[0].axis('off')
        
        # Plot after image with change overlay
        axs[1].imshow(image2)
        axs[1].imshow(mask_rgba)
        axs[1].axis('off')

        if show_titles:
            axs[0].set_title('Before (t1)')
            axs[1].set_title('After (t2) with Changes')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
    
class MultiClassChangeDetectionDataset(BinaryChangeDetectionDataset):

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        num_classes: int = 9,
    ) -> Figure:
        """Plot a sample from the MultiClass dataset.

        Creates a figure with two subplots showing the before and after images.
        The after image includes an overlay of the change mask in various colors based on the class.

        Args:
            sample: A sample returned by __getitem__ containing:
                   - image: Concatenated image tensor [6, H, W]
                   - mask: Multi-class change mask tensor [H, W]
            show_titles: Flag indicating whether to show titles above each panel
            suptitle: Optional suptitle to use for figure

        Returns:
            A matplotlib Figure with the rendered sample
        """
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

        # Normalize and prepare images ensuring RGB order
        def prepare_image(img_tensor: torch.Tensor) -> np.ndarray:
            """Prepare image for visualization with correct RGB ordering.
            
            Args:
                img_tensor: Input image tensor in [C, H, W] format
                
            Returns:
                Normalized numpy array in [H, W, C] format with values in [0,1]
            """
            # img_tensor is in (C,H,W) format with R=0, G=1, B=2
            img = img_tensor.permute(1, 2, 0).numpy()  # to (H,W,C)
            # Normalize each channel independently
            img = percentile_normalization(img, axis=(0, 1))
            # Ensure values are in [0,1] for proper display
            return np.clip(img, 0, 1)
        
        split_images = int(sample['image'].shape[0] / 2)

        image1 = prepare_image(sample['image'][:split_images])
        image2 = prepare_image(sample['image'][split_images:])

        # Create mask overlay
        mask = sample['mask'].numpy()

               # Plot before image
        axs[0].imshow(image1)
        axs[0].axis('off')

        # Plot after image
        axs[1].imshow(image2)
        axs[1].axis('off')

        # Plot multi-class mask using a colormap
        # Determine number of classes from unique values in mask

        vmin = 0
        vmax = num_classes
        
        cmap = plt.cm.get_cmap('viridis', num_classes)  # Use detected number of classes

        # Overlay mask on after image with transparency
        mask_overlay = np.ma.masked_where(mask == 0, mask)
        axs[2].imshow(mask_overlay, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
        

        if show_titles:
            axs[0].set_title('Before (t1)')
            axs[1].set_title('After (t2)')
            axs[2].set_title('Change Mask') # Updated title

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig