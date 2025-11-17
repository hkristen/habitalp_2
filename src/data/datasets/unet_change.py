import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
from torchgeo.datasets import GeoDataset, RasterDataset

matplotlib.use('Agg')


class RGBRasterDataset(RasterDataset):
    all_bands = ("R", "G", "B")


class CIRRasterDataset(RasterDataset):
    all_bands = ("NIR", "R", "G")


class MultiModalDataset(GeoDataset):
    """Dataset for multi-modal remote sensing data.
    Input layers include RGB, CIR, and elevation data and can be chosen freely.
    """

    def __init__(
        self,
        image_paths: dict[Path],
        mask_path: Path,
        res: float | None = None,
        transforms=None,
    ):
        """Initialize the MultiModalDataset.

        Args:
            image_paths (dict[Path]): Paths to input images
            mask_path (Path): Path to label mask image
            res (float, optional): Resolution of the dataset in units of CRS (defaults to the resolution of the first file found). Defaults to None.
            transforms: A function/transform that takes an input sample
                and returns a transformed version. Defaults to None.
        """
        super().__init__(transforms)

        # Set dataset properties
        self.image_paths = image_paths
        self.mask_path = mask_path
        self.bands = []
        self.is_image_band = []

        # Create list for image datasets
        self.image_ds_list = []
        
        # Add RGB dataset if available
        if "rgb" in self.image_paths:
            bands = ["R", "G", "B"]
            self.image_ds_list.append(
                RGBRasterDataset(
                    self.image_paths["rgb"], res=res, transforms=transforms, bands=bands
                )
            )
            self.bands += bands
            self.is_image_band += [True, True, True]

        # Add CIR dataset if available
        if "cir" in self.image_paths:
            bands = ["NIR"]
            self.image_ds_list.append(
                CIRRasterDataset(
                    self.image_paths["cir"], res=res, transforms=transforms, bands=bands
                )
            )
            self.bands += bands
            self.is_image_band += [True]

        # Add PAN dataset if available
        if "pan" in self.image_paths:
            self.image_ds_list.append(
                RasterDataset(
                    self.image_paths["pan"],
                    res=res,
                    transforms=transforms,
                )
            )
            self.bands += ["PAN"]
            self.is_image_band += [True]

        # Add Spot-4 dataset if available
        if "spot-4" in self.image_paths:
            self.image_ds_list.append(
                RasterDataset(
                    self.image_paths["spot-4"],
                    res=res,
                    transforms=transforms,
                )
            )
            self.bands += ["Spot_G", "Spot_R", "Spot_NIR", "Spot_SWIR"]
            self.is_image_band += [True, True, True, True]

        # Add elevation layers and forest attributes if available
        for elevation_layer in [
            "dtm",
            "dsm",
            "ndsm",
            "slope",
            "aspect",
            "tri",
            "tpi",
            "roughness_terrain",
            "roughness_canopy",
            "curvature",
            "planform_curvature",
            "profile_curvature",
            "fhd",
            "kurtosis",
            "percentile_90",
            "skewness",
            "stddev",
            "vertical_structure",
        ]:
            if elevation_layer in self.image_paths:
                self.image_ds_list.append(
                    RasterDataset(self.image_paths[elevation_layer], res=res)
                )
                self.bands += [elevation_layer]
                self.is_image_band += [False]

        # Add forest mask if available
        if "forest_mask" in self.image_paths:
            self.image_ds_list.append(
                RasterDataset(self.image_paths["forest_mask"], res=res)
            )
            self.bands += ["forest_mask"]
            self.is_image_band += [False]

        # Combine all image datasets
        if len(self.image_ds_list) == 0:
            raise ValueError("No valid image paths provided.")
        elif len(self.image_ds_list) == 1:
            self.image_ds = self.image_ds_list[0]
        else:
            self.image_ds = self.image_ds_list[0]
            for ds in self.image_ds_list[1:]:
                self.image_ds = self.image_ds & ds
        self.image_ds_list = None

        # Set label dataset
        self.mask_ds = RasterDataset(
            self.mask_path,
            res=res,
            transforms=transforms,
        )
        self.mask_ds.is_image = False

        # Create intersection of image and mask datasets
        self.dataset = self.image_ds & self.mask_ds

        # Set dataset crs, resolution, and index
        self.crs = self.dataset.crs
        self.res = self.dataset.res
        self.index = self.dataset.index

        # Set bands for plotting
        self.plot_bands = [0, 1, 2] if "rgb" in self.image_paths else 0

    def __getitem__(self, query):
        return self.dataset.__getitem__(query)

    def __str__(self):
        return self.dataset.__str__()

    def crs(self):
        return self.dataset.crs

    def plot(self, sample):
        """Plot a sample from the dataset.

        Creates a figure with three subplots showing image, labels and prediction.

        Args:
            sample: A sample returned by __getitem__ containing:
                   - image: Concatenated image tensor [6, H, W]
                   - mask: Binary change mask tensor [H, W]
                   - prediction: Predicted change mask tensor [H, W]

        Returns:
            A matplotlib Figure with the rendered sample
        """
        # Ensure the image tensor is in the range [0, 1] and permute to [H, W, C]
        image = (sample["image"] / 255.0).clamp(min=0, max=1).permute(1, 2, 0)

        # Select image bands
        image = image[:, :, self.plot_bands]

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
        axs[0].imshow(image, cmap="gray")
        axs[0].set_axis_off()
        axs[1].imshow(
            sample["mask"].squeeze().numpy(),
        )
        axs[1].set_axis_off()
        axs[2].imshow(
            sample["prediction"].squeeze().numpy(),
        )
        axs[2].set_axis_off()

        return fig

    def res(self):
        return self.dataset.res
