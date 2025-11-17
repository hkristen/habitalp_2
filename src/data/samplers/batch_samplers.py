from collections.abc import Iterator

from shapely import intersection
from shapely.geometry import box, Polygon
import torch
from torch import Generator

from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import BatchGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box, tile_to_chips

class RandomBatchGeoSamplerWithinArea(BatchGeoSampler):
    """Updated RandomBatchGeoSampler from torchgeo with support for ROI Polygon check.
    
    Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        batch_size: int,
        roi_shape: Polygon,
        length: int | None = None,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionchanged:: 0.4
           ``length`` parameter is now optional, a reasonable default will be used

        .. versionadded:: 0.7
            The *generator* parameter.

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            roi_shape: Polygon defining the region of interest to sample from
            batch_size: number of samples per batch
            length: number of samples per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
            generator: pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.generator = generator

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])

        self.batch_size = batch_size
        self.roi_shape = roi_shape
        self.length = 0
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[list[BoundingBox]]:
        """Return the indices of a dataset.

        Yields:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Check if the tile is fully within the ROI shape
            boxgeom = box(
                    minx=bounds.minx,
                    miny=bounds.miny,
                    maxx=bounds.maxx,
                    maxy=bounds.maxy,
                )
            cell_fully_labeled = self.roi_shape.contains(boxgeom)
            # If not fully labeled, calculate the labelled area
            if not cell_fully_labeled:
                labeled_cell_area = intersection(self.roi_shape, boxgeom)

            # Choose random indices within that tile
            batch = []
            search_count = 0
            while len(batch) < self.batch_size:
                search_count += 1
                bounding_box = get_random_bounding_box(
                    bounds, self.size, self.res, self.generator
                )
                if cell_fully_labeled or labeled_cell_area.contains(box(bounding_box.minx, bounding_box.miny, bounding_box.maxx, bounding_box.maxy)):
                    batch.append(bounding_box)
                elif search_count > (self.batch_size * 1000):
                    raise RuntimeError("Could not find enough patches in labeled cell area.")

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size
