import warnings
from collections.abc import Iterator, Iterable

import torch
from torch import Generator
from rtree.index import Index as RTreeIndex # Spatial index for polygons
from shapely import intersection
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.prepared import prep # Optional: speeds up repeated intersection checks
from torchgeo.samplers import GeoSampler, RandomGeoSampler

from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box

class RandomGeoSamplerIntersectingPolygons(GeoSampler):
    """Samples individual elements randomly, ensuring intersection with label polygons.

    This sampler yields :term:`patches <patch>` (defined by their BoundingBox)
    that are guaranteed to geometrically intersect with at least one polygon
    from a provided set of label polygons.

    It avoids loading pixel data during sampling, making it much faster than
    checking pixel counts, but requires a pre-defined set of label geometries.

    .. warning::
        This sampler assumes that the provided ``label_polygons`` are in the
        **same Coordinate Reference System (CRS)** as the ``dataset``. Mismatched
        CRSs will lead to incorrect intersection results.

    Suitable for use with the `sampler` argument of
    `torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        label_polygons: Iterable[Polygon | MultiPolygon],
        size: tuple[float, float] | float,
        length: int,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
        max_attempts_per_sample: int = 1000,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: GeoDataset to index from. Its CRS must match label_polygons.
            label_polygons: An iterable of Shapely Polygon or MultiPolygon objects
                representing the ground truth labeled areas.
            size: Dimensions of each :term:`patch`.
            length: Number of valid samples (intersecting polygons) per epoch.
            roi: BoundingBox region of interest to sample tiles from
                 (defaults to the bounds of ``dataset.index``).
            units: Defines if ``size`` is in pixel or CRS units.
            max_attempts_per_sample: Maximum number of random patches to try
                generating before giving up on finding an intersecting one for
                a single yield iteration.
        """
        super().__init__(dataset, roi) # Initializes self.index, self.res, self.roi
        self.size = _to_tuple(size)
        self.length = length
        self.max_attempts_per_sample = max_attempts_per_sample

        if units == Units.PIXELS:
            if self.res == 0:
                raise ValueError(
                    "Dataset resolution is not set, cannot use units=Units.PIXELS."
                )
            # Ensure size is in CRS units for get_random_bounding_box
            elif type(self.res) == tuple:
                self.size = (self.size[0] * self.res[0], self.size[1] * self.res[1])
            else:
                self.size = (self.size[0] * self.res, self.size[1] * self.res)

        # Build Spatial Index for Label Polygons
        self.label_polygons = list(label_polygons) # Store the polygons
        if not self.label_polygons:
             warnings.warn("`label_polygons` is empty. No samples will be generated.")
             self.label_index = None 
        else:
             self.label_index = RTreeIndex()
             for i, poly in enumerate(self.label_polygons):
                 if isinstance(poly, (Polygon, MultiPolygon)) and not poly.is_empty:
                     self.label_index.insert(i, poly.bounds)
                 else:
                     warnings.warn(f"Skipping invalid or empty geometry at index {i}: {poly}")

        # Calculate potential dataset tiles (hits) and their areas within the BBox ROI
        # Used for weighting the *initial tile* selection
        self.hits = []
        areas = []

        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            # Basic check if a tile is large enough
            if (
                bounds.maxx - bounds.minx >= float(self.size[1])
                and bounds.maxy - bounds.miny >= float(self.size[0])
            ):
                self.hits.append(hit)
                areas.append(max(bounds.area, 1e-9)) # Use area, prevent zero

        if not self.hits:
            raise ValueError(
                f"No dataset tiles found in the ROI {self.roi} that are large"
                f" enough to fit a patch of size {self.size}."
            )

        self.areas = torch.tensor(areas, dtype=torch.float)

        # Prepare geometries for faster intersection checks if many polygons
        self.prepared_polygons = [prep(p) for p in self.label_polygons]


    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the indices of a dataset.

        Yields:
            (minx, maxx, miny, maxy, mint, maxt) coordinates that define a patch
            guaranteed to intersect with at least one of the `label_polygons`.
        """
        if self.label_index is None or self.label_index.count == 0:
             warnings.warn("Label polygon index is empty, yielding no samples.")
             return # Stop iteration if no labels

        num_yielded = 0
        while num_yielded < self.length:
            found_intersecting_box = False
            for attempt in range(self.max_attempts_per_sample):
                # 1. Choose a random dataset tile, weighted by area
                try:
                    idx = torch.multinomial(self.areas, 1).item()
                except RuntimeError as e:
                     raise RuntimeError(f"Could not sample from areas: {self.areas}. Error: {e}")

                hit = self.hits[idx]
                tile_bounds = BoundingBox(*hit.bounds)

                # 2. Generate a random candidate bounding box within the tile
                candidate_bbox = get_random_bounding_box(
                    tile_bounds, self.size, self.res
                )

                # 3. Convert candidate BBox to Shapely geometry
                patch_geom = box(
                    candidate_bbox.minx, candidate_bbox.miny,
                    candidate_bbox.maxx, candidate_bbox.maxy
                )

                # 4. Check for intersection using the spatial index
                try:
                    # Find potentially intersecting polygons using their bounds
                    potential_indices = list(self.label_index.intersection(patch_geom.bounds))

                    # Perform precise intersection check only on potential hits
                    for poly_idx in potential_indices:
                        #label_poly = self.label_polygons[poly_idx]
                        # Use prepared geometry if available:
                        if self.prepared_polygons[poly_idx].intersects(patch_geom):
                        #if patch_geom.intersects(label_poly):
                            yield candidate_bbox
                            num_yielded += 1
                            found_intersecting_box = True
                            break # Found intersection, break polygon check loop

                    if found_intersecting_box:
                        break # Break attempt loop, move to next sample

                except Exception as e:
                    # Catch potential errors during intersection (e.g., invalid geometry)
                    warnings.warn(
                        f"Error during intersection check for bbox {candidate_bbox}: {e}. Skipping."
                    )
                    continue # Try another random box

            if not found_intersecting_box:
                raise RuntimeError(
                    f"Could not find an intersecting patch after "
                    f"{self.max_attempts_per_sample} attempts. "
                    f"Check CRS alignment between dataset and label_polygons, "
                    f"polygon validity, and density of labels within the ROI."
                )

    def __len__(self) -> int:
        """Return the number of samples in a single epoch."""
        return self.length


class RandomGeoSamplerWithinArea(RandomGeoSampler):
    """Updated RandomGeoSampler from torchgeo with support for ROI Polygon check.
    
    Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
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
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
            generator: pseudo-random number generator (PRNG).
        """
        super().__init__(
            dataset=dataset,
            size=size,
            length=length,
            roi=roi,
            units=units,
            generator=generator,
        )
        self.roi_shape = roi_shape

        self.cell_fully_labeled = []
        self.labelled_cell_area = []
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            # Check if the tile is fully within the ROI shape
            boxgeom = box(
                minx=bounds.minx,
                miny=bounds.miny,
                maxx=bounds.maxx,
                maxy=bounds.maxy,
            )
            fully_labeled = self.roi_shape.contains(boxgeom)
            self.cell_fully_labeled.append(fully_labeled)

            if fully_labeled:
                self.labelled_cell_area.append(None)
            else:
                # If not fully labeled, calculate the labelled area
                self.labelled_cell_area.append(intersection(self.roi_shape, boxgeom))

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Yields:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            search_count = 0
            found_cell = False
            while (search_count < 100) and not found_cell:
                search_count += 1
                # Choose a random tile, weighted by area
                idx = torch.multinomial(self.areas, 1)
                hit = self.hits[idx]
                bounds = BoundingBox(*hit.bounds)

                # Choose a random index within that tile
                bounding_box = get_random_bounding_box(
                    bounds, self.size, self.res, self.generator
                )

                if self.cell_fully_labeled[idx]:
                    found_cell = True

                    yield bounding_box
                else:
                    if self.labelled_cell_area[idx].contains(
                        box(
                            bounding_box.minx,
                            bounding_box.miny,
                            bounding_box.maxx,
                            bounding_box.maxy,
                        )
                    ):
                        found_cell = True

                        yield bounding_box
            if not found_cell:
                raise RuntimeError(
                    f"Could not find patch in labeled cell area after 100 iterations."
                )
