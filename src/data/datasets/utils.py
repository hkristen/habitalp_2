from collections.abc import Sequence
from copy import deepcopy
from math import floor, isclose

from rtree.index import Index, Property
from shapely import intersection
from shapely.geometry import box, Polygon
from torch import Generator, default_generator, randperm

from torchgeo.datasets import GeoDataset

def _fractions_to_lengths(fractions: Sequence[float], total: int) -> Sequence[int]:
    """Utility to divide a number into a list of integers according to fractions.

    Implementation based on :meth:`torch.utils.data.random_split`.

    Args:
        fractions: list of fractions
        total: total to be divided

    Returns:
        List of lengths.

    .. versionadded:: 0.5
    """
    lengths = [floor(frac * total) for frac in fractions]
    remainder = int(total - sum(lengths))
    # Add 1 to all the lengths in round-robin fashion until the remainder is 0
    for i in range(remainder):
        idx_to_add_at = i % len(lengths)
        lengths[idx_to_add_at] += 1
    return lengths

def random_grid_cell_assignment(
    dataset: GeoDataset,
    fractions: Sequence[float],
    roi_shape: Polygon,
    grid_size: int = 6,
    generator: Generator | None = default_generator,
    min_overlap: float = 0.4,
) -> list[GeoDataset]:
    """Updated function from torchgeo with roi intersection check:
    https://torchgeo.readthedocs.io/en/stable/api/datasets.html#torchgeo.datasets.random_grid_cell_assignment
    
    Overlays a grid over a GeoDataset and randomly assigns cells to new GeoDatasets.

    This function will go through each BoundingBox in the GeoDataset's index, overlay
    a grid over it, and randomly assign each cell to new GeoDatasets.

    Args:
        dataset: dataset to be split
        fractions: fractions of splits to be produced
        grid_size: number of rows and columns for the grid
        generator: generator used for the random permutation

    Returns:
        A list of the subset datasets.

    .. versionadded:: 0.5
    """
    if not isclose(sum(fractions), 1):
        raise ValueError('Sum of input fractions must equal 1.')

    if any(n <= 0 for n in fractions):
        raise ValueError('All items in input fractions must be greater than 0.')

    if grid_size < 2:
        raise ValueError('Input grid_size must be greater than 1.')

    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in fractions
    ]

    cells = []

    # Generate the grid's cells for each bbox in index
    for i, hit in enumerate(
        dataset.index.intersection(dataset.index.bounds, objects=True)
    ):
        minx, maxx, miny, maxy, mint, maxt = hit.bounds

        stridex = (maxx - minx) / grid_size
        stridey = (maxy - miny) / grid_size

        for x in range(grid_size):
            for y in range(grid_size):
                cell_minx = minx + x * stridex
                cell_maxx = minx + (x + 1) * stridex
                cell_miny = miny + y * stridey
                cell_maxy = miny + (y + 1) * stridey
                boxgeom = box(
                    minx=cell_minx,
                    miny=cell_miny,
                    maxx=cell_maxx,
                    maxy=cell_maxy,
                )
                cell_area = boxgeom.area
                intersection_area = intersection(roi_shape, boxgeom).area
                if intersection_area / cell_area > min_overlap:
                    cells.append(
                        (
                            (
                                cell_minx,
                                cell_maxx,
                                cell_miny,
                                cell_maxy,
                                mint,
                                maxt,
                            ),
                            hit.object,
                        )
                    )

    lengths = _fractions_to_lengths(fractions, len(dataset) * len(cells))

    # Randomly assign cells to each new index
    cells = [cells[i] for i in randperm(len(cells), generator=generator)]

    for i, length in enumerate(lengths): # 3
        for j in range(length):
            cell = cells.pop()
            new_indexes[i].insert(j, cell[0], cell[1])

    new_datasets = []
    for index in new_indexes:
        ds = deepcopy(dataset)
        ds.index = index
        new_datasets.append(ds)

    return new_datasets