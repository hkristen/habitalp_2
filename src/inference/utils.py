import numpy as np
from pathlib import Path
import rasterio
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

from ..trainers.utils import read_intersecting_area_of_two_rasters


def configure_metric_collections(
    num_classes: int,
    averaging: str = "macro",
    ignore_index: int = 255,
):
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

    return multiclass_metric_collection, metric_collection_each_label


def physical_constraints_module(
    mask_path: Path,
    prediction_path: Path,
    dtm_path: Path,
    slope_path: Path,
    experiment_dir: Path,
    output_name: str,
    export_mask_for_each_constraint: bool = False,
):
    """Calculates physical constraints based on the provided ground truth mask and model predictions.

    Args:
        mask_path (Path): Path to label mask image.
        prediction_path (Path): Path to the prediction file.
        dtm_path (Path): Path to the digital terrain model (DTM) file.
        slope_path (Path): Path to the slope raster file.
        experiment_dir (Path): Directory to save the constraint violation masks.
        output_name (str): Name of the resulting prediction raster file.
        export_mask_for_each_constraint (bool, optional): Exports violation constraint masks if true. Defaults to False.
    """    
    # Get mask and prediction arrays
    mask_array, prediction_array, output_meta = read_intersecting_area_of_two_rasters(
        mask_path, prediction_path
        )

    non_forest_classes = [1, 2, 3, 4, 5, 19, 21, 22, 23]
    alpine_vegetation = [20, 22]
    grassland = [21, 22]

    needle_dominating = [6, 8, 9, 11, 12, 15, 16, 20]
    conifer_dominating = [7, 10, 13, 14, 17, 18]
    young_growth = [6, 7]
    pole_wood = [8, 9, 10]
    mature_forest = [11, 12, 13, 14]
    old_forest = [15, 16, 17, 18]

    nodata_cells = (mask_array == output_meta["nodata"]) | (prediction_array == output_meta["nodata"])
    output_meta.update(
        {
            "compress": "lzw",
            "nodata": 255,
            "tiled": True,
            "predictor": 2,
            "bigtiff": True,
        }
    )

    # --- 1. Non-forest to mature/old forest ---
    condition_1_true = np.asarray(
            np.isin(mask_array, non_forest_classes) & np.isin(prediction_array, mature_forest + old_forest),
        )
    print(f"{np.count_nonzero(condition_1_true)} pixels violate constraint 1: Non-forest to mature/old forest")

    if export_mask_for_each_constraint:
        condition_1_true = np.where(nodata_cells, 255, condition_1_true).astype(np.uint8)  # Set nodata to 255
        with rasterio.open(
            experiment_dir / "constraint_violation_1_non_forest_to_mature_forest.tif",
            "w",
            **output_meta,
        ) as dst:
            dst.write(condition_1_true.astype(rasterio.uint8), 1)

    total_violations = condition_1_true.copy()
    del condition_1_true

    # --- 2. Young growth to old forest ---
    condition_2_true = np.asarray(
            np.isin(mask_array, young_growth) & np.isin(prediction_array, old_forest),
        )
    print(f"{np.count_nonzero(condition_2_true)} pixels violate constraint 2: Young growth to old forest")

    if export_mask_for_each_constraint:
        condition_2_true = np.where(nodata_cells, 255, condition_2_true).astype(np.uint8)  # Set nodata to 255
        with rasterio.open(
            experiment_dir / "constraint_violation_2_young_growth_to_old_forest.tif",
            "w",
            **output_meta,
        ) as dst:
            dst.write(condition_2_true.astype(rasterio.uint8), 1)

    total_violations = np.any([total_violations, condition_2_true], axis=0)
    del condition_2_true

    # --- 3. Forest setback ---
    condition_3_true = np.asarray(
        (np.isin(mask_array, pole_wood) & np.isin(prediction_array, young_growth)) |
        (np.isin(mask_array, mature_forest) & np.isin(prediction_array, young_growth + pole_wood)) |
        (np.isin(mask_array, old_forest) & np.isin(prediction_array, young_growth + pole_wood + mature_forest))
        )
    print(f"{np.count_nonzero(condition_3_true)} pixels violate constraint 3: Forest setback to younger stage")

    if export_mask_for_each_constraint:
        condition_3_true = np.where(nodata_cells, 255, condition_3_true).astype(np.uint8)  # Set nodata to 255
        with rasterio.open(
            experiment_dir / "constraint_violation_3_forest_setback_to_younger_stage.tif",
            "w",
            **output_meta,
        ) as dst:
            dst.write(condition_3_true.astype(rasterio.uint8), 1)

    total_violations = np.any([total_violations, condition_3_true], axis=0)
    del condition_3_true

    # --- 4. Rock to grassland ---
    condition_4_true = np.asarray(
            (mask_array == 5) & np.isin(prediction_array, grassland),
        )
    print(f"{np.count_nonzero(condition_4_true)} pixels violate constraint 4: Rock to grassland")

    if export_mask_for_each_constraint:
        condition_4_true = np.where(nodata_cells, 255, condition_4_true).astype(np.uint8)  # Set nodata to 255
        with rasterio.open(
            experiment_dir / "constraint_violation_4_rock_to_grassland.tif",
            "w",
            **output_meta,
        ) as dst:
            dst.write(condition_4_true.astype(rasterio.uint8), 1)

    total_violations = np.any([total_violations, condition_4_true], axis=0)
    del condition_4_true

    # --- 5. Water bodies on steep slopes ---
    with rasterio.open(slope_path) as slope_src:
        slope_window = slope_src.window(*output_meta["bounds"])
        slope_array = slope_src.read(1, window=slope_window, out_shape=mask_array.shape, resampling=rasterio.enums.Resampling.bilinear)

    condition_5_true = np.asarray(
            (slope_array > 35.0) & (mask_array != 1) & (prediction_array == 1),
        )
    del slope_array
    print(f"{np.count_nonzero(condition_5_true)} pixels violate constraint 5: Water bodies on steep slopes (> 35°)")

    if export_mask_for_each_constraint:
        condition_5_true = np.where(nodata_cells, 255, condition_5_true).astype(np.uint8)  # Set nodata to 255
        with rasterio.open(
            experiment_dir / "constraint_violation_5_water_bodies_on_steep_slopes.tif",
            "w",
            **output_meta,
        ) as dst:
            dst.write(condition_5_true.astype(rasterio.uint8), 1)

    total_violations = np.any([total_violations, condition_5_true], axis=0)
    del condition_5_true

    # --- 6. Alpine vegetation in lowland ---
    with rasterio.open(dtm_path) as dtm_src:
        dtm_window = dtm_src.window(*output_meta["bounds"])
        dtm_array = dtm_src.read(1, window=dtm_window, out_shape=mask_array.shape, resampling=rasterio.enums.Resampling.bilinear)

    condition_6_true = np.asarray(
            (dtm_array < 650.0) & ~np.isin(mask_array, alpine_vegetation) & np.isin(prediction_array, alpine_vegetation),
        )
    del dtm_array
    print(f"{np.count_nonzero(condition_6_true)} pixels violate constraint 6: Alpine vegetation in lowland (< 650m)")

    if export_mask_for_each_constraint:
        condition_6_true = np.where(nodata_cells, 255, condition_6_true).astype(np.uint8)  # Set nodata to 255
        with rasterio.open(
            experiment_dir / "constraint_violation_6_alpine_vegetation_in_lowland.tif",
            "w",
            **output_meta,
        ) as dst:
            dst.write(condition_6_true.astype(rasterio.uint8), 1)

    total_violations = np.any([total_violations, condition_6_true], axis=0)
    del condition_6_true

    # --- Reset violations by resetting predictions to ground truth ---
    prediction_array_with_resolved_physical_constraints = np.where(total_violations & ~nodata_cells, mask_array, prediction_array)
    with rasterio.open(
        experiment_dir / f"{output_name}.tif",
        "w",
        **output_meta,
    ) as dst:
        dst.write(prediction_array_with_resolved_physical_constraints.astype(rasterio.uint8), 1)

    if export_mask_for_each_constraint:
        total_violations = np.where(nodata_cells, 255, total_violations)  # Set nodata to 255
        with rasterio.open(
            experiment_dir / "constraint_violation_total.tif",
            "w",
            **output_meta,
        ) as dst:
            dst.write(total_violations.astype(rasterio.uint8), 1)
