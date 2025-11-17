import sys

sys.path.append("../..")

import warnings
from pathlib import Path

import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from terratorch.tasks import SemanticSegmentationTask

from src.data.datamodules.unet_change import SegmentationDataModule

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # Data paths
    data_folder = Path("/home/hkristen/Nextcloud/HabitAlp2.0/Originaldaten")
    input_data = {
        "rgb": data_folder / "processed/orthophoto_gis_stmk/flug_2013_2015_rgb.tif",
        "cir": data_folder / "processed/orthophoto_gis_stmk/falschfarben_2013_2015.tif",
        "dtm": data_folder / "processed/elevation_waldstmk/dtm.tif",
        # "dsm": data_folder / "processed/elevation_waldstmk/dsm.tif",
        # "slope": data_folder / "processed/elevation_waldstmk/slope.tif",
        # "aspect": data_folder / "processed/elevation_waldstmk/aspect.tif",
        # "tri": data_folder / "processed/elevation_waldstmk/tri.tif",
        # "tpi": data_folder / "processed/elevation_waldstmk/tpi.tif",
        # "roughness": data_folder / "processed/elevation_waldstmk/roughness.tif",
        # "curvature": data_folder / "processed/elevation_waldstmk/curvature.tif",
        # "planform_curvature": data_folder
        # / "processed/elevation_waldstmk/planform_curvature.tif",
        # "profile_curvature": data_folder
        # / "processed/elevation_waldstmk/profile_curvature.tif",
        "ndsm": data_folder / "processed/elevation_waldstmk/ndsm.tif",
    }
    mask_path = data_folder / "processed/mask/classes_v3.tif"
    roi_shape_path = data_folder / "roi/habitalp_2013_boundary.gpkg"
    target_class_definiton_path = (
        data_folder / "habitalp_target_classes/Zielklassen 2024 v3.csv"
    )
    experiment_dir_base = "/home/hkristen/habitalp2/src/models/experiments"

    # Training parameters
    lr = 1e-4
    accelerator = "auto"
    max_epochs = 60
    batch_size = 32

    # Create datamodule
    datamodule = SegmentationDataModule(
        input_data,
        mask_path,
        roi_shape_path,
        n_classes=23,
        batch_size=batch_size,
        patch_size=(224, 224),
        num_workers=4,
        train_batches_per_epoch=1024,
        val_batches_per_epoch=64,
        target_class_definition_path=target_class_definiton_path,
    )

    datamodule.setup(stage="fit")

    # Model configuration
    model_args = dict(
        backbone="prithvi_eo_v2_300",
        backbone_pretrained=True,
        backbone_num_frames=1,
        num_classes=datamodule.n_classes + 1,
        backbone_bands=[
            "BLUE",
            "GREEN",
            "RED",
            "NIR_NARROW",
            "NDSM",
            "DTM",
            # "DSM",
            # "SLOPE",
            # "ASPECT",
            # "TRI",
            # "TPI",
            # "ROUGHNESS",
            # "CURVATURE",
            # "PLANFORM_CURVATURE",
            # "PROFILE_CURVATURE",
        ],
        decoder="UNetDecoder",
        decoder_channels=[512, 256, 128, 64],
        necks=[
            {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
            {"name": "ReshapeTokensToImage"},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        head_dropout=0.1,
    )

    # Create task
    task = SemanticSegmentationTask(
        model_args,
        "EncoderDecoderFactory",
        loss="ce",
        lr=lr,
        ignore_index=None,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=True,
        plot_on_val=False,
        class_names=datamodule.class_names,
        class_weights=[
            3.05754852e-09,
            5.61588104e-08,
            2.55059813e-07,
            1.31145305e-08,
            1.18684274e-08,
            2.41300286e-09,
            1.39145003e-08,
            2.71112867e-08,
            1.55613875e-08,
            8.56254672e-09,
            3.42943184e-08,
            4.17894324e-09,
            4.42492833e-09,
            2.31283622e-08,
            1.43593138e-08,
            5.54229727e-09,
            1.03878198e-08,
            4.66937618e-08,
            1.60617068e-08,
            1.63495321e-08,
            2.86372574e-09,
            1.65979327e-08,
            6.29546752e-09,
            4.21235049e-08,
        ],
    )

    # Setup wandb logging
    wandb_project = "semantic-segmentation-terratorch"
    wandb_key = "YOUR_WANDB_API_KEY_HERE"  # Replace with your WandB API key
    experiment_name = "TERRATORCH_TEST_3_pretrained_true_frozen_ndsm_dtm"

    experiment_dir = Path(experiment_dir_base) / Path(experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    wandb.login(key=wandb_key)
    wb_logger = WandbLogger(
        name=experiment_name,
        project=wandb_project,
        log_model="all",
        save_dir=str(experiment_dir),
    )

    # Create trainer
    trainer = Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        logger=[wb_logger],
    )

    # Start training
    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    main()
