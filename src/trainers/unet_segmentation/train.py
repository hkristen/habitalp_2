import sys
from pathlib import Path

BASE_FOLDER = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_FOLDER))

import pandas as pd
import shutil
import torch
import wandb
import yaml

from src.data.datamodules.unet_change import SegmentationDataModule
from src.trainers import setup_training, MultiClassSemanticSegmentationTask


def train_model(
    logging: str,
    wandb_project: str,
    wandb_key: str,
    config: dict=None,
):
    """Prepare datamodule and SemanticSegmentationTask and perform model training while logging to wandb.

    Args:
        logging (str): Either 'local' or 'remote' for W&B logging
        wandb_project (str): W&B project name
        wandb_key (str): W&B API key (required if logging='remote')
        config (dict): Custom config dictionary. Will override parameters in config-defaults.yaml if exists.

    Returns:
        (SegmentationDataModule, SemanticSegmentationTask, Trainer): datamodule, task, trainer
    """
    # Init wandb. This will load the config file
    run = wandb.init(project=wandb_project, config=config)
    config = run.config
    wandb_run_id = run.id

    # Location to create experiment data folder inside
    experiment_dir = BASE_FOLDER / "models" / config.name

    # Add more config parameters to wandb
    run.name = config.name
    run.notes = config.notes

    datamodule = get_datamodule(
        config=config
    )
    config["in_channels"] = datamodule.in_channels

    task = get_task(
        hyperparameters=config.hyperparameters,
        in_channels=config.in_channels,
        num_classes=config.num_labels,
        ignore_index=config["ignore_index"],
        precalculated_weights=config.precalculated_weights,
    )

    # Set Internal precision of float32 matrix multiplications
    # See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision(config.matmul_precision)

    trainer = setup_training(
        experiment_name=config.name,
        experiment_dir=experiment_dir,
        min_epochs=config.hyperparameters["min_epochs"],
        max_epochs=config.hyperparameters["max_epochs"],
        gpu_id=config["gpu_id"],
        patience=config.hyperparameters["patience"],
        wandb_project=wandb_project,
        logging=logging,
        wandb_key=wandb_key,
        monitor_metric=config.hyperparameters["monitor_metric"],
        log_model=True,
    )

    # Train model
    _ = trainer.fit(model=task, datamodule=datamodule)
    config["device"] = str(task.device)

    # Area of training data for plots
    if config["train_data_area_path"]:
        train_data_area_path = Path(config["data_folder"]) / config["train_data_area_path"]
        train_data_area = (pd.read_csv(train_data_area_path)["Train"] / 10000).to_list()
    else:
        train_data_area = None

    # Evaluate using test set and log final metrics
    trainer.test(model=task, datamodule=datamodule)
    task.log_final_metrics(class_names=datamodule.class_names.copy(), train_data_area=train_data_area)

    # Save model
    torch.save(task.model.state_dict(), experiment_dir / f"{config.name}.pt")

    # Get sample images from test set and plot predictions
    task.plot_prediction_samples(
        datamodule=datamodule,
        experiment_dir=experiment_dir,
        n_samples=3,
        chnls=datamodule.dataset.plot_bands,
    )

    wandb.finish()

    # Save the config file to the experiment directory
    shutil.copy("config-defaults.yaml", experiment_dir / "config.yaml")

    # Save model paths
    best_model_path = Path(trainer.checkpoint_callback.best_model_path).name
    model_paths = {
        "best_model_path": best_model_path,
        "last_model_path": "last.ckpt",
        "wandb_project": wandb_project,
        "wandb_run_id": wandb_run_id,
    }
    with open(experiment_dir / "model_paths.yaml", "w") as file:
        yaml.dump(model_paths, file)

    return datamodule, task, trainer


def get_datamodule(
    config: dict,
    prediction_year: int | None=None,
):
    """Get datamodule for semantic segmentation.

    Args:
        config (dict): Configuration for the model
        prediction_year (int, optional): Which year to predict on (either 2020 or 2024)

    Returns:
        SegmentationDataModule: Data module
    """
    hyperparameters = config.get("hyperparameters", config) # If called from sweep, hyperparameters are in config
    data_folder = Path(config["data_folder"])

    image_data_paths = config["image_data_paths"]
    image_data_paths = {key: data_folder / image_data_paths[key] for key in hyperparameters["input_bands"] if key in image_data_paths}
    mask_path = data_folder / config["mask_path"]
    roi_shape_path = data_folder / config["roi_shape_path"]
    if prediction_year is not None:
        image_data_paths_pred = config.get(f"image_data_paths_pred_{prediction_year}")
        image_data_paths_pred = {key: data_folder / image_data_paths_pred[key] for key in hyperparameters["input_bands"] if key in image_data_paths_pred}
        prediction_mask_path = config.get(f"prediction_mask_path_{prediction_year}")
        prediction_mask_path = data_folder / prediction_mask_path
    else:
        image_data_paths_pred = None
        prediction_mask_path = None

    return SegmentationDataModule(
        image_data_paths,
        mask_path,
        n_classes=config["num_labels"],
        roi_shape_path=roi_shape_path,
        batch_size=hyperparameters["batch_size"],
        patch_size=(hyperparameters["patch_size"], hyperparameters["patch_size"]),
        num_workers=config["num_dataloader_workers"],
        res=hyperparameters["res"],
        train_batches_per_epoch=hyperparameters["train_batches_per_epoch"],
        val_batches_per_epoch=hyperparameters["val_batches_per_epoch"],
        class_names=config["class_names"],
        image_paths_pred=image_data_paths_pred,
        prediction_mask_path=prediction_mask_path
    )


def get_task(
    hyperparameters: dict,
    in_channels: int,
    num_classes: int,
    ignore_index: int=0,
    precalculated_weights: list=None,
):
    """Get MultiClassSemanticSegmentation Task for model training.

    We ignore the nodata class label which corresponds to the "not labeled" class.

    Args:
        hyperparameters (dict): Hyperparameters for the model
        in_channels (int): Number of input bands
        num_classes (int): Number of target classes without the nodata class
        ignore_index (int, optional): Index to ignore in the loss function (e.g., nodata class). Defaults to 0.
        precalculated_weights (list, optional): Precalculated class weights to use with crossentropy loss. Defaults to None.

    Returns:
        MultiClassSemanticSegmentationTask: Task
    """    
    return MultiClassSemanticSegmentationTask(
        model=hyperparameters["model"],
        backbone=hyperparameters["backbone"],
        weights=hyperparameters["use_pretrained_model_weights"],
        in_channels=in_channels,
        task="multiclass" if num_classes > 1 else "binary",
        num_classes=num_classes+1 if num_classes > 1 else None,
        loss=hyperparameters["loss"],
        class_weights=torch.Tensor(precalculated_weights) if precalculated_weights is not None else None,
        ignore_index=ignore_index,
        lr=hyperparameters["learning_rate"],
        patience=hyperparameters["patience"],
    )

def main():
    # Wandb settings
    logging = "remote"
    wandb_project = "unet_change"
    wandb_key = "YOUR_WANDB_API_KEY_HERE"  # Replace with your WandB API key

    # Call training function
    datamodule, task, trainer = train_model(
        logging=logging,
        wandb_project=wandb_project,
        wandb_key=wandb_key,
    )

if __name__ == '__main__':
    main()
