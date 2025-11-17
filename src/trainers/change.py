from typing import Any
from torch import nn
import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryJaccardIndex,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassJaccardIndex,
    MulticlassF1Score,
)
from torchgeo.datasets.utils import percentile_normalization
from torchgeo.trainers import SemanticSegmentationTask
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW


class BinaryChangeSemanticSegmentationTaskBinaryLoss(SemanticSegmentationTask):
    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet50",
        weights: bool = True,
        in_channels: int = 6,
        num_classes: int = 2,
        loss: str = "bce",
        pos_weight: Tensor | None = None,
        lr: float = 1e-3,
        patience: int = 10,
        weight_decay: float = 1e-4,
    ) -> None:
        # Store pos_weight before calling parent's __init__
        self.pos_weight = pos_weight

        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            num_classes=num_classes,
            loss=loss,
            lr=lr,
            patience=patience,
        )
        # Save hyperparameters including lr and weight_decay for optimizer/scheduler
        self.save_hyperparameters(ignore=["pos_weight"])

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        loss: str = self.hparams["loss"]
        if loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        elif loss == "jaccard":
            self.criterion = smp.losses.JaccardLoss(mode="binary")
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(mode="binary", normalized=True)
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'bce', 'jaccard', or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the binary classification metrics."""
        metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "jaccard": BinaryJaccardIndex(),
                "f1": BinaryF1Score(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Tensor:
        """Shared step for training, validation and testing."""
        x = batch["image"].to(self.device)
        y = (
            batch["mask"].unsqueeze(dim=1).float().to(self.device)
        )  # Add channel dim and convert to float
        y_hat = self(x).to(self.device)

        # Convert y_hat to have same shape as y (from [B,2,H,W] to [B,1,H,W])
        y_hat = y_hat[:, 1:2, :, :]  # Take only the positive class logits

        loss = self.criterion(y_hat, y)
        self.log(f"{stage}_loss", loss, batch_size=x.shape[0])

        # Get metrics for current stage
        metrics = getattr(self, f"{stage}_metrics")
        metrics(y_hat, y)
        self.log_dict(metrics, batch_size=x.shape[0])

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and scheduler.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, batch_idx, "train")

        # Plot first 10 predictions
        is_wandb = (
            isinstance(self.logger, WandbLogger)
            or "wandb" in str(self.logger.__class__).lower()
        )
        if batch_idx < 10 and is_wandb:
            x = batch["image"].to(self.device)
            with torch.no_grad():
                y_hat = self(x).to(self.device)
                y_hat = y_hat[:, 1:2, :, :]
                y_hat = (torch.sigmoid(y_hat) > 0.5).int()

            batch["prediction"] = y_hat
            sample = {k: v[0].cpu() for k, v in batch.items()}
            fig = self.plot(sample)

            self.logger.experiment.log(
                {f"train_predictions/sample_{batch_idx}": wandb.Image(fig)}
            )
            plt.close()

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in self.train_metrics.compute().items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss = self._shared_step(batch, batch_idx, "val")

        # Plot first 10 predictions
        is_wandb = (
            isinstance(self.logger, WandbLogger)
            or "wandb" in str(self.logger.__class__).lower()
        )
        if batch_idx < 10 and is_wandb:
            x = batch["image"].to(self.device)
            with torch.no_grad():
                y_hat = self(x).to(self.device)
                y_hat = y_hat[:, 1:2, :, :]
                y_hat = (torch.sigmoid(y_hat) > 0.5).int()

            batch["prediction"] = y_hat
            sample = {k: v[0].cpu() for k, v in batch.items()}
            fig = self.plot(sample)

            self.logger.experiment.log(
                {f"val_predictions/sample_{batch_idx}": wandb.Image(fig)}
            )
            plt.close()

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in self.val_metrics.compute().items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "test")

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Return binary predictions."""
        threshold = 0.5
        x = batch["image"].to(self.device)
        y_hat = self(x).to(self.device)
        y_hat = y_hat[:, 1:2, :, :]  # Take only the positive class logits
        y_hat_hard = (torch.sigmoid(y_hat) > threshold).int()
        return y_hat_hard

    def plot(self, sample):
        """Plot a sample with its prediction."""
        image1 = sample["image"][:3]
        image2 = sample["image"][3:]
        mask = sample["mask"]
        prediction = sample["prediction"].squeeze(0)  # Remove channel dimension

        # Normalize images for visualization
        def prepare_image(img_tensor):
            img = img_tensor.permute(1, 2, 0).numpy()
            img = percentile_normalization(img)
            return np.clip(img, 0, 1)

        image1 = prepare_image(image1)
        image2 = prepare_image(image2)

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(4 * 5, 5))
        axs[0].imshow(image1)
        axs[0].axis("off")
        axs[1].imshow(image2)
        axs[1].axis("off")
        axs[2].imshow(mask.numpy())
        axs[2].axis("off")
        axs[3].imshow(prediction.numpy())
        axs[3].axis("off")

        axs[0].set_title("Image 1")
        axs[1].set_title("Image 2")
        axs[2].set_title("Ground Truth")
        axs[3].set_title("Prediction")

        plt.tight_layout()
        return fig


class MultiClassChangeSemanticSegmentationTask(SemanticSegmentationTask):
    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet50",
        weights: bool = True,
        in_channels: int = 6,
        num_classes: int = 1,  # Default to 10 classes
        loss: str = "ce",  # Default to CrossEntropy
        lr: float = 1e-3,
        class_weights: Tensor | None = None,
        patience: int = 10,
        ignore_index: int | None = None,  # Optional index to ignore in loss/metrics
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        weight_decay: float = 1e-4,  # Added weight_decay
    ) -> None:
        # Store ignore_index before calling parent's __init__
        self.ignore_index = ignore_index

        # Call parent's __init__
        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            num_classes=num_classes,
            loss=loss,
            lr=lr,
            class_weights=class_weights,
            patience=patience,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
        )
        # Save hparams needed for metric calculation
        self.save_hyperparameters(
            ignore=["model", "criterion", "class_weights"]
        )  # Avoid saving model/criterion state

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        loss: str = self.hparams["loss"]
        class_weights = getattr(self, "class_weights", None)
        ignore_index = getattr(self, "ignore_index", None)  # Use stored ignore_index

        if loss == "ce":
            # Use ignore_index if provided
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_index if ignore_index is not None else -100,
                weight=class_weights,
            )
        elif loss == "jaccard":
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass",
                classes=self.hparams.num_classes,
                ignore_index=ignore_index,
            )
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                mode="multiclass", ignore_index=ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard', or 'focal' loss for multi-class."
            )

    def configure_metrics(self) -> None:
        """Initialize the multi-class classification metrics."""
        num_classes = self.hparams.num_classes
        ignore_index = getattr(self, "ignore_index", None)  # Use stored ignore_index
        metrics = MetricCollection(
            {
                # Use Multiclass metrics and provide num_classes and ignore_index
                # Average is set to 'macro' to give equal importance to all classes, these are quite important in this case
                "accuracy": MulticlassAccuracy(
                    num_classes=num_classes, ignore_index=ignore_index, average="macro"
                ),
                "jaccard": MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="macro"
                ),
                "f1": MulticlassF1Score(
                    num_classes=num_classes, ignore_index=ignore_index, average="macro"
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Tensor:
        """Shared step for training, validation and testing."""
        x = batch["image"].to(self.device)
        # Target y should be [B, H, W] with type long for CrossEntropyLoss
        y = batch["mask"].long().to(self.device)
        y_hat = self(x).to(self.device)  # Output is [B, C, H, W]

        # Ensure y_hat and y shapes are compatible with the loss function
        # CrossEntropyLoss expects y_hat: [B, C, H, W] and y: [B, H, W]
        loss = self.criterion(y_hat, y)
        self.log(f"{stage}_loss", loss, batch_size=x.shape[0])

        # Get metrics for current stage
        metrics = getattr(self, f"{stage}_metrics")
        # torchmetrics usually handles logits directly for multi-class
        metrics(y_hat, y)
        self.log_dict(metrics, batch_size=x.shape[0])

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, batch_idx, "train")

        # Plot first 10 predictions using argmax for multi-class
        is_wandb = (
            isinstance(self.logger, WandbLogger)
            or "wandb" in str(self.logger.__class__).lower()
        )
        if batch_idx < 10 and is_wandb:
            x = batch["image"].to(self.device)
            with torch.no_grad():
                y_hat_logits = self(x).to(self.device)
                # Get predicted class indices: [B, H, W]
                y_hat_pred = torch.argmax(y_hat_logits, dim=1)

            # Add prediction to batch for plotting (needs to be [B, 1, H, W] or similar)
            batch["prediction"] = y_hat_pred.unsqueeze(
                1
            ).cpu()  # Add channel dim back for consistency if needed by plot
            sample = {
                k: v[0].cpu() for k, v in batch.items()
            }  # Take first sample from batch
            fig = self.plot(sample)

            self.logger.experiment.log(
                {f"train_predictions/sample_{batch_idx}": wandb.Image(fig)}
            )
            plt.close()

        # Log metrics (using self.log_dict in _shared_step handles this now)
        # self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True) # Already logged in _shared_step
        # Metrics are logged via self.log_dict in _shared_step

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss = self._shared_step(batch, batch_idx, "val")

        # Plot first 10 predictions using argmax for multi-class
        is_wandb = (
            isinstance(self.logger, WandbLogger)
            or "wandb" in str(self.logger.__class__).lower()
        )
        if batch_idx < 10 and is_wandb:
            x = batch["image"].to(self.device)
            with torch.no_grad():
                y_hat_logits = self(x).to(self.device)
                # Get predicted class indices: [B, H, W]
                y_hat_pred = torch.argmax(y_hat_logits, dim=1)

            # Add prediction to batch for plotting (needs to be [B, 1, H, W] or similar)
            batch["prediction"] = y_hat_pred.unsqueeze(1).cpu()  # Add channel dim back
            sample = {
                k: v[0].cpu() for k, v in batch.items()
            }  # Take first sample from batch
            fig = self.plot(sample)

            self.logger.experiment.log(
                {f"val_predictions/sample_{batch_idx}": wandb.Image(fig)}
            )
            plt.close()

        # Log metrics (using self.log_dict in _shared_step handles this now)
        # self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True) # Already logged in _shared_step
        # Metrics are logged via self.log_dict in _shared_step

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and scheduler.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Return multi-class predictions."""
        x = batch["image"].to(self.device)
        y_hat_logits = self(x).to(self.device)  # Output: [B, C, H, W]
        # Get predicted class indices by taking argmax along the class dimension
        y_hat_pred = torch.argmax(y_hat_logits, dim=1)  # Output: [B, H, W]
        return y_hat_pred

    def plot(self, sample):
        """Plot a sample with its prediction."""
        image1 = sample["image"][:3]
        image2 = sample["image"][3:]
        # Mask should be [H, W] integer labels
        mask = sample["mask"]
        # Prediction should be [H, W] integer labels (remove channel dim if present)
        prediction = sample["prediction"].squeeze()

        # Normalize images for visualization
        def prepare_image(img_tensor):
            img = img_tensor.permute(1, 2, 0).numpy()
            img = percentile_normalization(img)
            return np.clip(img, 0, 1)

        image1 = prepare_image(image1)
        image2 = prepare_image(image2)

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(4 * 5, 5))
        axs[0].imshow(image1)
        axs[0].axis("off")
        axs[1].imshow(image2)
        axs[1].axis("off")
        # Use a colormap suitable for segmentation masks (e.g., 'viridis', 'tab10', 'tab20')
        # Ensure mask and prediction are numpy arrays
        axs[2].imshow(
            mask.numpy(), cmap="viridis", vmin=0, vmax=self.hparams.num_classes - 1
        )
        axs[2].axis("off")
        axs[3].imshow(
            prediction.numpy(),
            cmap="viridis",
            vmin=0,
            vmax=self.hparams.num_classes - 1,
        )
        axs[3].axis("off")

        axs[0].set_title("Image 1")
        axs[1].set_title("Image 2")
        axs[2].set_title("Ground Truth")
        axs[3].set_title("Prediction")

        plt.tight_layout()
        return fig
