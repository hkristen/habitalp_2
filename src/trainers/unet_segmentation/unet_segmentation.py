import kornia.augmentation as K
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.trainers import SemanticSegmentationTask
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    JaccardIndex,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)
from typing import Any
import wandb

class MultiClassSemanticSegmentationTask(SemanticSegmentationTask):
    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.Accuracy`: Overall accuracy
          (OA) using 'micro' averaging. The number of true positives divided by the
          dataset size. Higher values are better.
        * :class:`~torchmetrics.JaccardIndex`: Intersection
          over union (IoU). Uses 'micro' averaging. Higher valuers are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging, not used here, gives equal weight to each class, useful
             for balanced performance assessment across imbalanced classes.
        """
        kwargs = {
            'task': self.hparams['task'],
            'num_classes': self.hparams['num_classes'],
            'num_labels': self.hparams['num_labels'],
            'ignore_index': self.hparams['ignore_index'],
        }
        averaging = "macro"
        metrics = MetricCollection(
            [
                Accuracy(multidim_average='global', average=averaging, **kwargs),
                JaccardIndex(average=averaging, **kwargs),
                Precision(average=averaging, **kwargs),
                Recall(average=averaging, **kwargs),
                F1Score(average=averaging, **kwargs),
            ]
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.final_metrics = MetricCollection(
            {
                'ConfusionMatrix': ConfusionMatrix(
                    normalize='true', **kwargs
                ),
                'JaccardIndex': JaccardIndex(
                    average="none", **kwargs
                ),
                'Precision': Precision(
                    average="none", **kwargs
                ),
                'Recall': Recall(
                    average="none", **kwargs
                ),
                'F1Score': F1Score(
                    average="none", **kwargs
                ),
            },
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and scheduler.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4
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

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Updated function from torchgeo with support for wandb figure logging:
        https://torchgeo.readthedocs.io/en/stable/api/trainers.html#torchgeo.trainers.SemanticSegmentationTask.validation_step
    
        Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['mask']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, batch_size=batch_size)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=batch_size)

        if (
            batch_idx < 6
            and hasattr(self.trainer, 'datamodule')
            and hasattr(self.trainer.datamodule, 'plot')
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'log')
        ):
            datamodule = self.trainer.datamodule
            aug = K.AugmentationSequential(
                K.Denormalize(datamodule.mean, datamodule.std), # TODO original mean and std
                data_keys=None,
                keepdim=True,
            )
            batch = aug(batch)
            match self.hparams['task']:
                case 'binary' | 'multilabel':
                    batch['prediction'] = (y_hat.sigmoid() >= 0.5).long()
                case 'multiclass':
                    batch['prediction'] = y_hat.argmax(dim=1)

            for key in ['image', 'mask', 'prediction']:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.log(
                    {f"val_prediction/{batch_idx}": wandb.Image(fig)}
                )
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, batch_size=batch_size)

        if self.hparams["loss"] == "bce":
            y = y.float()

        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, batch_size=batch_size)

        self.final_metrics.update(y_hat, y)

    def log_final_metrics(
        self, class_names: list[str] = None, train_data_area: list[float] = None
    ):
        """Generate plots for metrics calculated on the test set.

        Args:
            class_names (list[str], optional): Class names to use as labels. Defaults to None.
            train_data_area (list[float], optional): Area of training data for each class. Defaults to None.
        """
        logger = self.logger.experiment

        if class_names is not None:
            class_names.insert(self.hparams["ignore_index"], "No Data")

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(24, 24), dpi=100)
        self.final_metrics['ConfusionMatrix'].plot(ax=ax, labels=class_names)
        ax.set_title("Confusion Matrix")
        logger.log({f"test/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        # Plot Jaccard Index
        fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
        self.final_metrics['JaccardIndex'].plot(ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        class_names = labels if class_names is None else class_names
        ax.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
        ax.set_title("Jaccard Index")
        fig.tight_layout()
        logger.log({f"test/jaccard_each_class": wandb.Image(fig)})
        plt.close(fig)

        # Plot Precision
        fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
        self.final_metrics['Precision'].plot(ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
        ax.set_title("Precision")
        fig.tight_layout()
        logger.log({f"test/precision_each_class": wandb.Image(fig)})
        plt.close(fig)

        # Plot Recall
        fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
        self.final_metrics['Recall'].plot(ax=ax)
        ax.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
        ax.set_title("Recall")
        fig.tight_layout()
        logger.log({f"test/recall_each_class": wandb.Image(fig)})
        plt.close(fig)

        # Plot F1 Score
        fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
        self.final_metrics['F1Score'].plot(ax=ax)
        ax.legend(class_names, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
        ax.set_title("F1 Score")
        fig.tight_layout()
        logger.log({f"test/f1score_each_class": wandb.Image(fig)})
        plt.close(fig)

        if train_data_area is not None:
            plt.style.use("seaborn-v0_8")

            train_data_area.insert(0, 0.0)

            # Plot Jaccard Index vs Train area
            metric = self.final_metrics["JaccardIndex"].compute()
            fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
            ax.scatter(train_data_area, metric)
            ax.set_title("JaccardIndex and Train Data Size")
            ax.set_xlabel("Train Data Area (ha)")
            ax.set_ylabel("JaccardIndex")

            for x, y, label in zip(train_data_area, metric, class_names):
                ax.text(x, y, label)

            logger.log({f"test/jaccard_train_area": wandb.Image(fig)})
            plt.close(fig)

            # Plot F1 Score vs Train Data Area
            metric = self.final_metrics["F1Score"].compute()
            fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
            ax.scatter(train_data_area, metric)
            ax.set_title("F1 Score and Train Data Size")
            ax.set_xlabel("Train Data Area (ha)")
            ax.set_ylabel("F1 Score")

            for x, y, label in zip(train_data_area, metric, class_names):
                ax.text(x, y, label)

            logger.log({f"test/f1score_train_area": wandb.Image(fig)})
            plt.close(fig)

    def plot_prediction_samples(
        self,
        datamodule,
        experiment_dir: str,
        n_samples: int=8,
        chnls: list[int] = [0, 1, 2],
    ):
        from matplotlib.colors import ListedColormap
        import rasterio

        ds_test = datamodule.test_dataloader()
        batch = next(iter(ds_test))

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #batch_device = batch.copy()
        #batch_device["image"] = batch_device["image"].to(device)

        y_hat = self.predict_step(batch, 0)#.cpu()
        match self.hparams['task']:
            case 'binary' | 'multilabel':
                batch['prediction'] = (y_hat >= 0.5).long().squeeze(1)
            case 'multiclass':
                batch['prediction'] = y_hat.argmax(dim=1)

        rows = n_samples
        cols = 3

        # Define colormap for labels
        if self.hparams["task"] == "binary":
            cmap = ListedColormap(["white", "blue"])
        else:
            cmap = "hsv"
        # Set max colors vor label maps
        vmax = self.hparams['num_classes'] - 1 if self.hparams['task'] == "multiclass" else 1

        # Create a figure with subplots
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=80)

        # Output folder for predicted samples
        sample_folder = Path(experiment_dir) / "prediction_samples_test_ds"
        Path(sample_folder).mkdir(exist_ok=True)

        res = datamodule.test_dataset.res

        # Plot and save each sample
        for row in range(rows):
            arr = torch.clamp(batch["image"][row] / 255.0, min=0, max=1).numpy()
            image = arr.transpose(1, 2, 0)[:, :, chnls]

            # Plot Image
            axs[row, 0].imshow(image, cmap="gray")
            axs[row, 0].axis("off")

            # Plot Labels
            axs[row, 1].imshow(
                batch["mask"][row].squeeze().numpy(),
                cmap=cmap,
                vmin=0,
                vmax=vmax,
            )
            axs[row, 1].axis("off")

            # Plot Prediction
            axs[row, 2].imshow(
                batch['prediction'][row].squeeze().numpy(),
                cmap=cmap,
                vmin=0,
                vmax=vmax,
            )
            axs[row, 2].axis("off")

            # Save predicted tif
            bbox = batch["bounds"][row]
            out_transform = rasterio.transform.from_origin(bbox.minx, bbox.maxy, res[0], res[1])

            with rasterio.open(
                sample_folder / f"{row}.tif",
                "w",
                driver="GTiff",
                transform=out_transform,
                crs=datamodule.test_dataset.crs,
                count=1,
                width=datamodule.patch_size[1],
                height=datamodule.patch_size[0],
                dtype=rasterio.uint8,
                compress="lzw",
            ) as dst:
                dst.write(batch['prediction'][row].unsqueeze(0))

        axs[0, 0].set_title("Image")
        axs[0, 1].set_title("Label")
        axs[0, 2].set_title("Prediction")

        plt.tight_layout()

        fig.savefig(experiment_dir / "prediction_sampels_test_ds.png")

        if self.logger:
            self.logger.experiment.log({
                f"test/prediction_samples": wandb.Image(fig)
            })

        return fig