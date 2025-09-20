from typing import Callable, Iterator

import lightning as L
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Optimizer

from binding_affinity.data.types import AtomicInterfaceGraph
from binding_affinity.utils.metrics import create_regression_metrics


class AffinityModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_fn: Callable[[Iterator[nn.Parameter]], Optimizer],
        loss: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer_fn(self.model.parameters())
        self.loss = loss
        self.train_metrics = create_regression_metrics(prefix="train")
        self.val_metrics = create_regression_metrics(prefix="val")
        self.test_metrics = create_regression_metrics(prefix="test")

    def training_step(self, batch: AtomicInterfaceGraph, batch_idx: int = 0) -> dict[str, Tensor]:
        predictions = self.model.forward(batch).flatten()
        loss = self.loss(predictions, batch.affinity)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metrics.update(predictions, batch.affinity)
        return {"loss": loss, "predictions": predictions}

    def validation_step(self, batch: AtomicInterfaceGraph, batch_idx: int = 0) -> dict[str, Tensor]:
        predictions = self.model.forward(batch).flatten()
        loss = self.loss(predictions, batch.affinity)

        self.val_metrics.update(predictions, batch.affinity)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "predictions": predictions}

    def test_step(self, batch: AtomicInterfaceGraph, batch_idx: int = 0) -> dict[str, Tensor]:
        predictions = self.model.forward(batch).flatten()
        loss = self.loss(predictions, batch.affinity)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.update(predictions, batch.affinity)
        return {"loss": loss, "predictions": predictions}

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.log_dict(self.val_metrics.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer
