import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor, optim
from abc import abstractmethod


class Residual(nn.Module):
    def __init__(self, features: int, activation=nn.ELU) -> None:
        super().__init__()
        self.res = nn.Sequential(
                nn.Linear(features, features),
                activation()
            )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.res(x)


class BaseSR(pl.LightningModule):
    reg_strength = 0.001
    learning_rate = 1e-3
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=0.2,
                        patience=20,
                        min_lr=5e-5)
        out = {"optimizer": optimizer,
               "lr_scheduler": scheduler,
               "monitor": "val_loss"}
        return out

    def training_step(self, batch, batch_idx) -> Tensor:
        loss =  self._loss(batch)
        loss += self._l2_regularization()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx) -> None:
        loss = self._loss(batch)
        self.log("test_loss", loss)

    def _l2_regularization(self):
        """
        Computes the L2 regularization term to be added to the loss

        Args:
            reg_strength (float, optional): Defaults to 0.005.
        """
        out = sum(torch.square(p).sum() for p in self.parameters())
        return self.reg_strength * out

    def _l1_regularization(self):
        out = sum(torch.abs(p).sum() for p in self.parameters())
        return self.reg_strength * out

    def turn_off_regularization(self):
        self.reg_strength = 0

    @abstractmethod
    def _loss(self, batch):
        raise NotImplementedError()

