"""
Inspired by https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""

import torch
import pytorch_lightning as pl

import torch.nn as nn
from torch import optim, Tensor

from typing import Any

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
    def __init__(self, features: int, activation = nn.ELU) -> None:
        super().__init__()
        self.res = nn.Sequential(
                nn.Linear(features, features),
                activation()
            )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.res(x)


class Encoder(pl.LightningModule):
    def __init__(self, n_in: int, latent_dim: int) -> None:
        super().__init__()
        activation = nn.ELU
        n_hidden = 256
        self.net = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, latent_dim)
            )

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
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
        loss =  self._reconstruction_loss(batch)
        loss += self._l2_regularization()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("test_loss", loss)

    def _l2_regularization(self, reg_strength=0.001):
        out = sum(torch.square(p).sum() for p in self.parameters())
        return reg_strength * out

    def _reconstruction_loss(self, batch):
        # TODO: Check Loss function
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss

    def _l2_regularization(self, reg_strength=0.001):
        out = sum(torch.square(p).sum() for p in self.parameters())
        return reg_strength * out


class Decoder(nn.Module):
    def __init__(self, n_out: int, latent_dim: int) -> None:
        super().__init__()
        activation = nn.ELU
        n_hidden = 256
        self.net = nn.Sequential(
                nn.Linear(latent_dim, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_out)
            )

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
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
        loss =  self._reconstruction_loss(batch)
        loss += self._l2_regularization()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("test_loss", loss)

    def _l2_regularization(self, reg_strength=0.001):
        out = sum(torch.square(p).sum() for p in self.parameters())
        return reg_strength * out

    def _reconstruction_loss(self, batch):
        # TODO: Check Loss function
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss

    def _l2_regularization(self, reg_strength=0.001):
        out = sum(torch.square(p).sum() for p in self.parameters())
        return reg_strength * out



class Autoencoder(pl.LightningModule):
    example_input_array = Tensor([1.0, 1.5, 2.5])

    def __init__(self,
                 n_in: int,
                 n_out: int,
                 latent_dim: int) -> None:
        super().__init__()

        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()

        self.encoder = Encoder(n_in, latent_dim)
        self.decoder = Decoder(n_out, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
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
        loss =  self._reconstruction_loss(batch)
        loss += self._l2_regularization()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("test_loss", loss)

    def _l2_regularization(self, reg_strength=0.001):
        out = sum(torch.square(p).sum() for p in self.parameters())
        return reg_strength * out

    def _l1_regularization(self, reg_strength=0.001):
        out = sum(torch.abs(p).sum() for p in self.parameters())
        return reg_strength * out

    def _reconstruction_loss(self, batch):
        # TODO: Check Loss function
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss

