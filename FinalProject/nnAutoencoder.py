"""
Inspired by https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""

import torch
import pytorch_lightning as pl

import torch.nn as nn
from torch import optim, Tensor

from typing import Any


class Encoder(nn.Module):
    def __init__(self, n_in: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(3, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 2),
                nn.Tanh()
            )

    def forward(self, x):
        x = self.net(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_out: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(2, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3)
            )

    def forward(self, x):
        x = self.net(x)
        return x


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

    def forward(self, x):
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
        # TODO: what does this return
        out = {"optimizer": optimizer,
               "lr_scheduler": scheduler,
               "monitor": "val_loss"}
        return out

    def training_step(self, batch, batch_idx) -> Tensor:
        loss = self._reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("test_loss", loss)

    def _reconstruction_loss(self, batch):
        # TODO: Check Loss function
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss

