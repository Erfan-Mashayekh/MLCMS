"""
Inspired by https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""

import torch
import pytorch_lightning as pl

import torch.nn as nn
from torch import optim, Tensor

from typing import Any, Tuple

from nnModules import Residual

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
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
                nn.Dropout(p=0.05),
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden),
                nn.Linear(n_hidden, latent_dim)
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x


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
                nn.Dropout(p=0.05),
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden),
                nn.Linear(n_hidden, n_out)
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):
    """
    Autoencoder used to learn embedding of Swiss Roll
    """
    example_input_array = Tensor([1.0, 1.5, 2.5])

    def __init__(self,
                 n_in: int,
                 n_out: int,
                 latent_dim: int) -> None:
        """
        Args:
            n_in (int): data dimension of input
            n_out (int): data dimension of output (redundant with n_in, but
                kept due to keep compatible with other methods)
            latent_dim (int): dimension of latent space / bottleneck
        """
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

    def _l2_regularization(self, reg_strength=0.005) -> Tensor:
        """
        Computes the L2 regularization term to be added to the loss

        Args:
            reg_strength (float, optional): Defaults to 0.005.
        """
        out = sum(torch.square(p).sum() for p in self.parameters())
        return reg_strength * out

    def _l1_regularization(self, reg_strength=0.005):
        out = sum(torch.abs(p).sum() for p in self.parameters())
        return reg_strength * out

    def _reconstruction_loss(self, batch) -> Tensor:
        """
        computes the MSE-loss of the reconstructed and original data
        """
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss

