"""
Inspired by https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""

import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor

from typing import Any, Tuple



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, n_in: int, latent_dim: int) -> None:
        super().__init__()
        self.activation = nn.ELU
        self.net = nn.Sequential(
                nn.Linear(n_in, 256),
                self.activation(),
                nn.Linear(256, 256),
                self.activation(),
                nn.Linear(256, 256),
                self.activation(),
                nn.Linear(256, 256),
                self.activation(),
                nn.Linear(256, 256),
                self.activation()
            )

        self.mu_layer   = nn.Linear(256, latent_dim)
        self.sigma_layer= nn.Linear(256, latent_dim)

        self.kl = 0 # (effective) KL-Divergence

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.net(x))
        mu = F.relu(self.mu_layer(x))
        sigma = torch.exp(self.sigma_layer(x))

        self.kl = ((sigma ** 2 + mu ** 2) / 2 - torch.log(sigma)).sum()
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, n_out: int, latent_dim: int) -> None:
        super().__init__()
        self.activation = nn.ELU
        self.net = nn.Sequential(
                nn.Linear(latent_dim, 256),
                self.activation(),
                nn.Linear(256, 256),
                self.activation(),
                nn.Linear(256, 256),
                self.activation(),
                nn.Linear(256, 256),
                self.activation(),
                nn.Linear(256, 256),
                self.activation(),
                nn.Linear(256, n_out),
                self.activation()
            )
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(self, z: Tensor) -> Tensor:
        x = self.net(z)
        return x


class VariationalAutoencoder(pl.LightningModule):
    example_input_array = Tensor([1.0, 1.5, 2.5])

    def __init__(self,
                 data_dim: int,
                 latent_dim: int) -> None:
        super().__init__()

        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()

        self.dim = data_dim
        self.encoder = Encoder(data_dim, latent_dim)
        self.decoder = Decoder(data_dim, latent_dim)

        self.N_dist = torch.distributions.Normal(0, 1)
        self.N_dist.loc = self.N_dist.loc.to(DEVICE)
        self.N_dist.scale = self.N_dist.scale.to(DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        mu, sigma = self.encoder(x)
        z = mu + sigma * self.N_dist.sample(mu.shape)
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
        sigma_decoder = torch.exp(self.decoder.log_sigma)
        loss = self._reconstruction_loss(batch) / (2 * sigma_decoder**2) \
                + self.encoder.kl \
                + batch.shape[0] * self.dim * self.decoder.log_sigma
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx) -> None:
        loss = self._reconstruction_loss(batch)
        self.log("test_loss", loss)

    def _l2_regularisation(self, reg_strength=0.001):
        out = sum(torch.square(p).sum() for p in self.parameters())
        return reg_strength * out

    def _l1_regularisation(self, reg_strength=0.001):
        out = sum(torch.abs(p).sum() for p in self.parameters())
        return reg_strength * out

    def _reconstruction_loss(self, batch):
        # TODO: Check Loss function
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss

