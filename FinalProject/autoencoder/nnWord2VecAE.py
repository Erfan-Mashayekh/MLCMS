import torch
import torch.nn as nn
from torch import Tensor, optim

import nnBase
from nnBase import Residual

from typing import Any


class Encoder(nn.Module):
    def __init__(self, n_in: int, latent_dim: int) -> None:
        super().__init__()
        activation = nn.ELU
        n_hidden = 512
        self.net = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation()
            )
        self.residual = nn.Sequential(
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden)
            )
        self.final = nn.Linear(n_hidden, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        x = self.residual(x)
        x = self.final(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_out: int, latent_dim: int) -> None:
        super().__init__()
        activation = nn.ELU
        n_hidden = 512
        self.net = nn.Sequential(
                nn.Linear(latent_dim, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation(),
                nn.Linear(n_hidden, n_hidden),
                activation()
            )
        self.residual = nn.Sequential(
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden)
            )
        self.final = nn.Linear(n_hidden, n_out)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        x = self.residual(x)
        x = self.final(x)
        return x


class Word2VecAutoencoder(nnBase.BaseSR):
    example_input_array = torch.ones(300)

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
               "monitor": "compression_loss"}
        return out

    def training_step(self, batch, batch_idx) -> Tensor:
        loss =  self._loss(batch)
        loss += self._l2_regularization()
        self.log("compression_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> Tensor:
        loss =  self._loss(batch)
        self.log("final_loss", loss)
        return loss

    def _loss(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss

