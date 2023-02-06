"""
Inspired by https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""
import torch.nn as nn
from torch import Tensor
from typing import Any

from nnModules import Residual, BaseSR


class Encoder(BaseSR):
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
                nn.BatchNorm1d(n_hidden),
                activation(),
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden),
                nn.Linear(n_hidden, latent_dim)
            )

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

    def _loss(self, batch):
        """
        computes the MSE-loss of the reconstructed and original data
        """
        x, z = batch
        z_hat = self.forward(x)
        loss = nn.functional.mse_loss(z, z_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss


class Decoder(BaseSR):
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
                nn.BatchNorm1d(n_hidden),
                activation(),
                Residual(n_hidden),
                Residual(n_hidden),
                Residual(n_hidden),
                nn.Linear(n_hidden, n_out)
            )

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

    def _loss(self, batch):
        """
        computes the MSE-loss of the reconstructed and original data
        """
        z, x = batch
        x_hat = self.forward(z)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss


class PretrainedAE(BaseSR):
    def __init__(self, encoder_path: str, decoder_path = None) -> None:
        super().__init__()
        self.encoder = Encoder.load_from_checkpoint(encoder_path)
        for param in self.encoder.parameters():
            param.requires_grad = False

        if decoder_path == None:
            self.decoder = Decoder(3, 2)
        else:
            self.decoder = Decoder.load_from_checkpoint(decoder_path)
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # x_hat = self.residual(x_hat)
        return x_hat

    def _loss(self, batch):
        """
        computes the MSE-loss of the reconstructed and original data
        """
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss