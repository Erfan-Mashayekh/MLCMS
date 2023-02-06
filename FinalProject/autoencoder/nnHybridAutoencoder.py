"""
Inspired by https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""
import torch.nn as nn
from torch import Tensor
from typing import Any

import nnBase
from nnBase import Residual


class Encoder(nnBase.BaseSR):
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


class Decoder(nnBase.BaseSR):
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


class PretrainedAE(nnBase.BaseSR):
    """
    Pretrained autoencoder used to learn embedding of Swiss Roll
    assumes input dimension of 3 and 2D latent space.

    Child class of nn.Module -> pl.LightningModule -> BaseSR
    """
    def __init__(self,
                 encoder_path: str,
                 decoder_path = None) -> None:
        """
        Args:
            encoder_path (str): path to encoder checkpoint from which to
                load weights
            decoder_path (_type_, optional): path to decoder checkpoint.
                if None then default initialize. Defaults to None.
        """
        super().__init__()
        self.encoder = Encoder.load_from_checkpoint(encoder_path)
        for param in self.encoder.parameters(): # freeze encoder params
            param.requires_grad = False

        if decoder_path == None:
            self.decoder = Decoder(3, 2)
        else:
            self.decoder = Decoder.load_from_checkpoint(decoder_path)
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
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

    def unfreeze_encoder(self) -> None:
        """
        Unfreezes the encoder parameters for training and reduces the
        learning rate.
        """
        self.learning_rate = 1e-5
        for param in self.encoder.parameters():
            param.requires_grad = True