"""
Inspired by https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""
import torch.nn as nn
from torch import Tensor

import nnBase
from nnBase import Residual


class Encoder(nn.Module):
    def __init__(self,
                 n_in: int,
                 latent_dim: int,
                 n_hidden: int,
                 activation: nn.Module) -> None:
        super().__init__()
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
    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 n_hidden: int,
                 activation: nn.Module) -> None:
        super().__init__()
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


class Autoencoder(nnBase.BaseSR):
    """
    Autoencoder used to learn embedding of Swiss Roll
    """
    example_input_array = Tensor([1.0, 1.5, 2.5])

    def __init__(self,
                 n_in: int,
                 n_out: int,
                 latent_dim: int,
                 n_hidden=256,
                 activation=nn.ELU) -> None:
        """
        Args:
            n_in (int): data dimension of input
            n_out (int): data dimension of output (redundant with n_in, but
                kept due to keep compatible with other methods)
            latent_dim (int): dimension of latent space / bottleneck
        """
        super().__init__()
        self.encoder = Encoder(n_in, latent_dim, n_hidden, activation)
        self.decoder = Decoder(n_out, latent_dim, n_hidden, activation)
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _loss(self, batch) -> Tensor:
        """
        computes the MSE-loss of the reconstructed and original data
        """
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=1).mean(dim=[0])  # batch dim: 0, data dim: 1
        return loss

