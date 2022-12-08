import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torch import Tensor
from typing import Tuple


class Encoder(nn.Module):
    """
    Encoder neural network consisting of two fully connected hidden layers
    of 256 units each and ReLU activation functions.
    """
    def __init__(self, latent_dims: int):
        super(Encoder, self).__init__()
        self.linear_hidden1 = nn.Linear(2, 256)
        self.linear_hidden2 = nn.Linear(256, 256)
        self.linear_hidden3 = nn.Linear(256, 256)
        self.linear_mu_out = nn.Linear(256, latent_dims)
        self.linear_sigma_out = nn.Linear(256, latent_dims)
        self.kl = 0  # (effective) KL-Divergence

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Maps image input to a gaussian distribution in lower dimensional
        latent space.
        @param x: input image
        @return: mean and standard deviation
        """
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear_hidden1(x))
        x = F.relu(self.linear_hidden2(x))
        x = F.relu(self.linear_hidden3(x))
        mu = self.linear_mu_out(x)
        sigma = torch.exp(self.linear_sigma_out(x))

        self.kl = ((sigma ** 2 + mu ** 2) / 2 - torch.log(sigma)).sum()
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dims: int):
        super(Decoder, self).__init__()
        self.linear_hidden1 = nn.Linear(latent_dims, 256)
        self.linear_hidden2 = nn.Linear(256, 256)
        self.linear_hidden3 = nn.Linear(256, 256)
        self.linear_out = nn.Linear(256, 2)
        # Use log_sigma to enforce positive sigma:
        self.log_sigma = nn.Parameter(torch.zeros(1))  # register standard deviation as trainable parameter

    def forward(self, z: Tensor) -> Tensor:
        z = F.relu(self.linear_hidden1(z))
        z = F.relu(self.linear_hidden2(z))
        x = F.relu(self.linear_hidden3(z))
        z = torch.sigmoid(self.linear_out(z))  # apply sigmoid to output layer to naturally squash z into range 0, 1
        return z


class VariationalAutoencoder(nn.Module):
    """
    Consists of an encoder and decoder.
    The encoder maps an input x to a distribution q(z) over a low dimensional latent space of
    independent variables.
    The decoder takes an input z drawn from the distribution q(z) over the latent space given
    an encoding input x and tries to reconstruct the original input x.
    """
    def __init__(self, latent_dims: int):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

        self.N_latent = torch.distributions.Normal(0, 1)
        self.N_latent.loc = self.N_latent.loc.cuda()  # hack to get sampling on the GPU
        self.N_latent.scale = self.N_latent.scale.cuda()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network. First encoding and decoding the data
        @param x: input used to compute latent distribution q(z) which tries to approximate p(z|x)
        @return: central value of posterior distribution p(x|z) where z is sampled from the latent distribution
        """
        mu, sigma = self.encoder(x)
        z = mu + sigma * self.N_latent.sample(mu.shape)  # sample input z from Gaussian Distribution
        mu_decoder = self.decoder(z)
        return mu_decoder
