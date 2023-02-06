import matplotlib.pyplot as  plt
import numpy as np
import torch
from torch import Tensor
from data import get_swiss_roll_data

plt.rcParams['figure.dpi'] = 200


def plot_swiss_roll(data: Tensor | np.ndarray, c=None) -> None:
    if type(data) == Tensor:
        data = data.detach().numpy()
    assert type(data) == np.ndarray
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(*data.T, s=0.1, c=c)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.show()


def plot_swiss_roll_error(x_hat: Tensor, x_true = None) -> None:
    x_hat = x_hat.detach()
    if x_true == None:
        x_true, _ = get_swiss_roll_data(x_hat.shape[0])
    error = (x_hat - x_true).norm(dim=1)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Relative Error of Reconstruction")
    ax.scatter3D(*x_true.T, s=0.1, c=error, cmap="Reds")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.show()


def plot_2d_latent(*data: Tensor | np.ndarray, c=None) -> None:
    for d in data:
        if type(d) == Tensor:
            d = d.detach().numpy()
        assert type(d) == np.ndarray
        plt.scatter(d[:,0], d[:,1], s=0.1, c=c)
    plt.title("Latent Space")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.show()