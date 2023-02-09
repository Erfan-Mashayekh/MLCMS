import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def plot_eigenfunctions(n_evecs, nr_samples, nr_samples_plot, part, evecs, evals, time):
    from datafold.utils.plot import plot_pairwise_eigenvector
    """
    Plot different graphs with respect to selected part.
    """
    # print(f"{n_evecs} Largest eigenvalues with respect this dataset: \n {evals}")
    idx_plot = np.random.permutation(nr_samples)[0:nr_samples_plot]

    if part == 1 or part == 2:
        plot_pairwise_eigenvector(
            eigenvectors=evecs[idx_plot, :],
            n=1,
            fig_params=dict(figsize=[7, 7]),
            scatter_params=dict(cmap=plt.cm.Spectral, c=time[idx_plot],s=0.5)
        ) 
    elif part == 3:
        pass
    
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    # ax.scatter(x, y, cmap=plt.cm.Spectral, c=points_color, s=0.5, alpha=0.8)
    ax.scatter(x, y, c=points_color, s=0.5, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_3d_scatter(ax, points, title=None):
    x, y, z = points.T
    # ax.scatter(x, y, cmap=plt.cm.Spectral, c=points_color, s=0.5, alpha=0.8)
    ax.scatter(x, y, z, s=0.5, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.zaxis.set_major_formatter(ticker.NullFormatter())

def plot_3d(points, title):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white", constrained_layout=True)
    ax = plt.axes(projection ="3d")
    fig.suptitle(title, size=16)
    add_3d_scatter(ax, points)
    plt.show()