import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

############################# TASK 3 #############################


def read_points():
    """
    Reads the file and extracts two vector of x0 and x1
    :return: x0, x1
    """
    x0 = np.genfromtxt("nonlinear_vectorfield_data_x0.txt", delimiter=" ")
    x1 = np.genfromtxt("nonlinear_vectorfield_data_x1.txt", delimiter=" ")
    print(f"x0 shape: {x0.shape}")
    print(f"x1 shape: {x1.shape}")

    plt.figure(figsize=(5, 5))
    plt.scatter(x0[:, 0], x0[:, 1], s=5, c="blue")
    plt.scatter(x1[:, 0], x1[:, 1], s=5, c="black")
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 2")
    plt.show()

    return x0, x1


def plot_phase_portrait_linear_matrix(A, X, Y, trj, ax):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    """
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)

    # fig = plt.figure(figsize=(7, 7))
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
    #gs = gridspec.GridSpec(nrows=1, ncols=1)

    #  Varying density along a streamline
    #ax0 = fig.add_subplot(gs[0, 0])
    # ax.quiver(X, Y, U, V, color='b', units='xy')
    ax.streamplot(X, Y, U, V, density=[0.9, 1])
    ax.scatter(trj[:, 0], trj[:, 1], s=5, c="red")
    ax.set_title('Streamplot for linear vector field A*x')
    ax.set_aspect(1)
    return ax


def plot_phase_portrait_linear(U, V, X, Y, ax):
    """
    Plots a non-linear vector field in a streamplot, defined with X and Y coordinates and the derivatives U and V.
    """
    #fig1 = plt.figure(figsize=(7, 7))
    #gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1])

    #  Varying density along a streamlne
    # ax1 = fig1.add_subplot(gs[0, 0])
    # ax.quiver(X, Y, U, V, color='black', units='xy')
    ax.streamplot(X, Y, U, V, density=[0.9, 1])
    ax.set_title(f'Streamplot for linear vector field')
    ax.set_aspect(1)
    return ax

def plot_vector_fields(U, V, X, Y, ax, title):
    """
    Plots a non-linear vector fields, defined with X and Y coordinates and the derivatives U and V.
    """
    ax.quiver(X, Y, U, V, color='black', units='xy')
    ax.set_title(title)
    ax.set_aspect(1)
    return ax

def plot_points(x0, x1, x):

    plt.figure(figsize=(5, 5))
    plt.scatter(x0[:, 0], x0[:, 1], s=5, c="blue")
    plt.scatter(x1[:, 0], x1[:, 1], s=5, c="black")
    plt.scatter(x[:, 0], x[:, 1], s=5, c="red")
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 2")
    plt.show()
