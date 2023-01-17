import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

############################# TASK 1 #############################

def basic_data_plot_task1(X: np.ndarray, F: np.ndarray) -> None:
    """
    configures a basic plot and adds data.

    Args:
        X (np.ndarray): Data array of independent parameters (input)
        F (np.ndarray): Data array of dependent parameters (label)
    """
    plt.scatter(X, F, s=2, c="red", label="data")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    plt.grid(True)

############################# TASK 3 #############################

def plot_phase_portrait_linear(w: int, A: np.ndarray, x):
    """
    Plots a linear vector field in a streamplot, defined with
    X and Y coordinates and the matrix A.
    """
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    UV = UV

    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    ax = plt.gca()
    ax.streamplot(X, Y, U, V, density=[1, 1])
    ax.set_aspect(1)
    ax.plot(x[0,:], x[1,:], label="trajectory")

    ax.legend(loc='lower left')
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_title("the trajectory of initial point(10,10)")
    ax.set_xlabel("coordinate 1")
    ax.set_ylabel("coordinate 2")
    return ax

############################# TASK 3 #############################

def t3_plot_points(*data: np.ndarray) -> None:

    plt.figure(figsize=(5, 5))
    for xy_array in data:
        plt.scatter(xy_array[:, 0], xy_array[:, 1], s=2)
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 2")
    plt.show()


def t3_plot_vector_fields(vec_field: np.ndarray,
                          pos: np.ndarray,
                          ax, title: str):
    """
    Plots a non-linear vector fields, defined with X and Y coordinates
    and the derivatives U and V.
    """
    U, V = vec_field[:,0], vec_field[:,1]
    X, Y = pos[:,0], pos[:,1]
    ax.quiver(X, Y, U, V, color='black', units='xy')
    ax.set_title(title)
    ax.set_aspect(1)
    ax.set_xlabel("coordinate 1")
    ax.set_ylabel("coordinate 2")
    return ax