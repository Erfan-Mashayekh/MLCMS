import numpy as np
import scipy
import matplotlib.pyplot as plt


############################# TASK 1 #############################

def linear_basis_lst_sqr_approx(X: np.ndarray, F: np.ndarray) -> float:
    """
    Calculates the optimal parameter A of the linear function f(x) = A * x.
    The entries x_i of X and f_i of F have to be scalars.
    The output is optimal in a least squares sense.

    Args:
        X (np.ndarray): Data array of independent parameters (input)
        F (np.ndarray): Data array of dependent parameters (label)

    Returns:
        float: slope a of the linear function f(x) = a * x such that
            ||F - a * X||^2 is minimized w.r.t. a.
    """
    Xt_X = X.dot(X)
    Xt_F = X.dot(F)
    return Xt_F / Xt_X


def radial_basis(X:     np.ndarray,
                 grid:  np.ndarray,
                 epsilon: float) -> np.ndarray:
    """
    Computes a matrix (i,j) with the values phi_j(x_i) of the j-th basis
    function at the i-th data point. The basis functions are defined as
    phi_j(x) = exp(-||grid_j - x||^2 / epsilon^2)
    The entries x_i of X have to be scalars.

    Args:
        X (np.ndarray, shape (N)): Data array of independent parameters
        grid (np.ndarray, shape (L)): Center points of radial basis functions
        epsilon (float): parameter governing the smoothness / peakedness of the
            gaussians.

    Returns:
        np.ndarray, shape (N, L): matrix (i,j) with entries phi_j(x_i)
    """
    return np.exp(-(scipy.spatial.distance_matrix(
                            X[:, None],
                            grid[:, None])
                    / epsilon)**2)


def radial_basis_lst_sqr_approx(X: np.ndarray,
                                F: np.ndarray,
                                grid: np.ndarray,
                                epsilon: float,
                                cond: float) -> np.ndarray:
    """
    Calculates the optimal coefficients C for the ansatz function:
    f(x) = sum_{l=1}^L c_l phi_l(x) such that C minimizes
    ||F - phi(X) @ C.T||^2 where phi_l are the radial basis functions used in
    the radial_basis method.

    Args:
        X (np.ndarray): Data array of independent parameters (input)
        F (np.ndarray): Data array of dependent parameters (label)
        grid (np.ndarray): Center points of radial basis functions
        epsilon (float): parameter governing the smoothness / peakedness of the
            gaussians.
        cond (float): Cutoff for 'small' singular values; used to determine
            effective rank of matrix (see below). Singular values smaller than
            cond * largest_singular_value are considered zero.

    Returns:
        np.ndarray: optimal coefficients w.r.t the basis defined by grid
            in a least squares sense
    """
    phi_X = radial_basis(X, grid, epsilon)
    matrix = phi_X.T @ phi_X
    target = phi_X.T @ F
    coefficients, _, _, _ = scipy.linalg.lstsq(matrix, target, cond=cond)
    return coefficients


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