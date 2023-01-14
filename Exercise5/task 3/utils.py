import numpy as np
import scipy
from scipy import spatial
from scipy.integrate import solve_ivp
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
    Xt_X = X.T @ X
    Xt_F = X.T @ F
    return Xt_F / Xt_X


def radial_basis(X: np.ndarray,
                 grid: np.ndarray,
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
    return np.exp(-(spatial.distance_matrix(X, grid) / epsilon) ** 2)


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

    print(f'matrix: {matrix.shape}')
    print(f'target: {target.shape}')
    print(f'coefficients: {coefficients.shape}')

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


############################# TASK 3 #############################

def vector_field(x0, x1, dt):
    """
    Compute vector fields from two sequential data points
    :param x0:
    :param x1:
    :param dt:
    :return:
    """
    return (x1 - x0) / dt


def compute_matrix_a(x0, vector_field):
    """
    Computes the coefficient matrix (A) in linear system of Ax=v using least square
    :param x0: x
    :param vector_field: v
    :return: A
    """
    return scipy.linalg.lstsq(x0, vector_field)[0].T


def solve_ivp_implicit(matrix, x, dt):
    """
    Computes the initial value problem using a first order implicit method
    :param matrix:
    :param x:
    :param dt:
    :return:
    """
    matrix_b = np.identity(2) - dt * matrix
    sol, _, _, _ = scipy.linalg.lstsq(matrix_b, x.T)

    print(f'matrix: {matrix.shape}')
    print(f'target: {x.shape}')
    print(f'coefficients: {matrix_b.shape}')

    return sol.T


class Solver:

    def __init__(self, x, t_start, t_end):
        self.x = x
        self.t_start = t_start
        self.t_end = t_end


class SolverLinear(Solver):

    def __init__(self, x, t_start, t_end, matrix_a):
        super().__init__(x, t_start, t_end)
        self.matrix_a = matrix_a

    def fun(self, t, y):
        return y @ self.matrix_a.T

    def solve_linear_system(self):
        x1 = np.zeros(self.x.shape)
        i = 0
        for x in self.x:
            x1[i, :] = solve_ivp(self.fun, t_span=np.array([self.t_start, self.t_end]), y0=x).y.T[-1]
            i += 1
        return x1


class SolverRadialBasis(Solver):

    def __init__(self, x, t_start, t_end, coefficients, GRID_b, EPSILON):
        super().__init__(x, t_start, t_end)
        self.coefficients = coefficients
        self.GRID_b = GRID_b
        self.EPSILON = EPSILON

    def fun(self, t, y):
        y = y.reshape((1, y.size))
        return radial_basis(y, self.GRID_b, self.EPSILON) @ self.coefficients

    def solve_linear_system(self):
        x1 = np.zeros(self.x.shape)
        i = 0
        for x in self.x:
            x1[i, :] = solve_ivp(self.fun, t_span=np.array([self.t_start, self.t_end]), y0=x).y.T[-1]
            i += 1
        return x1


def compute_error(x, x_predicted):
    return np.mean(np.linalg.norm(x-x_predicted, axis=0)**2 / x.shape[0])
