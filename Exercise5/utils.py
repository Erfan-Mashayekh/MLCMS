import numpy as np
from scipy import spatial, linalg, integrate
import matplotlib.pyplot as plt
import datafold.dynfold as dfold
import datafold.pcfold as pfold


def radial_basis(X:     np.ndarray,
                 grid:  np.ndarray,
                 epsilon: float) -> np.ndarray:
    """
    Computes a matrix (i,j) with the values phi_j(x_i) of the j-th basis
    function at the i-th data point. The basis functions are defined as
    phi_j(x) = exp(-||grid_j - x||^2 / epsilon^2)

    Args:
        X (np.ndarray, shape (N)): Data array of independent parameters
        grid (np.ndarray, shape (L)): Center points of radial basis functions
        epsilon (float): parameter governing the smoothness / peakedness of the
            gaussians.

    Returns:
        np.ndarray, shape (N, L): matrix (i,j) with entries phi_j(x_i)
    """
    if X.ndim == 1: # reshaping required in case of 1-dim input
        out = np.exp(-(spatial.distance_matrix(X[:, None],grid[:, None]) / epsilon)**2)
    else:
        out = np.exp(-(spatial.distance_matrix(X, grid) / epsilon) ** 2)
    return out


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
    coefficients, _, _, _ = linalg.lstsq(matrix, target, cond=cond)
    return coefficients


############################# TASK 1 #############################

def linear_basis_lst_sqr_approx(X: np.ndarray, F: np.ndarray) -> float:
    """
    Calculates the optimal parameter A of the linear function f(x) = A * x.
    The x_i in X have to be scalars.
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



############################# TASK 2 #############################

def t2_solve(x0: np.ndarray,
             start_time: float,
             end_time: float,
             A: np.ndarray) -> np.ndarray:
    linear_system = lambda t, x, A: A @ x
    x1_pre = np.zeros(x0.shape)
    for i in range(len(x0)):
        x1_pre[i, :] = integrate.solve_ivp(
                                    linear_system,
                                    t_span=[start_time, end_time],
                                    y0=x0[i, :],
                                    t_eval=[end_time],
                                    args=[A]
                                )["y"].reshape(2,)
    return x1_pre

def t2_trajectory(point: np.ndarray,
                    start_time: float,
                    end_time: float,
                    A: np.ndarray) -> np.ndarray:
    linear_system = lambda t, x, A: A @ x
    t = np.linspace(start_time, end_time, 10000)
    x1_pre = integrate.solve_ivp(
                            linear_system,
                            [start_time, end_time],
                            y0=point,
                            t_eval=t,
                            args=[A]
                        )["y"]
    return x1_pre


############################# TASK 3 #############################

def vector_field(x0: np.ndarray, x1: np.ndarray, dt:float) -> np.ndarray:
    """
    Compute vector fields from two sequential data points

    Args:
        x0 (np.ndarray): positions
        x1 (np.ndarray): next positions (advanced by evolution operator)
        dt (float): time step size (of evolution operator)

    Returns:
        np.ndarray: velocity vector field approximated by finite differences
    """
    return (x1 - x0) / dt


def compute_matrix_a(x0: np.ndarray, vector_field: np.ndarray) -> np.ndarray:
    """
    Computes the coefficient matrix (A) in linear system of Ax=v
    using least squares

    Args:
        x0 (np.ndarray): _description_
        vector_field (np.ndarray): _description_

    Returns:
        np.ndarray: optimal matrix in a least squares sense
    """
    return linalg.lstsq(x0, vector_field)[0].T


def solve_ivp_implicit(matrix: np.ndarray,
                       x: np.ndarray,
                       dt: float) -> np.ndarray:
    """
    Computes the initial value problem using a first order implicit method
    dx / dt = matrix * x

    Args:
        matrix (np.ndarray)
        x (np.ndarray): start value
        dt (float): time step size

    Returns:
        np.ndarray: _description_
    """
    matrix_b = np.identity(2) - dt * matrix
    sol, _, _, _ = linalg.lstsq(matrix_b, x.T)

    print(f'matrix: {matrix.shape}')
    print(f'target: {x.shape}')
    print(f'coefficients: {matrix_b.shape}')

    return sol.T

def compute_mse(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the mean squared error between to arrays
    Assumes that 0 is the batch axis and 1 is the instance axis

    Args:
        x1 (np.ndarray): shape (N, dim)
        x2 (np.ndarray): shape (N, dim)

    Returns:
        float: MSE Error
    """
    N = x1.shape[0]
    return np.sum((x1 - x2)**2)  / N


############################# TASK 4 #############################

def time_delay(X: np.ndarray,
               delta_t: int,
               is_periodic = False) -> np.ndarray:
    """
    applies time delay multiple times

    Args:
        X (np.ndarray): Data array of coordinate we want to do time-delay
            embedding
        delta_t (int): the delay time Delta t
        is_periodic (bool, optional): specifies, whether data is periodic.
            Defaults to False.

    Returns:
        np.ndarray: [x(t), x(t - delta_t), x(t - 2 * delta_t)]
    """
    if is_periodic:
        x1 = X
        x2 = np.roll(X, - delta_t)
        x3 = np.roll(X, - 2 * delta_t)
    else:
        x1 = X[:X.shape[0] - 2*delta_t]
        x2 = X[delta_t:X.shape[0] - delta_t]
        x3 = X[2*delta_t:]

    out = np.array([x1, x2, x3])
    return out


def _lorenz(t: float,
           X: np.ndarray,
           sigma: float,
           beta: float,
           rho: float) -> np.ndarray:
    """
    define the Lorenz equations (LE)

    Args:
        t (float): time (dummy parameter)
        X (np.ndarray): initial condition
        sigma (float): parameter of (LE)
        beta (float): parameter (LE)
        rho (float): parameter (LE)

    Returns:
        np.ndarray: derivative of X w.r.t time
    """
    x, y, z = X

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    dX_dt = [dx_dt, dy_dt, dz_dt]

    return dX_dt


def lorenz_trajectory(X, sigma, beta, rho, start_time, end_time):
    """
    integrate $x,y,z$ with respect to $t$ by \emph{solve_ivp} and plot the trajectory.
    :param X: the initial condition
    :param sigma, beta, rho: Lorenz parameters
    :param linewidth: the width of line in plot
    """
    t = np.linspace(start_time, end_time, 10000)

    # integrate the Lorenz equations
    # xyz = odeint(lorenz, X, t, args = (sigma, beta, rho))
    xyz = integrate.solve_ivp(_lorenz, t_span = [t[0], t[-1]], y0 = X, t_eval = t, args = (sigma, beta, rho))
    xyz = xyz["y"]

    # plot the trajectory in three-dimensional phase space
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(projection='3d')
    ax1.plot(xyz[0, :], xyz[1, :], xyz[2, :], label="trajectory")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend()
    ax1.set_title(fr"$\sigma={sigma}, \beta={round(beta,2)}, \rho={rho}, X_0=[{X[0]}, {X[1]}, {X[2]}]$")
    plt.show()
    return xyz


def t4_fun_radial_trajectory(X: np.ndarray,
                             GRID_b: np.ndarray,
                             epsilon,
                             coefficients,
                             start_time: float,
                             end_time: float) -> None:
    """
    integrate x,y,z with respect to t by solve_ivp and plot the trajectory.

    Args:
        X (np.ndarray): _description_
        GRID_b (np.ndarray): Grid points of radial basis
        epsilon (_type_): parameter controlling peakedness of radial basis
        coefficients (_type_): coefficients of model w.r.t. chosen basis
        start_time (float): start of trajectory
        end_time (float): end of trajectory
    """
    t = np.linspace(start_time, end_time, 10000)

    def fun_radial(t, y):
        y = y.reshape((1, y.size))
        return radial_basis(y, GRID_b, epsilon) @ coefficients

    # integrate the Lorenz equations
    # xyz = odeint(lorenz, X, t, args = (sigma, beta, rho))
    xyz = integrate.solve_ivp(fun_radial, t_span = [t[0], t[-1]], y0 = X, t_eval = t)
    xyz = xyz.y

    # plot the trajectory in three-dimensional phase space
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(projection='3d')
    ax1.plot(xyz[0, :], xyz[1, :], xyz[2, :], label="trajectory")
    ax1.plot(xyz[0, 0], xyz[1, 0], xyz[2, 0], 'o', color="r", label="initial point")
    ax1.set_xlabel("x()")
    ax1.set_ylabel("$x(t + /Delta t)$")
    ax1.set_zlabel("$x(t + 2*/Delta t)$")
    ax1.legend()
    ax1.set_title("trajectories of the training")


############################# TASK 5 #############################

def compute_eigenfunctions_datafold(x, sample):
    """
    Compute the eigenfunctions for the given dataset 'u_pca' using a Datafold.
    Returns: eigenfunctions = [eigenvectors, eigenvalues]
    """

    # Create the Manifold using Datafold lib
    positions_pcm = pfold.PCManifold(x)
    positions_pcm.optimize_parameters()

    # Compute the kernel and the eigenfunctions using Datafold lib
    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(epsilon=positions_pcm.kernel.epsilon),
        n_eigenpairs=sample,
        dist_kwargs=dict(cut_off=positions_pcm.cut_off),
    )
    dmap = dmap.fit(positions_pcm)
    evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

    return evecs, evals