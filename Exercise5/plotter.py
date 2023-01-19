import matplotlib.pyplot as plt
import numpy as np
from utils import time_delay, t2_trajectory


############################# TASK 1 #############################

def t1_basic_data_plot(X: np.ndarray, F: np.ndarray) -> None:
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


############################# TASK 2 #############################

def plot_phase_portrait_linear(w: int, A: np.ndarray):
    """
    Plots phase portrait in a streamplot, defined with
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
    # ax.plot(x[0,:], x[1,:], label="trajectory")

    # ax.legend(loc='lower left')
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_title("the phase portrait of Ax")
    ax.set_xlabel("coordinate 1")
    ax.set_ylabel("coordinate 2")
    return ax

def plot_trajectory_with_phase_portrait_linear(w: int, 
                                               A: np.ndarray, 
                                               x0_point: np.ndarray, 
                                               start_time: int, 
                                               end_time: int):
    """
    Visualize the trajectory of start point x0_point with the phase portrait
    """
    x1_pre = t2_trajectory(x0_point, start_time, end_time, A)

    ax = plot_phase_portrait_linear(w, A)
    ax.plot(x1_pre[0,:], x1_pre[1,:], label="trajectory")
    ax.set_title(f"the trajectory of {x0_point} with the phase portrait")

    ax.legend(loc='lower left')



############################# TASK 3 #############################

def t3_plot_points(*data: np.ndarray, fname: str = None) -> None:
    """
    Plots arbitrarily many datasets of two dimensional plots in one figure.
    Each set of points gets a different color.
    If filename is provided the figure is saved.

    Args:
        *data (np.ndarray):
        fname (str, optional): filename. Defaults to None.
    """
    plt.figure(figsize=(5, 5))
    for xy_array in data:
        plt.scatter(xy_array[:, 0], xy_array[:, 1], s=2)
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 2")
    if fname != None:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()


def t3_plot_vector_fields(vec_field: np.ndarray,
                          pos: np.ndarray,
                          ax,
                          title: str):
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


############################# TASK 4 #############################


def t4_takens(X: np.ndarray, 
              delta_t: np.ndarray, 
              name: str, 
              is_periodic = False) -> None:
    """
    Plots the specified coordinate against its delayed version with different delta_t

    Args:
        X (np.ndarray): the specified coordinate
        delta_t (np.ndarray): the delay time Delta t
        name(str): name of the specified coordinate
    """
    # X, Y, Z = time_delay(X, delta_t, is_periodic)

    # fig = plt.figure(figsize=(20, 10))
    fig = plt.figure(figsize=(20, 5*len(delta_t)))

    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # ax1.plot(X, Y, Z)
    # ax1.set_xlabel("$x(t)$")
    # ax1.set_ylabel("$x(t- \Delta t)$")
    # ax1.set_zlabel("$x(t- 2\Delta t)$")

    for i in range(len(delta_t)):
        x = time_delay(X, delta_t[i], is_periodic)
        ax = fig.add_subplot(len(delta_t), 4, i+1, projection='3d')
        ax.plot(x[0], x[1], x[2])
        ax.set_xlabel(name + "$(t)$")
        ax.set_ylabel(name + "$(t+ \Delta t)$")
        ax.set_zlabel(name + "$(t+ 2\Delta t)$")
        ax.set_title(f"$\Delta t ={delta_t[i]}$")
    plt.show()

############################# TASK 5 #############################

def plot_pca(x: np.ndarray,
             u: np.ndarray,
             window_shape: np.ndarray,
             row: int, col: int) -> None:
    """
    Plots Principal component analysis based on different measurements taken in different areas.
    """
    fig, ax = plt.subplots(row, col, figsize=(row * 5, col * 5), subplot_kw=dict(projection='3d'))
    for i in range(row):
        for j in range(col):
            ax[i][j].scatter(*x.T, s=1, c=u[:window_shape[0], row * i + j])
    plt.show()


def plot_arclength_velocities(vel: np.ndarray, arclength: np.ndarray) -> None:
    """
    Plots the velocity on archlength over arclength of the curve

    Args:
        vel (np.ndarray): velocity
        time (np.ndarray): time
    """
    plt.rcParams["figure.figsize"] = (10, 5)
    curve_arclength = 2 * np.pi * arclength / np.max(arclength)
    plt.plot(curve_arclength, vel)
    plt.xlabel("arclength of the curve")
    plt.ylabel("velocity on arclength")
    plt.xticks([0, 6.28], ['0', '2π'], rotation='horizontal')
    plt.show()


def plot_vector_field(v_field: np.ndarray, arclength: np.ndarray) -> None:
    """
    Plots the vector field in each period

    Args:
        v_field: the vector field
    """
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.plot(arclength, v_field)
    plt.xlabel("arclength")
    plt.ylabel("vector field")
    plt.show()