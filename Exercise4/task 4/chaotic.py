
import  numpy  as  np 
import  matplotlib.pyplot  as  plt
# from scipy.integrate import odeint 
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

def  logistic (r, x):
    """
    define the logistic function: x_n+1 = r * x_n * (1 - x_n)
    :param r: r in logistic function
    :param x: x_n in logistic function
    :returns: x_n+1 in lohistic function
    """
    x_next = r * x * (1 - x)
    return x_next

def plot_logistic_map(r, x0):
    """
    plot functions x_n+1 = r * x_n * (1 - x_n) and the change of x_n with n
    :param r: r in logistic function
    :param x0: the initial value of system
    """
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    iterations = 100 # 10 iterations of the logistic map
    t = np.linspace(0, 1)
    ax1.plot(t, logistic(r, t), 'k', lw=2) # plot function y = r * x * (1 - x)
    ax1.plot([0, 1], [0, 1], 'k', lw=2) # plot functions y = x

    x = x0
    y = []
    for i in range(iterations):
        y.append(x)
        x_new = logistic(r, x)
        # plot the updating process x_n+1 = r * x_n * (1 - x_n)
        ax1.plot([x, x], [x, x_new], 'k', lw=1) # (x_n, x_n) -> (x_n, x_n+1)
        ax1.plot([x, x_new], [x_new, x_new], 'k', lw=1) # (x_n, x_n+1) -> (x_n+1, x_n+1)
        # plot the positions with increasing opacity
        ax1.plot([x], [x_new], 'ok', ms=8, alpha=(i + 1) / iterations)
        x = x_new

    ax1.set_xlim(0, 1)
    ax1.set_xlabel("$x_n$")
    ax1.set_ylabel("$x_{n+1}$")
    ax1.set_title(f"$r={round(r,2)}, x_0={round(x0,2)}$")

    ax2.set_xlabel("n")
    ax2.set_ylabel("$x_n$")
    ax2.plot(np.arange(0, 100, 1), y, 'ok-')
    ax2.set_title(f"r={round(r,2)}")
    # ax2.set_yscale("log")

def plot_bifurcation(r_min, r_max, x0):
    """
    simulate the system and plot the bifurcation diagram
    :param r_min: the maximum of r
    :param r_max: the minimum of r
    :param x0: the initial value of system
    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()

    # rlist = []
    # xlist = []

    n = 10000 # 10000 values of r
    iterations = 1000  # 1000 iterations of the logistic map
    last = 100  # keep the last 100 iterations to display the bifurcation diagram

    # for r in np.linspace(r_min, r_max, n):
    #     for i in range(iterations):
    #         x = logistic(r, x)
    #         # plot the bifurcation diagram
    #         if i >= (iterations - last):
    #             rlist.append(r)
    #             xlist.append(x)
    # ax.plot(rlist, xlist, ',k', alpha=.2)
    # this method is too slow

    r = np.linspace(r_min, r_max, n)
    x = x0 * np.ones(n)

    # plot the bifurcation diagram
    for i in range(iterations):
        x = logistic(r, x)
        if i >= (iterations - last):
            ax.plot(r, x, ',k', alpha=.2)

    ax.set_xlim(r_min, r_max)
    ax.set_xlabel("r")
    ax.set_ylabel("x")
    ax.set_title("Bifurcation diagram")

def lorenz(t, X, sigma, beta, rho):
    """
    define the Lorenz equations
    :param X: the initial condition
    :param sigma, beta, rho: Lorenz paramters
    """
    x, y, z = X

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    dX_dt = [dx_dt, dy_dt, dz_dt]

    return dX_dt

def lorenz_trajectory(X, sigma, beta, rho, linewidth):
    """
    integrate $x,y,z$ with respect to $t$ by \emph{solve_ivp} and plot the trajectory.
    :param X: the initial condition
    :param sigma, beta, rho: Lorenz paramters
    :param linewidth: the width of line in plot
    """
    start_time = 0
    end_time = 1000
    t = np.linspace(start_time, end_time, 100000)

    # integrate the Lorenz equations
    # xyz = odeint(lorenz, X, t, args = (sigma, beta, rho))
    xyz = solve_ivp(lorenz, t_span = [t[0], t[-1]], y0 = X, t_eval = t, args = (sigma, beta, rho))
    xyz = xyz["y"]

    # plot the trajectory in three-dimensional phase space
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(projection='3d')
    ax1.plot(xyz[0, :], xyz[1, :], xyz[2, :], linewidth=linewidth, label="trajectory")
    ax1.plot(xyz[0, 0], xyz[1, 0], xyz[2, 0], 'o', color="r", label="initial point")
    ax1.plot(xyz[0, -1], xyz[1, -1], xyz[2, -1], 'o', color="y", label="end point")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend()
    ax1.set_title(f"$sigma={sigma}, beta={round(beta,2)}, rho={rho}, X_0=[{X[0]}, {X[1]}, {X[2]}]$")

    return xyz

def difference(X1, X2, sigma, beta, rho):
    """
    plot the Lorenz attractor
    :param X1, X2: two different initial conditions we want to compare
    :param sigma, beta, rho: Lorenz paramters
    """
    # when rho = 28, the plot is so dense. so we change the linewidth to 0.1
    if rho > 1:
        linewidth = 0.1
    else: linewidth = 1

    x1 = lorenz_trajectory(X1, sigma, beta, rho, linewidth)
    x2 = lorenz_trajectory(X2, sigma, beta, rho, linewidth)

    fig = plt.figure(figsize=(20, 10))

    # merge trajectories in one plot to compare
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(x1[0, :], x1[1, :], x1[2, :], linewidth=linewidth, label=f"$X_0=[{X1[0]}, {X1[1]}, {X1[2]}]$", color='red')
    ax1.plot(x2[0, :], x2[1, :], x2[2, :], linewidth=linewidth, label=f"$X_0=[{X2[0]}, {X2[1]}, {X2[2]}]$", color='green')
    ax1.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("compare the trajectories")

    # compute the difference
    x = x1.T - x2.T
    diff = np.linalg.norm(x, axis=1, keepdims=True) ** 2
    # plot the difference
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(np.arange(0, 1000, 0.01), diff, linewidth=0.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Difference")
    ax2.set_title("the difference between the points on the trajectories")

    # compute when the difference larger than 1
    iterations = 0
    for i, d in enumerate(diff):
        if d > 1:
            iterations = i
            print(f"The difference between the points on the trajectory larger than 1 at {round(0.01 * iterations, 2)}s")
            break