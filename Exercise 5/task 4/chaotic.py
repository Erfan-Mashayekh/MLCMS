
import  numpy  as  np 
import  matplotlib.pyplot  as  plt
from scipy.integrate import odeint 
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
    simulate the system and plot functions x_n+1 = r * x_n * (1 - x_n)
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
    ax1.set_title(f"$r={r:.1f}, x_0={x0:.1f}$")

    ax2.plot(np.arange(0, 1, 0.01), y, 'ok-')
    # ax2.set_yscale("log")

def plot_bifurcation(r_min, r_max, x0):
    """
    simulate the system and plot the bifurcation diagram
    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()

    # rlist = []
    # xlist = []

    n = 10000 # 10000 values of r
    iterations = 1000  # 1000 iterations of the logistic map
    last = 100  # keep the last 100 iterations to display the bifurcation diagram

    # for r in np.linspace(r_min, r_max, n):
    #     x = x0 # initial x0 = 0.1
    #     for i in range(iterations):
    #         x = logistic(r, x)
    #         # display the bifurcation diagram.
    #         if i >= (iterations - last):
    #             rlist.append(r)
    #             xlist.append(x)
    # ax.plot(rlist, xlist, ',k', alpha=.2)

    r = np.linspace(r_min, r_max, n)
    x = x0 * np.ones(n)

    for i in range(iterations):
        x = logistic(r, x)
        if i >= (iterations - last):
            ax.plot(r, x, ',k', alpha=.2)

    ax.set_xlim(r_min, r_max)
    ax.set_xlabel("r")
    ax.set_ylabel("x")
    ax.set_title("Bifurcation diagram")

def lorenz(X, t, sigma, beta, rho):
    """
    define the Lorenz equations
    """
    x, y, z = X

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    X_new = [dx_dt, dy_dt, dz_dt]

    return X_new

def lorenz_trajectory(X, sigma, beta, rho):
    """
    plot the lorenz attractor
    """
    start_time = 0
    end_time = 1000
    t = np.linspace(start_time, end_time, 100000)

    xyz = odeint(lorenz, X, t, args = (sigma, beta, rho))

    return xyz

def difference(X1, X2, sigma, beta, rho):
    x1 = lorenz_trajectory(X1, sigma, beta, rho)
    x2 = lorenz_trajectory(X2, sigma, beta, rho)

    # plot the lorenz attractor in three-dimensional phase space
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(x1[:, 0], x1[:, 1], x1[:, 2])
    ax1.set_title(f"$sigma={sigma}, beta={round(beta,2)}, rho={rho}, X_0=[{X1[0]}, {X1[1]}, {X1[2]}]$")
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(x2[:, 0], x2[:, 1], x2[:, 2])
    ax2.set_title(f"$sigma={sigma}, beta={round(beta,2)}, rho={rho}, X_0=[{X2[0]}, {X2[1]}, {X2[2]}]$")

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot(x1[:, 0], x1[:, 1], x1[:, 2], label=f"$X_0=[{X1[0]}, {X1[1]}, {X1[2]}]$", color='red')
    ax3.plot(x2[:, 0], x2[:, 1], x2[:, 2], label=f"$X_0=[{X2[0]}, {X2[1]}, {X2[2]}]$", color='green')
    ax3.legend()

    x = x1 - x2
    diff = np.linalg.norm(x, axis=1, keepdims=True) ** 2
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(np.arange(0, 1000, 0.01), diff)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Difference")

    iterations = 0
    for i, d in enumerate(diff):
        if d > 1:
            iterations = i
            print(f"The difference between the points on the trajectory larger than 1 at t = {0.01 * iterations}")
            break