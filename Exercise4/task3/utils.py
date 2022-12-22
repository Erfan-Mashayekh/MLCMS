import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1]-time[0]
    for k in range(1, len(time)):
        yt[k, :] = yt[k-1, :] + step_size * f_ode(yt[k-1, :])
    return yt, time


def plot_phase_portrait_nonlinear(U, V, X, Y, alpha):
    """
    Plots a non-linear vector field in a streamplot, defined with X and Y coordinates and the derivatives U and V.
    """
    fig1 = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(nrows=1, ncols=1, height_ratios=[1])

    #  Varying density along a streamlne
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.streamplot(X, Y, U, V, density=[0.9, 1])
    ax1.set_title(f'Streamplot for non-linear vector field for alpha = {alpha}');
    ax1.set_aspect(1)
    return ax1

