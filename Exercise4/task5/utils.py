import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from config import *
from sir_model import model

time = np.linspace(t_0, t_end, steps)


def plot_bifurcation(b_array: ArrayLike,
                     initial_states: list,
                     xlim=[192, 200],
                     ylim=[0, 0.09],
                     figheight=40,
                     figwidth=20,
                     save_fig=False,
                     columns=3) -> None:
    for case, initial_state in enumerate(initial_states):
        fig, ax = plt.subplots(int(len(b_array) / columns), columns)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        st = fig.suptitle("$S_0=${}, $I_0=${}, $R_0=${}".format(initial_state[0],
                                                                  initial_state[1],
                                                                  initial_state[2])
                            , fontsize=30)
        for i, b in enumerate(b_array):
            sol = solve_ivp(model, t_span=[time[0],time[-1]],
                            y0=initial_state, t_eval=time,
                            args=(mu0, mu1, beta, A, d, nu, b),
                            method='DOP853', rtol=rtol, atol=atol)

            if columns > 1:
                sub_plot = ax[int(i / columns), i % columns]
            else:
                sub_plot = ax[int(i / columns)]
            sub_plot.grid(True)
            sub_plot.plot(sol.y[0], sol.y[1], 'r-')
            sub_plot.scatter(sol.y[0], sol.y[1], s=1, c=time, cmap='bwr')
            sub_plot.set_title("$b=${0:.{1}f}".format(b,3))
            sub_plot.set_xlabel("S")
            sub_plot.set_ylabel("I")
            sub_plot.set_xlim(*xlim)
            sub_plot.set_ylim(*ylim)

        st.set_y(0.99)
        fig.subplots_adjust(top=0.95)
        if save_fig:
            plt.savefig("./case{}.png".format(case), dpi='figure')

        plt.show()

def plot_phase(mu0, mu1, beta, A, d, nu, b):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')

    X, Y, Z  = np.meshgrid(np.linspace(195, 205, 10), np.linspace(0, 0.1, 10), np.linspace(0, 10, 10))
    U, V, W = np.zeros(X.shape), np.zeros(Y.shape), np.zeros(Z.shape)
    NI, NJ, NK = X.shape

    for i in range(NI):
        for j in range(NJ):
            for k in range(NK):
                x, y, z = X[0,i,0], Y[j,0,0], Z[0,0,k]
                mu = mu0 + (mu1 - mu0) * (b/(y+b))
                u = A - d * x - beta * x * y /(x + y + z)
                v = -(d + nu) * y - mu * y + beta * x * y /(x + y + z)
                w = mu * y - d * z
                ax.quiver(x, y, z, u, v, w, length=0.8, normalize=False)

    ax.plot(A/d, 0, 0,'o', color="r", label=f"$E_0$")
    ax.set_xlabel("S")
    ax.set_ylabel("I")
    ax.set_zlabel("R")
    ax.legend()