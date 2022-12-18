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
