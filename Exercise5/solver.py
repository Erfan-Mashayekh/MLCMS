import numpy as np
from scipy import integrate
from utils import radial_basis


class Solver:

    def __init__(self, x, t_start, t_end):
        self.x = x
        self.t_start = t_start
        self.t_end = t_end


class SolverLinear(Solver):

    def __init__(self, x, t_start, t_end, matrix_a):
        super().__init__(x, t_start, t_end)
        self.matrix_a = matrix_a

    def _fun(self, t, y):
        return y @ self.matrix_a.T

    def solve_linear_system(self):
        x1 = np.zeros(self.x.shape)
        i = 0
        for x in self.x:
            x1[i, :] = integrate.solve_ivp(
                                self._fun,
                                t_span=np.array([self.t_start, self.t_end]),
                                y0=x
                            ).y.T[-1]
            i += 1
        return x1


class SolverRadialBasis(Solver):

    def __init__(self, x, t_start, t_end, coefficients, GRID_b, EPSILON):
        super().__init__(x, t_start, t_end)
        self.coefficients = coefficients
        self.GRID_b = GRID_b
        self.EPSILON = EPSILON

    def _fun(self, t, y):
        y = y.reshape((1, y.size))
        return radial_basis(y, self.GRID_b, self.EPSILON) @ self.coefficients

    def solve_linear_system(self):
        x1 = np.zeros(self.x.shape)
        i = 0
        for x in self.x:
            x1[i, :] = integrate.solve_ivp(
                                self._fun,
                                t_span=np.array([self.t_start, self.t_end]),
                                y0=x
                            ).y.T[-1]
            i += 1
        return x1