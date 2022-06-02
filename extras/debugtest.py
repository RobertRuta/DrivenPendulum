import numpy as np
import scipy as sp

from scipy.integrate import solve_ivp
from numpy import sin


def deriv(t, y):
    theta = y[0]
    thetadot = y[1]
    thetaddot = - g/L * sin(theta)

    return np.array([thetadot, thetaddot])


g = 10
L = 1
y0 = np.array([np.radians(45), np.radians(0)])

sol = solve_ivp(deriv, (0, 30), y0, 'RK45', np.linspace(0, 30, 1000))



