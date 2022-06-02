import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos
from scipy.integrate import solve_ivp

def deriv(t, y):
    theta = y[0]
    thetadot = y[1]

    thetaddot = -thetadot / q - sin(theta) + g * cos(w * t)

    return np.array([thetadot, thetaddot])


g = 1.5
q = 2
w = 2/3

N = 1000
t_0 = 0
t_f = 100

y_0 = np.array([0,0])

sol = solve_ivp(deriv, (t_0, t_f), y_0, 'RK45', np. linspace(t_0, t_f, N))
#Y = np.asarray(sol.y)

plt.scatter(Y[:,0], Y[:,1])
plt.show()
