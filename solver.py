import numpy as np
import matplotlib.pyplot as plt

def deriv(t,y):
    theta = y[0]
    thetadot = y[1]

    q = 1.15
    

    thetaddot = -thetadot/q

    return np.array(thetadot, thetaddot)

steps = 10**8
t_0 = 0
t_f = 10

T = np.linspace(t_0, t_f, steps)
print("number of bytes occupied by T: ", T.nbytes)