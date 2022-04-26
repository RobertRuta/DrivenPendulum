import numpy as np
import matplotlib.pyplot as plt

from numpy import sin, cos, pi

def deriv(t,y):
    theta = y[0]
    thetadot = y[1]


    
    thetaddot = -thetadot/q - sin(theta) + g*cos(w*t)
    
    return np.array([thetadot, thetaddot])

# PARAMS
q = 2
g = 1.5
w = 2/3

steps = 10**7
t_0 = 0
t_f = 10
dt = 2*pi/w

#INIT CONDITIONS
y_0 = np.array([0,0])

t = t_0
y = y_0
Y = np.empty([steps,2])

for step in range(steps):
    
    dy = deriv(t,y)*dt
    y = y + dy
    Y[step,0] = y[0]
    Y[step,1] = y[1]
    
    t += dt
    print("step: ", step)
    
print("Y array size: ", Y.nbytes)
    
plt.scatter(Y[:,0],Y[:,1])
plt.show()
