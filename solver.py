import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

steps = 10**5
t_0 = 0
t_f = 10
dt = (2*pi/w)/ 800

count = 0

table = []

for dt in [(2*pi/w)/ 800, (2*pi/w)/ 2000]:
    
    #INIT CONDITIONS
    y_0 = np.array([0,0])

    t = t_0
    y = y_0
    Y = np.empty([steps,2])

    for step in tqdm(range(steps)):
        
        dy = deriv(t,y)*dt
        y = y + dy
        Y[step,0] = y[0]
        Y[step,1] = y[1]
        
        t += dt
        #print("step: ", step)
        
    table.append(Y)
        
    count += 1
    
    t += dt
    #print("step: ", step, "    ", step / steps * 100, "%")
    
        
    
fig = plt.figure(figsize=(6,6), dpi=70)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
Y1 = table[0]
Y2 = table[1]

ax1.scatter(Y1[:,0],Y1[:,1], s=0.1, linewidths=0)
ax1.scatter(Y2[:,0],Y2[:,1], s=0.1, linewidths=0)


ax2.plot()

plt.show()
