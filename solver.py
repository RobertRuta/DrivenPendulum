import numpy as np
import matplotlib.pyplot as plt

from numpy import sin, cos, pi, linspace
from time import time


def timing(func):
    def wrapper(*args, **kwargs):
        start = time()
        output = func(*args, **kwargs)
        end = time()
                
        print(f"Time take: {end - start}")
        
        return output     
       
    return wrapper


def deriv(t,y):
    theta = y[0]
    thetadot = y[1]
    
    thetaddot = -thetadot/q - sin(theta) + g*cos(w*t)
    
    return np.array([thetadot, thetaddot])


@timing
def solver(y_0, t_0, t_f, N):
    
    dt = (t_f - t_0) / N
    
    def euler(y):
        dy = deriv(t,y)*dt
        return y + dy
    
    def rk12(y):
        k1 = deriv(t,y)
        dy1 = y + k1*dt/2
        k2 = deriv(t+dt/2, dy1)
        dy2 = y + k2*dt/2
        k3 = deriv(t+dt/2, dy2)
        dy3 = y + k3*dt
        k4 = deriv(t+dt, dy3)
        
        dy = 1/6*dt*(k1+ 2*k2 + 2*k3 + k4)
        
        return y + dy
    
    y = y_0
    t = t_0
    
    
    Y = np.empty([N,2])  
    Y[0,0] = y_0[0]  
    Y[0,1] = y_0[1]
    i = 1
    while i < N:
        y = euler(y)
        t += dt
        if i == 49998:
            print("")
        
        Y[i,0] = y[0]
        Y[i,1] = y[1]
        
        i += 1
        
    return Y
    

# PARAMS
q = 2
g = 1.5
w = 2/3

steps = 10**5
t_0 = 0
t_f = 10
dt = (2*pi/w)/ 800

N1 = 10000
N2 = 20000

Y1 = solver(np.array([0,0]), 0, 10, N1)
Y2 = solver(np.array([0,0]), 0, 10, N2)



# count = 0

# table = []

# for dt in [(2*pi/w)/ 5000, (2*pi/w)/10000]:
    
#     #INIT CONDITIONS
#     y_0 = np.array([0,0])

#     t = t_0
#     t_f = 50
#     y = y_0
#     Y1 = np.empty([steps,2])

#     step = 0
#     while t < t_f:
        
#         dy = deriv(t,y)*dt
#         y = y + dy
#         Y1[step,0] = y[0]
#         Y1[step,1] = y[1]
        
#         t += dt
#         step+=1
#         print("step: ", step)
        
    
fig = plt.figure(figsize=(6,6), dpi=70)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

T1 = linspace(t_0,t_f,N1)
T2 = linspace(t_0,t_f,N2)
t = linspace(t_0,t_f,int(N2/2))

ax1.scatter(T1, Y1[:,0], s=2, linewidths=0)
ax1.scatter(T2, Y2[:,0], s=2, linewidths=0)


Y1_interp = np.interp(t, T1, Y1[:,0])
Y2_interp = np.interp(t, T2, Y2[:,0])

ax2.scatter(t, (Y1_interp-Y2_interp), s=2)

plt.show()