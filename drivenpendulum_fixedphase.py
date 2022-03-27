import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from numpy import sin, cos, pi

def deriv(t, y, w, g, q, phi):
    theta = y[0]
    thetadot = y[1]
    
    thetaddot = - thetadot / q - sin(theta) + g*cos(phi)
    
    return np.array([thetadot, thetaddot])

def simulate(y0, params, t0, tf, N):
    w = params[0]
    q = params[1]
    g = params[2]
    phi = params[3]
    
    t0 = phi + 2*pi*10
    tf = phi + 2*pi*N
        
    sol = solve_ivp(deriv, (0, tf), y0, t_eval=np.arange(t0, tf, 2*pi), args=(w, g, q, phi ))
    
    return np.array([sol.y[0], sol.y[1], sol.t])
    
def MakeFrames(PHI):
    for phi in PHI:
        print("phi = {}".format(phi))    
        
        params = np.array([w, q, g, phi])
        sol1 = simulate(y0, params, t0, tf, N)
        
        fig = plt.figure(figsize=(7,7))
        ax1 = fig.add_subplot(111)

        i = 0        
        for angle in sol1[0]:    
            sgn = angle / np.abs(angle)
            result = angle - sgn*int((np.abs(np.degrees(angle)) + 180)/360)*2*np.pi
            sol1[0,i] = result            
            i += 1

        ax1.scatter(sol1[0], sol1[1], s=1, c='black')
        ax1.set_xlim([-pi,pi])
        ax1.axhline(color='black')
        ax1.axvline(color='black')
        ax1.set_xlabel("\u03b8 [rad]")
        ax1.set_ylabel("\u03c9 [rad/s]")
        ax1.set_title("Phase Diagram | g = {:.3f}".format(g), fontsize=20)

        
        fig.subplots_adjust(hspace=0.4)
        fig.savefig("PoincareFixedPhase_frames/poincare_fixdphs{}.png".format(int(phi*100)))
        plt.close(fig)
        
    
w = 2*pi
q = 2
g = 1.5

PHI = np.arange(0, 6*pi/5, pi/5)


theta_0 = 0
thetadot_0 = 0.2

t0 = 0
tf = 5000

N = 1000
#SetSim()

y0 = np.array([theta_0, thetadot_0])

MakeFrames(PHI)