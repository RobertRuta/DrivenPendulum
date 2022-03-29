import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from numpy import sin, cos, pi

def deriv(t, y, w, g, q):
    theta = y[0]
    thetadot = y[1]
    
    phi = w*t
    
    thetaddot = - thetadot / q - sin(theta) + g*cos(phi)
    
    return np.array([thetadot, thetaddot])

def simulate(y0, params, t0, tf, N):
    w = params[0]
    q = params[1]
    g = params[2]
        
    sol = solve_ivp(deriv, (0, tf), y0, t_eval=np.linspace(t0, tf, N), args=(w, g, q, ))
    
    return np.array([sol.y[0], sol.y[1], sol.t])

def SetSim():
    global w
    global q
    global g
    global theta_0
    global thetadot_0
    
    print("Enter equation paramters.")
    w = float(input("Omega: "))
    q = float(input("q: "))
    g = float(input("g: "))

    print()

    print("Enter initial conditions.")
    theta_0 = float(input("Enter initial angle: "))
    thetadot_0 = float(input("Enter initial angular velocity: "))

    print()

    tf = input("Enter simulation duration: ")
    
    def CleanUp(sol):
<<<<<<< HEAD

=======
        i = 0        
        for angle in sol.y[0]:    
            sgn = angle / np.abs(angle)
            result = angle - sgn*int((np.abs(np.degrees(angle)) + 180)/360)*2*np.pi
            sol.y[0,i] = result            
            i += 1
        
        return sol
>>>>>>> 1679afedcb72eb7db944621d5d8e3118d3d29792
def MakeFrames(G):
    for g in G:
        print("g = {}".format(g))    
        
        params = np.array([w, q, g])
        sol1 = simulate(y0, params, t0, tf, N)
        sol2 = simulate(y0, params, 30*2*pi/w, 5030*2*pi/w, 5001)
        
        fig = plt.figure(figsize=(7,7))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        i = 0        
        for angle in sol1[0]:    
            sgn = angle / np.abs(angle)
            result = angle - sgn*int((np.abs(np.degrees(angle)) + 180)/360)*2*np.pi
            sol1[0,i] = result            
            i += 1
            
        i = 0
        for angle in sol2[0]:    
            sgn = angle / np.abs(angle)
            result = angle - sgn*int((np.abs(np.degrees(angle)) + 180)/360)*2*np.pi
            sol2[0,i] = result            
            i += 1

        ax1.scatter(sol1[0], sol1[1], s=1, c='black')
        ax1.set_xlim([-pi,pi])
        ax1.axhline(color='black')
        ax1.axvline(color='black')
        ax1.set_xlabel("\u03b8 [rad]")
        ax1.set_ylabel("\u03c9 [rad/s]")
        ax1.set_title("Phase Diagram | g = {:.3f}".format(g), fontsize=20)
        
        ax2.scatter(sol2[0], sol2[1], s=1, c='black', marker='s')
        ax2.set_xlim([-pi,pi])
        ax2.axhline(color='black')
        ax2.axvline(color='black')
        ax2.set_xlabel("\u03b8 [rad]")
        ax2.set_ylabel("\u03c9 [rad/s]")
        ax2.set_title("Poincare Map | g = {:.3f}".format(g), fontsize=20)
        
        fig.subplots_adjust(hspace=0.4)
        fig.savefig("PhaseAndPoincare_frames/specific/phase_poincare_diag{}.png".format(int(g*100)))
        plt.close(fig)
        
    
w = 2/3
q = 2
#G = np.linspace(0.7, 2, 200)
G = [0.9, 1.07, 1.15, 1.35, 1.45, 1.50]


theta_0 = 0
thetadot_0 = 0.2

t0 = 4000
tf = 5000

N = 10000
#SetSim()

y0 = np.array([theta_0, thetadot_0])

MakeFrames(G)