import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from numpy import sin, cos, pi
from matplotlib.patches import Circle

def deriv(t, y, w, g, q):
    theta = y[0]
    thetadot = y[1]
    
    phi = w*t
    
    thetaddot = - thetadot / q - sin(theta) + g*cos(phi)
    
    return np.array([thetadot, thetaddot])

def simulate(y0, t0, tf, N):
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
    
def MakePlots(G):
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
        
def MakeFrames():

    sol = solve_ivp(deriv, (0, tf), y0, t_eval=np.linspace(t0, tf, N), args=(w, g, q, ))
    
    th = sol.y[0]
    X = cos(th)
    Y = sin(th)
    t = sol.t

    dt = t[1] - t[0]
    R = 0.05
    trail_duration = 1
    trail_length = int(trail_duration / dt)

    # Make an image every di time points, corresponding to a frame rate of fps
    # frames per second.
    # Frame rate, s-1
    fps = 30
    di = round(1/fps/dt)

    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    for i in range(0, t.size, di):
        print("Step: {}".format(i))
        ax.plot([0, X[i]], [0, Y[i]], lw=2, c='k')        

        # Draw Circles
        c_pivot = Circle((0,0), R/2, fc='k', zorder=10)
        c_ball = Circle((X[i],Y[i]), R, fc='b', zorder=10)
        ax.add_patch(c_pivot)
        ax.add_patch(c_ball)

        # Draw Trail
        ns = 20
        s = trail_length // ns
        
        for j in range(ns):
            imin = i - (ns-j)*s
            if imin < 0:
                continue
            imax = imin + s + 1
            alpha = (j/ns)**2
            ax.plot(X[imin:imax], Y[imin:imax], c='b', solid_capstyle='butt', lw=3, alpha=alpha)

        # Plot Forcing Arrow
        arrowLen = 0.2*cos(w*t[i])
        ax.plot([X[i], X[i] + arrowLen*sin(-th[i])], [Y[i], Y[i] + arrowLen*cos(th[i])], '-r')
            
        # Centre the image on the fixed anchor point, and ensure the axes are equal
        ax.set_xlim(-1-R, R+1)
        ax.set_ylim(-1-R, 1+R)
        ax.set_aspect('equal', adjustable='box')

        # Text-box printing total time elapsed
        textstr = '\u03b8 = {:.2f} rads \n t = {:.2f} s'.format(th[i], t[i])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(-0.3, 1.1, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        plt.axis('off')
        plt.savefig('anim_frames/_img_{frame:04d}.png'.format(frame = i//di), dpi=72)
        plt.cla()

w = 2/3
q = 4
#G = np.linspace(0.7, 2, 200)
#G = [0.9, 1.07, 1.15, 1.35, 1.45, 1.50]
g = 1.5

theta_0 = 0
thetadot_0 = 0

t0 = 0
tf = 30

N = 10000
#SetSim()

y0 = np.array([theta_0, thetadot_0])

MakeFrames()