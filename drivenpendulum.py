import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from scipy.integrate import solve_ivp, OdeSolver
from numpy import sin, cos, pi
from tqdm import tqdm


def deriv(t, y, w, g, q):
    theta = y[0]
    thetadot = y[1]
    
    phi = w*t
    
    thetaddot = - thetadot / q - sin(theta) + g*cos(phi)
    
    return np.array([thetadot, thetaddot])


def AddProgressBar():
    old_init = OdeSolver.__init__
    old_step = OdeSolver.step
    # define our own methods
    def new_init(self, fun, t0, y0, t_bound, vectorized, support_complex=False):

        # define the progress bar
        self.pbar = tqdm(total=t_bound - t0, unit='ut', initial=t0, ascii=True, desc='IVP')
        self.last_t = t0
        
        # call the old method - we still want to do the old things too!
        old_init(self, fun, t0, y0, t_bound, vectorized, support_complex)


    def new_step(self):
        # call the old method
        old_step(self)
        
        # update the bar
        tst = self.t - self.last_t
        self.pbar.update(tst)
        self.last_t = self.t

        # close the bar if the end is reached
        if self.t >= self.t_bound:
            self.pbar.close()


    # overwrite the old methods with our customized ones
    OdeSolver.__init__ = new_init
    OdeSolver.step = new_step


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
        i = 0        
        for angle in sol.y[0]:    
            sgn = angle / np.abs(angle)
            result = angle - sgn*int((np.abs(np.degrees(angle)) + 180)/360)*2*np.pi
            sol.y[0,i] = result            
            i += 1
        
        return sol

def CleanUp(solution):
    i = 0        
    for angle in solution.y[0]:    
        sgn = angle / np.abs(angle)
        result = angle - sgn*int((np.abs(np.degrees(angle)) + 180)/360)*2*np.pi
        solution.y[0][i] = result
        i += 1

def MakePhaseDiagram(axes, X, Y, g):
    axes.scatter(X, Y, s=1, c='black')
    axes.set_xlim([-pi,pi])
    axes.axhline(color='black')
    axes.axvline(color='black')
    axes.set_xlabel("\u03b8 [rad]")
    axes.set_ylabel("\u03c9 [rad/s]")
    axes.set_title("Phase Diagram | g = {:.3f}".format(g), fontsize=20)  

    return axes  

def MakePoincareDiagram(axes, X, Y, g):
    #axes.axhline(color='black', lw=0.3)
    #axes.axvline(color='black', lw=0.3)

    axes.scatter(X, Y, s=0.05, linewidth=0, c='black', marker='.')
    axes.set_xlim([-pi,pi])
    axes.set_ylim([-1,pi])
    axes.set_xlabel("\u03b8 [rad]")
    axes.set_ylabel("\u03c9 [rad/s]")
    axes.set_title("Poincare Map | g = {:.3f}, q = 4".format(g), fontsize=16)

    return axes  
    
def CreateDiagrams(G):
    for g in G:
        print("g = {}".format(g))
        
        sol1 = solve_ivp(deriv, (0, tf), y0, t_eval=np.linspace(t0, tf, N), args=(w, g, q, ))
        sol2 = solve_ivp(deriv, (0, 5030*2*pi/w), y0, t_eval=np.linspace(30*2*pi/w, 5030*2*pi/w, 5001), args=(w, g, q, ))
        
        fig = plt.figure(figsize=(7,7))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        sol1 = CleanUp(sol1)
        sol2 = CleanUp(sol2)

        ax1 = MakePhaseDiagram(ax1, sol1.y[0], sol1.y[1]. g)
        ax2 = MakePoincareDiagram(ax2, sol2.y[0], sol2.y[1]. g)
        
        fig.subplots_adjust(hspace=0.4)
        fig.savefig("PhaseAndPoincare_frames/specific/phase_poincare_diag{}.png".format(int(g*100)))
        plt.close(fig)
        
# Create frames to be used in gif
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
    i = 0

def SaveData(N):

    data_to_save = []
    data_to_save = np.empty((len(N), 4))

    i=0
    for n in N:
        print("Trying to save {}x2 data points".format(n))

        start_time = time.time()
        sol = solve_ivp(deriv, (0, (n+30)*2*pi/w), y0, t_eval=np.linspace(30*2*pi/w, (n+30)*2*pi/w, n+1), args=(w, g, q, ), method='RK45')
        solve_time = time.time() - start_time
        print("Total solve time: {}".format(solve_time))

        start_time = time.time()
        CleanUp(sol)
        clean_time = time.time() - start_time
        print("Total clean time: {}".format(clean_time))

        start_time = time.time()
        with open("phasedata_q{}_g{}_{}e3.pickle".format(int(q*10), int(g*10), int(n/1000)), 'wb') as file:
            pickle.dump(sol, file)
        pickle_time = time.time() - start_time
        print("Total pickle time: {}".format(pickle_time))

        data_to_save[i,:] = [n, solve_time, clean_time, pickle_time]
        i += 1

    np.savetxt("timings.csv", data_to_save, delimiter=', ')

    return sol

def LoadData(N):
    with open("Pickles/phasedata_q20_g15_{}e3.pickle".format(int(N/1000)), 'rb') as file:
        sol = pickle.load(file)
    return sol

AddProgressBar()

w = 2/3
q = 2
#G = np.linspace(0.7, 2, 200)
#G = [0.9, 1.07, 1.15, 1.35, 1.45, 1.50]
g = 1.5

theta_0 = 0
thetadot_0 = 0
y0 = np.array([theta_0, thetadot_0])

t0 = 4000
tf = 5000

N = np.geomspace(10**7, 10**8, 3)
N = N.astype(int)


#sol = SaveData([10**8])

sol = LoadData(10**8)
#print(sol.t)

th = sol.y[0]
omega = sol.y[1]

fig = plt.figure(figsize=(7,7), dpi=128)
ax = fig.add_subplot(111)
#MakePoincareDiagram(ax,th,omega,1.5)
ax.scatter(th, omega, marker='.', linewidth=0, s=0.1, c='k')
plt.show()

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)

# ax = MakePoincareDiagram(ax, sol.y[0], sol.y[1], g)
# #fig.savefig("phase_poincare_diag{}.png".format(int(g*100)))
# plt.show()  

#SetSim()
#CreateDiagrams(G)
#MakePlots