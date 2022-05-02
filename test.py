import numpy as np
import matplotlib.pyplot as plt

xp = np.linspace(0,10,10)
fp = np.sin(xp)
x = np.linspace(0, 10, 1000)

Y = np.interp(x, xp, fp)

plt.plot(xp,fp)
plt.scatter(x,Y)
plt.show()