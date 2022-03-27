import numpy as np
from numpy import pi, cos

N = 1000

phi = pi/10

t0 = phi + 2*pi*10
tf = phi + 2*pi*N

A = np.arange(t0, tf, 2*pi)

print(cos(A[10]))
print(cos(A[20]))
