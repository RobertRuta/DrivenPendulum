import numpy as np
from numpy import pi, cos
import time
import pickle

a = np.array([[1,2,3], [7,8,9]])
b = np.array([4,5,6])

c = [[a], [b]]

print(np.stack((a,b)))