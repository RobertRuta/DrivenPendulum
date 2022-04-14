import numpy as np
from numpy import pi, cos
import time
import pickle

start_time = time.time()

N = 1000000



data = np.random.uniform(0, 1, size=(N,2))

with open('test_data.pickle', 'wb') as file:
    pickle.dump(data, file)

with open('test_data.pickle', 'rb') as file:
    data_loaded = pickle.load(file)

print(data_loaded)
print(data)

#np.savetxt('test_data1.csv', data, delimiter=', ')
#file = open('test_data', 'w')
#file.write()

## Conclusion:: Pickle fast af

print("--- %s seconds ---" % (time.time() - start_time))