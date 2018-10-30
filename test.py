import numpy as np

a = np.ones((3,3,3))


a[1,2,:] = 0
print(a)