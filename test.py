import numpy as np

mask = np.identity(5,int)

print(mask)

tmp = mask[0]
mask[0] = mask[4]
mask[4] = tmp

print(mask) 