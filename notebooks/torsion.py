import numpy as np

G = 80e9
D = 0.020
k = 2500

l = G*np.pi*D**4/(k*32)

print(l)