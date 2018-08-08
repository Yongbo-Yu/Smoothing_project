import numpy as np 

yf = np.array([1,2,3,4,5,6])
yc=[sum(yf[current: current+2]) for current in xrange(0, len(yf), 2)]
print(yc)