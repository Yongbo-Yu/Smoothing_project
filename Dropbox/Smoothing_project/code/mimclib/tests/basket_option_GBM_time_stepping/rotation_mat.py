
# Implementation of the Aguilera-Perez Algorithm.
# Aguilera, Antonio, and Ricardo Perez-Aguila. "General n-dimensional rotations." (2004).

import numpy as np
import math
def rotmnd(v,theta):

    n = np.size(v,1);
    M = np.identity(n);
    for c in range(0,n-2):
        for r in range(c+1,0,-1):
            t = atan2(v(r,c),v(r-1,c))
            R = np.identity(n)
            R[(r r-1),(r r-1)] = np.array([cos(t) -sin(t)]; [sin(t) cos(t)])
            v = R.dot(v)
            M = R.dot(M)
        
    
    R = np.identity(n)
    R[[n-1 n],[n-1 n] ] = np.array([cos(theta) -sin(theta)]; [sin(theta) cos(theta)])
    M = np.linalg.solve(M,R.dot(M))