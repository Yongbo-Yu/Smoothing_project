import numpy as np
theta = np.radians(15)
c, s = np.cos(theta), np.sin(theta)
R1 = np.array(((1,0,0),(0, c,-s), (0, s, c)))
print(R1) 

R2 = np.array((( c,-s,0),(0,1,0), (-s, 0, c)))
print(R2) 

R3= np.array((( c,-s,0), ( s, c,0),(0, 0, 1)))
print(R3) 

R=R3.dot(R2.dot(R1))

print R

