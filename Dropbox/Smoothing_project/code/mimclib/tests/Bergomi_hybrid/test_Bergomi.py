import fftw3
import RBergomi
from RBergomi import *
import numpy as np

x=0.235**2;
HIn=Vector(1)
HIn[0]= 0.43
e=Vector(1)
e[0]=1.9
r=Vector(1)
r[0]=-0.9
t=Vector(1)
t[0]=1.0
k=Vector(1)
k[0]=0.8
NIn=2
MIn=1

z=RBergomi.RBergomiST( x,  HIn, e,  r,  t, k,  NIn, MIn)

a=np.zeros(1000000)
W1 = Vector(NIn)
W2 = Vector(NIn)
for m in range(0,1000000):
	for i in range(0,NIn):
		W1[i] = (np.random.normal(loc=0.0, scale=1.0))
		W2[i] = np.random.normal(loc=0.0, scale=1.0)
		#print W1[i]
	a[m]=z.ComputePayoffRT_single(W1,W2);
	
print(np.mean(a))