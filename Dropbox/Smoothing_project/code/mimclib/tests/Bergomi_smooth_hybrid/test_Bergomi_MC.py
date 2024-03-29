import fftw3
import RBergomi
from RBergomi import *
import numpy as np
import time

x=0.235**2;
#x=0.1;
HIn=Vector(1)
HIn[0]= 0.07
#HIn[0]=0.02
e=Vector(1)
e[0]=1.9
#e[0]=0.4
r=Vector(1)
r[0]=-0.9
#r[0]=-0.7
t=Vector(1)
t[0]=1.0
k=Vector(1)
k[0]=1.0
NIn=2
MIn=1

start=time.time()
z=RBergomi.RBergomiST( x,  HIn, e,  r,  t, k,  NIn, MIn)

a=np.zeros((1*(10**4)))
W1 = Vector(NIn)
W2 = Vector(NIn)
for m in range(0,(1*(10**4))):
	for i in range(0,NIn):
		W1[i] = (np.random.normal(loc=0.0, scale=1.0))
		W2[i] = np.random.normal(loc=0.0, scale=1.0)
		#print W1[i]
	a[m]=z.ComputePayoffRT_single(W1,W2)
	
print ('Relative Bias')	
print(np.mean(a))
elapsed_time_qoi=time.time()-start;
print ('Time')
print  (elapsed_time_qoi)    
print ('Stat Error')
stand=np.std(a, axis = 0)/np.sqrt((1*(10**4)))
print(stand)
