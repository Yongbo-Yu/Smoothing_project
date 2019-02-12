import RBergomi
from RBergomi import *
import numpy as np
import time

xi=0.235**2;
#x=0.1;
H= 0.07
#HIn[0]=0.02
eta=1.9
#e[0]=0.4
rho=-0.9
#r[0]=-0.7
t=1.0
k=1
N=2


start=time.time()


a=np.zeros((1*(10**4)))
W1 = Vector(N)
Wtilde = Vector(N)
for m in range(0,(8*(10**6))):
	for i in range(0,NIn):
		Wtilde[i] = (np.random.normal(loc=0.0, scale=1.0))
		W1[i] = np.random.normal(loc=0.0, scale=1.0)

	a[m]=updatePayoff_cholesky(Wtilde,W1,v,eta,H,rho,xi,t,K,N)
	
print ('Relative Bias')	
print(np.mean(a))
elapsed_time_qoi=time.time()-start;
print ('Time')
print  (elapsed_time_qoi)    
print ('Stat Error')
stand=np.std(a, axis = 0)/np.sqrt((1*(10**4)))
print(stand)
