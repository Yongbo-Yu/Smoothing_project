from RBergomi import *
from RfBm import *
from RNorm  import *
import random
import numpy as np
import time

xi=0.235**2;
#xi=0.1;
H= 0.07
#H=0.02
eta=1.9
#eta=0.4
rho=-0.9
#rho=-0.7
t=1.0
K=1
N=128

start=time.time()

rnorm=RNorm(3)

rfbm= RfBm(N,H,rnorm)

a=np.zeros((1*(10**6)))
W1 = Vector(N)
Wtilde = Vector(N)
v=Vector(N)

#for i in range(0,N):
#	W1[i]=i
#	Wtilde[i]=i



for m in range(0,(1*(10**6))):
	#for i in range(0,N):
	mean=np.zeros(2*N)
	cov=np.identity(2*N)
	x=np.random.multivariate_normal(mean,cov)

	
	rfbm.generate(x,W1, Wtilde);
	
		#Wtilde[i] = (np.random.normal(loc=0.0, scale=1.0))
		#W1[i] = np.random.normal(loc=0.0, scale=1.0)

	a[m]=updatePayoff_cholesky(Wtilde,W1,v,eta,H,rho,xi,t,K,N)


print ('Relative Bias')	
print(np.mean(a))
elapsed_time_qoi=time.time()-start;
print ('Time')
print  (elapsed_time_qoi)    
print ('Stat Error')
stand=np.std(a, axis = 0)/np.sqrt((1*(10**6)))
print(stand)
