# vectorized vs iterative brownian bridge construction (almost same cost)

import numpy as np 
import time

Nsteps=32
T=1.0
dt=T/float(Nsteps) # time steps length
d=int(np.log2(Nsteps))

mean = np.zeros(Nsteps)
covariance= np.identity(Nsteps)
y = np.random.multivariate_normal(mean, covariance) 
y1_1=y[1:Nsteps]
y1=y[0]
def brownian_increments(y1,y,Nsteps):
        t=np.linspace(0, T, Nsteps+1)     
        h=Nsteps
        j_max=1
        bb= np.zeros((1,Nsteps+1))
        bb[0,h]=np.sqrt(T)*y1
       
        
         
        for k in range(1,d+1):
            i_min=h//2
            i=i_min
            l=0
            r=h
            for j in range(1,j_max+1):
                a=((t[r]-t[i])* bb[0,l]+(t[i]-t[l])*bb[0,r])/float(t[r]-t[l])
                b=np.sqrt((t[i]-t[l])*(t[r]-t[i])/float(t[r]-t[l]))
                bb[0,i]=a+b*y[i-1]
                i=i+h
                l=l+h
                r=r+h
            j_max=2*j_max
            h=i_min 
        return bb    
def brownian_increments_2(y1,y,Nsteps):
        t=np.linspace(0, self.T, Nsteps+1)     

        cov=np.zeros((Nsteps+1,Nsteps+1))
        for i in range(0,Nsteps+1):
        	for j in range(i,Nsteps+1):
        		cov[i,j]=min (t[i],t[j])
        cov=cov+np.transpose(cov)-np.diag(np.diag(cov)) 		
        
        C=np.linalg.cholesky(cov)

        bb= np.zeros((1,Nsteps+1))
        y=np.array([y1,y])

        bb=C.dot(y)
       
        return bb         
start_time=time.time()       
bb1=brownian_increments(y1,y1_1,Nsteps)     
elapsed_time_qoi=time.time()-start_time

print  elapsed_time_qoi
start_time2=time.time()   
bb2=brownian_increments(y1,y1_1,Nsteps) 
elapsed_time_qoi2=time.time()-start_time2
print  elapsed_time_qoi2
print bb1
print    'Hello'
print bb2 