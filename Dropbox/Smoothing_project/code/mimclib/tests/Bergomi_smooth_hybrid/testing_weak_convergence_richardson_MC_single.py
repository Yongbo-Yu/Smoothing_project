import numpy as np
import time
import scipy.stats as ss

import random

#from joblib import Parallel, delayed
#import multiprocessing


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import numpy as np


import fftw3
import RBergomi
from RBergomi import *
import mimclib.misc as misc

class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    
  
    # for the values of below paramters, we need to see the paper as well check with Christian 
    x=0.235**2;   # this will provide the set of xi parameter values 
    #x=0.1;
    HIn=Vector(1)    # this will provide the set of H parameter values
    #HIn[0]=0.43
    HIn[0]=0.07
    #HIn[0]=0.02
    e=Vector(1)    # This will provide the set of eta paramter values
    e[0]=1.9
    #e[0]=0.4
    r=Vector(1)   # this will provide the set of rho paramter values
    r[0]=-0.9
    #r[0]=-0.7
    T=Vector(1)     # this will provide the set of T(time to maturity) parameter value
    T[0]=1.0
    k=Vector(1)     # this will provide the set of K (strike ) paramter value
    k[0]=1.0
   # y1perp = Vector(N)
    MIn=1        # number of samples M (I think we do not need this paramter here by default in our case it should be =1)


    #methods
    # this method initializes 
    def __init__(self,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
         #Here we need to use the C++ code to compute the payoff 
        self.z=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  Nsteps, self.MIn)
        self.zf=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  2*Nsteps, self.MIn)

        self.dt=self.T[0]/float(Nsteps) # time steps length
        


   
    


     # objfun: 
    def objfun(self,Nsteps):
        mean = np.zeros(4*Nsteps)
        covariance= np.identity(4*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)

        

        yperp_1f=y[2*Nsteps+1:4*Nsteps]
        yperp1f=y[2*Nsteps]

        bbperp_f=self.brownian_increments(yperp1f,yperp_1f,2*Nsteps)
        W1perp_f= [(bbperp_f[i+1]-bbperp_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bbperp_f)-1)]


        yperp_1c=yperp_1f[0:Nsteps-1]
        yperp1c=yperp1f
        bbperp_c=self.brownian_increments(yperp1c,yperp_1c,Nsteps)
        W1perp_c= [(bbperp_c[i+1]-bbperp_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp_c)-1)]
        


         #non hierarchical
        #W1=y[0:self.N]
        #hierarchical way
        y_1f=y[1:2*Nsteps]
        y1f=y[0]
        bb_f=self.brownian_increments(y1f,y_1f,2*Nsteps)
        W1_f= [(bb_f[i+1]-bb_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bb_f)-1)]

        y_1c=y_1f[0:Nsteps-1]
        y1c=y1f
        bb_c=self.brownian_increments(y1c,y_1c,Nsteps)
        W1_c= [(bb_c[i+1]-bb_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bb_c)-1)]





         #level 1: richardson extrapol (new version)
        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )
        d[0,0] = self.z.ComputePayoffRT_single(W1_c,W1perp_c); 
        d[1,0]=self.zf.ComputePayoffRT_single(W1_f,W1perp_f); 

        d[1,1] = 2*d[1,0] - d[0,0]
        QoI=d[1,1] 
        return QoI


    # This function implements the brownian bridge construction in 1 D, the argument y here represent independent  multivariate gaussian  rdv  given by
    #mean = np.zeros(self.N)
    #covariance= np.identity(self.N)
    #y = np.random.multivariate_normal(mean, covariance)
    #y1: is the first direction given by W(T)/sqrt(T)/   #This function gives  the brownian motion increments built from Brownian bridge construction: # the composition of  BB and brownian_increments give us
    # the function \phi in our notes(discussion)
        
    def brownian_increments(self,y1,y,Nsteps):
        t=np.linspace(0, self.T, Nsteps+1)     
        h=Nsteps
        j_max=1
        bb= np.zeros(Nsteps+1)
        bb[h]=np.sqrt(self.T)*y1
       
        d=int(np.log2(Nsteps)) #power 2 number steps
        for k in range(1,d+1):
            i_min=h//2
            i=i_min
            l=0
            r=h
            for j in range(1,j_max+1):
                a=((t[r]-t[i])* bb[l]+(t[i]-t[l])*bb[r])/float(t[r]-t[l])
                b=np.sqrt((t[i]-t[l])*(t[r]-t[i])/float(t[r]-t[l]))
                bb[i]=a+b*y[i-1]
                i=i+h
                l=l+h
                r=r+h
            j_max=2*j_max
            h=i_min 
        return bb





def weak_convergence_differences():    
        start_time=time.time()
        #exact= 0.0712073 #exact value of K=1, H=0.43_xi_0.235^2_eta_1_9_r__09
        exact= 0.0791  #exact value of K=1, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact= 0.224905759853  #exact value of K=0.8, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact=0.124762643828  #exact value of K=1, H=0.02_xi_01_eta_0_4_r__07
        #exact=0.0568394  #exact value of K=1.2, H=0.02_xi_01_eta_0_4_r__07
        #exact= 0.00993973310944  #exact value of K=1.2, H=0.07_xi_0.235^2_eta_1_9_r__09
     

        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([2])
       
        dt_arr=1.0/(Nsteps_arr)
     
        elapsed_time_qoi=np.zeros(1)
        error=np.zeros(1)
        stand=np.zeros(1)
        Ub=np.zeros(1)
        Lb=np.zeros(1)
    
        values=np.zeros((2*(10**3),1)) 
        for i in range(0,1):
            print i
            start_time=time.time()
            prb = Problem(Nsteps_arr[i]) 
            for j in range(2*(10**3)):
                
                values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)
       
            elapsed_time_qoi[i]=time.time()-start_time
            print  elapsed_time_qoi[i]

        
        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(2*(10**3)))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand


        print elapsed_time_qoi
        print(error)   
        print(stand)
        print Lb
        print Ub
        
         
       
       

weak_convergence_differences()