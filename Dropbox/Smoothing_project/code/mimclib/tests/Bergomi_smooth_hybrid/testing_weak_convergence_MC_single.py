# In this file, we plot the weak convergence rate for MC without Richardson extrapolation without doing the partial change of measure


import numpy as np
import time
import scipy.stats as ss

import random

#from joblib import Parallel, delayed
#import multiprocessing


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


import fftw3
import RBergomi
from RBergomi import *
import mimclib.misc as misc


import pathos.multiprocessing as mp
import pathos.pools as pp

class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    
  
    # for the values of below paramters, we need to see the paper as well check with Christian 
    #x=0.235**2;   # this will provide the set of xi parameter values 
    x=0.1;
    HIn=Vector(1)    # this will provide the set of H parameter values
    #HIn[0]=0.43
    #HIn[0]=0.07
    HIn[0]=0.02
    e=Vector(1)    # This will provide the set of eta paramter values
    #e[0]=1.9
    e[0]=0.4
    r=Vector(1)   # this will provide the set of rho paramter values
    #r[0]=-0.9
    r[0]=-0.7
    T=Vector(1)     # this will provide the set of T(time to maturity) parameter value
    T[0]=1.0
    k=Vector(1)     # this will provide the set of K (strike ) paramter value
    k[0]=1.2
   # y1perp = Vector(N)
    MIn=1        # number of samples M (I think we do not need this paramter here by default in our case it should be =1)


    #methods
    # this method initializes 
    def __init__(self,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
         #Here we need to use the C++ code to compute the payoff 
        # self.z=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  Nsteps, self.MIn)

        # self.dt=self.T[0]/float(Nsteps) # time steps length
        self.d=int(np.log2(Nsteps)) #power 2 number steps


     # objfun: 
    def objfun(self,Nsteps):
        mean = np.zeros(2*Nsteps)
        covariance= np.identity(2*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)

        #non hierarchical
        #W1perp=y[Nsteps:]
        #hierarchical way
        yperp_1=y[Nsteps+1:2*Nsteps]
        yperp1=y[Nsteps]
        bbperp=self.brownian_increments(yperp1,yperp_1,Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp)-1)]

         #non hierarchical
        #W1=y[0:Nsteps]
        #hierarchical way
        y_1=y[1:Nsteps]
        y1=y[0]
        bb=self.brownian_increments(y1,y_1,Nsteps)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(Nsteps) for i in range(0,len(bb)-1)]

        QoI=self.z.ComputePayoffRT_single(W1,W1perp); 
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
       
        
        for k in range(1,self.d+1):
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
        #exact= 0.0712073 #exact value of K=1, H=0.43_xi_0.235^2_eta_1_9_r__09
        #exact=  0.0791   #0.0792047  #exact value of K=1, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact= 0.224905759853  #exact value of K=0.8, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact= 0.00993973310944  #exact value of K=1.2, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact=0.124756301225  #exact value of K=1, H=0.02_xi_01_eta_0_4_r__07
        #exact=0.2407117  #exact value of K=0.8, H=0.02_xi_01_eta_0_4_r__07
        exact=0.0568  #exact value of K=1.2, H=0.02_xi_01_eta_0_4_r__07

        marker=['>', 'v', '^', 'o', '*','+','-',':']
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([2])
        dt_arr=1.0/(Nsteps_arr)
        
        error=np.zeros(1)
        stand=np.zeros(1)
        elapsed_time_qoi=np.zeros(1)
        Ub=np.zeros(1)
        Lb=np.zeros(1)
       
        values=np.zeros(((1*(10**6),1))) 
        num_cores = mp.cpu_count()
        for i in range(0,1):
            print i
        
            start_time=time.time()

            prb = Problem(Nsteps_arr[i])     
            prb.z=RBergomi.RBergomiST( prb.x,  prb.HIn, prb.e,  prb.r,  prb.T, prb.k, Nsteps_arr[i], prb.MIn) 
            for j in range(1*(10**6)):
                  #Here we need to use the C++ code to compute the payoff 
                
                values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)
                


            elapsed_time_qoi[i]=time.time()-start_time
         
            print (np.mean(values[:,i],axis=0) *exact)
           


              

       
        start_time_2=time.time()
        error=np.abs(np.mean(values,axis=0) - 1) 
        elapsed_time_qoi=time.time()-start_time_2+elapsed_time_qoi

        stand=np.std(values, axis = 0)/  float(np.sqrt((1*(10**6))))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand

        print (elapsed_time_qoi)
        print(error)   
        print(stand)
        print Lb
        print Ub
         
        
#weak_convergence_rate_plotting()
weak_convergence_differences() 
    



