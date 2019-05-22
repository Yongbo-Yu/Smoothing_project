#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

# In this file, we plot the weak convergence rate for MC without Richardson extrapolation without doing the partial change of measure


import numpy as np
import time
import scipy.stats as ss
from scipy.stats import norm

import random


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


import fftw3
import RBergomi
from RBergomi import *
import mimclib.misc as misc


import pathos.multiprocessing as mp
import pathos.pools as pp

from latticeseq_b2 import latticeseq_b2

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
        self.zff=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  4*Nsteps, self.MIn)

        
        # random shifting
        self.M = 2**3
        random.seed(1) # Mersenne Twister
        self.shifts = np.random.rand(self.M, 8*Nsteps)



     # objfun: 
    def objfun(self,x,Nsteps):
        
        # now use quasi-random samplesY 
     
    	y=norm.ppf(x)

       
        yperp_1ff=y[4*Nsteps+1:8*Nsteps]
        yperp1ff=y[4*Nsteps]

        bbperp_ff=self.brownian_increments(yperp1ff,yperp_1ff,4*Nsteps)
        W1perp_ff= [(bbperp_ff[i+1]-bbperp_ff[i]) *np.sqrt(4*Nsteps) for i in range(0,len(bbperp_ff)-1)]
        

        yperp_1f=yperp_1ff[0:2*Nsteps-1]
        yperp1f=yperp1ff

        bbperp_f=self.brownian_increments(yperp1f,yperp_1f,2*Nsteps)
        W1perp_f= [(bbperp_f[i+1]-bbperp_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bbperp_f)-1)]


        yperp_1c=yperp_1ff[0:Nsteps-1]
        yperp1c=yperp1ff
        bbperp_c=self.brownian_increments(yperp1c,yperp_1c,Nsteps)
        W1perp_c= [(bbperp_c[i+1]-bbperp_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp_c)-1)]
        



        #hierarchical way

        y_1ff=y[1:4*Nsteps]
        y1ff=y[0]

        bb_ff=self.brownian_increments(y1ff,y_1ff,4*Nsteps)
        W1_ff= [(bb_ff[i+1]-bb_ff[i]) *np.sqrt(4*Nsteps) for i in range(0,len(bb_ff)-1)]


        y_1f=y_1ff[0:2*Nsteps-1]
        y1f=y1ff
        bb_f=self.brownian_increments(y1f,y_1f,2*Nsteps)
        W1_f= [(bb_f[i+1]-bb_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bb_f)-1)]

        y_1c=y_1ff[0:Nsteps-1]
        y1c=y1ff
        bb_c=self.brownian_increments(y1c,y_1c,Nsteps)
        W1_c= [(bb_c[i+1]-bb_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bb_c)-1)]


         #level 2: richardson extrapol (new version)
    
        dc = self.z.ComputePayoffRT_single(W1_c,W1perp_c); 
        df=self.zf.ComputePayoffRT_single(W1_f,W1perp_f); 
        dff=self.zff.ComputePayoffRT_single(W1_ff,W1perp_ff); 

        d11=2*df-dc
        d21=2*dff-df
        d=((2**2) *d21-d11)/float(2**2-1)

        QoI=d 

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
        #exact= 0.0712073 #exact value of K=1, H=0.43_xi_0.235^2_eta_1_9_r__09
        exact=   0.0791   #0.0792047  #exact value of K=1, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact= 0.224905759853  #exact value of K=0.8, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact= 0.00993973310944  #exact value of K=1.2, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact=0.124756301225  #exact value of K=1, H=0.02_xi_01_eta_0_4_r__07
        #exact=0.2407117  #exact value of K=0.8, H=0.02_xi_01_eta_0_4_r__07
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

        nprev = 0
        m=17
        N = 2**m
        
    
        for i in range(0,1):
            print i
        
            start_time=time.time()

            prb = Problem(Nsteps_arr[i])    
            latgen = latticeseq_b2('exod2_base2_m20_CKN.txt', s=8*Nsteps_arr[i])
            n = N/prb.M
            print n
           

            acc = np.zeros((prb.M,)) # accumulator for each shift  
            for j in range(0,n):
                  #Here we need to use the C++ code to compute the payoff 
                 
                
               
                # QMC generator (truncated to correct number of dimensions)
                
                x = latgen.next(); # next point, evaluate in all shifts
                #print x
                acc += [ prb.objfun((x+shift) % 1, Nsteps_arr[0]) for shift in prb.shifts ]
            acc=acc/float(n)    
            #print acc

            elapsed_time_qoi[i]=time.time()-start_time
            
          
             
       
        start_time_2=time.time()
        error=np.abs(np.mean(acc)) 
        elapsed_time_qoi=time.time()-start_time_2+elapsed_time_qoi

        stand=np.std(acc)/ float(np.sqrt(prb.M))
        Ub=np.abs(np.mean(acc) )+1.96*stand
        Lb=np.abs(np.mean(acc) )-1.96*stand

        print (elapsed_time_qoi)
        print(error)   
        print(stand)
        print Lb
        print Ub
      
        
weak_convergence_differences()      

