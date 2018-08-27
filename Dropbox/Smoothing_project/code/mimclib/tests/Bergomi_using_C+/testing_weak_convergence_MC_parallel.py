# In this file, we plot the weak convergence rate for MC without Richardson extrapolation without doing the partial change of measure


import numpy as np
import time
import scipy.stats as ss

import random

#from joblib import Parallel, delayed
#import multiprocessing
from numba import autojit, prange

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import numpy as np
import time
import scipy.stats as ss

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
    x=0.235**2;   # this will provide the set of xi parameter values 
    #x=0.00001;
    HIn=Vector(1)    # this will provide the set of H parameter values
    HIn[0]=0.07
    e=Vector(1)    # This will provide the set of eta paramter values
    e[0]=1.9
    r=Vector(1)   # this will provide the set of rho paramter values
    r[0]=-0.9
    #r[0]=0
    T=Vector(1)     # this will provide the set of T(time to maturity) parameter value
    T[0]=1.0
    k=Vector(1)     # this will provide the set of K (strike ) paramter value
    k[0]=1
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

        

        yperp_1=y[Nsteps+1:2*Nsteps]
        yperp1=y[Nsteps]
        bbperp=self.brownian_increments(yperp1,yperp_1,Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp)-1)]

         #non hierarchical
        #W1=y[0:self.N]
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


    

    def Quit(self):
        pass

    def __exit__(self, type, value, traceback):
        pass

    def __enter__(self):
        return self

    @staticmethod
    def Init():
        import sys #This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter
        count = len(sys.argv)  #sys.argv is a list in Python, which contains the command-line arguments passed to the script. With the len(sys.argv) function you can count the number of arguments. 
        #arr = (ct.c_char_p * len(sys.argv))()
        arr = sys.argv


# def weak_convergence_rate_plotting():    
#     #num_cores = multiprocessing.cpu_count()
#         exact= 0.0712073 #exact value of K=1, H=0.43 
#         marker=['>', 'v', '^', 'o', '*','+','-',':']
#         ax = figure().gca()
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         # # feed parameters to the problem
#         Nsteps_arr=np.array([2,4,8,16,32,64])
#         dt_arr=1.0/(Nsteps_arr)
#         error=np.zeros(6)
#         stand=np.zeros(6)

#         for i in range(0,6):
#             values=np.zeros(10000)
#             for j in prange(10000):

#                 #andom.seed( j)
                
#                 prb = Problem(Nsteps_arr[i]) 
#                 values[j]=prb.objfun(Nsteps_arr[i]) 
#             error[i]=np.abs(np.mean(values) - exact) 
#             print error[i] 
#             stand[i]=np.std(values, axis = 0)/(10000)

#         print(error)   
#         print(stand)
        

#         z= np.polyfit(np.log(dt_arr), np.log(error), 1)
#         fit=np.exp(z[0]*np.log(dt_arr))
#         print z[0]
        
#         plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
#         plt.yscale('log')
#         plt.xscale('log')
#         plt.xlabel(r'$\Delta t$',fontsize=14)

#         plt.plot(dt_arr, fit,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--', marker='o')
        
#         plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
#         plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
#         plt.legend(loc='upper left')
#         plt.savefig('./results/weak_convergence_order_Bergomi_H_007_K_1.eps', format='eps', dpi=1000)     
    



def weak_convergence_differences():    
        #exact= 0.0712073 #exact value of K=1, H=0.43 
        exact= 0.0792047  #exact value of K=1, H=0.07
        marker=['>', 'v', '^', 'o', '*','+','-',':']
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([2,4,8,16,32])
        dt_arr=1.0/(Nsteps_arr)
        error_diff=np.zeros(4)
        stand_diff=np.zeros(4)
        error=np.zeros(5)
        stand=np.zeros(5)
        Ub=np.zeros(5)
        Lb=np.zeros(5)
        Ub_diff=np.zeros(4)
        Lb_diff=np.zeros(4)
        values=np.zeros((10**6,5)) 
        num_cores = mp.cpu_count()
        for i in range(0,5):
            print i
            

            def processInput(j):
                  #Here we need to use the C++ code to compute the payoff 
                prb = Problem(Nsteps_arr[i])   
                prb.z=RBergomi.RBergomiST( prb.x,  prb.HIn, prb.e,  prb.r,  prb.T, prb.k, Nsteps_arr[i], prb.MIn)
                return prb.objfun(Nsteps_arr[i])/float(exact)

            # results = Parallel(n_jobs=num_cores)(delayed(processInput)(j) for j in inputs)
            p =  pp.ProcessPool(num_cores)  # Processing Pool with four processors
            
            values[:,i]= p.map(processInput, range((10**6)))    
           


              

       

        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(10**5))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand
        print(error)   
        print(stand)
        print Lb
        print Ub
         
        differences= [values[:,i]-values[:,i+1] for i in range(0,4)]
        error_diff=np.abs(np.mean(differences,axis=1))
        print error_diff 
        stand_diff=np.std(differences, axis = 1)/ float(np.sqrt(10**6))
        print stand_diff
        Ub_diff=np.abs(np.mean(differences,axis=1))+1.96*stand_diff
        Lb_diff=np.abs(np.mean(differences,axis=1))-1.96*stand_diff
        print Ub_diff
        print Lb_diff

       
        z= np.polyfit(np.log(dt_arr), np.log(error), 1)
        fit=np.exp(z[0]*np.log(dt_arr))
        print z[0]

        z_diff= np.polyfit(np.log(dt_arr[0:4]), np.log(error_diff), 1)
        fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:4]))
        print z_diff[0]
        
        fig = plt.figure()

        plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' , marker='>',hold=True) 
        plt.plot(dt_arr, Lb,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
        plt.plot(dt_arr, Ub,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr, fit*10,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--')
        
        
        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_Bergomi_H_007_K_1_M_10_6_CI_relative.eps', format='eps', dpi=1000)  

        fig = plt.figure()
        plt.plot(dt_arr[0:4], error_diff,linewidth=2.0,label='weak_error' , marker='>', hold=True) 
        plt.plot(dt_arr[0:4], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
        plt.plot(dt_arr[0:4], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr[0:4], fit_diff*10,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--')
        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_differences_Bergomi_H_007_K_1_M_10_6_CI_relative.eps', format='eps', dpi=1000)  

#weak_convergence_rate_plotting()
weak_convergence_differences()