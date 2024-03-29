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
        self.z=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  Nsteps, self.MIn)
        self.zf=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  2*Nsteps, self.MIn)
        self.zff=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  4*Nsteps, self.MIn)

       
        


   
    


     # objfun: 
    def objfun(self,Nsteps):
        mean = np.zeros(8*Nsteps)
        covariance= np.identity(8*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)
  


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
        #exact= 0.0712073 #exact value of K=1, H=0.43 
        exact= 0.0792047  #exact value of K=1, H=0.07
        marker=['>', 'v', '^', 'o', '*','+','-',':']
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([1,2])
        dt_arr=1.0/(Nsteps_arr)
        error=np.zeros(2)
        stand=np.zeros(2)
        Ub=np.zeros(2)
        elapsed_time_qoi=np.zeros(2)
        Lb=np.zeros(2)
        values=np.zeros((5*(10**5),2)) 
        for i in range(0,2):
            print i
            start_time=time.time()
            prb = Problem(Nsteps_arr[i]) 
            for j in range(5*(10**5)):
                values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)



            elapsed_time_qoi[i]=time.time()-start_time
            print  elapsed_time_qoi[i]    
        
        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(5*(10**5)))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand



        print (elapsed_time_qoi)
        print(error)   
        print(stand)
        print Lb
        print Ub


        differences= [values[:,i]-values[:,i+1] for i in range(0,1)]
        error_diff=np.abs(np.mean(differences,axis=1))
        print error_diff 
        stand_diff=np.std(differences, axis = 1)/ float(np.sqrt(5*(10**5)))
        print stand_diff
        Ub_diff=np.abs(np.mean(differences,axis=1))+1.96*stand_diff
        Lb_diff=np.abs(np.mean(differences,axis=1))-1.96*stand_diff
        print Ub_diff
        print Lb_diff
    
       
        z= np.polyfit(np.log(dt_arr), np.log(error), 1)
        fit=np.exp(z[0]*np.log(dt_arr))
        print z[0]

        z3=np.zeros(2)
        z3[0]=3.0
        z3[1]=np.log(error[0])
        fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])

        z_diff= np.polyfit(np.log(dt_arr[0:1]), np.log(error_diff), 1)
        fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:1]))
        print z_diff[0]

        z3diff=np.zeros(2)
        z3diff[0]=3.0
        z3diff[1]=np.log(error_diff[0])
        fit3diff=np.exp(z3diff[0]*np.log(dt_arr[0:1])+z3diff[1])

    
        fig = plt.figure()

        plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
        plt.plot(dt_arr, Lb,linewidth=2.0,label='Lb' ,linestyle = '--', hold=True) 
        plt.plot(dt_arr, Ub,linewidth=2.0,label='Ub' ,linestyle = '--', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)
        plt.plot(dt_arr, fit,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--', marker='o')
        plt.plot(dt_arr, fit3,linewidth=2.0,label=r'rate= %s' % format(z3[0]  , '.2f'), linestyle = '--', marker='o')
        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_Bergomi_H_007_K_1_richardson_level2_relative_M_10_5.eps', format='eps', dpi=1000)  

        fig = plt.figure()
        plt.plot(dt_arr[0:1], error_diff,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
        plt.plot(dt_arr[0:1], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = '--', hold=True) 
        plt.plot(dt_arr[0:1], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = '--', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr[0:1], fit_diff,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--', marker='o')
        plt.plot(dt_arr[0:1], fit3diff,linewidth=2.0,label=r'rate= %s' % format(z3diff[0]  , '.2f'), linestyle = '--', marker='o')


        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_differences_Bergomi_H_043_K_1_richardson_level2_relative_M_10_5.eps', format='eps', dpi=1000)



weak_convergence_differences()