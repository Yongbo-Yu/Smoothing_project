# In this file, we plot the weak convergence rate for MC without Richardson extrapolation without doing the partial change of measure


import numpy as np
import time


import random


#from joblib import Parallel, delayed
#import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


from RBergomi import *
from RfBm import *
from RNorm  import *


import pathos.multiprocessing as mp
import pathos.pools as pp

class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    
  
    # for the values of below paramters, we need to see the paper as well check with Christian 
    xi=0.235**2;   # this will provide the set of xi parameter values 
    #x=0.1;
    # this will provide the set of H parameter values
    #H=0.43
    H=0.07
    #HIn[0]=0.02
      # This will provide the set of eta paramter values
    eta=1.9
    #e[0]=0.4
     # this will provide the set of rho paramter values
    rho=-0.9
    #rho=-0.7
         # this will provide the set of T(time to maturity) parameter value
    T=1.0
    # this will provide the set of K (strike ) paramter value
    K=1.0
   # y1perp = Vector(N)

   
  


    #methods
    # this method initializes 
    def __init__(self,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
         #Here we need to use the C++ code to compute the payoff 
        # self.z=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  Nsteps, self.MIn)

        # self.dt=self.T[0]/float(Nsteps) # time steps length
        self.d=int(np.log2(Nsteps)) #power 2 number steps

        self.rnorm=RNorm()

        self.rfbm= RfBm(Nsteps,self.H,self.rnorm)
        self.L=np.array(self.rfbm.GetL())  #getting the L matrix
        L11=self.L[0:Nsteps]
        
        self.L1=L11[:,0:Nsteps]
        self.L1_inv=np.linalg.inv(self.L1)

        self.W1 = Vector(Nsteps)
        self.Wtilde = Vector(Nsteps)  
        self.v=Vector(Nsteps)
    

     # objfun: 
    def objfun(self,Nsteps):
        # step 1: Generate 2N Gaussian vector
        mean = np.zeros(2*Nsteps)
        covariance= np.identity(2*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)

        # step 2: Construct the hierarchical transformation: y -> X(such that X remains Gaussian), construct L1^{-1}
        X = Vector(2*Nsteps)
        #bb=self.brownian_increments(y[0],y[1:Nsteps],Nsteps)
        #W= [(bb[i+1]-bb[i]) *np.sqrt(Nsteps) for i in range(0,len(bb)-1)]
        #X[0:Nsteps]=(self.L1_inv).dot(W)   #constructing X[0:Nsteps-1]

        X[Nsteps:]=y[Nsteps:]   #constructing X[Nsteps:]
        X[0:Nsteps]=y[0:Nsteps]

        
        # Step 3:  given x compute W1 and Wtilde
        
        self.rfbm.generate(X,self.W1, self.Wtilde); # need to adjust rfbm code
        # Step 4:  compute price
        QoI=updatePayoff_cholesky(self.Wtilde,self.W1,self.v,self.eta,self.H,self.rho,self.xi,self.T,self.K,Nsteps) 
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
        exact= 0.0791  #exact value of K=1, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact= 0.224905759853  #exact value of K=0.8, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact= 0.00993973310944  #exact value of K=1.2, H=0.07_xi_0.235^2_eta_1_9_r__09
        #exact=0.124756301225  #exact value of K=1, H=0.02_xi_01_eta_0_4_r__07
        #exact=0.2407117  #exact value of K=0.8, H=0.02_xi_01_eta_0_4_r__07
        #exact=0.0568394  #exact value of K=1.2, H=0.02_xi_01_eta_0_4_r__07

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
        elapsed_time_qoi=np.zeros(5)
        Ub=np.zeros(5)
        Lb=np.zeros(5)
        Ub_diff=np.zeros(4)
        Lb_diff=np.zeros(4)
        values=np.zeros(((4*(10**2),5))) 
        num_cores = mp.cpu_count()
        for i in range(0,5):
            print i
            

            start_time=time.time()
            def processInput(j):
                  #Here we need to use the C++ code to compute the payoff 
                prb = Problem(Nsteps_arr[i])          
                return prb.objfun(Nsteps_arr[i])/float(exact)

            # results = Parallel(n_jobs=num_cores)(delayed(processInput)(j) for j in inputs)
            p =  pp.ProcessPool(num_cores)  # Processing Pool with four processors
            
            values[:,i]= p.map(processInput, range(((4*(10**2)))))    

 
            elapsed_time_qoi[i]=time.time()-start_time
            print  elapsed_time_qoi[i]
            print (np.mean(values[:,i],axis=0) *0.0791)

    
        
    
        start_time_2=time.time()

       
        error=np.abs(np.mean(values,axis=0) - 1) 
        elapsed_time_qoi=time.time()-start_time_2+elapsed_time_qoi

        stand=np.std(values, axis = 0)/  float(np.sqrt((4*(10**2))))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand

        print (elapsed_time_qoi)
        print(error)   
        print(stand)
        print Lb
        print Ub
         
        differences= [values[:,i]-values[:,i+1] for i in range(0,4)]
        error_diff=np.abs(np.mean(differences,axis=1))
        print error_diff 
        stand_diff=np.std(differences, axis = 1)/ float(np.sqrt((4*(10**2))))
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



        
        z3=np.zeros(5)
        z3[0]=1.0
        z3[1]=np.log(error[0])
        fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])


        z3diff=np.zeros(4)
        z3diff[0]=1.0
        z3diff[1]=np.log(error_diff[0])
        fit3diff=np.exp(z3diff[0]*np.log(dt_arr[0:4])+z3diff[1])
        
        fig = plt.figure()

        plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' , marker='>',hold=True) 
        plt.plot(dt_arr, Lb,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
        plt.plot(dt_arr, Ub,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr, fit/10,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--')
        plt.plot(dt_arr, fit3*10,linewidth=2.0,label=r'rate= %s' % format(z3[0]  , '.2f'), linestyle = '--')
        
        
        plt.ylabel(r'$\mid  E[g(X_{\Delta t})-  g(X)] \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_Bergomi_H_007_K_1_M_4_10_2_CI_relative_cholesky_non_hierarchical_parallel.eps', format='eps', dpi=1000)  

        fig = plt.figure()
        plt.plot(dt_arr[0:4], error_diff,linewidth=2.0,label='weak_error' , marker='>', hold=True) 
        plt.plot(dt_arr[0:4], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
        plt.plot(dt_arr[0:4], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr[0:4], fit_diff/10,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--')
        plt.plot(dt_arr[0:4], fit3diff*10,linewidth=2.0,label=r'rate= %s' % format(z3diff[0]  , '.2f'), linestyle = '--')
        plt.ylabel(r'$\mid E[g(X_{\Delta t})-  g(X_{\Delta t/2})] \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_differences_Bergomi_H_007_K_1_M_4_10_2_CI_relative_cholesky_non_hierarchical_parallel.eps', format='eps', dpi=1000)  

#weak_convergence_rate_plotting()
weak_convergence_differences()