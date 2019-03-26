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

class Problem(object):
 
# attributes
    # attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    basket_d=2     # number of assets in the basket
    c= (1/float(basket_d))*np.ones(basket_d)     # weigths
    sigma=None    # vector of volatilities
    K=None         # Strike price
    T=1.0                      # maturity
    rho=None                  #correlation matrix
    exact=12.900784  # 2-d, sigma=0.4, S_0=K=100, T=1, r=0,rho=0.3
    #exact=11.447  # 2-d, sigma=0.4, S_0=K=100, T=1, r=0,rho=0.0
   
 
 
#methods
    # this method initializes 
    def __init__(self,Nsteps,nested=False):
        self.nested = nested
   
        self.random_gen = None or np.random
        
        self.S0=100*np.ones(self.basket_d) 
       
        self.sigma=0.4*np.ones(self.basket_d) #vector of volatilities
    
        self.K= 100                        # Strike price and coeff determine if we have in/at/out the money option
    
        from scipy.linalg import toeplitz 
        self.rho=toeplitz([1,0.3]) #correlation matrix
        #self.rho=toeplitz([1,0.0]) #correlation matrix
        self.dt=self.T/float(Nsteps) # time steps length
        self.d=int(np.log2(Nsteps)) #power 2 number steps

       
 

     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):

        mean = np.zeros(self.basket_d*Nsteps)
        covariance= np.identity(self.basket_d*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)    
        y_1=y[1:Nsteps]
        y1=y[0]
        y2=y[Nsteps]
        y_2=y[Nsteps+1:]
    
        X=self.stock_price_trajectory_basket_BS(y1,y_1,y2,y_2,Nsteps)
           
            
        QoI= self.payoff(X)
        
        return QoI

    def brownian_increments(self,y1,y,Nsteps):
        t=np.linspace(0, self.T, Nsteps+1)     
        h=Nsteps
        j_max=1
        bb= np.zeros((1,Nsteps+1))
        bb[0,h]=np.sqrt(self.T)*y1
       
        
         
        for k in range(1,self.d+1):
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
     
     
    def stock_price_trajectory_basket_BS(self,y1,yvec_1,y2,yvec_2,Nsteps):

        y=np.array([y1,y2])

        #building the brownian bridge increments
        bb1=self.brownian_increments(y[0],yvec_1,Nsteps)
        bb2=self.brownian_increments(y[1],yvec_2,Nsteps)

        dW1= [bb1[0,i+1]-bb1[0,i]  for i in range(0,Nsteps)] 
        dW2= [bb2[0,i+1]-bb2[0,i] for i in range(0,Nsteps)] 
        dW=np.array([dW1 ,dW2])        
          # construct the correlated  brownian bridge increments
        lower_triang_cholesky = np.linalg.cholesky(self.rho)
     
        dW=np.dot(lower_triang_cholesky,dW)  
    
        dW1=dW[0,:]
        dW2=dW[1,:]
     
      

        X=np.zeros((self.basket_d,Nsteps+1)) #here will store the BS trajectory
      
        X[:,0]=self.S0
        for n in range(1,Nsteps+1):
            #X[0,n]=X[0,n-1]*(1+self.sigma[0]*((self.dt/float(np.sqrt(self.T)))*(self.A_inv[0,0]*y1+ self.A_inv[0,1:].dot(y2)) +  dbb1[n-1] ))  
            #X[1,n]=X[1,n-1]*(1+self.sigma[1]*((self.dt/float(np.sqrt(self.T)))*(self.A_inv[1,0]*y1+ self.A_inv[1,1:].dot(y2)) +  dbb2[n-1] ) )  
            X[0,n]=X[0,n-1]*(1+self.sigma[0]*dW1[n-1])
            X[1,n]=X[1,n-1]*(1+self.sigma[1]*dW2[n-1])

      
        return X[:,-1]
         
         
      
         # this function defines the payoff function used here
    def payoff(self,x): 
       #print(x)
       g=(x.dot(self.c)-self.K)
       if g<0:
           g=0
       
       return g
         


def weak_convergence_differences():    
        start_time=time.time()
        exact= 12.900784
        marker=['>', 'v', '^', 'o', '*','+','-',':']
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([2,4,8,16])
        dt_arr=1.0/(Nsteps_arr)
        error_diff=np.zeros(3)
        stand_diff=np.zeros(3)
        elapsed_time_qoi=np.zeros(4)
        error=np.zeros(4)
        stand=np.zeros(4)
        Ub=np.zeros(4)
        Lb=np.zeros(4)
        Ub_diff=np.zeros(3)
        Lb_diff=np.zeros(3)
        values=np.zeros((1*(10**5),4)) 
        for i in range(0,4):
            print i
            start_time=time.time()
            prb = Problem(Nsteps_arr[i]) 
            for j in range(1*(10**5)):
              
                values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)

            elapsed_time_qoi[i]=time.time()-start_time
            print  elapsed_time_qoi[i]  
            print np.mean(values[:,i]*float(exact))
          
        
             
        print elapsed_time_qoi

        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(1*(10**5)))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand
        print(error)   
        print(stand)
        print Lb
        print Ub
         
        differences= [values[:,i]-values[:,i+1] for i in range(0,3)]
        error_diff=np.abs(np.mean(differences,axis=1))
        print error_diff 
        stand_diff=np.std(differences, axis = 1)/ float(np.sqrt(1*(10**5)))
        print stand_diff
        Ub_diff=np.abs(np.mean(differences,axis=1))+1.96*stand_diff
        Lb_diff=np.abs(np.mean(differences,axis=1))-1.96*stand_diff
        print Ub_diff
        print Lb_diff

       
        z= np.polyfit(np.log(dt_arr), np.log(error), 1)
        fit=np.exp(z[0]*np.log(dt_arr))
        print z[0]


        z3=np.zeros(2)
        z3[0]=1.0
        z3[1]=np.log(error[0]*2)
        fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])




        z_diff= np.polyfit(np.log(dt_arr[0:3]), np.log(error_diff), 1)
        fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:3]))
        print z_diff[0]

        z3diff=np.zeros(2)
        z3diff[0]=1.0
        z3diff[1]=np.log(error_diff[0]*2)
        fit3diff=np.exp(z3diff[0]*np.log(dt_arr[0:3])+z3diff[1])
        
        fig = plt.figure()

        plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
        plt.plot(dt_arr, Lb,linewidth=2.0,label='Lb' ,linestyle = '--', hold=True) 
        plt.plot(dt_arr, Ub,linewidth=2.0,label='Ub' ,linestyle = '--', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr, fit,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--', marker='o')
        plt.plot(dt_arr, fit3,linewidth=2.0,label=r'rate= %s' % format(z3[0]  , '.2f'), linestyle = '--', marker='o')
        
        
        plt.ylabel(r'$\mid  E[g(X_{\Delta t})-  g(X)] \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_basket_2d_option_plain_MC_M_10_3.eps', format='eps', dpi=1000)  

        fig = plt.figure()
        plt.plot(dt_arr[0:3], error_diff,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
        plt.plot(dt_arr[0:3], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = '--', hold=True) 
        plt.plot(dt_arr[0:3], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = '--', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr[0:3], fit_diff,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--', marker='o')
        plt.plot(dt_arr[0:3], fit3diff,linewidth=2.0,label=r'rate= %s' % format(z3diff[0]  , '.2f'), linestyle = '--', marker='o')
        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_differences_cbasket_2d_option_plain_MC_M_10_3.eps', format='eps', dpi=1000)  


weak_convergence_differences()   

    



