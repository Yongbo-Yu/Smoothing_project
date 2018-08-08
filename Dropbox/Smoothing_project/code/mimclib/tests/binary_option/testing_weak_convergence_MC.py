import numpy as np
import time
import scipy.stats as ss
from scipy.stats import sem

import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



class Problem(object):

# attributes
 
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    T=1.0                      # maturity
    sigma=None    # volatility
    d=None
    dt=None



#methods
    # this method initializes 
    def __init__(self,coeff,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money optio
        self.sigma=0.4
        self.dt=self.T/float(Nsteps) # time steps length
        self.d=int(np.log2(Nsteps)) #power 2 number steps

    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y,Nsteps):
        Y = np.array(Y)
        goal=self.objfun(Y,Nsteps);
        return goal


     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):

            mean = np.zeros(Nsteps)
            covariance= np.identity(Nsteps)
            y = np.random.multivariate_normal(mean, covariance)    

            y_1=y[1:Nsteps]
            y1=y[0]
           
            bar_z=self.newtons_method(0,y_1,Nsteps)
            from scipy.stats import norm
            QoI= norm.sf(bar_z) 
                  
            return QoI

        
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


    # This function simulates a 1D BS trajectory for stock price, it plays the role of f_1 in our notes
    def stock_price_trajectory_1D_BS(self,y1,y,Nsteps):
        bb=self.brownian_increments(y1,y,Nsteps)
        dW= [bb[i+1]-bb[i] for i in range(0,len(bb)-1)] 
        dbb=dW-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
      
        X=np.zeros(Nsteps+1) #here will store the BS trajectory
        X[0]=self.S0
        for n in range(1,Nsteps+1):
            X[n]=X[n-1]*(1+self.sigma*dW[n-1])
         
        return X[-1],dbb
       
    # Root solving procedure
 
    #Now we set up the methods used for newton iteration
    def dx(self,x,y,Nsteps):
        P1,dP1=self.f(x,y,Nsteps)
        return abs(0-P1)


    def f(self,y1,y,Nsteps):# need to check this
        X,dbb=self.stock_price_trajectory_1D_BS(y1,y,Nsteps) # right version
        #dbb=y # to check  analiticity wrt to \delta B
        fi=1+(self.sigma/float(np.sqrt(self.T)))*y1*(self.dt)+self.sigma*dbb
        product=np.prod(fi)
        summation=np.sum(1/fi)
        Py=product-(self.K/float(self.S0))
        dPy=(self.sigma/float(np.sqrt(self.T)))*(self.dt)*product*summation
        return Py,dPy    
        
        
    def newtons_method(self,x0,y,Nsteps,eps=1e-10):
        delta = self.dx(x0,y,Nsteps)
        while delta > eps:
        
            #(self.f(x0,y))
            P_value,dP=self.f(x0,y,Nsteps)
            x0 = x0 - 0.1*P_value/dP
            delta = self.dx(x0,y,Nsteps)
            
        return x0     



def weak_convergence_differences():    
        exact=  0.4207
        marker=['>', 'v', '^', 'o', '*','+','-',':']
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([2,4,8,16])
        dt_arr=1.0/(Nsteps_arr)
        error_diff=np.zeros(3)
        stand_diff=np.zeros(3)
        error=np.zeros(4)
        stand=np.zeros(4)
        Ub=np.zeros(4)
        Lb=np.zeros(4)
        Ub_diff=np.zeros(3)
        Lb_diff=np.zeros(3)
        values=np.zeros((10**4,4)) 
        for i in range(0,4):
            print i
            for j in range(10**4):
                prb = Problem(1,Nsteps_arr[i]) 
                values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)
        

        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(10**4))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand
        print(error)   
        print(stand)
        print Lb
        print Ub
         
        differences= [values[:,i]-values[:,i+1] for i in range(0,3)]
        error_diff=np.abs(np.mean(differences,axis=1))
        print error_diff 
        stand_diff=np.std(differences, axis = 1)/ float(np.sqrt(10**4))
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
        
        
        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_binary_option_relative.eps', format='eps', dpi=1000)  

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
        plt.savefig('./results/weak_convergence_order_differences_binary_option_relative.eps', format='eps', dpi=1000)  


weak_convergence_differences()   
    

