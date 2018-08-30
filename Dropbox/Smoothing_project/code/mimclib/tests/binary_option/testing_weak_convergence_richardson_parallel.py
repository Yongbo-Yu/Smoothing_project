import numpy as np
import time
import scipy.stats as ss
from scipy.stats import sem

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


from joblib import Parallel, delayed
import multiprocessing


class Problem_non_smooth_richardson_extrapolation(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    T=1.0                      # maturity
    sigma=None    # volatility
    #N=4    # number of time steps which will be equal to the number of brownian bridge components (we set is a power of 2)
    #N=2         # discretization resolution
    d=None
    dt=None



#methods
    # this method initializes 
    def __init__(self,coeff,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
        #self.S0=np.random.uniform(8,20,1) # initial stock price
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        #self.sigma=np.random.uniform(0.3,0.4,1)  #volatility
        self.sigma=0.4
        
    

     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):


        mean = np.zeros(2*Nsteps)
        covariance= np.identity(2*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)    
        y_1f=y[1:2*Nsteps]
        y1f=y[0]
        
        y_1c=y_1f[0:Nsteps-1]
        y1c=y1f
        
        #level 1: richardson extrapol

        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )
     
        #kink points by newton method
        bar_z_f=self.newtons_method(0,y_1f,2*Nsteps)
        bar_z_c=self.newtons_method(0,y_1c,Nsteps)

        from scipy.stats import norm
            
        d[1,0] = norm.sf(bar_z_f) 

        d[0,0] =norm.sf(bar_z_c) 


        #Richardson
        d[1,1] = 2*d[1,0] - d[0,0]
        QoI=d[1,1]
                
        return QoI

    def brownian_increments(self,y1,y,Nsteps):
        t=np.linspace(0, self.T, Nsteps+1)     
        h=Nsteps
        j_max=1
        bb= np.zeros(Nsteps+1)
        bb[h]=np.sqrt(self.T)*y1
       
        ds=int(np.log2(Nsteps)) #power 2 number steps
        for k in range(1,ds+1):
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
        dt_s=self.T/float(Nsteps)
    
        X=np.zeros(Nsteps+1) #here will store the BS trajectory
        X[0]=self.S0
        for n in range(1,Nsteps+1):
            X[n]=X[n-1]*(1+self.sigma*dW[n-1])
        dbb=dW-(dt_s/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point
        return X[-1],dbb

        

 

    # Root solving procedure
 
    #Now we set up the methods used for newton iteration
    def dx(self,x,y,Nsteps):
        P1,dP1=self.f(x,y,Nsteps)
        return abs(0-P1)


    def f(self,y1,y,Nsteps):# need to check this
        X,dbb=self.stock_price_trajectory_1D_BS(y1,y,Nsteps) # right version
        dt_s=self.T/float(Nsteps)
        fi=1+(self.sigma/float(np.sqrt(self.T)))*y1*(dt_s)+self.sigma*dbb
        product=np.prod(fi)
        summation=np.sum(1/fi)
        Py=product-(self.K/float(self.S0))
        dPy=(self.sigma/float(np.sqrt(self.T)))*(dt_s)*product*summation
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
    exact=  0.420740290561
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # # feed parameters to the problem
    Nsteps_arr=np.array([1,2,4,8])
   
    dt_arr=1.0/(Nsteps_arr)
    error_diff=np.zeros(3)
    stand_diff=np.zeros(3)
    error=np.zeros(4)
    stand=np.zeros(4)
    elapsed_time_qoi=np.zeros(4)
    Ub=np.zeros(4)
    Lb=np.zeros(4)
    Ub_diff=np.zeros(3)
    Lb_diff=np.zeros(3)
    values=np.zeros((4*(10**6),4)) 
    inputs = range(4*(10**6))
    num_cores = multiprocessing.cpu_count()
              
    for i in range(0,4):
        print i
        start_time=time.time()
        
        prb = Problem_non_smooth_richardson_extrapolation(1,Nsteps_arr[i]) 
        
        def processInput(j):
            return prb.objfun(Nsteps_arr[i])/float(exact)

        results = Parallel(n_jobs=num_cores)(delayed(processInput)(j) for j in inputs)
        values[:,i]=results

        elapsed_time_qoi[i]=time.time()-start_time
        print  elapsed_time_qoi[i]
          
    

    
    start_time_2=time.time()
    error=np.abs(np.mean(values,axis=0) - 1) 
    elapsed_time_qoi=time.time()-start_time_2+elapsed_time_qoi

    stand=np.std(values, axis = 0)/  float(np.sqrt((4*(10**6))))
    Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
    Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand

    print (elapsed_time_qoi)
    print(error)   
    print(stand)
    print Lb
    print Ub
    
     
    differences= [values[:,i]-values[:,i+1] for i in range(0,3)]
    error_diff=np.abs(np.mean(differences,axis=1))
    print error_diff 
    stand_diff=np.std(differences, axis = 1)/ float(np.sqrt((4*(10**6))))
    print stand_diff
    Ub_diff=np.abs(np.mean(differences,axis=1))+1.96*stand_diff
    Lb_diff=np.abs(np.mean(differences,axis=1))-1.96*stand_diff
    print Ub_diff
    print Lb_diff

   
    z= np.polyfit(np.log(dt_arr), np.log(error), 1)
    fit=np.exp(z[0]*np.log(dt_arr))
    print z[0]


    z3=np.zeros(2)
    z3[0]=2.0
    z3[1]=np.log(error[0])
    fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])

    z_diff= np.polyfit(np.log(dt_arr[0:3]), np.log(error_diff), 1)
    fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:3]))
    print z_diff[0]

    z3diff=np.zeros(2)
    z3diff[0]=2.0
    z3diff[1]=np.log(error_diff[0])
    fit3diff=np.exp(z3diff[0]*np.log(dt_arr[0:3])+z3diff[1])
    
    fig = plt.figure()

    plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
    plt.plot(dt_arr, Lb,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
    plt.plot(dt_arr, Ub,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\Delta t$',fontsize=14)

    plt.plot(dt_arr, fit*10,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--')
    plt.plot(dt_arr, fit3*10,linewidth=2.0,label=r'rate= %s' % format(z3[0]  , '.2f'), linestyle = '--')
    
    
    plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
    plt.legend(loc='upper left')
    plt.savefig('./results/weak_convergence_order_binary_richardson_relative_M_10_5.eps', format='eps', dpi=1000)  

    fig = plt.figure()
    plt.plot(dt_arr[0:3], error_diff,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
    plt.plot(dt_arr[0:3], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
    plt.plot(dt_arr[0:3], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\Delta t$',fontsize=14)

    plt.plot(dt_arr[0:3], fit_diff*10,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--')
    plt.plot(dt_arr[0:3], fit3diff*10,linewidth=2.0,label=r'rate= %s' % format(z3diff[0]  , '.2f'), linestyle = '--')
    plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
    plt.legend(loc='upper left')
    plt.savefig('./results/weak_convergence_order_differences_binary_richardson_relative_M_10_5.eps', format='eps', dpi=1000)  


weak_convergence_differences()
