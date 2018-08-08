import numpy as np
import time
import scipy.stats as ss

#from joblib import Parallel, delayed
#import multiprocessing
from numba import autojit, prange

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator





class Problem_richardson_extrapolation(object):

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
    smooth=1


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
        self.dt=self.T/float(Nsteps) # time steps length
        self.d_c=int(np.log2(Nsteps)) #power 2 number steps
        self.d_f=int(np.log2(2*Nsteps))
        

    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y,Nsteps):
        Y = np.array(Y)
        goal=self.objfun(Y,Nsteps);
        return goal


     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):
        start_time=time.time()
        #print(y)
        #x=np.asarray(self.richardson(y,self.N, n=1))
        #QoI=self.payoff_smooth2(x)
        #print(y)
        #level 1: richardson extrapol (old version)
        # n=1
        # d = np.array( [[0] * (n + 1)] * (n + 1), float )
        # d[0,0] = self.payoff_smooth2(self.stock_price_trajectory_1D_BS(Nsteps)) # approximation
        # powerOf4 = 1  # values of 4^j
 
        # d[1,0] = self.payoff_smooth2(self.stock_price_trajectory_1D_BS(2*Nsteps)) # approximation

        # powerOf4 = 4 * powerOf4
        # d[1,1] = d[1,0] + ( d[1,0] - d[0,0] ) / ( powerOf4 - 1 )
        # QoI=d[1,1] 
       
         #level 1: richardson extrapol (new version)

        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )
        price_c,price_f=self.stock_price_trajectory_1D_BS(Nsteps)
        d[0,0] = self.payoff_smooth2(price_c) # approximation
        d[1,0]=self.payoff_smooth2(price_f) # approximation


        d[1,1] = 2*d[1,0] - d[0,0]
        QoI=d[1,1] 

        elapsed_time_qoi=time.time()-start_time;

        self.elapsed_time=self.elapsed_time+elapsed_time_qoi

                
        return QoI


    
     
     # This function implements the richardson extrapolation scheme

    def richardson( self,y,Nsteps_r, n): 

        #n: number of levels of extrapolation
        #Nsteps: initial stepsize

        d = np.array( [[0] * (n + 1)] * (n + 1), float )

        # for i in xrange( n + 1 ):
            
        #     y1=y[Nsteps-1]
        #     y_aux=y[0:Nsteps-1]
        #     d[i,0] = self.stock_price_trajectory_1D_BS(y1,y_aux,Nsteps) # approximation

        #     powerOf4 = 1  # values of 4^j
        #     for j in xrange( 1, i + 1 ):
        #         powerOf4 = 4 * powerOf4
        #         d[i,j] = d[i,j-1] + ( d[i,j-1] - d[i-1,j-1] ) / ( powerOf4 - 1 )

        #     Nsteps = 2 * Nsteps
        #     #self.N=Nsteps
        

        #level 1: richardson extrapol

        y1=y[Nsteps_r-1]
        y_aux=y[0:Nsteps_r-1]
        d[0,0] = self.stock_price_trajectory_1D_BS(y1,y_aux,Nsteps_r) # approximation
        powerOf4 = 1  # values of 4^j
        y1=y[2*Nsteps_r-1]
        y_aux=y[0:2*Nsteps_r-1]
        d[1,0] = self.stock_price_trajectory_1D_BS(y1,y_aux,2*Nsteps_r) # approximation

        powerOf4 = 4 * powerOf4
        d[1,1] = d[1,0] + ( d[1,0] - d[0,0] ) / ( powerOf4 - 1 )
        return d[n,n]

# This function implements the brownian bridge construction in 1 D, the argument y here represent independent  multivariate gaussian  rdv  given by
    #mean = np.zeros(self.N)
    #covariance= np.identity(self.N)
    #y = np.random.multivariate_normal(mean, covariance)
    #y1: is the first direction given by W(T)/sqrt(T)/   #This function gives  the brownian motion increments built from Brownian bridge construction: # the composition of  BB and brownian_increments give us
    # the function \phi in our notes(discussion)

    def brownian_increments(self,Nsteps):
        t_c=np.linspace(0, self.T, Nsteps+1)     
        t_f=np.linspace(0, self.T, 2*Nsteps+1)  
        h_c=Nsteps
        h_f=2*Nsteps
        j_max_c=1
        j_max_f=1

        mean = np.zeros(2*Nsteps)
        covariance= np.identity(2*Nsteps)
        yf = np.random.multivariate_normal(mean, covariance)
        yc=[sum(yf[current: current+2]) for current in xrange(0, len(yf), 2)]


        y1_f=yf[h_f-1]
        
        y1_c=yc[h_c-1]

        bb_f= np.zeros(2*Nsteps+1)
        bb_c= np.zeros(Nsteps+1)

        bb_f[h_f]=np.sqrt(self.T)*y1_f
        bb_c[h_c]=np.sqrt(self.T)*y1_c
      
       
        #ds=int(np.log2(Nsteps)) #power 2 number steps
        for k_c in range(1,self.d_c+1):
            i_min_c=h_c//2
            i_c=i_min_c
            l_c=0
            r_c=h_c
            for j_c in range(1,j_max_c+1):
                a_c=((t_c[r_c]-t_c[i_c])* bb_c[l_c]+(t_c[i_c]-t_c[l_c])*bb_c[r_c])/float(t_c[r_c]-t_c[l_c])
                b_c=np.sqrt((t_c[i_c]-t_c[l_c])*(t_c[r_c]-t_c[i_c])/float(t_c[r_c]-t_c[l_c]))
                bb_c[i_c]=a_c+b_c*yc[i_c-1]
                i_c=i_c+h_c
                l_c=l_c+h_c
                r_c=r_c+h_c
            j_max_c=2*j_max_c
            h_c=i_min_c 

        for k_f in range(1,self.d_f+1):
            i_min_f=h_f//2
            i_f=i_min_f
            l_f=0
            r_f=h_f
            for j_f in range(1,j_max_f+1):
                a_f=((t_f[r_f]-t_f[i_f])* bb_f[l_f]+(t_f[i_f]-t_f[l_f])*bb_f[r_f])/float(t_f[r_f]-t_f[l_f])
                b_f=np.sqrt((t_f[i_f]-t_f[l_f])*(t_f[r_f]-t_f[i_f])/float(t_f[r_f]-t_f[l_f]))
                bb_f[i_f]=a_f+b_f*yf[i_f-1]
                i_f=i_f+h_f
                l_f=l_f+h_f
                r_f=r_f+h_f
            j_max_f=2*j_max_f
            h_f=i_min_f     


        return bb_c,bb_f



    # This function simulates a 1D BS trajectory for stock price, it plays the role of f_1 in our notes
    def stock_price_trajectory_1D_BS(self,Nsteps):
        bb_c,bb_f=self.brownian_increments(Nsteps)
        dW_c= [bb_c[i+1]-bb_c[i] for i in range(0,len(bb_c)-1)] 
        dW_f= [bb_f[i+1]-bb_f[i] for i in range(0,len(bb_f)-1)] 
       
      
        X_c=np.zeros(Nsteps+1) #here will store the BS trajectory
        X_f=np.zeros(2*Nsteps+1) #here will store the BS trajectory

        X_c[0]=self.S0
        X_f[0]=self.S0

        for nc in range(1,Nsteps+1):
            X_c[nc]=X_c[nc-1]*(1+self.sigma*dW_c[nc-1])
        for nf in range(1,2*Nsteps+1):
            X_f[nf]=X_f[nf-1]*(1+self.sigma*dW_f[nf-1])    
        
        return X_c[-1],X_f[-1]
           
        
    
    # this function defines the payoff function used here
    def payoff(self,x): 
       print(x)
       g=(x-self.K)
       g[g < 0] = 0
       return g  

 
    # this function defines the smooth approximation of the  original payoff function  using BS formual
    def payoff_smooth1(self,x): 
       #Black and Scholes   
       if x>0:
            
            d1=(np.log(x/self.K) + (self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T))
            d2=d1-self.sigma*np.sqrt(self.T)
            #d2=(np.log(x/ self.K) + (- self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
            g=  x * ss.norm.cdf(d1) - self.K * ss.norm.cdf(d2)
       else:
            g=0
        
       return g  
   
  

      # this function defines the smooth approximation of the  original payoff function  (second approximation)
    def payoff_smooth2(self,x): 
       #Black and Scholes        
        g=0.5*(x-self.K+np.sqrt(((x-self.K)**2)+(100000)))
        return g  

    @staticmethod
    def Init():
        import sys #This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter
        count = len(sys.argv)  #sys.argv is a list in Python, which contains the command-line arguments passed to the script. With the len(sys.argv) function you can count the number of arguments. 
        #arr = (ct.c_char_p * len(sys.argv))()
        arr = sys.argv








def weak_convergence_rate_plotting():    
    
    exact= 160
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # # feed parameters to the problem
    Nsteps_arr=np.array([2,4,8,16,32,64])
    dt_arr=1.0/(Nsteps_arr+1)
    error=np.zeros(6)
    var=np.zeros(6)
    for i in range(0,6):
    	values=np.zeros(100000)
        for j in prange(100000):
        	print(j)
	        prb = Problem_richardson_extrapolation(1,Nsteps_arr[i]) 
	        values[j]=prb.objfun(Nsteps_arr[i]) 
    	error[i]=np.mean(values) 
    	var[i]=np.var(values)   

 
    print(error)   
    print(var)
    weak_error=np.abs(np.diff(error))
    print(weak_error)




    plt.plot(dt_arr[:-1], weak_error,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\Delta t$',fontsize=14)
    #ax.axis([0.1, 0.6, 0.0001, 0.1])
    plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
    #plt.subplot_tool()
    plt.legend(loc='upper left')
    plt.savefig('./results/weak_convergence_order_1D_BS_smooth_richardson.eps', format='eps', dpi=1000)  


weak_convergence_rate_plotting()
   



