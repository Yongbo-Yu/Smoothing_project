import numpy as np
import time
 
import random
 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
 
import pathos.multiprocessing as mp
import pathos.pools as pp


 
class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    #T=10.0                      # maturity
    T=1.0                      
    sigma=None    # volatility
    d=None
    dt=None

    #exact=13.0847 #  S_0=K=100, T=10, r=0,rho=-0.9, v_0=0.04, theta=0.04, xi=1,\kapp=0.5 (does not satisfies Feller condition)
    exact=7.5789 #  S_0=K=100, T=1, r=0,rho=-0.9, v_0=0.04, theta=0.04, xi=0.5,\kapp=1 (satisfies Feller condition)
    yknots_right=[]
    yknots_left=[]


#methods
    # this method initializes 
    def __init__(self,coeff,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        
        self.rho=-0.9

        #self.kappa= 0.5
        self.kappa= 5.0
        self.theta=0.04

        #self.xi=1
        self.xi=0.5
        self.v0=0.04
        
       # self.K= coeff*self.S0   
        self.dt=self.T/float(Nsteps) # time steps length
        self.d=int(np.log2(Nsteps)) #power 2 number steps

        
 

     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):

        mean = np.zeros(2*Nsteps)
        covariance= np.identity(2*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)    

        # step 1 # get the two partitions of coordinates y_1 for the volatility path  and y_s for  the asset path  
        y1=y[0:Nsteps] # this points are related to the volatility path
        y2=y[Nsteps:]

       
    
        X=self.stock_price_trajectory_1D_heston(y1[0],y1[1:],y2[0],y2[1:],Nsteps)
           
            
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

    

    # This function simulates a 1D heston trajectory for stock price and volatility paths
    def stock_price_trajectory_1D_heston(self,y1,y,yv1,yv,Nsteps):
        bb=self.brownian_increments(y1,y,Nsteps)
      
        dW= [bb[0,i+1]-bb[0,i] for  i in range(0,Nsteps)] 
        
        #hierarhcical
       # bb_v=self.brownian_increments(yv1,yv,Nsteps)
       # dW_v= [bb_v[0,i+1]-bb_v[0,i] for i in range(0,Nsteps)] 
       
        # # non hierarhcical
        dW_v=[]
        dW_v.append(yv1)
        dW_v[1:]=[np.array(yv[i]) for i in range(0,len(yv))]
        dW_v=np.array(dW_v)

        dW_s= self.rho *np.array(dW_v)*np.sqrt(self.dt)  + np.sqrt(1-self.rho**2) * np.array(dW)
      
        X=np.zeros(Nsteps+1) #here will store the asset trajectory
        V=np.zeros(Nsteps+1) #here will store the  volatility trajectory

        X[0]=self.S0
        V[0]=self.v0
        
        for n in range(1,Nsteps+1):
            
            X[n]=X[n-1]*(1+np.sqrt(V[n-1])*dW_s[n-1])
         
            V[n]=V[n-1]- self.kappa *self.dt* max(V[n-1],0)+ self.xi *np.sqrt(max(V[n-1],0))*dW_v[n-1]*np.sqrt(self.dt) + self.kappa*self.theta*self.dt
            V[n]=max(V[n],0)
            
        return X[-1]
       
    # this function defines the payoff function used here
    def payoff(self,x): 

       g=(x-self.K)
       if g<0:
           g=0
       return g
    

         


def weak_convergence_differences():    
        start_time=time.time()
        exact=7.5789 #  S_0=K=100, T=1, r=0,rho=-0.9, v_0=0.04, theta=0.04, xi=0.5,\kapp=1 (satisfies Feller condition)
        marker=['>', 'v', '^', 'o', '*','+','-',':']
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([8])
        dt_arr=1.0/(Nsteps_arr)
    
        elapsed_time_qoi=np.zeros(1)
        error=np.zeros(1)
        stand=np.zeros(1)
        Ub=np.zeros(1)
        Lb=np.zeros(1)
   
        values=np.zeros((9*(10**3),1)) 
        for i in range(0,1):
            print i
            start_time=time.time()
            prb = Problem(1,Nsteps_arr[i]) 
            for j in range(9*(10**3)):
              
                values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)

            elapsed_time_qoi[i]=time.time()-start_time
            print  elapsed_time_qoi[i]    
            print np.mean(values[:,i]*float(exact))
        
             
        print elapsed_time_qoi

        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(9*(10**3)))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand
        print(error)   
        print(stand)
        print Lb
        print Ub
        


weak_convergence_differences()   

    



