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
    basket_d=4   # number of assets in the basket
    c= (1/float(basket_d))*np.ones(basket_d)     # weigths
    
    sigma=None    # vector of volatilities
    K=None         # Strike price
    T=1.0                      # maturity
    rho=None                  #correlation matrix
    nelem=None;               # discretization
    exact=11.04 # 4-d, sigma=0.4, S_0=K=100, T=1, r=0,rho=0.3

#methods
    # this method initializes the class of basket 
    def __init__(self,Nsteps, nested=False):
        self.nested = nested

        self.random_gen = None or np.random
        
        self.S0=100*np.ones(self.basket_d) 
       
        self.sigma=0.4*np.ones(self.basket_d) #vector of volatilities
     
        self.K= 100                        # Strike price and coeff determine if we have in/at/out the money option
        
        # correltion matrix
        from scipy.linalg import toeplitz  
        self.rho=toeplitz([1,0.3,0.3,0.3])

       
        


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
        y_2=y[Nsteps+1:2*Nsteps]
        y3=y[2*Nsteps]
        y_3=y[2*Nsteps+1:3*Nsteps]
        y4=y[3*Nsteps]
        y_4=y[3*Nsteps+1:]
    
        X=self.stock_price_trajectory_basket_BS(y1,y_1,y2,y_2,y3,y_3,y4,y_4,Nsteps)
           
            
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
     
     
    def stock_price_trajectory_basket_BS(self,y1,yvec_1,y2,yvec_2,y3,yvec_3,y4,yvec_4,Nsteps):

    	y=np.array([y1,y2,y3,y4])
    	

        #building the brownian bridge increments
        bb1=self.brownian_increments(y[0],yvec_1,Nsteps)
        bb2=self.brownian_increments(y[1],yvec_2,Nsteps)
        bb3=self.brownian_increments(y[2],yvec_3,Nsteps)
        bb4=self.brownian_increments(y[3],yvec_4,Nsteps)

        dW1= [bb1[0,i+1]-bb1[0,i]  for i in range(0,Nsteps)] 
        dW2= [bb2[0,i+1]-bb2[0,i] for i in range(0,Nsteps)] 
        dW3= [bb3[0,i+1]-bb3[0,i]  for i in range(0,Nsteps)] 
        dW4= [bb4[0,i+1]-bb4[0,i] for i in range(0,Nsteps)] 

        dW=np.array([dW1 ,dW2,dW3,dW4])

        
          # construct the correlated  brownian bridge increments
        lower_triang_cholesky = np.linalg.cholesky(self.rho)
     
        dW=np.dot(lower_triang_cholesky,dW)  


    
        dW1=dW[0,:]
        dW2=dW[1,:]
        dW3=dW[2,:]
        dW4=dW[3,:]

        
        


        X=np.zeros((self.basket_d,Nsteps+1)) #here will store the BS trajectory
      
        X[:,0]=self.S0
        for n in range(1,Nsteps+1):
            #X[0,n]=X[0,n-1]*(1+self.sigma[0]*((self.dt/float(np.sqrt(self.T)))*(self.A_inv[0,0]*y1+ self.A_inv[0,1:].dot(y2)) +  dbb1[n-1] ))  
            #X[1,n]=X[1,n-1]*(1+self.sigma[1]*((self.dt/float(np.sqrt(self.T)))*(self.A_inv[1,0]*y1+ self.A_inv[1,1:].dot(y2)) +  dbb2[n-1] ) )  
            X[0,n]=X[0,n-1]*(1+self.sigma[0]*dW1[n-1])
            X[1,n]=X[1,n-1]*(1+self.sigma[1]*dW2[n-1])
            X[2,n]=X[2,n-1]*(1+self.sigma[2]*dW3[n-1])
            X[3,n]=X[3,n-1]*(1+self.sigma[3]*dW4[n-1])

      
        return X[:,-1]
         
         
      
    def payoff(self,x): 
       #print(x)
       g=(x.dot(self.c)-self.K)
       
       if g<0:
           g=0
       return g
         


 
 
def weak_convergence_differences():    
        start_time=time.time()
        exact=11.04 #S_0=K=100, sigma =0.4, corr=0.3, T=1
       
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
    
        values=np.zeros((1*(10**8),1)) 
         
      
        
 
        num_cores = mp.cpu_count()
   
        for i in range(0,1):
            print i
            start_time=time.time()

            prb = Problem(Nsteps_arr[i]) 

            #for j in range(1*(10**1)):
             #     #Here we need to use the C++ code to compute the payoff             
              #  values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)
             
            #prb = Problem(Nsteps_arr[i]) 
            def processInput(j):
                return prb.objfun(Nsteps_arr[i])/float(exact)
 
            
            p =  pp.ProcessPool(num_cores)  # Processing Pool with four processors
            
            values[:,i]= p.map(processInput, range(((1*(10**8))))  )

            elapsed_time_qoi[i]=time.time()-start_time
            print np.mean(values[:,i]*float(exact))
            print  elapsed_time_qoi[i]


 
 
        
        print elapsed_time_qoi
 
        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(1*(10**8)))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand
        print(error)  
        print(stand)
        print Lb
        print Ub
          
      
 
        
        
 
 
weak_convergence_differences()   