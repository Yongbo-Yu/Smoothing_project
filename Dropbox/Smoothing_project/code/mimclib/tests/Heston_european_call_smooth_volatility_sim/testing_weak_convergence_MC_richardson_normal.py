import numpy as np
import time

import random

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

    exact=6.332542 #  S_0=K=100, T=1, r=0,rho=-0.9, v_0=0.04, theta=0.0025, xi=0.1,\kapp=1  
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
        self.kappa= 1.0
        self.xi=0.1
        self.v0=0.04
        self.theta=(self.xi**2)/(4*self.kappa)

        self.d=int(np.log2(Nsteps)) #power 2 number steps

    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y,Nsteps):
        Y = np.array(Y)
        goal=self.objfun(Y,Nsteps);
        return goal


     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):

        mean = np.zeros(4*Nsteps)
        covariance= np.identity(4*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)    


          #Richardson level
        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )   


        y1f=y[0:2*Nsteps] # this points are related to the volatility path

        y2f=[2*Nsteps]
        y2f[0]=y[2*Nsteps]
        y2f[1:]=y[2*Nsteps+1:]
        
        X_f=self.stock_price_trajectory_1D_heston(y2f[0],y2f[1:],y1f[0],y1f[1:2*Nsteps],2*Nsteps)        
        QoI_f= self.payoff(X_f)

        d[1,0] =QoI_f


        y1c=y1f[0:Nsteps] # this points are related to the volatility path

        y2c=[Nsteps]
        y2c[0]=y2f[0]
        y2c[1:]=y2f[1:Nsteps]
        
        X_c=self.stock_price_trajectory_1D_heston(y2c[0],y2c[1:],y1c[0],y1c[1:Nsteps],Nsteps)        
        QoI_c= self.payoff(X_c)

        d[0,0] =QoI_c

        d[1,1] = 2*d[1,0] - d[0,0]
        
        QoI=d[1,1]
        return QoI

    def brownian_increments(self,y1,y,Nsteps):
        t=np.linspace(0, self.T, Nsteps+1)     
        h=Nsteps
        j_max=1
        bb= np.zeros((1,Nsteps+1))
        bb[0,h]=np.sqrt(self.T)*y1
        ds=int(np.log2(Nsteps)) #power 2 number steps      
        for k in range(1,ds+1):
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

        dt=self.T/float(Nsteps)
         #  hierarhcical
        bb=self.brownian_increments(y1,y,Nsteps)
        dW= [bb[0,i+1]-bb[0,i] for i in range(0,Nsteps)] 

        # # # non hierarhcical
        # dW=[]
        # dW.append(y1)
        # dW[1:]=[np.array(y[i]) for i in range(0,len(y))]
        # dW=np.array(dW)
    
        #  hierarhcical
        bb_v=self.brownian_increments(yv1,yv,Nsteps)
        dW_v= [bb_v[0,i+1]-bb_v[0,i] for i in range(0,Nsteps)] 

        # # # non hierarhcical
        # dW_v=[]
        # dW_v.append(yv1)
        # dW_v[1:]=[np.array(yv[i]) for i in range(0,len(yv))]
        # dW_v=np.array(dW_v)
        

        
        dW_s= self.rho *np.array(dW_v) + np.sqrt(1-self.rho**2) * np.array(dW) 
        #dW_s= self.rho *np.array(dW_v)*np.sqrt(self.dt) + np.sqrt(1-self.rho**2) * np.array(dW)
       

       


        X=np.zeros(Nsteps+1) #here will store the asset trajectory
        V=np.zeros(Nsteps+1) #here will store the  volatility trajectory

        X[0]=self.S0
        V[0]=self.v0
   
        
        
        for n in range(1,Nsteps+1):
            X[n]=X[n-1]*(1+np.sqrt(V[n-1])*dW_s[n-1])
            V[n]=np.abs(V[n-1])- self.kappa *dt* max(V[n-1],0)+ self.xi *np.sqrt(max(V[n-1],0))*dW_v[n-1]*np.sqrt(dt)+ self.kappa*self.theta*dt
            V[n]=max(V[n],0)
            
        return X[-1]
        
        
        
     
    # this function defines the payoff function used here
    def payoff(self,x): 
       #print(x)
       
       g=(x-self.K)
       if g>0:
         return g
       else:
         return 0
 



  


def weak_convergence_differences():    
        start_time=time.time()
        exact=6.332542 #  S_0=K=100, T=1, r=0,rho=-0.9, v_0=0.04, theta=0.0025, xi=0.1,\kapp=1
        marker=['>', 'v', '^', 'o', '*','+','-',':']
        
        # # feed parameters to the problem
        Nsteps_arr=np.array([4])
        dt_arr=1.0/(Nsteps_arr)
    
        elapsed_time_qoi=np.zeros(1)
        error=np.zeros(1)
        stand=np.zeros(1)
        Ub=np.zeros(1)
        Lb=np.zeros(1)
        num_cores = mp.cpu_count()
   
        values=np.zeros((1*(10**8),1)) 
        for i in range(0,1):
            print i
            start_time=time.time()
            prb = Problem(1,Nsteps_arr[i]) 
            # for j in range(3*(10**5)):
              
            #     values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)


            

            def processInput(j):
                return prb.objfun(Nsteps_arr[i])/float(exact)
 
            
            p =  pp.ProcessPool(num_cores)  # Processing Pool with four processors
            
            values[:,i]= p.map(processInput, range(((1*(10**8))))  )    

            elapsed_time_qoi[i]=time.time()-start_time
          
            print np.mean(values[:,i]*float(exact))
        
             
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

    



