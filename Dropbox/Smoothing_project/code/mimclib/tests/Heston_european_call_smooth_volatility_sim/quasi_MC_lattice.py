#!/usr/bin/env python

import numpy as np
import time
import scipy.stats as ss
from scipy.stats import norm

import random


import pathos.multiprocessing as mp
import pathos.pools as pp

from latticeseq_b2 import latticeseq_b2

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
    # exact=6.445535 #  S_0=K=100, T=1, r=0,rho=-0.9, v_0=0.04, theta=0.005, xi=0.1,\kapp=1  (n=2)
    # exact=10.86117 #  S_0=K=100, T=1, r=0,rho=-0.3, v_0=0.09, theta=0.09, xi=1,\kapp=2.7778; n=1 ;   set 3

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
        
        # paramters for the bessel process
        self.beta=self.xi/float(2)
        self.alpha=-self.kappa/float(2)

       # self.K= coeff*self.S0   
        self.dt=self.T/float(Nsteps) # time steps length
        self.d=int(np.log2(Nsteps)) #power 2 number steps

        # For less than 185 points
        beta=128
        self.yknots_right=np.polynomial.laguerre.laggauss(beta)
        self.yknots_left=self.yknots_right

        # self.dt=self.T[0]/float(Nsteps) # time steps length
        self.d=int(np.log2(Nsteps)) #power 2 number steps
        # random shifting
        self.M = 2**4
        random.seed(1) # Mersenne Twister
        self.shifts = np.random.rand(self.M, 2*Nsteps-1)



     # objfun: 
    def objfun(self,x,Nsteps):
        
        # now use quasi-random samplesY 
     
    	y=norm.ppf(x)
       
        # step 1 # get the two partitions of coordinates y_1 for the volatility path  and y_s for  the asset path  
        y1=y[0:Nsteps] # this points are related to the volatility path

        y2=[Nsteps]
        y2[0]=0.0
        y2[1:]=y[Nsteps:]

        y2s=y2[1:]
        
        # step 2: computing the location of the kink
        bar_z=self.newtons_method(y2[0],y2s,y1[0],y1[1:],Nsteps)
        
        # step 3: performing the pre-intgeration step wrt kink point
    
        mylist_left=[]
        mylist_left.append(self.yknots_left[0])
        mylist_left[1:]=[np.array(y2s[i]) for i in range(0,len(y2s))]
        points_left=self.cartesian(mylist_left)

        x_l=np.asarray([self.stock_price_trajectory_1D_heston(bar_z-points_left[i,0],points_left[i,1:],y1[0],y1[1:Nsteps],Nsteps)[0]  for i in range(0,len(self.yknots_left[0]))])
        QoI_left= self.yknots_left[1].dot(self.payoff(x_l)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z-points_left[:,0])**2)/2)* np.exp(points_left[:,0])))


        mylist_right=[]
        mylist_right.append(self.yknots_right[0])
        mylist_right[1:]=[np.array(y2s[i]) for i in range(0,len(y2s))]
        points_right=self.cartesian(mylist_right)
        x_r=np.asarray([self.stock_price_trajectory_1D_heston(points_right[i,0]+bar_z,points_right[i,1:],y1[0],y1[1:Nsteps],Nsteps)[0] for i in range(0,len(self.yknots_right[0]))])
        QoI_right= self.yknots_right[1].dot(self.payoff(x_r)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right[:,0]+bar_z)**2)/2)* np.exp(points_right[:,0]))

        QoI=QoI_left+QoI_right
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
        dW= [bb[0,i+1]-bb[0,i] for i in range(0,Nsteps)] 
    
        #  hierarhcical
       # bb_v=self.brownian_increments(yv1,yv)
        #dW_v= [bb_v[0,i+1]-bb_v[0,i] for i in range(0,self.N)] 

        # # non hierarhcical
        dW_v=[]
        dW_v.append(yv1)
        dW_v[1:]=[np.array(yv[i]) for i in range(0,len(yv))]
        dW_v=np.array(dW_v)
        

        
        dW_s= self.rho *np.array(dW_v)*np.sqrt(self.dt) + np.sqrt(1-self.rho**2) * np.array(dW)
        y1s= self.rho *yv1 + np.sqrt(1-self.rho**2) * y1


        #option1 
        # dbb1=dW-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        # dbbv=dW_v*np.sqrt(self.dt) -(self.dt/np.sqrt(self.T))*yv1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        # dbb_s= self.rho *np.array(dbbv) + np.sqrt(1-self.rho**2) * np.array(dbb1)
        # #option2
        dbb_s=dW_s-(self.dt/np.sqrt(self.T))*y1s



        X=np.zeros(Nsteps+1) #here will store the asset trajectory
        X_v=np.zeros(Nsteps+1) #here will store the  Bessel process trajectory
        V=np.zeros(Nsteps+1) #here will store the  volatility trajectory

        X[0]=self.S0
        V[0]=self.v0
        X_v[0]=np.sqrt(self.v0)
        
        
        for n in range(1,Nsteps+1):
            X[n]=X[n-1]*(1+np.sqrt(V[n-1])*dW_s[n-1])
            X_v[n]=X_v[n-1]*(1+self.alpha*self.dt)+self.beta*dW_v[n-1]*np.sqrt(self.dt)
            V[n]=X_v[n]**2
            
        return X[-1],dbb_s,V
       
    # this function defines the payoff function used here
    def payoff(self,x): 

       g=(x-self.K)
       g[g < 0] = 0
       return g  


    # Root solving procedure
 
    #Now we set up the methods used for newton iteration
    def dx(self,x,y,yv1,yv,Nsteps):
        P1,dP1=self.f(x,y,yv1,yv,Nsteps)
        return abs(0-P1)

  
    def f(self,y1,y,yv1,yv,Nsteps):
        X,dbb,V=self.stock_price_trajectory_1D_heston(y1,y,yv1,yv,Nsteps) # right version
        fi=np.zeros((1,len(dbb)))
        
        y1s= self.rho *yv1 + np.sqrt(1-self.rho**2) * y1
     
        fi=1+(np.sqrt(V[0:Nsteps])/float(np.sqrt(self.T)))*y1s*(self.dt)+(np.sqrt(V[0:Nsteps]))*dbb
        product=np.prod(fi)
        Py=product-(self.K/float(self.S0))
        
        summation=np.sum(np.sqrt(V[0:Nsteps])/fi)
        dPy=(1/float(np.sqrt(self.T)))*(self.dt)*product*summation
        return Py,dPy    
        

                    
  
    def newtons_method(self,x0,y,yv1,yv,Nsteps,eps=1e-10):
        delta = self.dx(x0,y,yv1,yv,Nsteps)
        while delta > eps:
    
            P_value,dP=self.f(x0,y,yv1,yv,Nsteps)
            x0 = x0 - 0.1*P_value/dP
            delta = self.dx(x0,y,yv1,yv,Nsteps) 
        return x0     




    def cartesian(self,arrays, out=None):
        """
        Generate a cartesian product of input arrays.

        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])

        """

        arrays = [np.asarray(x) for x in arrays]
        dtype = float

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = n / arrays[0].size
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out=out[0:m,1:])
            for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m,1:] = out[0:m,1:]
        return out               
 



def weak_convergence_differences():    
        
        exact=6.332542 #  S_0=K=100, T=1, r=0,rho=-0.9, v_0=0.04, theta=0.0025, xi=0.1,\kapp=1  

        # # feed parameters to the problem
        Nsteps_arr=np.array([4])
        dt_arr=1.0/(Nsteps_arr)
        
        error=np.zeros(1)
        stand=np.zeros(1)
        elapsed_time_qoi=np.zeros(1)
        Ub=np.zeros(1)
        Lb=np.zeros(1)

        nprev = 0
        m=11
        N = 2**m
        
    
        for i in range(0,1):
            print i
        
            start_time=time.time()

            prb = Problem(1,Nsteps_arr[i]) 
            latgen = latticeseq_b2('exod2_base2_m20_CKN.txt', s=2*Nsteps_arr[i]-1)
            n = N/prb.M
            print n
           

            acc = np.zeros((prb.M,)) # accumulator for each shift  
            for j in range(0,n):
                  #Here we need to use the C++ code to compute the payoff 
                 
                
               
                # QMC generator (truncated to correct number of dimensions)
                
                x = latgen.next(); # next point, evaluate in all shifts
                #print x
                acc += [ prb.objfun((x+shift) % 1, Nsteps_arr[0]) for shift in prb.shifts ]
            acc=acc/float(n)    
            #print acc

            elapsed_time_qoi[i]=time.time()-start_time
            
          
             
       
        start_time_2=time.time()
        error=np.abs(np.mean(acc)) 
        elapsed_time_qoi=time.time()-start_time_2+elapsed_time_qoi

        stand=np.std(acc)/ float(np.sqrt(prb.M))
        Ub=np.abs(np.mean(acc) )+1.96*stand
        Lb=np.abs(np.mean(acc) )-1.96*stand

        print (elapsed_time_qoi)
        print(error)   
        print(stand)
        print Lb
        print Ub
      
        
weak_convergence_differences()      

