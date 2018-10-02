import numpy as np
import os
#import sys
import time
import scipy.stats as ss


#sys.setrecursionlimit(10000)
class Problem_non_smooth_richardson_extrapolation_basket(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    N=4   # number of time steps which will be equal to the number of brownian bridge components (we set is a power of 2)
    d=None
    dt=None

    basket_d=2     # number of assets in the basket
    c= (1/float(basket_d))*np.ones(basket_d)     # weigths
    sigma=None    # vector of volatilities
    K=None         # Strike price
    T=1.0                      # maturity
    rho=None                  #correlation matrix
    

#methods
    # this method initializes 
    def __init__(self,params,nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random

        self.S0=100*np.ones(self.basket_d)
        self.sigma=0.2*np.ones(self.basket_d) #vector of volatilities
       
        self.K= 100                        # Strike price and coeff determine if we have in/at/out the money option
        from scipy.linalg import toeplitz 
        self.rho=toeplitz([1,0.3]) #correlation matrix

        self.dt=self.T/float(self.N) # time steps length
        self.d=int(np.log2(self.N)) #power 2 number steps
        
 

    def BeginRuns(self,ind, N):
        self.elapsed_time=0.0
        self.nelem = np.array(self.params.h0inv * self.params.beta**(np.array(ind)), dtype=np.uint32)
        if self.nested:
            self.nelem -= 1
        assert(len(self.nelem) == self.GetDim())
        return self.nelem


    def EndRuns(self):
        elapsed_time=self.elapsed_time;
        self.elapsed_time=0.0;
        return elapsed_time;

    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y):
        Y = np.array(Y)
        goal=self.objfun(self.nelem,Y);
        return goal


     # objfun:  beta #number of points in the first direction
    def objfun(self,nelem,y):
        start_time=time.time()

        beta=16
        yknots_right=np.polynomial.laguerre.laggauss(beta)
        yknots_left=yknots_right


         #Richardson level 

        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )    
     

        #finer level
        # step 1
        y2_1f=y[2*self.N-1:2*self.basket_d*(self.N-1)]
        #y21f=y[2*self.N]
   
        
        y1_1f=y[0:2*self.N-1]
        #y1f=y[0]


        #step2        #kink points by newton method

        bar_z_f=self.newtons_method(np.zeros(self.basket_d),y1_1f,y2_1f,2*self.N)



        mylist_left1_f=[]
        mylist_left1_f.append(yknots_left[0])
        mylist_left1_f[1:]=[np.array(y1_1f[i]) for i in range(0,len(y1_1f))]


        mylist_left2_f=[]
        mylist_left2_f.append(yknots_left[0])
        mylist_left2_f[1:]=[np.array(y2_1f[i]) for i in range(0,len(y2_1f))]

        mylist_right1_f=[]
        mylist_right1_f.append(yknots_right[0])
        mylist_right1_f[1:]=[np.array(y1_1f[i]) for i in range(0,len(y1_1f))]


        mylist_right2_f=[]
        mylist_right2_f.append(yknots_right[0])
        mylist_right2_f[1:]=[np.array(y2_1f[i]) for i in range(0,len(y2_1f))]

        # first quadrant (left,right)
        
        mylist_left_right_f=mylist_left1_f+mylist_right2_f
        weights_list_left_right_f=[yknots_left[1],yknots_right[1]]
     

        points_left_right_f=self.cartesian(mylist_left_right_f)
        weights_left_right_aux_f=self.cartesian(weights_list_left_right_f)
        weights_left_right_f=np.prod(weights_left_right_aux_f,1)


        x_lr_f=np.asarray([self.stock_price_trajectory_basket_BS(bar_z_f[0]-points_left_right_f[i,0],points_left_right_f[i,1:],points_left_right_f[i,2*self.N]\
            +bar_z_f[1],points_left_right_f[i,2*self.N+1:],2*self.N)  [0]  for i in range(0,len(yknots_left[0])*len(yknots_right[0]))])

        
        pay_lr_f=np.asarray( [self.payoff(x_lr_f[i,:]) for i in range(0,len(yknots_left[0])*len(yknots_right[0]))] ) 
        

        pay_lr_f=np.multiply(pay_lr_f,( ((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z_f[0]-points_left_right_f[:,0])**2)/2)\
            * np.exp(points_left_right_f[:,0]) * np.exp(-((points_left_right_f[:,2*self.N]+bar_z_f[1])**2)/2)* np.exp(points_left_right_f[:,2*self.N])))
      
        QoI_lr_f= weights_left_right_f.dot(pay_lr_f)
        




        # second quadrant (right,right)
        
        mylist_right_right_f=mylist_right1_f+mylist_right2_f
        weights_list_right_right_f=[yknots_right[1],yknots_right[1]]


        points_right_right_f=self.cartesian(mylist_right_right_f)
        weights_right_right_aux_f=self.cartesian(weights_list_right_right_f)
        weights_right_right_f=np.prod(weights_right_right_aux_f,1)


        x_rr_f=np.asarray([self.stock_price_trajectory_basket_BS(points_right_right_f[i,0]+bar_z_f[0],points_right_right_f[i,1:] ,points_right_right_f[i,2*self.N]\
            +bar_z_f[1],points_right_right_f[i,2*self.N+1:], 2*self.N)[0]  for i in range(0,len(yknots_right[0])*len(yknots_right[0]))])

        pay_rr_f=np.asarray( [self.payoff(x_rr_f[i,:])for i in range(0,len(yknots_right[0])*len(yknots_right[0]))])
        pay_rr_f=np.multiply(pay_rr_f,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_right_right_f[:,0]+bar_z_f[0])**2)/2)\
            * np.exp(points_right_right_f[:,0]) * np.exp(-((points_right_right_f[:,2*self.N]+bar_z_f[1])**2)/2)* np.exp(points_right_right_f[:,2*self.N]) ))

        
        QoI_rr_f= weights_right_right_f.dot(pay_rr_f)
       

        # third   quadrant (left,left)
        
        mylist_left_left_f=mylist_left1_f+mylist_left2_f
        weights_list_left_left_f=[yknots_left[1],yknots_left[1]]


        points_left_left_f=self.cartesian(mylist_left_left_f)
        weights_left_left_aux_f=self.cartesian(weights_list_left_left_f)
        weights_left_left_f=np.prod(weights_left_left_aux_f,1)


        x_ll_f=np.asarray([self.stock_price_trajectory_basket_BS(bar_z_f[0]-points_left_left_f[i,0],points_left_left_f[i,1:],bar_z_f[1]-points_left_left_f[i,2*self.N]\
            ,points_left_left_f[i,2*self.N+1:] , 2*self.N )[0]  for i in range(0,len(yknots_left[0])*len(yknots_left[0]))])
        
        pay_ll_f=np.asarray( [self.payoff(x_ll_f[i,:]) for i in range(0,len(yknots_left[0])*len(yknots_left[0]))])
        pay_ll_f=np.multiply(pay_ll_f,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z_f[0]-points_left_left_f[:,0])**2)/2)\
            * np.exp(points_left_left_f[:,0])* np.exp(-((bar_z_f[1]-points_left_left_f[:,2*self.N])**2)/2)* np.exp(points_left_left_f[:,2*self.N])   ))

        QoI_ll_f= weights_left_left_f.dot( pay_ll_f)



  

        # fourth quadrant (right,left)
        
        mylist_right_left_f=mylist_right1_f+mylist_left2_f
        weights_list_right_left_f=[yknots_right[1],yknots_left[1]]


        points_right_left_f=self.cartesian(mylist_right_left_f)
        weights_right_left_aux_f=self.cartesian(weights_list_right_left_f)
        weights_right_left_f=np.prod(weights_right_left_aux_f,1)

        x_rl_f=np.asarray([self.stock_price_trajectory_basket_BS(points_right_left_f[i,0]+bar_z_f[0],points_right_left_f[i,1:] ,bar_z_f[1]-points_right_left_f[i,2*self.N]\
            ,points_right_left_f[i,2*self.N+1:], 2*self.N )[0]  for i in range(0,len(yknots_right[0])*len(yknots_left[0]))])
        
        pay_rl_f=np.asarray( [self.payoff(x_rl_f[i,:])  for i in range(0,len(yknots_right[0])*len(yknots_left[0]))])
        pay_rl_f=np.multiply(pay_rl_f,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_right_left_f[:,0]+bar_z_f[0])**2)/2)\
            * np.exp(points_right_left_f[:,0]) * np.exp(-((bar_z_f[1]-points_right_left_f[:,2*self.N])**2)/2)* np.exp(points_right_left_f[:,2*self.N]) ))
          

        QoI_rl_f= weights_right_left_f.dot(pay_rl_f)
        

            
        d[1,0] =QoI_ll_f+QoI_rr_f+QoI_rl_f+QoI_lr_f
        
        #coarse level
        
         #coarser level points
        y_aux_c=y[0:self.basket_d*self.N]

         # step 1
        y2_1_c=y_aux_c[self.N-1:self.basket_d*(self.N-1)]
        #y21_c=y_aux_c[self.N]
   
        
        y1_1_c=y_aux_c[0:self.N-1]
        #y1_c=y_aux_c[0]
       
        
        #kink points by newton method

        bar_z_c=self.newtons_method(np.zeros(self.basket_d),y1_1_c,y2_1_c,self.N)


        mylist_left1_c=[]
        mylist_left1_c.append(yknots_left[0])
        mylist_left1_c[1:]=[np.array(y1_1_c[i]) for i in range(0,len(y1_1_c))]


        mylist_left2_c=[]
        mylist_left2_c.append(yknots_left[0])
        mylist_left2_c[1:]=[np.array(y2_1_c[i]) for i in range(0,len(y2_1_c))]

        mylist_right1_c=[]
        mylist_right1_c.append(yknots_right[0])
        mylist_right1_c[1:]=[np.array(y1_1_c[i]) for i in range(0,len(y1_1_c))]


        mylist_right2_c=[]
        mylist_right2_c.append(yknots_right[0])
        mylist_right2_c[1:]=[np.array(y2_1_c[i]) for i in range(0,len(y2_1_c))]




        # first quadrant (left,right)
        
        mylist_left_right_c=mylist_left1_c+mylist_right2_c
        weights_list_left_right_c=[yknots_left[1],yknots_right[1]]
     

        points_left_right_c=self.cartesian(mylist_left_right_c)
        weights_left_right_aux_c=self.cartesian(weights_list_left_right_c)
        weights_left_right_c=np.prod(weights_left_right_aux_c,1)


        x_lr_c=np.asarray([self.stock_price_trajectory_basket_BS(bar_z_c[0]-points_left_right_c[i,0],points_left_right_c[i,1:],points_left_right_c[i,self.N]\
            +bar_z_c[1],points_left_right_c[i,self.N+1:],self.N)  [0]  for i in range(0,len(yknots_left[0])*len(yknots_right[0]))])

        
        pay_lr_c=np.asarray( [self.payoff(x_lr_c[i,:]) for i in range(0,len(yknots_left[0])*len(yknots_right[0]))] ) 
        

        pay_lr_c=np.multiply(pay_lr_c,( ((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z_c[0]-points_left_right_c[:,0])**2)/2)\
            * np.exp(points_left_right_c[:,0]) * np.exp(-((points_left_right_c[:,self.N]+bar_z_c[1])**2)/2)* np.exp(points_left_right_c[:,self.N])))
      
        QoI_lr_c= weights_left_right_c.dot(pay_lr_c)
        


        # second quadrant (right,right)
        
        mylist_right_right_c=mylist_right1+mylist_right2_c
        weights_list_right_right_c=[yknots_right[1],yknots_right[1]]


        points_right_right_c=self.cartesian(mylist_right_right_c)
        weights_right_right_aux_c=self.cartesian(weights_list_right_right_c)
        weights_right_right_c=np.prod(weights_right_right_aux_c,1)


        x_rr_c=np.asarray([self.stock_price_trajectory_basket_BS(points_right_right_c[i,0]+bar_z_c[0],points_right_right_c[i,1:] ,points_right_right_c[i,self.N]\
            +bar_z_c[1],points_right_right_c[i,self.N+1:], self.N)[0]  for i in range(0,len(yknots_right[0])*len(yknots_right[0]))])

        pay_rr_c=np.asarray( [self.payoff(x_rr_c[i,:])for i in range(0,len(yknots_right[0])*len(yknots_right[0]))])
        pay_rr_c=np.multiply(pay_rr_c,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_right_right_c[:,0]+bar_z_c[0])**2)/2)\
            * np.exp(points_right_right_c[:,0]) * np.exp(-((points_right_right_c[:,self.N]+bar_z_c[1])**2)/2)* np.exp(points_right_right_c[:,self.N]) ))

        
        QoI_rr_c= weights_right_right_c.dot(pay_rr_c)


       

        # third   quadrant (left,left)
        
        mylist_left_left_c=mylist_left1_c+mylist_left2_c
        weights_list_left_left_c=[yknots_left[1],yknots_left[1]]


        points_left_left_c=self.cartesian(mylist_left_left_c)
        weights_left_left_aux_c=self.cartesian(weights_list_left_left_c)
        weights_left_left_c=np.prod(weights_left_left_aux_c,1)


        x_ll_c=np.asarray([self.stock_price_trajectory_basket_BS(bar_z_c[0]-points_left_left_c[i,0],points_left_left_c[i,1:],bar_z_c[1]-points_left_left_c[i,self.N]\
            ,points_left_left_c[i,self.N+1:] ,self.N )[0]  for i in range(0,len(yknots_left[0])*len(yknots_left[0]))])
        
        pay_ll_c=np.asarray( [self.payoff(x_ll_c[i,:]) for i in range(0,len(yknots_left[0])*len(yknots_left[0]))])
        pay_ll_c=np.multiply(pay_ll_c,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z_c[0]-points_left_left_c[:,0])**2)/2)\
            * np.exp(points_left_left_c[:,0])* np.exp(-((bar_z_c[1]-points_left_left_c[:,self.N])**2)/2)* np.exp(points_left_left_c[:,self.N])   ))

        QoI_ll_c= weights_left_left_c.dot( pay_ll_c)


        # fourth quadrant (right,left)
        
        mylist_right_left_c=mylist_right1_c+mylist_left2_c
        weights_list_right_left_c=[yknots_right[1],yknots_left[1]]


        points_right_left_c=self.cartesian(mylist_right_left_c)
        weights_right_left_aux_c=self.cartesian(weights_list_right_left_c)
        weights_right_left_c=np.prod(weights_right_left_aux_c,1)

        x_rl_c=np.asarray([self.stock_price_trajectory_basket_BS(points_right_left_c[i,0]+bar_z_c[0],points_right_left_c[i,1:] ,bar_z_c[1]-points_right_left_c[i,self.N]\
            ,points_right_left_c[i,self.N+1:],self.N )[0]  for i in range(0,len(yknots_right[0])*len(yknots_left[0]))])
        
        pay_rl_c=np.asarray( [self.payoff(x_rl_c[i,:])  for i in range(0,len(yknots_right[0])*len(yknots_left[0]))])
        pay_rl_c=np.multiply(pay_rl_c,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_right_left_c[:,0]+bar_z[0])**2)/2)\
            * np.exp(points_right_left_c[:,0]) * np.exp(-((bar_z_c[1]-points_right_left_c[:,self.N])**2)/2)* np.exp(points_right_left_c[:,self.N]) ))
          

        QoI_rl_c= weights_right_left_c.dot(pay_rl_c)
        



        d[0,0] =QoI_ll_c+QoI_rr_c+QoI_rl_c+QoI_lr_c

            
        d[1,1] = 2*d[1,0] - d[0,0]
        QoI=d[1,1] 

                    
        
    

        elapsed_time_qoi=time.time()-start_time;
        self.elapsed_time=self.elapsed_time+elapsed_time_qoi                
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


    # This function simulates a basket BS trajectory for stock price, it plays the role of f_1 in our notes
    def stock_price_trajectory_basket_BS(self,y1,yvec_1,y2,yvec_2,Nsteps):
        bb1=self.brownian_increments(y1,yvec_1,Nsteps)
        bb2=self.brownian_increments(y2,yvec_2,Nsteps)

    

        dW1= [bb1[0,i+1]-bb1[0,i]  for i in range(0,Nsteps)] 
        dW2= [bb2[0,i+1]-bb2[0,i] for i in range(0,Nsteps)] 

        dW=np.array([dW1 ,dW2])

        # construct the correlated  brownian bridge increments
        lower_triang_cholesky = np.linalg.cholesky(self.rho)
     
        dW=np.dot(lower_triang_cholesky,dW)  

        
        dW1=dW[0,:]
        dW2=dW[1,:]

        dbb1=dW1-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        dbb2=dW2-(self.dt/np.sqrt(self.T))*y2 # brownian bridge increments dbb_i (used later for the location of the kink point)
        
        
        X=np.zeros((self.basket_d,Nsteps+1)) #here will store the BS trajectory
      
        X[:,0]=self.S0
        for n in range(1,Nsteps+1):
            X[0,n]=X[0,n-1]*(1+self.sigma[0]*dW[0,n-1])
            X[1,n]=X[1,n-1]*(1+self.sigma[1]*dW[1,n-1])
      
        
        return X[:,-1],dbb1,dbb2

        
       # this function defines the payoff function used here
    def payoff(self,x): 
       #print(x)
       g=(self.c.dot(x)-self.K)
   
       if g>0:
          return g
       else:
          return 0
  

    # Root solving procedure
  
     
    #Now we set up the methods used for newton iteration
    def dx(self,x,yvec_1,yvec_2,Nsteps):
     
      
            P1,dP1=self.f(x[0],yvec_1,x[1],yvec_2,Nsteps)
            
       
            return np.abs(0-P1[0])  ,  np.abs(0-P1[1]) 
 
    def f(self,y1,yvec_1,y2,yvec_2,Nsteps):# need to check this
        X,dbb1,dbb2=self.stock_price_trajectory_basket_BS(y1,yvec_1,y2,yvec_2,Nsteps) 
        fi=np.zeros((self.basket_d,len(dbb1)))
        product=np.zeros(self.basket_d)
        summation=np.zeros(self.basket_d)
        Py=np.zeros(self.basket_d)
        dPy=np.zeros(self.basket_d)

        fi[0,:]=  1+(self.sigma[0]/float(np.sqrt(self.T)))*y1*(self.dt)+self.sigma[0]*dbb1
        fi[1,:]=  1+(self.sigma[1]/float(np.sqrt(self.T)))*y2*(self.dt)+self.sigma[1]*dbb2
        
        product[0]=np.prod(fi[0,:])
        product[1]=np.prod(fi[1,:])

        Py[0]=product[0]-(self.K/(float(self.S0[0]*self.c[0]*self.basket_d)))
        Py[1]=product[1]-(self.K/(float(self.S0[1]*self.c[1]*self.basket_d)))

        summation[0]=np.sum(1/fi[0,:])
        summation[1]=np.sum(1/fi[1,:])

        
        dPy[0]= (self.sigma[0]/float(np.sqrt(self.T)))*(self.dt)*product[0]*summation[0]
        dPy[1]=  (self.sigma[1]/float(np.sqrt(self.T)))*(self.dt)*product[1]*summation[1]
        return Py,dPy         
         
 
    def newtons_method(self,x0,yvec_1,yvec_2,Nsteps,eps=1e-10):
        
        delta1, delta2 = self.dx(x0,yvec_1,yvec_2,Nsteps)


      
        while (delta1 > eps) | (delta2 > eps):
        
            P_value,dP=self.f(x0[0],yvec_1,x0[1],yvec_2,Nsteps)
       
            x0[0] = x0[0] - 0.1*P_value[0]/dP[0]
            x0[1] = x0[1] - 0.1*P_value[1]/dP[1]

            delta1,delta2 = self.dx(x0,yvec_1,yvec_2,Nsteps)

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


    def Quit(self):
        pass

    def __exit__(self, type, value, traceback):
        pass

    def __enter__(self):
        return self

    @staticmethod
    def Init():
        import sys #This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter
        count = len(sys.argv)  #sys.argv is a list in Python, which contains the command-line arguments passed to the script. With the len(sys.argv) function you can count the number of arguments. 
        #arr = (ct.c_char_p * len(sys.argv))()
        arr = sys.argv

    @staticmethod
    def Final():
        pass

    def GetDim(self):
        return 0
