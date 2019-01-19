import numpy as np
import os
import sys
import time



class Problem(object):
# attributes
    random_gen=None;
    elapsed_time=0.0;
    N=2
    S0=None     # vector of initial stock prices
    basket_d=2     # number of assets in the basket
    c= (1/float(basket_d))*np.ones(basket_d)     # weigths
    sigma=None    # vector of volatilities
    K=None         # Strike price
    T=1.0                      # maturity
    rho=None                  #correlation matrix
  
    nelem=None;               # discretization
    exact=12.900784  # 2-d, sigma=0.4, S_0=K=100, T=1, r=0,rho=0.3

#methods
    # this method initializes the class of basket 
    def __init__(self,params, nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random
        
        self.S0=100*np.ones(self.basket_d) 
       
        self.sigma=0.4*np.ones(self.basket_d) #vector of volatilities
     
        self.K= 100                        # Strike price and coeff determine if we have in/at/out the money option
        
        #self.rho=self.correlation()                                                  #correlation matrx
        #self.rho=np.array([[1 , 0.3,  0.3],[ 0.3,  1 , 0.3],[0.3 , 0.3 ,1] ])
        #from numpy import concatenate, zeros
        #from scipy.linalg import toeplitz
        #self.rho=toeplitz([1,0.8,0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.8, 0.8,0.8,0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.8])
        
        self.rho=toeplitz([1,0.3])
        # self.Sigma=np.zeros((self.basket_d,self.basket_d))
        # for i in range(0,self.basket_d):
        #         for j in range(i,self.basket_d):
        #               self.Sigma[i,j]=self.sigma[i]*self.sigma[j]*self.rho[i,j]*self.T
        # self.Sigma=self.Sigma+np.transpose(self.Sigma)-np.diag(np.diag(self.Sigma))
       

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

    # objfun
    def objfun(self,nelem,y):

        start_time=time.time();
         
        # step 1 # get the two partitions of coordinates \mathbf{Z}_1 and \mathbf{Z}_{-1} for y which is a vector of N \times basket_d
        z1=y[0:-1:basket_d:] # getting \mathbf{Z}_1 
        z__1=np.setdiff1d(y, z1) # getting \mathbf{Z}_{-1}

        # step 2: doing the rotation from  \mathbf{Z}_1  to \mathbf{Y}_1
        A=self.rotation_matrix()
        y1=np.dot(A,z1) # getting \mathbf{Y}_1 by rotation using matrix A (to be defined)
        A_inv=np.transpose(A) # since A is  a rotation matrix than A^{-1}=A^T
        y__1=y1[1:]        # getting \mathbf{Y}_{-1}



        # step 3: computing the location of the kink
        bar_y1=self.newtons_method(0,y__1,z__1) 


        # step 4: performing the pre-intgeration step wrt kink point
        yknots_right=np.polynomial.laguerre.laggauss(beta)
        yknots_left=yknots_right
        
        mylist_left_y=[]
        mylist_left_z=[]

        mylist_left_y.append(yknots_left_y[0])
        mylist_left_y[1:]=[np.array(y__1[i]) for i in range(0,len(y__1))]
        mylist_left_z=[np.array(z__1[i]) for i in range(0,len(z__1))]
        mylist_left=np.concatenate(mylist_left_y,mylist_left_z)

        points_left=self.cartesian(mylist_left)
            
        # to be updated    
        # x_l=np.asarray([self.stock_price_trajectory_1D_BS(bar_z-points_left[i,0],points_left[i,1:])[0]  for i in range(0,len(yknots_left[0]))])
        # QoI_left= yknots_left[1].dot(self.payoff(x_l)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z-points_left[:,0])**2)/2)* np.exp(points_left[:,0])))

        mylist_right_y=[]
        mylist_right_z=[]
        mylist_right_y.append(yknots_right[0])
        mylist_right_y[1:]=[np.array(y[i]) for i in range(0,len(y__1))]
        mylist_right_z=[np.array(z__1[i]) for i in range(0,len(z__1))]
        mylist_right=np.concatenate(mylist_right_y,mylist_right_z)
        points_right=self.cartesian(mylist_right)

# to be updated    
        # x_r=np.asarray([self.stock_price_trajectory_1D_BS(points_right[i,0]+bar_z,points_right[i,1:])[0] for i in range(0,len(yknots_right[0]))])
        # QoI_right= yknots_right[1].dot(self.payoff(x_r)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right[:,0]+bar_z)**2)/2)* np.exp(points_right[:,0]))

        
        QoI=QoI_left+QoI_right

        
        elapsed_time_qoi=time.time()-start_time;
        self.elapsed_time=self.elapsed_time+elapsed_time_qoi;
        return QoI


    
    
    def brownian_increments(self,y1,y):
        t=np.linspace(0, self.T, self.N+1)     
        h=self.N
        j_max=1
        bb= np.zeros((1,self.N+1))
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

    
    # This function creates the desired rotation matrix A
    def rotation_matrix(self)    

    return A


      # This function simulates a basket BS trajectory for stock price, it plays the role of f_1 in our notes
    def stock_price_trajectory_basket_BS(self,y1,yvec_1,y2,yvec_2):
        bb1=self.brownian_increments(y1,yvec_1)
        bb2=self.brownian_increments(y2,yvec_2)

    

        dW1= [bb1[0,i+1]-bb1[0,i]  for i in range(0,self.N)] 
        dW2= [bb2[0,i+1]-bb2[0,i] for i in range(0,self.N)] 

 

        dW=np.array([dW1 ,dW2])

        # construct the correlated  brownian bridge increments
        lower_triang_cholesky = np.linalg.cholesky(self.rho)
     
        dW=np.dot(lower_triang_cholesky,dW)  
          
    
       
        dW1=dW[0,:]
        dW2=dW[1,:]


 

        dbb1=dW1-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        dbb2=dW2-(self.dt/np.sqrt(self.T))*y2 # brownian bridge increments dbb_i (used later for the location of the kink point)
        
          
    


        X=np.zeros((self.basket_d,self.N+1)) #here will store the BS trajectory
      
        X[:,0]=self.S0
        for n in range(1,self.N+1):
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
     
   

   # this function generates the correlation matrix rho ( think it is Ok but as I increase d the elements in the diagonal are not exactly 1, I need to check that more)
    def correlation(self):
        x = np.random.uniform(0.8,1,self.basket_d-1)
        cp=[np.prod(x[:i]) for i in range(1,self.basket_d) ]
        diag_terms=np.zeros(self.basket_d)
        diag_terms[0]=1
        diag_terms[1:]=[np.sqrt(1-(x[i]**2)) for i in range(0,self.basket_d-1)]
        tau=np.eye(self.basket_d)
        for i in range(0,self.basket_d):
                tau[i+1:,i]=cp[:self.basket_d-1-i]
        tau=tau.dot(np.diag(diag_terms))
        rho=tau.dot(np.transpose(tau))
        return rho
       


    
    


       # Root solving procedure
 
    #Now we set up the methods used for newton iteration


       #Now we set up the methods used for newton iteration
    # def dx(self,x,yvec_1,yvec_2,j):
    #     #print x
    #     if j==1:
    #         P1,dP1=self.f(x[0],yvec_1,x[1],yvec_2,1)
    #         return np.abs(0-P1)
    #     else:
        
    #         P1,dP1=self.f(x[0],yvec_1,x[1],yvec_2,2)
    #         return np.abs(0-P1)    

    def dx(self,x,yvec_1,yvec_2):
        #print x
      
            P1,dP1=self.f(x[0],yvec_1,x[1],yvec_2)
            
       
            return np.abs(0-P1[0])  ,  np.abs(0-P1[1]) 


    # def f(self,y1,yvec_1,y2,yvec_2,j):# need to check this
    #     X,dbb1,dbb2=self.stock_price_trajectory_basket_BS(y1,yvec_1,y2,yvec_2) 
    #     fi=np.zeros((self.basket_d,len(dbb1)))
    #     # product=np.zeros(self.basket_d)
    #     # summation=np.zeros(self.basket_d)
    #     # Py=np.zeros(self.basket_d)
    #     # dPy=np.zeros(self.basket_d)
    #     if j==1:
    #         fi=  1+(self.sigma[0]/float(np.sqrt(self.T)))*y1*(self.dt)+self.sigma[0]*dbb1
    #         product=np.prod(fi)
            
    #         Py=product-(self.K/(float(self.S0[0]*self.c[0])))
    #         summation=np.sum(1/fi)
    #         dPy=  (self.sigma[0]/float(np.sqrt(self.T)))*(self.dt)*product*summation
    #     else:    
        
    #         fi=  1+(self.sigma[1]/float(np.sqrt(self.T)))*y2*(self.dt)+self.sigma[1]*dbb2
    #         product=np.prod(fi)
    #         Py=product-(self.K/(float(self.S0[1]*self.c[1])))
    #         summation=np.sum(1/fi)
    #         dPy=  (self.sigma[1]/float(np.sqrt(self.T)))*(self.dt)*product*summation
    #     return Py,dPy    
    


    def f(self,y1,yvec_1,y2,yvec_2):# need to check this
        X,dbb1,dbb2=self.stock_price_trajectory_basket_BS(y1,yvec_1,y2,yvec_2) 
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
    

        
    def newtons_method(self,x0,yvec_1,yvec_2,eps=1e-10):
        
        delta1, delta2 = self.dx(x0,yvec_1,yvec_2)


      
        while (delta1 > eps) | (delta2 > eps):
        
            P_value,dP=self.f(x0[0],yvec_1,x0[1],yvec_2)
       
            x0[0] = x0[0] - 0.1*P_value[0]/dP[0]
            x0[1] = x0[1] - 0.1*P_value[1]/dP[1]

            delta1,delta2 = self.dx(x0,yvec_1,yvec_2)

        return x0     

    # def newtons_method(self,x0,yvec_1,yvec_2,eps=1e-5):
        
    #     delta1= self.dx(x0,yvec_1,yvec_2,1)
    #     delta2= self.dx(x0,yvec_1,yvec_2,2)
        
      
    #     while (delta1 > eps): 
        
    #         P_value,dP=self.f(x0[0],yvec_1,x0[1],yvec_2,1)
       
    #         x0[0] = x0[0] - 0.1*P_value/dP
           
    #         delta1= self.dx(x0,yvec_1,yvec_2,1)

    #     while (delta2 > eps): 
        
    #         P_value,dP=self.f(x0[1],yvec_1,x0[1],yvec_2,2)
       
    #         x0[1] = x0[1] - 0.1*P_value/dP

    #         delta2 = self.dx(x0 ,yvec_1,yvec_2,2)    

    #     return x0   




    




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
