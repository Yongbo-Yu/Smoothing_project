import numpy as np
import time
import scipy.stats as ss

class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    T=1.0                      # maturity
    sigma=None    # volatility
    N=7   # discretization resolution
    d=None
    dt=None


#methods
    # this method initializes 
    def __init__(self,params,coeff,nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random
        #self.S0=np.random.uniform(8,20,1) # initial stock price
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
  
        #self.sigma=np.random.uniform(0.3,0.4,1)  #volatility
        self.sigma=0.4
        self.dt=self.T/float(self.N+1) # time steps length
        self.d=int(np.log2(self.N+1)) #power 2 number steps
        

  

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

        y=y[0:self.N] 
        bar_z,dP=self.newtons_method(0,y)

        from scipy.stats import norm
        QoI= norm.sf(bar_z)   
        #QoI= np.sqrt(2* np.pi)*   norm.sf(bar_z)   
       # * (1/(100.0*dP))
       
       
        elapsed_time_qoi=time.time()-start_time;
        self.elapsed_time=self.elapsed_time+elapsed_time_qoi
                
        return QoI


    # This function implements the brownian bridge construction in 1 D, the argument y here represent independent  multivariate gaussian  rdv  given by
    #mean = np.zeros(self.N)
    #covariance= np.identity(self.N)
    #y = np.random.multivariate_normal(mean, covariance)
    #y1: is the first direction given by W(T)/sqrt(T)/   #This function gives  the brownian motion increments built from Brownian bridge construction: # the composition of  BB and brownian_increments give us
    # the function \phi in our notes(discussion)
        
    def brownian_increments(self,y1,y):
        t=np.linspace(0, self.T, self.N+2)     
        h=self.N+1
        j_max=1
        bb= np.zeros(self.N+2)
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
    def stock_price_trajectory_1D_BS(self,y1,y):
        bb=self.brownian_increments(y1,y)
        dW= [bb[i+1]-bb[i] for i in range(0,len(bb)-1)] 
        dbb=dW-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        
      
        X=np.zeros(self.N+2) #here will store the BS trajectory
        X[0]=self.S0
        for n in range(1,self.N+2):
            X[n]=X[n-1]*(1+self.sigma*dW[n-1])
        return X[-1],dbb
      
        
           
    # Root solving procedure
 
    #Now we set up the methods used for newton iteration
    def dx(self,x,y):
        P1,dP1=self.f(x,y)
        return abs(0-P1)




    def f(self,y1,y):# need to check this
        X,dbb=self.stock_price_trajectory_1D_BS(y1,y) # right version
        fi=1+(self.sigma/float(np.sqrt(self.T)))*y1*(self.dt)+self.sigma*dbb
        product=np.prod(fi)
        summation=np.sum(1/fi)
        Py=product-(self.K/float(self.S0))
        dPy=(self.sigma/float(np.sqrt(self.T)))*(self.dt)*product*summation
        return Py,dPy    
        

    def newtons_method(self,x0,y,eps=1e-10):
        delta = self.dx(x0,y)
        DP=1.0
        while delta > eps:
            #global DP
            #(self.f(x0,y))
            P_value,dP=self.f(x0,y)
            x0 = x0 - 0.1*P_value/dP
            DP=dP
            delta = self.dx(x0,y)
            
            
        return x0,DP




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
