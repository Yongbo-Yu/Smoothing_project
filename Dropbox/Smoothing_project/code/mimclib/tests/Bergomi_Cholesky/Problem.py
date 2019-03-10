import numpy as np
import time


from RBergomi import *
from RfBm import *
from RNorm  import *



class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    N=2# Number of time steps N, discretization resolution
  
    # for the values of below paramters, we need to see the paper as well check with Christian 
    xi=0.235**2;  # this will provide the set of xi parameter values 
    #x=0.1;
        # this will provide the set of H parameter values
    H=0.07
    #HIn[0]=0.02
    #HIn[0]=0.43
        # This will provide the set of eta paramter values
    eta=1.9
    #e[0]=0.4
      # this will provide the set of rho paramter values
    rho=-0.9
    #r[0]=-0.7
      # this will provide the set of T(time to maturity) parameter value
    T=1.0
      # this will provide the set of K (strike ) paramter value
    K=1.0
    #k[0]=1.2
    

  
#methods
    # this method initializes 
    def __init__(self,params,nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random
         #Here we need to use the C++ code to compute the payoff 
        

        self.dt=self.T/float(self.N) # time steps length
        self.d=int(np.log2(self.N)) #power 2 number steps
        self.rnorm=RNorm()

        self.rfbm= RfBm(self.N,self.H,self.rnorm)
        self.L=np.array(self.rfbm.GetL())  #getting the L matrix
        L11=self.L[0:self.N]
        
        self.L1=L11[:,0:self.N]
        self.L1_inv=np.linalg.inv(self.L1)

        self.W1 = Vector(self.N)
        self.Wtilde = Vector(self.N)  
        self.v=Vector(self.N)
        
   
        
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


     # objfun: 
    def objfun(self,nelem,y):
      
      

        # step 2: Construct the hierarchical transformation: y -> X(such that X remains Gaussian), construct L1^{-1}
        X = Vector(2*self.N)
        bb=self.brownian_increments(y[0],y[1:self.N])
        W= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        X[0:self.N]=(self.L1_inv).dot(W)   #constructing X[0:Nsteps-1]

        X[self.N:]=y[self.N:]   #constructing X[Nsteps:]
        #X[0:Nsteps]=y[0:Nsteps] #(Non  hierarchical)

        
        # Step 3:  given x compute W1 and Wtilde
        
        self.rfbm.generate(X,self.W1, self.Wtilde); # need to adjust rfbm code
        # Step 4:  compute price
        QoI=updatePayoff_cholesky(self.Wtilde,self.W1,self.v,self.eta,self.H,self.rho,self.xi,self.T,self.K,self.N) 
        return QoI


    

    def brownian_increments(self,y1,y):
        t=np.linspace(0, self.T, self.N+1)     
        h=self.N
        j_max=1
        bb= np.zeros(self.N+1)
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
