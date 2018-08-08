import numpy as np
import time
import scipy.stats as ss

import fftw3
import RBergomi
from RBergomi import *




class Problem_richardson_extrapolation_level2(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    N= 4# Number of time steps N, discretization resolution
  
    # for the values of below paramters, we need to see the paper as well check with Christian 
    x=0.235**2;   # this will provide the set of xi parameter values 
    HIn=Vector(1)    # this will provide the set of H parameter values
    HIn[0]=0.07
    e=Vector(1)    # This will provide the set of eta paramter values
    e[0]=1.9
    r=Vector(1)   # this will provide the set of rho paramter values
    r[0]=-0.9
    T=Vector(1)     # this will provide the set of T(time to maturity) parameter value
    T[0]=1.0
    k=Vector(1)     # this will provide the set of K (strike ) paramter value
    k[0]=1.0
    
    MIn=1            # number of samples M (I think we do not need this paramter here by default in our case it should be =1)


#methods
    # this method initializes 
    def __init__(self,params,nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random
         #Here we need to use the C++ code to compute the payoff 
        self.z=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  self.N, self.MIn)
        self.zf=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  2*self.N, self.MIn)
        self.zff=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  4*self.N, self.MIn)

        
    
      
        

  

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

        yperp_1ff=y[4*self.N+1:8*self.N]
        yperp1ff=y[4*self.N]

        bbperp_ff=self.brownian_increments(yperp1ff,yperp_1ff,4*self.N)
        W1perp_ff= [(bbperp_ff[i+1]-bbperp_ff[i]) *np.sqrt(4*self.N) for i in range(0,len(bbperp_ff)-1)]
        

        yperp_1f=yperp_1ff[0:2*self.N-1]
        yperp1f=yperp1ff

        bbperp_f=self.brownian_increments(yperp1f,yperp_1f,2*self.N)
        W1perp_f= [(bbperp_f[i+1]-bbperp_f[i]) *np.sqrt(2*self.N) for i in range(0,len(bbperp_f)-1)]


        yperp_1c=yperp_1ff[0:self.N-1]
        yperp1c=yperp1ff
        bbperp_c=self.brownian_increments(yperp1c,yperp_1c,self.N)
        W1perp_c= [(bbperp_c[i+1]-bbperp_c[i]) *np.sqrt(self.N) for i in range(0,len(bbperp_c)-1)]
        
        #hierarchical way

        y_1ff=y[1:4*self.N]
        y1ff=y[0]

        bb_ff=self.brownian_increments(y1ff,y_1ff,4*self.N)
        W1_ff= [(bb_ff[i+1]-bb_ff[i]) *np.sqrt(4*self.N) for i in range(0,len(bb_ff)-1)]


        y_1f=y_1ff[0:2*self.N-1]
        y1f=y1ff
        bb_f=self.brownian_increments(y1f,y_1f,2*self.N)
        W1_f= [(bb_f[i+1]-bb_f[i]) *np.sqrt(2*self.N) for i in range(0,len(bb_f)-1)]

        y_1c=y_1ff[0:self.N-1]
        y1c=y1ff
        bb_c=self.brownian_increments(y1c,y_1c,self.N)
        W1_c= [(bb_c[i+1]-bb_c[i]) *np.sqrt(self.N) for i in range(0,len(bb_c)-1)]


         #level 2: richardson extrapol (new version)
    
        dc = self.z.ComputePayoffRT_single(W1_c,W1perp_c); 
        df=self.zf.ComputePayoffRT_single(W1_f,W1perp_f); 
        dff=self.zff.ComputePayoffRT_single(W1_ff,W1perp_ff); 

        d11=2*df-dc
        d21=2*dff-df
        d=((2**2) *d21-d11)/float(2**2-1)
        QoI=d
        return QoI

    def brownian_increments(self,y1,y,Nsteps):
        t=np.linspace(0, self.T, Nsteps+1)     
        h=Nsteps
        j_max=1
        bb= np.zeros(Nsteps+1)
        bb[h]=np.sqrt(self.T)*y1
       
        d=int(np.log2(Nsteps)) #power 2 number steps
        for k in range(1,d+1):
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
