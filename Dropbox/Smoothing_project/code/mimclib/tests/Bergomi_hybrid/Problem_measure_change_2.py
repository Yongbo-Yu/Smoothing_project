import numpy as np
import time
import scipy.stats as ss

import fftw3
import RBergomi
from RBergomi import *
import mimclib.misc as misc
import numdifftools as nd

class Problem_measure_change_2(object):
# attributes
    random_gen=None;
    elapsed_time=0.0;
    N=1 # Number of time steps N, discretization resolution
    # for the values of below paramters, we need to see the paper as well check with Christian 
    x=0.235**2;   # this will provide the set of xi parameter values 
    #x=0.00001;
    HIn=Vector(1)    # this will provide the set of H parameter values
    HIn[0]=0.07
    e=Vector(1)    # This will provide the set of eta paramter values
    e[0]=1.9
    r=Vector(1)   # this will provide the set of rho paramter values
    r[0]=-0.9
    #r[0]=0
    T=Vector(1)     # this will provide the set of T(time to maturity) parameter value
    T[0]=1.0
    k=Vector(1)     # this will provide the set of K (strike ) paramter value
    k[0]=1
   # y1perp = Vector(N)
    MIn=1        # number of samples M (I think we do not need this paramter here by default in our case it should be =1)

#methods
    # this method initializes 
    def __init__(self,params,nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random
         #Here we need to use the C++ code to compute the payoff 
        self.z=RBergomi.RBergomiST(self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  self.N, self.MIn)
        self.dt=self.T[0]/float(self.N) # time steps length
        self.d=int(np.log2(self.N)) #power 2 number steps

        if self.N<=4:
            self.Hfun = nd.Hessian(self.fun) 
            self.z_bar=np.ones((self.N))
        

            from scipy.optimize import minimize    
            mini = minimize(self.fun2, self.z_bar, method='Nelder-Mead', options={'xatol': 1e-5, 'disp': False})
            self.z_bar1=mini.x

            

            Hfun_mode=np.linalg.inv(-self.Hfun(self.z_bar1))
            

            #  # Using Cholesky decomposition
            #self.L=np.linalg.cholesky(Hfun_mode)
         
            # #using Spectral decompositon
            e_vals, e_vecs = np.linalg.eig(Hfun_mode)
            self.L=e_vecs.dot(np.diag(np.sqrt(e_vals)))
        else:
            self.Hfun = nd.Hessian(self.fun_4) 
            self.z_bar=np.ones((4))
            from scipy.optimize import minimize    
            mini = minimize(self.fun2_4, self.z_bar, method='Nelder-Mead', options={'xatol': 1e-5, 'disp': False})
            self.z_bar1=mini.x
            Hfun_mode=np.linalg.inv(-self.Hfun(self.z_bar1))
            

            #  # Using Cholesky decomposition
            #self.L=np.linalg.cholesky(Hfun_mode)
         
            # #using Spectral decompositon
            e_vals, e_vecs = np.linalg.eig(Hfun_mode)
            self.L=e_vecs.dot(np.diag(np.sqrt(e_vals)))


  

       
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


    def fun(self,y11):        
        yperp=np.zeros((self.N))
    
        y=np.array([y11,yperp]).reshape(2*self.N)
        #hierarchical
        yperp_1=yperp[1:self.N]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y11[1:self.N]
        y1=y11[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=np.log((QoI))    
     
        return QoI    

    

    def fun_4(self,y11):        
        
        yperp=np.zeros((self.N))
        if self.N==8:
            y12=np.zeros((4))
            y_bar_1=np.array([y11,y12]).reshape(self.N)
            y=np.array([y_bar_1,yperp]).reshape(2*self.N)
        else:
            y12=np.zeros((4))
            y13=np.zeros((4))
            y14=np.zeros((4))
        
            y_bar_1=np.array([y11,y12,y13,y14]).reshape(self.N)

            y=np.array([y_bar_1,yperp]).reshape(2*self.N)    

        #hierarchical
        yperp_1=yperp[1:self.N]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y_bar_1[1:self.N]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=np.log((QoI))
     
        return QoI        


    
    def fun2(self,y11):
        yperp=np.zeros((self.N))
    
        y=np.array([y11,yperp]).reshape(2*self.N)
        #hierarchical
        yperp_1=yperp[1:self.N]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y11[1:self.N]
        y1=y11[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y)))
        

        

        #print QoI
        QoI=-np.log(QoI)
        
        return QoI  

    def fun2_4(self,y11):
        
        yperp=np.zeros((self.N))
        if self.N==8:
            y12=np.zeros((4))
            y_bar_1=np.array([y11,y12]).reshape(self.N)
            y=np.array([y_bar_1,yperp]).reshape(2*self.N)
        else:
            y12=np.zeros((4))
            y13=np.zeros((4))
            y14=np.zeros((4))
        
            y_bar_1=np.array([y11,y12,y13,y14]).reshape(self.N)

            y=np.array([y_bar_1,yperp]).reshape(2*self.N)    

        
        #hierarchical
        yperp_1=yperp[1:self.N]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y_bar_1[1:self.N]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        

        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        #print y
    
        #print QoI
        QoI=-np.log(QoI)
        
        return QoI  
     # objfun: 
    def objfun(self,nelem,y):

        start_time=time.time()
    

        if self.N<=4:
            y_bar_1=np.sqrt(2)*np.dot(self.L,y[0:self.N])+self.z_bar1
            bar_y=y_bar_1
            cst=np.linalg.det(self.L)*(2**(self.N/2))*(np.exp(0.5*y[0:self.N].dot(y[0:self.N])))
        else:
            bar_y=np.sqrt(2)*np.dot(self.L,y[0:4])+self.z_bar1
            if self.N==8:
                #hierarchical way
                y12=y[4:8]
                y_bar_1=np.array([bar_y,y12]).reshape(self.N)
                cst=np.linalg.det(self.L)*(2**(self.N/4))*(np.exp(0.5*y[0:4].dot(y[0:4])))
            else:
                #hierarchical way
                y12=y[4:8]
                y13=y[8:12]
                y14=y[12:16]
                y_bar_1=np.array([bar_y,y12,y13,y14]).reshape(self.N)
                cst=np.linalg.det(self.L)*(2**(self.N/8))*(np.exp(0.5*y[0:4].dot(y[0:4])))
        #hermite with importance sampling
       
       
        yperp=y[self.N:2*self.N]
        #hierarchical
        yperp_1=y[self.N+1:2*self.N]
        yperp1=y[self.N]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        y_1=y_bar_1[1:self.N]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        
      
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*(np.exp(-0.5*bar_y.dot(bar_y)))*cst
        #*(np.exp(-0.5*yperp.transpose().dot(yperp)))

        elapsed_time_qoi=time.time()-start_time;
        self.elapsed_time=self.elapsed_time+elapsed_time_qoi

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