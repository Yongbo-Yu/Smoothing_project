import numpy as np
import time
import scipy.stats as ss

import fftw3
import RBergomi
from RBergomi import *

import numdifftools as nd




class Problem_richardson_extrapolation_change_measure_corrected(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    N= 2# Number of time steps N, discretization resolution
  
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
        self.z=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  self.N, self.MIn)
        self.zf=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  2*self.N, self.MIn)

        

        if self.N<=2:

            Nsteps=self.N
            self.z_bar=np.zeros((2*Nsteps))
            
            from scipy.optimize import minimize    

            #mini = minimize(self.fun2, self.z_bar, method='CG')
            mini= minimize(self.fun2, self.z_bar, method='Nelder-Mead', options={'xatol': 1e-5, 'disp': False})
           
            self.z_bar=mini.x
            
            self.Hfun = nd.Hessian(self.fun) 
            Hfun_mode=np.linalg.inv(-self.Hfun(self.z_bar))
             # spectral
            #e_vals, e_vecs = np.linalg.eig(Hfun_mode)
            #self.L=e_vecs.dot(np.diag(np.sqrt(e_vals)))

            #Cholesky
            self.L=np.linalg.cholesky(Hfun_mode)

        else:
            
            Nsteps=self.N
            self.z_bar=np.zeros((4))
       
            
            from scipy.optimize import minimize    

            #mini = minimize(self.fun2, self.z_bar, method='CG')
            mini= minimize(self.fun2_4, self.z_bar, method='Nelder-Mead', options={'xatol': 1e-5, 'disp': False})

            self.z_bar=mini.x
            
            self.Hfun = nd.Hessian(self.fun_4) 
            Hfun_mode=np.linalg.inv(-self.Hfun(self.z_bar))

            #Cholesky
            #self.L=np.linalg.cholesky(Hfun_mode)
            
            # spectral
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
        Nsteps=self.N
        yperp=np.zeros((2*self.N))
        y=np.array([y11,yperp]).reshape(4*self.N)

        yperp_1f=y[2*Nsteps+1:4*Nsteps]
        yperp1f=y[2*Nsteps]

        bbperp_f=self.brownian_increments(yperp1f,yperp_1f,2*Nsteps)
        W1perp_f= [(bbperp_f[i+1]-bbperp_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bbperp_f)-1)]


        yperp_1c=yperp_1f[0:Nsteps-1]
        yperp1c=yperp1f
        bbperp_c=self.brownian_increments(yperp1c,yperp_1c,Nsteps)
        W1perp_c= [(bbperp_c[i+1]-bbperp_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp_c)-1)]
        

         #non hierarchical
        #W1=y[0:self.N]
        #hierarchical way
        y_1f=y[1:2*Nsteps]
        y1f=y[0]
        bb_f=self.brownian_increments(y1f,y_1f,2*Nsteps)
        W1_f= [(bb_f[i+1]-bb_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bb_f)-1)]

        y_1c=y_1f[0:Nsteps-1]
        y1c=y1f
        bb_c=self.brownian_increments(y1c,y_1c,Nsteps)
        W1_c= [(bb_c[i+1]-bb_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bb_c)-1)]

         #level 1: richardson extrapol (new version)
       
        dc = self.z.ComputePayoffRT_single(W1_c,W1perp_c); 
        df=self.zf.ComputePayoffRT_single(W1_f,W1perp_f); 

        QoI = (2*df- dc) *((2*np.pi)**(-2*Nsteps))*(np.exp(-0.5*y.dot(y)))


        
        QoI=np.log((np.abs(QoI)))
     
        return QoI   



     
    def fun2(self,y11):

        Nsteps=self.N

        yperp=np.zeros((2*self.N))
        y=np.array([y11,yperp]).reshape(4*self.N)
        
        #hierarchical
        yperp_1f=y[2*Nsteps+1:4*Nsteps]
        yperp1f=y[2*Nsteps]

        bbperp_f=self.brownian_increments(yperp1f,yperp_1f,2*Nsteps)
        W1perp_f= [(bbperp_f[i+1]-bbperp_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bbperp_f)-1)]


        yperp_1c=yperp_1f[0:Nsteps-1]
        yperp1c=yperp1f
        bbperp_c=self.brownian_increments(yperp1c,yperp_1c,Nsteps)
        W1perp_c= [(bbperp_c[i+1]-bbperp_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp_c)-1)]
        

         #non hierarchical
        #W1=y[0:self.N]
        #hierarchical way
        y_1f=y[1:2*Nsteps]
        y1f=y[0]
        bb_f=self.brownian_increments(y1f,y_1f,2*Nsteps)
        W1_f= [(bb_f[i+1]-bb_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bb_f)-1)]

        y_1c=y_1f[0:Nsteps-1]
        y1c=y1f
        bb_c=self.brownian_increments(y1c,y_1c,Nsteps)
        W1_c= [(bb_c[i+1]-bb_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bb_c)-1)]

         #level 1: richardson extrapol (new version)
       
        dc = self.z.ComputePayoffRT_single(W1_c,W1perp_c); 
        df=self.zf.ComputePayoffRT_single(W1_f,W1perp_f); 

        QoI = (2*df- dc) *((2*np.pi)**(-2*Nsteps))*(np.exp(-0.5*y.dot(y)))
    
        QoI=-np.log((np.abs(QoI)))      
        return QoI   
    


    def fun_c_4(self,y11):        
        yperp=np.zeros((self.N))
        if self.N==8:
            y12=np.zeros((4))
            y_bar_1=np.array([y11,y12]).reshape(self.N)
            y=np.array([y_bar_1,yperp]).reshape(2*self.N)
        else:
           
            y_bar_1=np.array(y11).reshape(self.N)

            y=np.array([y_bar_1,yperp]).reshape(2*self.N)    

        #hierarchical
        yperp_1=yperp[1:self.N]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,self.N)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y_bar_1[1:self.N]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1,self.N)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=np.log((QoI))
     
        return QoI   
    
    def fun_f_4(self,y11):        
        yperp=np.zeros((2*self.N))

        if self.N==4:
            y12=np.zeros((4))
            y_bar_1=np.array([y11,y12]).reshape(2*self.N)
            y=np.array([y_bar_1,yperp]).reshape(4*self.N)
        else:
            y12=np.zeros((4))
            y13=np.zeros((4))
            y14=np.zeros((4))
           
            y_bar_1=np.array([y11,y12,y13,y14]).reshape(2*self.N)

            y=np.array([y_bar_1,yperp]).reshape(4*self.N)   


        
        #hierarchical
        yperp_1=yperp[1:2*self.N]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,2*self.N)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(2*self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y_bar_1[1:2*self.N]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1,2*self.N)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(2*self.N) for i in range(0,len(bb)-1)]
        
        QoI=(self.zf.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-2*self.N))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=np.log((QoI))
     
        return QoI         
    
    


    def fun2_f_4(self,y11):        
        yperp=np.zeros((2*self.N))

        if self.N==4:
            y12=np.zeros((4))
            y_bar_1=np.array([y11,y12]).reshape(2*self.N)
            y=np.array([y_bar_1,yperp]).reshape(4*self.N)
        else:
            y12=np.zeros((4))
            y13=np.zeros((4))
            y14=np.zeros((4))
           
            y_bar_1=np.array([y11,y12,y13,y14]).reshape(2*self.N)

            y=np.array([y_bar_1,yperp]).reshape(4*self.N)   


        
        #hierarchical
        yperp_1=yperp[1:2*self.N]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,2*self.N)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(2*self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y_bar_1[1:2*self.N]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1,2*self.N)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(2*self.N) for i in range(0,len(bb)-1)]
        
        QoI=(self.zf.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-2*self.N))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=-np.log((QoI))
     
        return QoI     

     # objfun:  
    def objfun(self,nelem,y):

        Nsteps=self.N
        if self.N<=2:
            
            bar_y_f=np.sqrt(2)*np.dot(self.L,y[0:2*Nsteps])+self.z_bar
            bar_y_c=bar_y_f[0:Nsteps]

            #bar_y_c=np.sqrt(2)*np.dot(self.Lc,y[0:Nsteps])+self.z_bar_c_1

            
            
            y_bar_1_f=bar_y_f
            y_bar_1_c=bar_y_c
            #
            cst_f=np.linalg.det(self.L)*(2**(Nsteps))*(np.exp(0.5*y[0:2*Nsteps].dot(y[0:2*Nsteps])))
            #cst_c=np.linalg.det(self.Lc)*(2**(Nsteps/2))*(np.exp(0.5*y[0:Nsteps].dot(y[0:Nsteps])))
            #cst_c=np.linalg.det(self.Lf)*(2**(self.N))*(np.exp(0.5*y[0:Nsteps].dot(y[0:Nsteps])))





        else:
            
            bar_y_f=np.sqrt(2)*np.dot(self.Lf,y[0:4])+self.z_bar_f_1
            bar_y_c=np.sqrt(2)*np.dot(self.Lc,y[0:4])+self.z_bar_c_1
            #bar_y_c=bar_y_f
            #bar_y_c=bar_y_f[0:2]
            if self.N==4:
                #hierarchical way
                y12f=y[4:8]
                #y12c=y[2:4]
                y_bar_1_f=np.array([bar_y_f,y12f]).reshape(2*self.N)
                #y_bar_1_c=np.array([bar_y_c,y12c]).reshape(self.N)
                y_bar_1_c=np.array([bar_y_c]).reshape(self.N)

                cst_f=np.linalg.det(self.Lf)*(2**(self.N/2))*(np.exp(0.5*y[0:4].dot(y[0:4])))
                cst_c=np.linalg.det(self.Lc)*(2**(self.N/2))*(np.exp(0.5*y[0:4].dot(y[0:4])))
                #cst_c=np.linalg.det(self.Lf)*(2**(self.N/2))*(np.exp(0.5*y[0:4].dot(y[0:4])))
                
            else:
                #hierarchical way
                y12_f=y[4:8]
                y13_f=y[8:12]
                y14_f=y[12:16]
                y12_c=y[4:8]
                
                y_bar_1_f=np.array([bar_y_f,y12_f,y13_f,y14_f]).reshape(2*self.N)
                y_bar_1_c=np.array([bar_y_c,y12_c]).reshape(self.N)
                
                cst_f=np.linalg.det(self.Lf)*(2**(self.N/4))*(np.exp(0.5*y[0:4].dot(y[0:4])))
                cst_c=np.linalg.det(self.Lc)*(2**(self.N/4))*(np.exp(0.5*y[0:4].dot(y[0:4])))



      

        yperp_1f=y[2*Nsteps+1:4*Nsteps]
        yperp1f=y[2*Nsteps]
        bbperp_f=self.brownian_increments(yperp1f,yperp_1f,2*Nsteps)
        W1perp_f= [(bbperp_f[i+1]-bbperp_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bbperp_f)-1)]


        yperp_1c=yperp_1f[0:Nsteps-1]
        yperp1c=yperp1f
        bbperp_c=self.brownian_increments(yperp1c,yperp_1c,Nsteps)
        W1perp_c= [(bbperp_c[i+1]-bbperp_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp_c)-1)]
        

        #hierarchical way

        y_1f=y_bar_1_f[1:2*Nsteps]
        y1f=y_bar_1_f[0]
        bb_f=self.brownian_increments(y1f,y_1f,2*Nsteps)
        W1_f= [(bb_f[i+1]-bb_f[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bb_f)-1)]


        y_1c=y_bar_1_c[0:Nsteps-1]
        y1c=y_bar_1_c[0]
        bb_c=self.brownian_increments(y1c,y_1c,Nsteps)
        W1_c= [(bb_c[i+1]-bb_c[i]) *np.sqrt(Nsteps) for i in range(0,len(bb_c)-1)]

        


         #level 1: richardson extrapol (new version)
        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )
        

       
        #cst_c=np.linalg.det(self.Lc)*(2**(self.N/2))*(np.exp(0.5*y[0:Nsteps].dot(y[0:Nsteps])))

        #d[0,0] = self.z.ComputePayoffRT_single(W1_c,W1perp_c)*(np.exp(-0.5*bar_y_c.dot(bar_y_c)))*cst_c;
        # d[0,0] = self.z.ComputePayoffRT_single(W1_c,W1perp_c)*(np.exp(-0.5*bar_y_c.dot(bar_y_c)))*cst_c; 
        # d[1,0]=self.zf.ComputePayoffRT_single(W1_f,W1perp_f)*(np.exp(-0.5*bar_y_f.dot(bar_y_f)))*cst_f; 

        # d[1,1] = 2*d[1,0] - d[0,0]
        # QoI=d[1,1]


        d[0,0] = self.z.ComputePayoffRT_single(W1_c,W1perp_c); 
        d[1,0]=self.zf.ComputePayoffRT_single(W1_f,W1perp_f);

        d[1,1] = 2*d[1,0] - d[0,0]
        QoI=d[1,1] *(np.exp(-0.5*bar_y_f.dot(bar_y_f)))*cst_f
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
