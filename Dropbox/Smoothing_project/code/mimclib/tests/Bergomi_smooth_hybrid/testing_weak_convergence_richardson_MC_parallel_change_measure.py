# In this file, we plot the weak convergence rate for MC with Richardson extrapolation (level1) after doing the partial change of measure


# used  modules
import numpy as np
import random

# used for plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

#used for using rBergomi code
import fftw3
import RBergomi
from RBergomi import *

#used for parallelisation
import pathos.multiprocessing as mp
import pathos.pools as pp

# used for getting the hessian
import numdifftools as nd

class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    

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
    def __init__(self,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random

        self.z=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  Nsteps, self.MIn)
        self.zf=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  2*Nsteps, self.MIn)
        #self.d=int(np.log2(Nsteps)) #power 2 number steps
        if Nsteps<=2:

            self.N=Nsteps
            self.z_bar_f=np.ones((2*Nsteps))
            self.z_bar_c=np.ones((Nsteps))
        
            from scipy.optimize import minimize    

            #mini = minimize(self.fun2, self.z_bar, method='CG')
            mini_f= minimize(self.fun2_f, self.z_bar_f, method='Nelder-Mead', options={'xatol': 1e-5, 'disp': False})
            self.z_bar_f_1=mini_f.x
            
            self.Hfun_f = nd.Hessian(self.fun_f) 
            Hfun_mode_f=np.linalg.inv(-self.Hfun_f(self.z_bar_f_1))
             # spectral
            e_vals_f, e_vecs_f = np.linalg.eig(Hfun_mode_f)
            self.Lf=e_vecs_f.dot(np.diag(np.sqrt(e_vals_f)))

            #Cholesky
            #self.Lf=np.linalg.cholesky(Hfun_mode_f)
            
            #print self.Lf


            mini_c= minimize(self.fun2_c, self.z_bar_c, method='Nelder-Mead', options={'xatol': 1e-5, 'disp': False})
            self.z_bar_c_1=mini_c.x
            
            
        
            self.Hfun_c = nd.Hessian(self.fun_c) 
            Hfun_mode_c=np.linalg.inv(-self.Hfun_c(self.z_bar_c_1))
            # spectral
            e_vals_c, e_vecs_c = np.linalg.eig(Hfun_mode_c)
            self.Lc=e_vecs_c.dot(np.diag(np.sqrt(e_vals_c)))

            #Cholesky
            #self.Lc=np.linalg.cholesky(Hfun_mode_c)
            #print self.Lc
           
        else:
            
            self.N=Nsteps
            self.z_bar_f=np.zeros((4))
            self.z_bar_c=np.zeros((4))
            
            from scipy.optimize import minimize    

            #mini = minimize(self.fun2, self.z_bar, method='CG')
            mini_f= minimize(self.fun2_f_4, self.z_bar_f, method='Nelder-Mead', options={'xatol': 1e-5, 'disp': False})
            self.z_bar_f_1=mini_f.x
            
            self.Hfun_f = nd.Hessian(self.fun_f_4) 
            Hfun_mode_f=np.linalg.inv(-self.Hfun_f(self.z_bar_f_1))

            #Cholesky
            #self.Lf=np.linalg.cholesky(Hfun_mode_f)
            #print self.Lf

            # spectral
            e_vals_f, e_vecs_f = np.linalg.eig(Hfun_mode_f)
            self.Lf=e_vecs_f.dot(np.diag(np.sqrt(e_vals_f)))


            mini_c= minimize(self.fun2_c_4, self.z_bar_c, method='Nelder-Mead', options={'xatol': 1e-5, 'disp': False})
            self.z_bar_c_1=mini_c.x
        
            self.Hfun_c = nd.Hessian(self.fun_c_4) 
            Hfun_mode_c=np.linalg.inv(-self.Hfun_c(self.z_bar_c_1))

            #Cholesky
            #self.Lc=np.linalg.cholesky(Hfun_mode_c)
            # spectral
            e_vals_c, e_vecs_c = np.linalg.eig(Hfun_mode_c)
            self.Lc=e_vecs_c.dot(np.diag(np.sqrt(e_vals_c)))    

        
    

    def fun_c(self,y11):       
        Nsteps=len(y11) 
        
        yperp=np.zeros((Nsteps))
        y=np.array([y11,yperp]).reshape(2*Nsteps)
        #hierarchical
        yperp_1=yperp[1:Nsteps]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y11[1:Nsteps]
        y1=y11[0]
        bb=self.brownian_increments(y1,y_1,Nsteps)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(Nsteps) for i in range(0,len(bb)-1)]
        
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-Nsteps))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=np.log((QoI))
     
        return QoI  
    



    def fun_f(self,y11):        
        Nsteps=len(y11) 
       
        yperp=np.zeros((Nsteps))
        y=np.array([y11,yperp]).reshape(2*Nsteps)
        #hierarchical
        yperp_1=yperp[1:Nsteps]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y11[1:Nsteps]
        y1=y11[0]
        bb=self.brownian_increments(y1,y_1,Nsteps)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(Nsteps) for i in range(0,len(bb)-1)]
        
        QoI=(self.zf.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-Nsteps))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=np.log((QoI))
 
        return QoI

    



    def fun2_c(self,y11):
        Nsteps=len(y11) 
        yperp=np.zeros((Nsteps))
        y=np.array([y11,yperp]).reshape(2*Nsteps)
        
        #hierarchical
        yperp_1=yperp[1:Nsteps]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y11[1:Nsteps]
        y1=y11[0]
        bb=self.brownian_increments(y1,y_1,Nsteps)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(Nsteps) for i in range(0,len(bb)-1)]
        

        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-Nsteps))*(np.exp(-0.5*y.dot(y)))
        QoI=-np.log(QoI)
        return QoI
       
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        #print y


    

        
    def fun2_f(self,y11):

        Nsteps=len(y11) 
        
        yperp=np.zeros((Nsteps))
       
        y=np.array([y11,yperp]).reshape(2*Nsteps)
        
        #hierarchical
        yperp_1=yperp[1:Nsteps]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y11[1:Nsteps]
        y1=y11[0]
        bb=self.brownian_increments(y1,y_1,Nsteps)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(Nsteps) for i in range(0,len(bb)-1)]
        

        QoI=(self.zf.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-Nsteps))*(np.exp(-0.5*y.dot(y)))
    
        QoI=-np.log(QoI)        
        return QoI  


    
    def fun_c_4(self,y11):        
        Nsteps=self.N
        yperp=np.zeros((Nsteps))
        if Nsteps==8:
            y12=np.zeros((4))
            y_bar_1=np.array([y11,y12]).reshape(Nsteps)
            y=np.array([y_bar_1,yperp]).reshape(2*Nsteps)
        else:
           
            y_bar_1=np.array(y11).reshape(Nsteps)

            y=np.array([y_bar_1,yperp]).reshape(2*Nsteps)    

        #hierarchical
        yperp_1=yperp[1:Nsteps]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y_bar_1[1:Nsteps]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1,Nsteps)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(Nsteps) for i in range(0,len(bb)-1)]
        
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-Nsteps))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=np.log((QoI))
     
        return QoI   

    def fun2_c_4(self,y11):  
        Nsteps=self.N      
        yperp=np.zeros((Nsteps))
        if Nsteps==8:
            y12=np.zeros((4))
            y_bar_1=np.array([y11,y12]).reshape(Nsteps)
            y=np.array([y_bar_1,yperp]).reshape(2*Nsteps)
        else:
           
            y_bar_1=np.array(y11).reshape(Nsteps)

            y=np.array([y_bar_1,yperp]).reshape(2*Nsteps)    

        #hierarchical
        yperp_1=yperp[1:Nsteps]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(Nsteps) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y_bar_1[1:Nsteps]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1,Nsteps)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(Nsteps) for i in range(0,len(bb)-1)]
        
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-Nsteps))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=-np.log((QoI))
     
        return QoI       
    


    def fun_f_4(self,y11):        
        Nsteps=self.N    
        yperp=np.zeros((2*Nsteps))

        if Nsteps==4:
            y12=np.zeros((4))
            y_bar_1=np.array([y11,y12]).reshape(2*Nsteps)
            y=np.array([y_bar_1,yperp]).reshape(4*Nsteps)
        else:
            y12=np.zeros((4))
            y13=np.zeros((4))
            y14=np.zeros((4))
           
            y_bar_1=np.array([y11,y12,y13,y14]).reshape(2*Nsteps)

            y=np.array([y_bar_1,yperp]).reshape(4*Nsteps)   


        
        #hierarchical
        yperp_1=yperp[1:2*Nsteps]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,2*Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y_bar_1[1:2*Nsteps]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1,2*Nsteps)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bb)-1)]
        
        QoI=(self.zf.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-2*Nsteps))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=np.log((QoI))
     
        return QoI   

    def fun2_f_4(self,y11):      
        Nsteps=self.N  
        yperp=np.zeros((2*Nsteps))

        if Nsteps==4:
            y12=np.zeros((4))
            y_bar_1=np.array([y11,y12]).reshape(2*Nsteps)
            y=np.array([y_bar_1,yperp]).reshape(4*Nsteps)
        else:
            y12=np.zeros((4))
            y13=np.zeros((4))
            y14=np.zeros((4))
           
            y_bar_1=np.array([y11,y12,y13,y14]).reshape(2*Nsteps)

            y=np.array([y_bar_1,yperp]).reshape(4*Nsteps)   


        
        #hierarchical
        yperp_1=yperp[1:2*Nsteps]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1,2*Nsteps)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y_bar_1[1:2*Nsteps]
        y1=y_bar_1[0]
        bb=self.brownian_increments(y1,y_1,2*Nsteps)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(2*Nsteps) for i in range(0,len(bb)-1)]
        
        QoI=(self.zf.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-2*Nsteps))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=-np.log((QoI))
     
        return QoI     
     # objfun: 
    def objfun(self,Nsteps):
        mean = np.zeros(4*Nsteps)
        covariance= np.identity(4*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)


        if Nsteps<=2:
            
            bar_y_f=np.sqrt(2)*np.dot(self.Lf,y[0:2*Nsteps])+self.z_bar_f_1
            bar_y_c=np.sqrt(2)*np.dot(self.Lc,y[0:Nsteps])+self.z_bar_c_1

            #bar_y_c=bar_y_f[0:Nsteps]
            
            y_bar_1_f=bar_y_f
            y_bar_1_c=bar_y_c
            #
            cst_f=np.linalg.det(self.Lf)*(2**(Nsteps))*(np.exp(0.5*y[0:2*Nsteps].dot(y[0:2*Nsteps])))
            cst_c=np.linalg.det(self.Lc)*(2**(Nsteps/2))*(np.exp(0.5*y[0:Nsteps].dot(y[0:Nsteps])))
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
                y_bar_1_f=np.array([bar_y_f,y12f]).reshape(2*Nsteps)
                #y_bar_1_c=np.array([bar_y_c,y12c]).reshape(self.N)
                y_bar_1_c=np.array([bar_y_c]).reshape(Nsteps)

                cst_f=np.linalg.det(self.Lf)*(2**(self.N/2))*(np.exp(0.5*y[0:4].dot(y[0:4])))
                cst_c=np.linalg.det(self.Lc)*(2**(self.N/2))*(np.exp(0.5*y[0:4].dot(y[0:4])))
                #cst_c=np.linalg.det(self.Lf)*(2**(self.N/2))*(np.exp(0.5*y[0:4].dot(y[0:4])))
                
            else:
                #hierarchical way
                y12_f=y[4:8]
                y13_f=y[8:12]
                y14_f=y[12:16]
                y12_c=y[4:8]
                
                y_bar_1_f=np.array([bar_y_f,y12_f,y13_f,y14_f]).reshape(2*Nsteps)
                y_bar_1_c=np.array([bar_y_c,y12_c]).reshape(Nsteps)
                
                cst_f=np.linalg.det(self.Lf)*(2**(Nsteps/4))*(np.exp(0.5*y[0:4].dot(y[0:4])))
                cst_c=np.linalg.det(self.Lc)*(2**(Nsteps/4))*(np.exp(0.5*y[0:4].dot(y[0:4])))



      

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
        d[0,0] = self.z.ComputePayoffRT_single(W1_c,W1perp_c)*(np.exp(-0.5*bar_y_c.dot(bar_y_c)))*cst_c; 
        d[1,0]=self.zf.ComputePayoffRT_single(W1_f,W1perp_f)*(np.exp(-0.5*bar_y_f.dot(bar_y_f)))*cst_f;  

        d[1,1] = 2*d[1,0] - d[0,0]
        QoI=d[1,1]
        
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


    

   



def weak_convergence_differences():    
        #exact= 0.0712073 #exact value of K=1, H=0.43 
        exact= 0.0792047  #exact value of K=1, H=0.07
        marker=['>', 'v', '^', 'o', '*','+','-',':']
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([1,2,4,8])
        dt_arr=1.0/(Nsteps_arr)
        error_diff=np.zeros(3)
        stand_diff=np.zeros(3)
        error=np.zeros(4)
        stand=np.zeros(4)
        Ub=np.zeros(4)
        Lb=np.zeros(4)
        Ub_diff=np.zeros(3)
        Lb_diff=np.zeros(3)
        values=np.zeros((10**5,4)) 
        num_cores = mp.cpu_count()
        for i in range(0,4):
            print i
            

            def processInput(j):
            #       #Here we need to use the C++ code to compute the payoff 
                    
                prb = Problem(Nsteps_arr[i])   
                
                return prb.objfun(Nsteps_arr[i])/float(exact)

                    # # results = Parallel(n_jobs=num_cores)(delayed(processInput)(j) for j in inputs)
                    #p =  pp.ProcessPool(num_cores)  # Processing Pool with four processors
                    
            p =  pp.ProcessPool(num_cores)  # Processing Pool with four processors
            
            values[:,i]= p.map(processInput, range((10**5)))      
            
               
            # prb = Problem(Nsteps_arr[i])   
                
            # prb.objfun(Nsteps_arr[i])/float(exact)
            # print prb.objfun(Nsteps_arr[i])/float(exact)

       

        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(10**5))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand
        print(error)   
        print(stand)
        print Lb
        print Ub
         
        differences= [values[:,i]-values[:,i+1] for i in range(0,3)]
        error_diff=np.abs(np.mean(differences,axis=1))
        print error_diff 
        stand_diff=np.std(differences, axis = 1)/ float(np.sqrt(10**5))
        print stand_diff
        Ub_diff=np.abs(np.mean(differences,axis=1))+1.96*stand_diff
        Lb_diff=np.abs(np.mean(differences,axis=1))-1.96*stand_diff
        print Ub_diff
        print Lb_diff

       
        z= np.polyfit(np.log(dt_arr), np.log(error), 1)
        fit=np.exp(z[0]*np.log(dt_arr))
        print z[0]

        z_diff= np.polyfit(np.log(dt_arr[0:3]), np.log(error_diff), 1)
        fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:3]))
        print z_diff[0]


        z3=np.zeros(4)
        z3[0]=2.0
        z3[1]=np.log(error[0])
        fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])


        z3diff=np.zeros(3)
        z3diff[0]=2.0
        z3diff[1]=np.log(error_diff[0])
        fit3diff=np.exp(z3diff[0]*np.log(dt_arr[0:3])+z3diff[1])
        
        fig = plt.figure()

        plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' , marker='>',hold=True) 
        plt.plot(dt_arr, Lb,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
        plt.plot(dt_arr, Ub,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr, fit*10,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--')
        plt.plot(dt_arr, fit3*10,linewidth=2.0,label=r'rate= %s' % format(z3[0]  , '.2f'), linestyle = '--')
        
        
        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_Bergomi_H_007_K_1_M_10_5_CI_relative_measure_change_level_1_spec.eps', format='eps', dpi=1000)  

        fig = plt.figure()
        plt.plot(dt_arr[0:3], error_diff,linewidth=2.0,label='weak_error' , marker='>', hold=True) 
        plt.plot(dt_arr[0:3], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
        plt.plot(dt_arr[0:3], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr[0:3], fit_diff*10,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--')
        plt.plot(dt_arr[0:3], fit3diff*10,linewidth=2.0,label=r'rate= %s' % format(z3diff[0]  , '.2f'), linestyle = '--')
        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_differences_Bergomi_H_007_K_1_M_10_5_CI_relative_measure_change_level_1_spec.eps', format='eps', dpi=1000)  

#weak_convergence_rate_plotting()
weak_convergence_differences()