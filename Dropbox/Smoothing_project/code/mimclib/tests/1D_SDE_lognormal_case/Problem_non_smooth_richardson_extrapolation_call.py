import numpy as np
import os
#import sys
import time
import scipy.stats as ss


#sys.setrecursionlimit(10000)
class Problem_non_smooth_richardson_extrapolation_call(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    T=1.0                      # maturity
    sigma=None    # volatility
    #N=4    # number of time steps which will be equal to the number of brownian bridge components (we set is a power of 2)
    N=8   # discretization resolution
    d=None
    dt=None
    

#methods
    # this method initializes 
    def __init__(self,params,coeff,nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        self.sigma=0.4
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

        beta=10
        yknots_right=np.polynomial.laguerre.laggauss(beta)
        yknots_left=yknots_right


         #Richardson level 

        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )    
     

        #finer level QoI
        #finer level points 
        y_aux_f=y[0:2*self.N-1]
        #kink points by newton method
        bar_z_f=self.newtons_method(0,y_aux_f,2*self.N)

        mylist_left_f=[]
        mylist_left_f.append(yknots_left[0])
        mylist_left_f[1:]=[np.array(y_aux_f[i]) for i in range(0,len(y_aux_f))]
        points_left_f=self.cartesian(mylist_left_f)
        x_l_f=np.asarray([self.stock_price_trajectory_1D_BS(bar_z_f-points_left_f[i,0],points_left_f[i,1:],2*self.N)[0]  for i in range(0,len(yknots_left[0]))])
        QoI_left_f= yknots_left[1].dot(self.payoff(x_l_f)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z_f-points_left_f[:,0])**2)/2)* np.exp(points_left_f[:,0])))

        mylist_right_f=[]
        mylist_right_f.append(yknots_right[0])
        mylist_right_f[1:]=[np.array(y_aux_f[i]) for i in range(0,len(y_aux_f))]
        points_right_f=self.cartesian(mylist_right_f)
        x_r_f=np.asarray([self.stock_price_trajectory_1D_BS(points_right_f[i,0]+bar_z_f,points_right_f[i,1:],2*self.N)[0] for i in range(0,len(yknots_right[0]))])
        QoI_right_f= yknots_right[1].dot(self.payoff(x_r_f)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right_f[:,0]+bar_z_f)**2)/2)* np.exp(points_right_f[:,0]))

        d[1,0] =QoI_left_f+QoI_right_f



         #coarse level QoI
        #coarser level points
        y_aux_c=y_aux_f[0:self.N-1]
        #kink points by newton method
        bar_z_c=self.newtons_method(0,y_aux_c,self.N)

        mylist_left_c=[]
        mylist_left_c.append(yknots_left[0])
        mylist_left_c[1:]=[np.array(y_aux_c[i]) for i in range(0,len(y_aux_c))]
        points_left_c=self.cartesian(mylist_left_c)
        x_l_c=np.asarray([self.stock_price_trajectory_1D_BS(bar_z_c-points_left_c[i,0],points_left_c[i,1:],self.N)[0]  for i in range(0,len(yknots_left[0]))])
        QoI_left_c= yknots_left[1].dot(self.payoff(x_l_c)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z_c-points_left_c[:,0])**2)/2)* np.exp(points_left_c[:,0])))

        mylist_right_c=[]
        mylist_right_c.append(yknots_right[0])
        mylist_right_c[1:]=[np.array(y_aux_c[i]) for i in range(0,len(y_aux_c))]
        points_right_c=self.cartesian(mylist_right_c)
        x_r_c=np.asarray([self.stock_price_trajectory_1D_BS(points_right_c[i,0]+bar_z_c,points_right_c[i,1:],self.N)[0] for i in range(0,len(yknots_right[0]))])
        QoI_right_c= yknots_right[1].dot(self.payoff(x_r_c)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right_c[:,0]+bar_z_c)**2)/2)* np.exp(points_right_c[:,0]))

        d[0,0] =QoI_left_c+QoI_right_c

            
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


    # This function simulates a 1D BS trajectory for stock price, it plays the role of f_1 in our notes
    def stock_price_trajectory_1D_BS(self,y1,y,Nsteps):
        bb=self.brownian_increments(y1,y,Nsteps)
        dW= [bb[i+1]-bb[i] for i in range(0,len(bb)-1)] 
        dt_s=self.T/float(Nsteps)
    
        X=np.zeros(Nsteps+1) #here will store the BS trajectory
        X[0]=self.S0
        for n in range(1,Nsteps+1):
            X[n]=X[n-1]*(1+self.sigma*dW[n-1])
        dbb=dW-(dt_s/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point
        return X[-1],dbb

    # this function defines the payoff function used here
    def payoff(self,x): 
       #print(x)
       g=(x-self.K)
       g[g < 0] = 0
       return g  

    # Root solving procedure
 
    #Now we set up the methods used for newton iteration
    def dx(self,x,y,Nsteps):
        P1,dP1=self.f(x,y,Nsteps)
        return abs(0-P1)


    def f(self,y1,y,Nsteps):# need to check this
        X,dbb=self.stock_price_trajectory_1D_BS(y1,y,Nsteps) # right version
        dt_s=self.T/float(Nsteps)
        fi=1+(self.sigma/float(np.sqrt(self.T)))*y1*(dt_s)+self.sigma*dbb
        product=np.prod(fi)
        summation=np.sum(1/fi)
        Py=product-(self.K/float(self.S0))
        dPy=(self.sigma/float(np.sqrt(self.T)))*(dt_s)*product*summation
        return Py,dPy    



        
    def newtons_method(self,x0,y,Nsteps,eps=1e-10):
        delta = self.dx(x0,y,Nsteps)
        while delta > eps:
            #(self.f(x0,y))
            P_value,dP=self.f(x0,y,Nsteps)
            x0 = x0 - 0.1*P_value/dP
            delta = self.dx(x0,y,Nsteps)
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
