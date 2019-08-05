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
    N=2  # number of time steps which will be equal to the number of brownian bridge components (we set is a power of 2)
    d=None
    dt=None

    exact=6.332542 #  S_0=K=100, T=1, r=0,rho=-0.9, v_0=0.04, theta=0.0025, xi=0.1,\kapp=1 
    exact=10.86117 #  S_0=K=100, T=1, r=0,rho=-0.3, v_0=0.09, theta=0.09, xi=1,\kapp=2.7778; n=1 ;   set 3

    yknots_right=[]
    yknots_left=[]


#methods
    # this method initializes 
    def __init__(self,coeff,params,nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        
        
        self.rho=-0.3

        
        self.kappa= 2.7778
        

        self.xi=1.0
        #self.xi=0.5
        self.v0=0.09
        self.theta=0.09


        # self.theta=(self.xi**2)/(4*self.kappa)
        
        # paramters for the bessel process
        self.beta=self.xi/float(2)
        self.alpha=-self.kappa/float(2)

        
       # self.K= coeff*self.S0   
        self.dt=self.T/float(self.N) # time steps length
        self.d=int(np.log2(self.N)) #power 2 number steps

        # # # For less than 185 points
        # beta=128
        # self.yknots_right=np.polynomial.laguerre.laggauss(beta)
      
        # # For more than 185 points
        beta=512
        from Parser import Parser
        fx = open('lag_512_x.txt', 'r')
        Element_properties_x = Parser('./lag_512_x.txt')
        Element_properties_x.parse_file(fx.read(),'\n')
        x=np.array([float(i) for i in Element_properties_x.element_list])
       
        Element_properties_x.close_file()   
        fw = open('lag_512_w.txt', 'r')
        Element_properties_w = Parser('./lag_512_w.txt')
        Element_properties_w.parse_file(fw.read(),'\n')
        w=np.array([float(i) for i in Element_properties_w.element_list])
   
        Element_properties_x.close_file()   
        self.yknots_right.append(x[:360])
        self.yknots_right.append(w[:360])
     
       
        self.yknots_left=self.yknots_right
        

  

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

        

        # step 1 # get the two partitions of coordinates y_1 for the volatility path  and y_s for  the asset path  
        y1=y[0:self.N] # this points are related to the volatility path

        y2=[self.N]
        y2[0]=0.0
        y2[1:]=y[self.N:]

        #y_s= self.rho *np.array(y1)+ np.sqrt(1-self.rho**2) * np.array(y2) 
        #ys=y_s[1:]

        y2s=y2[1:]
        
        # step 2: computing the location of the kink
        #bar_z=self.newtons_method(y_s[0],ys,y1[0],y1[1:self.N])
        bar_z=self.newtons_method(y2[0],y2s,y1[0],y1[1:self.N])
        
        # step 3: performing the pre-intgeration step wrt kink point
    
        mylist_left=[]
        mylist_left.append(self.yknots_left[0])
        mylist_left[1:]=[np.array(y2s[i]) for i in range(0,len(y2s))]
        points_left=self.cartesian(mylist_left)

        x_l=np.asarray([self.stock_price_trajectory_1D_heston(bar_z-points_left[i,0],points_left[i,1:],y1[0],y1[1:self.N])[0]  for i in range(0,len(self.yknots_left[0]))])

       
        QoI_left= self.yknots_left[1].dot(self.payoff(x_l)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z-points_left[:,0])**2)/2)* np.exp(points_left[:,0])))


        mylist_right=[]
        mylist_right.append(self.yknots_right[0])
        mylist_right[1:]=[np.array(y2s[i]) for i in range(0,len(y2s))]
        points_right=self.cartesian(mylist_right)
        x_r=np.asarray([self.stock_price_trajectory_1D_heston(points_right[i,0]+bar_z,points_right[i,1:],y1[0],y1[1:self.N])[0] for i in range(0,len(self.yknots_right[0]))])
        QoI_right= self.yknots_right[1].dot(self.payoff(x_r)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right[:,0]+bar_z)**2)/2)* np.exp(points_right[:,0]))

        QoI=QoI_left+QoI_right

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

    

    # This function simulates a 1D heston trajectory for stock price and volatility paths
    def stock_price_trajectory_1D_heston(self,y1,y,yv1,yv):
        bb=self.brownian_increments(y1,y)
        dW= [bb[0,i+1]-bb[0,i] for i in range(0,self.N)] 
    
        #  hierarhcical
        bb_v=self.brownian_increments(yv1,yv)
        dW_v= [bb_v[0,i+1]-bb_v[0,i] for i in range(0,self.N)] 

        # # non hierarhcical
        # dW_v=[]
        # dW_v.append(yv1)
        # dW_v[1:]=[np.array(yv[i]) for i in range(0,len(yv))]
        # dW_v=np.array(dW_v)
        

        
        # dW_s= self.rho *np.array(dW_v)*np.sqrt(self.dt) + np.sqrt(1-self.rho**2) * np.array(dW)    #used for non hierarchical
        dW_s= self.rho *np.array(dW_v)+ np.sqrt(1-self.rho**2) * np.array(dW)
        y1s= self.rho *yv1 + np.sqrt(1-self.rho**2) * y1


        #option1 
        # dbb1=dW-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        # dbbv=dW_v*np.sqrt(self.dt) -(self.dt/np.sqrt(self.T))*yv1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        # dbb_s= self.rho *np.array(dbbv) + np.sqrt(1-self.rho**2) * np.array(dbb1)
        # #option2
        dbb_s=dW_s-(self.dt/np.sqrt(self.T))*y1s



        X=np.zeros(self.N+1) #here will store the asset trajectory
        V=np.zeros(self.N+1) #here will store the  volatility trajectory
        X_v=np.zeros(self.N+1) #here will store the  Bessel process trajectory
     

        X[0]=self.S0
        V[0]=self.v0
        X_v[0]=np.sqrt(self.v0)
        
        
        for n in range(1,self.N+1):
            X[n]=X[n-1]*(1+np.sqrt(V[n-1])*dW_s[n-1])
            X_v[n]=X_v[n-1]*(1+self.alpha*self.dt)+self.beta*dW_v[n-1] 
            #*np.sqrt(self.dt)  # used if we use non hierarchical version
            V[n]=X_v[n]**2
            
        return X[-1],dbb_s,V
       
    # this function defines the payoff function used here
    def payoff(self,x): 

       g=(x-self.K)
       g[g < 0] = 0
       return g  


    # Root solving procedure
 
    #Now we set up the methods used for newton iteration
    def dx(self,x,y,yv1,yv):
        P1,dP1=self.f(x,y,yv1,yv)
        return abs(0-P1)

  
    def f(self,y1,y,yv1,yv):
        X,dbb,V=self.stock_price_trajectory_1D_heston(y1,y,yv1,yv) # right version
        fi=np.zeros((1,len(dbb)))
        

        y1s= self.rho *yv1 + np.sqrt(1-self.rho**2) * y1
        
        fi=1+(np.sqrt(V[0:self.N])/float(np.sqrt(self.T)))*y1s*(self.dt) +(np.sqrt(V[0:self.N]))*dbb
        product=np.prod(fi)
        Py=product-(self.K/float(self.S0))


        summation=np.sum(np.sqrt(V[0:self.N])/fi)
        dPy=(1/float(np.sqrt(self.T)))*(self.dt)*product*summation
       # print dPy
        #dPy=1.0
        return Py,dPy    
        

                    
  
    def newtons_method(self,x0,y,yv1,yv,eps=1e-10):
        delta = self.dx(x0,y,yv1,yv)
        while delta > eps:
    
            P_value,dP=self.f(x0,y,yv1,yv)
            x0 = x0 - 0.1*P_value/dP
            delta = self.dx(x0,y,yv1,yv) 
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
