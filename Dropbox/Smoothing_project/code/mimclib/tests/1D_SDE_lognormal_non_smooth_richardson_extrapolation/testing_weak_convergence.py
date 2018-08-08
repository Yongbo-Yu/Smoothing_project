import numpy as np
import time
import scipy.stats as ss

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


class Problem_non_smooth_richardson_extrapolation(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    T=1.0                      # maturity
    sigma=None    # volatility
    #N=4    # number of time steps which will be equal to the number of brownian bridge components (we set is a power of 2)
    #N=2         # discretization resolution
    d=None
    dt=None
    smooth=1


#methods
    # this method initializes 
    def __init__(self,coeff,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
        #self.S0=np.random.uniform(8,20,1) # initial stock price
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        #self.sigma=np.random.uniform(0.3,0.4,1)  #volatility
        self.sigma=0.4
        self.dt_c=self.T/float(Nsteps) # time steps length
        self.dt_f=self.T/float(2*Nsteps) # time steps length
        self.d_c=int(np.log2(Nsteps)) #power 2 number steps
        self.d_f=int(np.log2(2*Nsteps))
    

     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):
        start_time=time.time()
        #level 1: richardson extrapol
        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )
        beta=10
        deg_leg = 2**3

     
        #finer level points 
    
        #kink points by newton method
        bar_z_c,bar_z_f=self.newtons_method(0,Nsteps)
    
        z_min=np.minimum(bar_z_f,bar_z_c)
        z_max=np.maximum(bar_z_f,bar_z_c)
        
        # finer level QoI
        yknots_right_f=np.polynomial.laguerre.laggauss(beta)
        yknots_left_f=yknots_right_f
        #left_side finer
        
        points_left_f=yknots_left_f[0]
        x_l_f=np.asarray([self.stock_price_trajectory_1D_BS(z_min-points_left_f[i],Nsteps)[1]  for i in range(0,len(yknots_left_f[0]))])
        QoI_left_f= yknots_left_f[1].dot(self.payoff(x_l_f)*((1/np.sqrt(2 * np.pi)) * np.exp(-((z_min-points_left_f)**2)/2)* np.exp(points_left_f)))

        #right_side finer
    
        points_right_f=yknots_right_f[0]
        x_r_f=np.asarray([self.stock_price_trajectory_1D_BS(points_right_f[i]+z_max,Nsteps)[1] for i in range(0,len(yknots_right_f[0]))])
        QoI_right_f= yknots_right_f[1].dot(self.payoff(x_r_f)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right_f+z_max)**2)/2)* np.exp(points_right_f))

        
        # #middle_side finer
        # # Gauss-Legendre (default interval is [-1, 1])
        # yknots_middle = np.polynomial.legendre.leggauss(deg_leg)
        # # Translate x values from the interval [-1, 1] to [a, b]
        # yknots_middle_f= 0.5*(yknots_middle[0] + 1)*(z_max - z_min) + z_min

    
        # points_middle_f=yknots_middle_f
        # x_mid_f=np.asarray([self.stock_price_trajectory_1D_BS(points_middle_f[i],Nsteps)[1] for i in range(0,len(yknots_middle_f))])
        # QoI_middle_f= (yknots_middle[1].dot(self.payoff(x_mid_f))) * 0.5*(z_max - z_min)
        # # final finer QoI
        #d[1,0] =QoI_left_f+QoI_right_f+QoI_middle_f
        d[1,0] =QoI_left_f+QoI_right_f
        print(d[1,0])

       # coarser level
        yknots_right_c=yknots_right_f
        yknots_left_c=yknots_left_f
        
        #left_side coarser
       
        points_left_c=yknots_left_c[0]
        x_l_c=np.asarray([self.stock_price_trajectory_1D_BS(z_min-points_left_c[i],Nsteps)[0]  for i in range(0,len(yknots_left_c[0]))])
        QoI_left_c= yknots_left_c[1].dot(self.payoff(x_l_c)*((1/np.sqrt(2 * np.pi)) * np.exp(-((z_min-points_left_c)**2)/2)* np.exp(points_left_c)))

        #x_l_c=np.asarray([self.stock_price_trajectory_1D_BS(z_min-points_left_c[i,0],points_left_c[i,1:],self.N)[0]  for i in range(0,len(yknots_left_c[0]))])
        #QoI_left_c= yknots_left_c[1].dot(self.payoff(x_l_c)*((1/np.sqrt(2 * np.pi)) * np.exp(-((z_min-points_left_c[:,0])**2)/2)* np.exp(points_left_c[:,0])))
        #right_side coarser
  
        points_right_c=yknots_right_c[0]
        x_r_c=np.asarray([self.stock_price_trajectory_1D_BS(points_right_c[i]+z_max,Nsteps)[0] for i in range(0,len(yknots_right_c[0]))])
        QoI_right_c= yknots_right_c[1].dot(self.payoff(x_r_c)*(1/np.sqrt(2 * np.pi)) * np.exp(-(( points_right_c+z_max)**2)/2)* np.exp(points_right_c))
        #x_r_c=np.asarray([self.stock_price_trajectory_1D_BS(points_right_c[i,0]+z_max,points_right_c[i,1:],self.N)[0] for i in range(0,len(yknots_right_c[0]))])
        #QoI_right_c= yknots_right_c[1].dot(self.payoff(x_r_c)*(1/np.sqrt(2 * np.pi)) * np.exp(-(( points_right_c[:,0]+z_max)**2)/2)* np.exp(points_right_c[:,0]))
        
        # #middle_side coarser
        # # Gauss-Legendre (default interval is [-1, 1])
        # yknots_middle_c = yknots_middle_f
      
        # points_middle_c=yknots_middle_c
        # x_mid_c=np.asarray([self.stock_price_trajectory_1D_BS(points_middle_c[i],Nsteps)[0] for i in range(0,len(yknots_middle_c))])
        # #x_mid_c=np.asarray([self.stock_price_trajectory_1D_BS(points_middle_c[i,0] ,points_middle_c[i,1:],self.N)[0] for i in range(0,len(yknots_middle_c))])
        # QoI_middle_c= (yknots_middle[1].dot(self.payoff(x_mid_c))) * 0.5*(z_max - z_min)
        # final coarser QoI
        #d[0,0] =QoI_left_c+QoI_right_c+QoI_middle_c
        d[0,0] =QoI_left_c+QoI_right_c
        print(d[0,0])
        #Richardson
        d[1,1] = 2*d[1,0] - d[0,0]
        QoI=d[1,1] 


        elapsed_time_qoi=time.time()-start_time;

        self.elapsed_time=self.elapsed_time+elapsed_time_qoi

                
        return QoI




# This function implements the brownian bridge construction in 1 D, the argument y here represent independent  multivariate gaussian  rdv  given by
    #mean = np.zeros(self.N)
    #covariance= np.identity(self.N)
    #y = np.random.multivariate_normal(mean, covariance)
    #y1: is the first direction given by W(T)/sqrt(T)/   #This function gives  the brownian motion increments built from Brownian bridge construction: # the composition of  BB and brownian_increments give us
    # the function \phi in our notes(discussion)

    def brownian_increments(self, y1_f,Nsteps):
        t_c=np.linspace(0, self.T, Nsteps+1)     
        t_f=np.linspace(0, self.T, 2*Nsteps+1)  
        h_c=Nsteps
        h_f=2*Nsteps
        j_max_c=1
        j_max_f=1

        mean = np.zeros(2*Nsteps-1)
        covariance= np.identity(2*Nsteps-1)
        yf = np.random.multivariate_normal(mean, covariance)
   
        yc=[sum(yf[current: current+2])*(1/np.sqrt(2)) for current in xrange(0, len(yf)-1, 2)]
     

        
        
        y1_c=y1_f

        bb_f= np.zeros(2*Nsteps+1)
        bb_c= np.zeros(Nsteps+1)

        bb_f[h_f]=np.sqrt(self.T)*y1_f
        bb_c[h_c]=np.sqrt(self.T)*y1_c
      
       
        #ds=int(np.log2(Nsteps)) #power 2 number steps
        for k_c in range(1,self.d_c+1):
            i_min_c=h_c//2
            i_c=i_min_c
            l_c=0
            r_c=h_c
            for j_c in range(1,j_max_c+1):
                a_c=((t_c[r_c]-t_c[i_c])* bb_c[l_c]+(t_c[i_c]-t_c[l_c])*bb_c[r_c])/float(t_c[r_c]-t_c[l_c])
                b_c=np.sqrt((t_c[i_c]-t_c[l_c])*(t_c[r_c]-t_c[i_c])/float(t_c[r_c]-t_c[l_c]))
                bb_c[i_c]=a_c+b_c*yc[i_c-1]
                i_c=i_c+h_c
                l_c=l_c+h_c
                r_c=r_c+h_c
            j_max_c=2*j_max_c
            h_c=i_min_c 

        for k_f in range(1,self.d_f+1):
            i_min_f=h_f//2
            i_f=i_min_f
            l_f=0
            r_f=h_f
            for j_f in range(1,j_max_f+1):
                a_f=((t_f[r_f]-t_f[i_f])* bb_f[l_f]+(t_f[i_f]-t_f[l_f])*bb_f[r_f])/float(t_f[r_f]-t_f[l_f])
                b_f=np.sqrt((t_f[i_f]-t_f[l_f])*(t_f[r_f]-t_f[i_f])/float(t_f[r_f]-t_f[l_f]))
                bb_f[i_f]=a_f+b_f*yf[i_f-1]
                i_f=i_f+h_f
                l_f=l_f+h_f
                r_f=r_f+h_f
            j_max_f=2*j_max_f
            h_f=i_min_f     


        return bb_c,bb_f,y1_c

    


    # def brownian_increments(self, y1_f,Nsteps):
        
    #     mean = np.zeros(2*Nsteps-1)
    #     covariance= np.identity(2*Nsteps-1)
    #     yf = np.random.multivariate_normal(mean, covariance)
    #     yc=np.asarray([sum(yf[current: current+2]) for current in xrange(0, len(yf)-1, 2)])


    #     y1_c=(yf[-1]+y1_f)

    #     dW_f= np.zeros(2*Nsteps)
    #     dW_c= np.zeros(Nsteps)

    #     dW_f[0:2*Nsteps-1]=np.sqrt(self.dt_f)*yf
    #     dW_f[2*Nsteps-1]=np.sqrt(self.dt_f)*y1_f
       
    #     dW_c[0: Nsteps-1]=np.sqrt(self.dt_f)*yc
    #     dW_c[Nsteps-1]=np.sqrt(self.dt_f)*y1_c
        
    #     return dW_c,dW_f,y1_c




    
    # This function simulates a 1D BS trajectory for stock price, it plays the role of f_1 in our notes
    def stock_price_trajectory_1D_BS(self,y1_f,Nsteps):
    	bb_c,bb_f,y1_c=self.brownian_increments(y1_f,Nsteps)
        dW_f= [bb_f[i+1]-bb_f[i] for i in range(0,len(bb_f)-1)] 
        dW_c= [(bb_c[i+1]-bb_c[i]) for i in range(0,len(bb_c)-1)] 
       
        dbb_f=dW_f-(self.dt_f/np.sqrt(self.T))*y1_f
        dbb_c=dW_c-(self.dt_c/np.sqrt(self.T))*y1_c
      
        X_c=np.zeros(Nsteps+1) #here will store the BS trajectory
        X_f=np.zeros(2*Nsteps+1) #here will store the BS trajectory

        X_c[0]=self.S0
        X_f[0]=self.S0

        for nc in range(1,Nsteps+1):
            X_c[nc]=X_c[nc-1]*(1+self.sigma*dW_c[nc-1])
        for nf in range(1,2*Nsteps+1):
            X_f[nf]=X_f[nf-1]*(1+self.sigma*dW_f[nf-1])    
        
        return X_c[-1],X_f[-1],dbb_c,dbb_f,y1_c
           
    


    # Root solving procedure
 
    #Now we set up the methods used for newton iteration
    def dx(self,x,Nsteps):
        P1_c,dP1_c,P1_f,dP1_f=self.f(x,Nsteps)
        return abs(0-P1_c),abs(0-P1_f)

    def f(self,y1_f,Nsteps):# need to check this
        X_c,X_f,dbb_c,dbb_f,y1_c=self.stock_price_trajectory_1D_BS(y1_f,Nsteps) # right version
        
        fi_f=1+(self.sigma/float(np.sqrt(self.T)))*y1_f*(self.dt_f)+self.sigma*dbb_f
        fi_c=1+(self.sigma/float(np.sqrt(self.T)))*y1_c*(self.dt_c)+self.sigma*dbb_c

        product_f=np.prod(fi_f)
        product_c=np.prod(fi_c)

        summation_f=np.sum(1/fi_f)
        summation_c=np.sum(1/fi_c)

        Py_f=product_f-(self.K/float(self.S0))
        Py_c=product_c-(self.K/float(self.S0))


        dPy_f=(self.sigma/float(np.sqrt(self.T)))*(self.dt_f)*product_f*summation_f
        dPy_c=(self.sigma/float(np.sqrt(self.T)))*(self.dt_c)*product_c*summation_c


        return Py_c,dPy_c,Py_f,dPy_f    
        
        
    def newtons_method(self,x0,Nsteps,eps=1e-4):
        delta_c,delta_f = self.dx(x0,Nsteps)
        x0_c=x0
        x0_f=x0

        while delta_c > eps:
            P_value_c,dP_c=self.f(x0_c,Nsteps)[0:2]
            x0_c= x0_c - 0.1*P_value_c/dP_c
            delta_c = self.dx(x0_c,Nsteps)[0]
            #print(delta_c)
        
        while delta_f > eps:
        
            #(self.f(x0,y))
            P_value_f,dP_f=self.f(x0_f,Nsteps)[2:]
            x0_f = x0_f - 0.1*P_value_f/dP_f
            delta_f = self.dx(x0_f,Nsteps)[1]  
            
        return x0_c,x0_f    




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
    
    
    # this function defines the payoff function used here
    def payoff(self,x): 
       g=(x-self.K)
       g[g < 0] = 0
       return g  

    @staticmethod
    def Init():
        import sys #This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter
        count = len(sys.argv)  #sys.argv is a list in Python, which contains the command-line arguments passed to the script. With the len(sys.argv) function you can count the number of arguments. 
        #arr = (ct.c_char_p * len(sys.argv))()
        arr = sys.argv








def weak_convergence_rate_plotting():    
    
    exact= 15.85
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # # feed parameters to the problem
    Nsteps_arr=np.array([2,4,8,16,32])
    dt_arr=1.0/(Nsteps_arr+1)
    error=np.zeros(5)
    var=np.zeros(5)
    for i in range(0,5):
    	values=np.zeros(10000)
        for j in range(10000):
            print(j)
            prb = Problem_non_smooth_richardson_extrapolation(1,Nsteps_arr[i]) 
	    values[j]=prb.objfun(Nsteps_arr[i]) 
    	error[i]=np.mean(values) 
    	var[i]=np.var(values)   

 
    print(error)   
    print(var)
    weak_error=np.abs(error-exact)
    print(weak_error)




    plt.plot(dt_arr, weak_error,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$\Delta t$',fontsize=14)
    #ax.axis([0.1, 0.6, 0.0001, 0.1])
    plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
    #plt.subplot_tool()
    plt.legend(loc='upper left')
    plt.savefig('./results/weak_convergence_order_1D_BS_nonsmooth_richardson_without_middle.eps', format='eps', dpi=1000)  


weak_convergence_rate_plotting()
   



