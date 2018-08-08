import numpy as np
import time
import scipy.stats as ss
from scipy.stats import sem

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

class Problem_richardson_level2_call(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    T=1.0                      # maturity
    sigma=None    # volatility
    d=None
    dt=None


    #methods
    # this method initializes 
    def __init__(self,coeff,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        self.sigma=0.4
        

     # objfun: 
    def objfun(self,Nsteps):
        mean = np.zeros(4*Nsteps)
        covariance= np.identity(4*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)
  

        beta=10
        yknots_right=np.polynomial.laguerre.laggauss(beta)
        yknots_left=yknots_right


         #Richardson level2

        #finest level  

        #finer level points 
        y_aux_ff=y[0:4*Nsteps-1]
        #kink points by newton method
        bar_z_ff=self.newtons_method(0,y_aux_ff,4*Nsteps)

        mylist_left_ff=[]
        mylist_left_ff.append(yknots_left[0])
        mylist_left_ff[1:]=[np.array(y_aux_ff[i]) for i in range(0,len(y_aux_ff))]
        points_left_ff=self.cartesian(mylist_left_ff)
        x_l_ff=np.asarray([self.stock_price_trajectory_1D_BS(bar_z_ff-points_left_ff[i,0],points_left_ff[i,1:],4*Nsteps)[0]  for i in range(0,len(yknots_left[0]))])
        QoI_left_ff= yknots_left[1].dot(self.payoff(x_l_ff)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z_ff-points_left_ff[:,0])**2)/2)* np.exp(points_left_ff[:,0])))

        mylist_right_ff=[]
        mylist_right_ff.append(yknots_right[0])
        mylist_right_ff[1:]=[np.array(y_aux_ff[i]) for i in range(0,len(y_aux_ff))]
        points_right_ff=self.cartesian(mylist_right_ff)
        x_r_ff=np.asarray([self.stock_price_trajectory_1D_BS(points_right_ff[i,0]+bar_z_ff,points_right_ff[i,1:],4*Nsteps)[0] for i in range(0,len(yknots_right[0]))])
        QoI_right_ff= yknots_right[1].dot(self.payoff(x_r_ff)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right_ff[:,0]+bar_z_ff)**2)/2)* np.exp(points_right_ff[:,0]))

        dff =QoI_left_ff+QoI_right_ff


        
     

        #finer level QoI
        #finer level points 
        y_aux_f=y_aux_ff[0:2*Nsteps-1]
        #kink points by newton method
        bar_z_f=self.newtons_method(0,y_aux_f,2*Nsteps)

        mylist_left_f=[]
        mylist_left_f.append(yknots_left[0])
        mylist_left_f[1:]=[np.array(y_aux_f[i]) for i in range(0,len(y_aux_f))]
        points_left_f=self.cartesian(mylist_left_f)
        x_l_f=np.asarray([self.stock_price_trajectory_1D_BS(bar_z_f-points_left_f[i,0],points_left_f[i,1:],2*Nsteps)[0]  for i in range(0,len(yknots_left[0]))])
        QoI_left_f= yknots_left[1].dot(self.payoff(x_l_f)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z_f-points_left_f[:,0])**2)/2)* np.exp(points_left_f[:,0])))

        mylist_right_f=[]
        mylist_right_f.append(yknots_right[0])
        mylist_right_f[1:]=[np.array(y_aux_f[i]) for i in range(0,len(y_aux_f))]
        points_right_f=self.cartesian(mylist_right_f)
        x_r_f=np.asarray([self.stock_price_trajectory_1D_BS(points_right_f[i,0]+bar_z_f,points_right_f[i,1:],2*Nsteps)[0] for i in range(0,len(yknots_right[0]))])
        QoI_right_f= yknots_right[1].dot(self.payoff(x_r_f)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right_f[:,0]+bar_z_f)**2)/2)* np.exp(points_right_f[:,0]))

        df=QoI_left_f+QoI_right_f



         #coarse level QoI
        #coarser level points
        y_aux_c=y_aux_ff[0:Nsteps-1]
        #kink points by newton method
        bar_z_c=self.newtons_method(0,y_aux_c,Nsteps)

        mylist_left_c=[]
        mylist_left_c.append(yknots_left[0])
        mylist_left_c[1:]=[np.array(y_aux_c[i]) for i in range(0,len(y_aux_c))]
        points_left_c=self.cartesian(mylist_left_c)
        x_l_c=np.asarray([self.stock_price_trajectory_1D_BS(bar_z_c-points_left_c[i,0],points_left_c[i,1:],Nsteps)[0]  for i in range(0,len(yknots_left[0]))])
        QoI_left_c= yknots_left[1].dot(self.payoff(x_l_c)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z_c-points_left_c[:,0])**2)/2)* np.exp(points_left_c[:,0])))

        mylist_right_c=[]
        mylist_right_c.append(yknots_right[0])
        mylist_right_c[1:]=[np.array(y_aux_c[i]) for i in range(0,len(y_aux_c))]
        points_right_c=self.cartesian(mylist_right_c)
        x_r_c=np.asarray([self.stock_price_trajectory_1D_BS(points_right_c[i,0]+bar_z_c,points_right_c[i,1:],Nsteps)[0] for i in range(0,len(yknots_right[0]))])
        QoI_right_c= yknots_right[1].dot(self.payoff(x_r_c)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right_c[:,0]+bar_z_c)**2)/2)* np.exp(points_right_c[:,0]))

        dc =QoI_left_c+QoI_right_c

            
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
   


    




def weak_convergence_differences():    
        exact=  15.8519
        marker=['>', 'v', '^', 'o', '*','+','-',':']
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([1,2,4])
        dt_arr=1.0/(Nsteps_arr)
        error=np.zeros(3)
        stand=np.zeros(3)
        Ub=np.zeros(3)
        Lb=np.zeros(3)
        values=np.zeros((10**6,3)) 
        for i in range(0,3):
            print i
            prb = Problem_richardson_level2_call(1,Nsteps_arr[i]) 
            for j in range(10**6):
                values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)
        
        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(10**6))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand
        print(error)   
        print(stand)
        print Lb
        print Ub


        differences= [values[:,i]-values[:,i+1] for i in range(0,2)]
        error_diff=np.abs(np.mean(differences,axis=1))
        print error_diff 
        stand_diff=np.std(differences, axis = 1)/ float(np.sqrt(10**6))
        print stand_diff
        Ub_diff=np.abs(np.mean(differences,axis=1))+1.96*stand_diff
        Lb_diff=np.abs(np.mean(differences,axis=1))-1.96*stand_diff
        print Ub_diff
        print Lb_diff
    
       
        z= np.polyfit(np.log(dt_arr), np.log(error), 1)
        fit=np.exp(z[0]*np.log(dt_arr))
        print z[0]

        z3=np.zeros(2)
        z3[0]=3.0
        z3[1]=np.log(error[0])
        fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])

        z_diff= np.polyfit(np.log(dt_arr[0:2]), np.log(error_diff), 1)
        fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:2]))
        print z_diff[0]

        z3diff=np.zeros(2)
        z3diff[0]=3.0
        z3diff[1]=np.log(error_diff[0])
        fit3diff=np.exp(z3diff[0]*np.log(dt_arr[0:2])+z3diff[1])

    
        fig = plt.figure()

        plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
        plt.plot(dt_arr, Lb,linewidth=2.0,label='Lb' ,linestyle = '--', hold=True) 
        plt.plot(dt_arr, Ub,linewidth=2.0,label='Ub' ,linestyle = '--', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)
        plt.plot(dt_arr, fit,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--', marker='o')
        plt.plot(dt_arr, fit3,linewidth=2.0,label=r'rate= %s' % format(z3[0]  , '.2f'), linestyle = '--', marker='o')
        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_Call_richardson_level2_relative.eps', format='eps', dpi=1000)  

        fig = plt.figure()
        plt.plot(dt_arr[0:2], error_diff,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
        plt.plot(dt_arr[0:2], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = '--', hold=True) 
        plt.plot(dt_arr[0:2], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = '--', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)

        plt.plot(dt_arr[0:2], fit_diff,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--', marker='o')
        plt.plot(dt_arr[0:2], fit3diff,linewidth=2.0,label=r'rate= %s' % format(z3diff[0]  , '.2f'), linestyle = '--', marker='o')


        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_differences_Call_richardson_level2_relative.eps', format='eps', dpi=1000)

weak_convergence_differences()