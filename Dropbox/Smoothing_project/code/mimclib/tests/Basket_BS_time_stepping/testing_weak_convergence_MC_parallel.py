import numpy as np
import time
import scipy.stats as ss
 
import random
 
 
 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
 
import pathos.multiprocessing as mp
import pathos.pools as pp

 
      
 
 
class Problem(object):
 
# attributes
    # attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    basket_d=2     # number of assets in the basket
    c= (1/float(basket_d))*np.ones(basket_d)     # weigths
    sigma=None    # vector of volatilities
    K=None         # Strike price
    T=1.0                      # maturity

    rho=None                  #correlation matrix
   
 
 
#methods
    # this method initializes 
    def __init__(self,Nsteps,nested=False):
        self.nested = nested
   
        self.random_gen = None or np.random
        
        self.S0=100*np.ones(self.basket_d) 
       
        self.sigma=0.2*np.ones(self.basket_d) #vector of volatilities
        #self.K= coeff*self.c.dot(self.S0)  
        self.K= 100                        # Strike price and coeff determine if we have in/at/out the money option
    
        from scipy.linalg import toeplitz 
        self.rho=toeplitz([1,0.3]) #correlation matrix
        self.dt=self.T/float(Nsteps) # time steps length
        self.d=int(np.log2(Nsteps)) #power 2 number steps
 
    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y,Nsteps):
        Y = np.array(Y)
        goal=self.objfun(Y,Nsteps);
        return goal
 
 
     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):
 
        mean = np.zeros(self.basket_d*(Nsteps-1))
        covariance= np.identity(self.basket_d*(Nsteps-1))
        y = np.random.multivariate_normal(mean, covariance)    
        
        # step 1
        y2_1=y[Nsteps-1:self.basket_d*(Nsteps-1)]
        #y21=y[Nsteps]
   
        
        y1_1=y[0:Nsteps-1]
        #y1=y[0]
       
        #step2

        bar_z=self.newtons_method(np.zeros(self.basket_d),y1_1,y2_1,Nsteps)
        

        #step3


        beta=32
        yknots_right=np.polynomial.laguerre.laggauss(beta)
        yknots_left=yknots_right

     


        mylist_left1=[]
        mylist_left1.append(yknots_left[0])
        mylist_left1[1:]=[np.array(y1_1[i]) for i in range(0,len(y1_1))]


        mylist_left2=[]
        mylist_left2.append(yknots_left[0])
        mylist_left2[1:]=[np.array(y2_1[i]) for i in range(0,len(y2_1))]

        mylist_right1=[]
        mylist_right1.append(yknots_right[0])
        mylist_right1[1:]=[np.array(y1_1[i]) for i in range(0,len(y1_1))]


        mylist_right2=[]
        mylist_right2.append(yknots_right[0])
        mylist_right2[1:]=[np.array(y2_1[i]) for i in range(0,len(y2_1))]

        # first quadrant (left,right)
        
        mylist_left_right=mylist_left1+mylist_right2
        weights_list_left_right=[yknots_left[1],yknots_right[1]]
     

        points_left_right=self.cartesian(mylist_left_right)
        weights_left_right_aux=self.cartesian(weights_list_left_right)
        weights_left_right=np.prod(weights_left_right_aux,1)


        x_lr=np.asarray([self.stock_price_trajectory_basket_BS(bar_z[0]-points_left_right[i,0],points_left_right[i,1:],points_left_right[i,Nsteps]\
            +bar_z[1],points_left_right[i,Nsteps+1:],Nsteps)  [0]  for i in range(0,len(yknots_left[0])*len(yknots_right[0]))])

        
        pay_lr=np.asarray( [self.payoff(x_lr[i,:]) for i in range(0,len(yknots_left[0])*len(yknots_right[0]))] ) 
        

        pay_lr=np.multiply(pay_lr,( ((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z[0]-points_left_right[:,0])**2)/2)\
            * np.exp(points_left_right[:,0]) * np.exp(-((points_left_right[:,Nsteps]+bar_z[1])**2)/2)* np.exp(points_left_right[:,Nsteps])))
      
        QoI_lr= weights_left_right.dot(pay_lr)
        




        # second quadrant (right,right)
        
        mylist_right_right=mylist_right1+mylist_right2
        weights_list_right_right=[yknots_right[1],yknots_right[1]]


        points_right_right=self.cartesian(mylist_right_right)
        weights_right_right_aux=self.cartesian(weights_list_right_right)
        weights_right_right=np.prod(weights_right_right_aux,1)


        x_rr=np.asarray([self.stock_price_trajectory_basket_BS(points_right_right[i,0]+bar_z[0],points_right_right[i,1:] ,points_right_right[i,Nsteps]\
            +bar_z[1],points_right_right[i,Nsteps+1:], Nsteps)[0]  for i in range(0,len(yknots_right[0])*len(yknots_right[0]))])

        pay_rr=np.asarray( [self.payoff(x_rr[i,:])for i in range(0,len(yknots_right[0])*len(yknots_right[0]))])
        pay_rr=np.multiply(pay_rr,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_right_right[:,0]+bar_z[0])**2)/2)\
            * np.exp(points_right_right[:,0]) * np.exp(-((points_right_right[:,Nsteps]+bar_z[1])**2)/2)* np.exp(points_right_right[:,Nsteps]) ))

        
        QoI_rr= weights_right_right.dot(pay_rr)


       

        # third   quadrant (left,left)
        
        mylist_left_left=mylist_left1+mylist_left2
        weights_list_left_left=[yknots_left[1],yknots_left[1]]


        points_left_left=self.cartesian(mylist_left_left)
        weights_left_left_aux=self.cartesian(weights_list_left_left)
        weights_left_left=np.prod(weights_left_left_aux,1)


        x_ll=np.asarray([self.stock_price_trajectory_basket_BS(bar_z[0]-points_left_left[i,0],points_left_left[i,1:],bar_z[1]-points_left_left[i,Nsteps]\
            ,points_left_left[i,Nsteps+1:] ,Nsteps )[0]  for i in range(0,len(yknots_left[0])*len(yknots_left[0]))])
        
        pay_ll=np.asarray( [self.payoff(x_ll[i,:]) for i in range(0,len(yknots_left[0])*len(yknots_left[0]))])
        pay_ll=np.multiply(pay_ll,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z[0]-points_left_left[:,0])**2)/2)\
            * np.exp(points_left_left[:,0])* np.exp(-((bar_z[1]-points_left_left[:,Nsteps])**2)/2)* np.exp(points_left_left[:,Nsteps])   ))

        QoI_ll= weights_left_left.dot( pay_ll)



  

        # fourth quadrant (right,left)
        
        mylist_right_left=mylist_right1+mylist_left2
        weights_list_right_left=[yknots_right[1],yknots_left[1]]


        points_right_left=self.cartesian(mylist_right_left)
        weights_right_left_aux=self.cartesian(weights_list_right_left)
        weights_right_left=np.prod(weights_right_left_aux,1)

        x_rl=np.asarray([self.stock_price_trajectory_basket_BS(points_right_left[i,0]+bar_z[0],points_right_left[i,1:] ,bar_z[1]-points_right_left[i,Nsteps]\
            ,points_right_left[i,Nsteps+1:],Nsteps )[0]  for i in range(0,len(yknots_right[0])*len(yknots_left[0]))])
        
        pay_rl=np.asarray( [self.payoff(x_rl[i,:])  for i in range(0,len(yknots_right[0])*len(yknots_left[0]))])
        pay_rl=np.multiply(pay_rl,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_right_left[:,0]+bar_z[0])**2)/2)\
            * np.exp(points_right_left[:,0]) * np.exp(-((bar_z[1]-points_right_left[:,Nsteps])**2)/2)* np.exp(points_right_left[:,Nsteps]) ))
          

        QoI_rl= weights_right_left.dot(pay_rl)
        

     
        
        QoI=QoI_ll+QoI_rr+QoI_rl+QoI_lr

  
        return QoI
 
    def brownian_increments(self,y1,y,Nsteps):
        t=np.linspace(0, self.T, Nsteps+1)     
        h=Nsteps
        j_max=1
        bb= np.zeros((1,Nsteps+1))
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
     
      # This function simulates a basket BS trajectory for stock price, it plays the role of f_1 in our notes
    def stock_price_trajectory_basket_BS(self,y1,yvec_1,y2,yvec_2,Nsteps):
        bb1=self.brownian_increments(y1,yvec_1,Nsteps)
        bb2=self.brownian_increments(y2,yvec_2,Nsteps)

    

        dW1= [bb1[0,i+1]-bb1[0,i]  for i in range(0,Nsteps)] 
        dW2= [bb2[0,i+1]-bb2[0,i] for i in range(0,Nsteps)] 

        dW=np.array([dW1 ,dW2])

        # construct the correlated  brownian bridge increments
        lower_triang_cholesky = np.linalg.cholesky(self.rho)
     
        dW=np.dot(lower_triang_cholesky,dW)  

          
    
       
        dW1=dW[0,:]
        dW2=dW[1,:]


 

        dbb1=dW1-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        dbb2=dW2-(self.dt/np.sqrt(self.T))*y2 # brownian bridge increments dbb_i (used later for the location of the kink point)
        
        

        

        X=np.zeros((self.basket_d,Nsteps+1)) #here will store the BS trajectory
      
        X[:,0]=self.S0
        for n in range(1,Nsteps+1):
            X[0,n]=X[0,n-1]*(1+self.sigma[0]*dW[0,n-1])
            X[1,n]=X[1,n-1]*(1+self.sigma[1]*dW[1,n-1])
      
        
        return X[:,-1],dbb1,dbb2
         
         
      
        # this function defines the payoff function used here
    def payoff(self,x): 
       #print(x)
       g=(self.c.dot(x)-self.K)
   
       if g>0:
          return g
       else:
          return 0
  
 
 
 
    # Root solving procedure
  
     
    #Now we set up the methods used for newton iteration
    def dx(self,x,yvec_1,yvec_2,Nsteps):
     
      
            P1,dP1=self.f(x[0],yvec_1,x[1],yvec_2,Nsteps)
            
       
            return np.abs(0-P1[0])  ,  np.abs(0-P1[1]) 
 
    def f(self,y1,yvec_1,y2,yvec_2,Nsteps):# need to check this
        X,dbb1,dbb2=self.stock_price_trajectory_basket_BS(y1,yvec_1,y2,yvec_2,Nsteps) 
        fi=np.zeros((self.basket_d,len(dbb1)))
        product=np.zeros(self.basket_d)
        summation=np.zeros(self.basket_d)
        Py=np.zeros(self.basket_d)
        dPy=np.zeros(self.basket_d)

        fi[0,:]=  1+(self.sigma[0]/float(np.sqrt(self.T)))*y1*(self.dt)+self.sigma[0]*dbb1
        fi[1,:]=  1+(self.sigma[1]/float(np.sqrt(self.T)))*y2*(self.dt)+self.sigma[1]*dbb2
        
        product[0]=np.prod(fi[0,:])
        product[1]=np.prod(fi[1,:])

        Py[0]=product[0]-(self.K/(float(self.S0[0]*self.c[0]*self.basket_d)))
        Py[1]=product[1]-(self.K/(float(self.S0[1]*self.c[1]*self.basket_d)))

        summation[0]=np.sum(1/fi[0,:])
        summation[1]=np.sum(1/fi[1,:])

        
        dPy[0]= (self.sigma[0]/float(np.sqrt(self.T)))*(self.dt)*product[0]*summation[0]
        dPy[1]=  (self.sigma[1]/float(np.sqrt(self.T)))*(self.dt)*product[1]*summation[1]
        return Py,dPy         
         
 
    def newtons_method(self,x0,yvec_1,yvec_2,Nsteps,eps=1e-10):
        
        delta1, delta2 = self.dx(x0,yvec_1,yvec_2,Nsteps)


      
        while (delta1 > eps) | (delta2 > eps):
        
            P_value,dP=self.f(x0[0],yvec_1,x0[1],yvec_2,Nsteps)
       
            x0[0] = x0[0] - 0.1*P_value[0]/dP[0]
            x0[1] = x0[1] - 0.1*P_value[1]/dP[1]

            delta1,delta2 = self.dx(x0,yvec_1,yvec_2,Nsteps)

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
        start_time=time.time()
        exact= 6.437977 #S_0=K=100, sigma =0.2, corr=0.3, T=1
        marker=['>', 'v', '^', 'o', '*','+','-',':']
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # # feed parameters to the problem
        Nsteps_arr=np.array([2,4,8,16])
        dt_arr=1.0/(Nsteps_arr)
        error_diff=np.zeros(3)
        stand_diff=np.zeros(3)
        error=np.zeros(4)
        stand=np.zeros(4)
        elapsed_time_qoi=np.zeros(4)
        Ub=np.zeros(4)
        Lb=np.zeros(4)
        Ub_diff=np.zeros(3)
        Lb_diff=np.zeros(3)
        values=np.zeros((5*(10**4),4)) 
         
      
        
 
        num_cores = mp.cpu_count()
   
        for i in range(0,4):
            print i
            start_time=time.time()
             
            prb = Problem(Nsteps_arr[i]) 
            def processInput(j):
                return prb.objfun(Nsteps_arr[i])/float(exact)
 
            
            p =  pp.ProcessPool(num_cores)  # Processing Pool with four processors
            
            values[:,i]= p.map(processInput, range(((5*(10**4)))))  
           
            elapsed_time_qoi[i]=time.time()-start_time
            print np.mean(values[:,i])
            print  elapsed_time_qoi[i]





         
 
 
        
        print elapsed_time_qoi
 
        error=np.abs(np.mean(values,axis=0) - 1) 
        stand=np.std(values, axis = 0)/  float(np.sqrt(5*(10**4)))
        Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
        Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand
        print(error)   
        print(stand)
        print Lb
        print Ub
          
        differences= [values[:,i]-values[:,i+1] for i in range(0,3)]
        error_diff=np.abs(np.mean(differences,axis=1))
        print error_diff 
        stand_diff=np.std(differences, axis = 1)/ float(np.sqrt(5*(10**4)))
        print stand_diff
        Ub_diff=np.abs(np.mean(differences,axis=1))+1.96*stand_diff
        Lb_diff=np.abs(np.mean(differences,axis=1))-1.96*stand_diff
        print Ub_diff
        print Lb_diff
 
        
        z= np.polyfit(np.log(dt_arr), np.log(error), 1)
        fit=np.exp(z[0]*np.log(dt_arr))
        print z[0]
 
 
        z3=np.zeros(2)
        z3[0]=1.0
        z3[1]=np.log(error[0]*2)
        fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])
 
 
 
 
        z_diff= np.polyfit(np.log(dt_arr[0:3]), np.log(error_diff), 1)
        fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:3]))
        print z_diff[0]
 
        z3diff=np.zeros(2)
        z3diff[0]=1.0
        z3diff[1]=np.log(error_diff[0]*2)
        fit3diff=np.exp(z3diff[0]*np.log(dt_arr[0:3])+z3diff[1])
         
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
        plt.savefig('./results/weak_convergence_order_basket_option_2d_1_relative_M_5_10_4_beta_32.eps', format='eps', dpi=1000)  
 
        fig = plt.figure()
        plt.plot(dt_arr[0:3], error_diff,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
        plt.plot(dt_arr[0:3], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = '--', hold=True) 
        plt.plot(dt_arr[0:3], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = '--', hold=True) 
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\Delta t$',fontsize=14)
 
        plt.plot(dt_arr[0:3], fit_diff,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--', marker='o')
        plt.plot(dt_arr[0:3], fit3diff,linewidth=2.0,label=r'rate= %s' % format(z3diff[0]  , '.2f'), linestyle = '--', marker='o')
        plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        plt.legend(loc='upper left')
        plt.savefig('./results/weak_convergence_order_differences_basket_option_2d_1_relative_M_5_10_4_beta_32.eps', format='eps', dpi=1000)  
 
 
weak_convergence_differences()   