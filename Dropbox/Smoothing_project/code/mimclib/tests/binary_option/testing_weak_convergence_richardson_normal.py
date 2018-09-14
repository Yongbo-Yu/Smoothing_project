import numpy as np
import time
import scipy.stats as ss
from scipy.stats import sem

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
        
    

     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):


        mean = np.zeros(2*Nsteps)
        covariance= np.identity(2*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)    
        y_1f=y[1:2*Nsteps]
        y1f=y[0]
        
        y_1c=y_1f[0:Nsteps-1]
        y1c=y1f
        
        #level 1: richardson extrapol

        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )



        Xf=self.stock_price_trajectory_1D_BS(y1f,y_1f,2*Nsteps)[0]

        Xc=self.stock_price_trajectory_1D_BS(y1c,y_1c,Nsteps)[0]
           
     
            
        d[1,0] = self.payoff(Xf)

        d[0,0] =self.payoff(Xc) 


        #Richardson
        d[1,1] = 2*d[1,0] - d[0,0]
        QoI=d[1,1]
                
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

        
    def payoff(self,x): 
       #print(x)
       g=(x-self.K)
       if g>0:
         g=1
       else:
         g=0
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
   


    



def weak_convergence_differences():    
    exact=  0.420740290561
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # # feed parameters to the problem
    Nsteps_arr=np.array([1,2,4,8])
   
    dt_arr=1.0/(Nsteps_arr)
    error_diff=np.zeros(3)
    stand_diff=np.zeros(3)
    elapsed_time_qoi=np.zeros(4)
    error=np.zeros(4)
    stand=np.zeros(4)
    Ub=np.zeros(4)
    Lb=np.zeros(4)
    Ub_diff=np.zeros(3)
    Lb_diff=np.zeros(3)
    values=np.zeros((3*(10**5),4)) 
    for i in range(0,4):
        print i
        start_time=time.time()

        prb = Problem_non_smooth_richardson_extrapolation(1,Nsteps_arr[i]) 
        for j in range(3*(10**5)):
            
            values[j,i]=prb.objfun(Nsteps_arr[i])/float(exact)

        

        elapsed_time_qoi[i]=time.time()-start_time
        print  elapsed_time_qoi[i]    
          


     


    start_time_2=time.time()
    error=np.abs(np.mean(values,axis=0) - 1) 
    elapsed_time_qoi=time.time()-start_time_2+elapsed_time_qoi
    
    stand=np.std(values, axis = 0)/  float(np.sqrt(3*(10**5)))
    Ub=np.abs(np.mean(values,axis=0) - 1)+1.96*stand
    Lb=np.abs(np.mean(values,axis=0) - 1)-1.96*stand

    print elapsed_time_qoi
    print(error)   
    print(stand)
    print Lb
    print Ub
    
     
    differences= [values[:,i]-values[:,i+1] for i in range(0,3)]
    error_diff=np.abs(np.mean(differences,axis=1))
    print error_diff 
    stand_diff=np.std(differences, axis = 1)/ float(np.sqrt(3*(10**5)))
    print stand_diff
    Ub_diff=np.abs(np.mean(differences,axis=1))+1.96*stand_diff
    Lb_diff=np.abs(np.mean(differences,axis=1))-1.96*stand_diff
    print Ub_diff
    print Lb_diff

   
    z= np.polyfit(np.log(dt_arr), np.log(error), 1)
    fit=np.exp(z[0]*np.log(dt_arr))
    print z[0]


    z3=np.zeros(2)
    z3[0]=2.0
    z3[1]=np.log(error[0])
    fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])

    z_diff= np.polyfit(np.log(dt_arr[0:3]), np.log(error_diff), 1)
    fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:3]))
    print z_diff[0]

    z3diff=np.zeros(2)
    z3diff[0]=2.0
    z3diff[1]=np.log(error_diff[0])
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
    plt.savefig('./results/weak_convergence_order_binary_richardson_relative_M_5_10_5.eps', format='eps', dpi=1000)  

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
    plt.savefig('./results/weak_convergence_order_differences_binary_richardson_relative_M_5_10_5.eps', format='eps', dpi=1000)  


weak_convergence_differences()


#     def brownian_increments(self, y1_f,Nsteps):
#         t_c=np.linspace(0, self.T, Nsteps+1)     
#         t_f=np.linspace(0, self.T, 2*Nsteps+1)  
#         h_c=Nsteps
#         h_f=2*Nsteps
#         j_max_c=1
#         j_max_f=1

#         mean = np.zeros(2*Nsteps-1)
#         covariance= np.identity(2*Nsteps-1)
#         yf = np.random.multivariate_normal(mean, covariance)
#         yc=yf[0:Nsteps-1]
   
            
#         y1_c=y1_f

#         bb_f= np.zeros(2*Nsteps+1)
#         bb_c= np.zeros(Nsteps+1)

#         bb_f[h_f]=np.sqrt(self.T)*y1_f
#         bb_c[h_c]=np.sqrt(self.T)*y1_c
      
       
#         #ds=int(np.log2(Nsteps)) #power 2 number steps
#         for k_c in range(1,self.d_c+1):
#             i_min_c=h_c//2
#             i_c=i_min_c
#             l_c=0
#             r_c=h_c
#             for j_c in range(1,j_max_c+1):
#                 a_c=((t_c[r_c]-t_c[i_c])* bb_c[l_c]+(t_c[i_c]-t_c[l_c])*bb_c[r_c])/float(t_c[r_c]-t_c[l_c])
#                 b_c=np.sqrt((t_c[i_c]-t_c[l_c])*(t_c[r_c]-t_c[i_c])/float(t_c[r_c]-t_c[l_c]))
#                 bb_c[i_c]=a_c+b_c*yc[i_c-1]
#                 i_c=i_c+h_c
#                 l_c=l_c+h_c
#                 r_c=r_c+h_c
#             j_max_c=2*j_max_c
#             h_c=i_min_c 

#         for k_f in range(1,self.d_f+1):
#             i_min_f=h_f//2
#             i_f=i_min_f
#             l_f=0
#             r_f=h_f
#             for j_f in range(1,j_max_f+1):
#                 a_f=((t_f[r_f]-t_f[i_f])* bb_f[l_f]+(t_f[i_f]-t_f[l_f])*bb_f[r_f])/float(t_f[r_f]-t_f[l_f])
#                 b_f=np.sqrt((t_f[i_f]-t_f[l_f])*(t_f[r_f]-t_f[i_f])/float(t_f[r_f]-t_f[l_f]))
#                 bb_f[i_f]=a_f+b_f*yf[i_f-1]
#                 i_f=i_f+h_f
#                 l_f=l_f+h_f
#                 r_f=r_f+h_f
#             j_max_f=2*j_max_f
#             h_f=i_min_f     


#         return bb_c,bb_f,y1_c

    

    
#     # This function simulates a 1D BS trajectory for stock price, it plays the role of f_1 in our notes
#     def stock_price_trajectory_1D_BS(self,y1_f,Nsteps):
#     	bb_c,bb_f,y1_c=self.brownian_increments(y1_f,Nsteps)
#         dW_f= [bb_f[i+1]-bb_f[i] for i in range(0,len(bb_f)-1)] 
#         dW_c= [(bb_c[i+1]-bb_c[i]) for i in range(0,len(bb_c)-1)] 
       
#         dbb_f=dW_f-(self.dt_f/np.sqrt(self.T))*y1_f
#         dbb_c=dW_c-(self.dt_c/np.sqrt(self.T))*y1_c
      
#         X_c=np.zeros(Nsteps+1) #here will store the BS trajectory
#         X_f=np.zeros(2*Nsteps+1) #here will store the BS trajectory

#         X_c[0]=self.S0
#         X_f[0]=self.S0

#         for nc in range(1,Nsteps+1):
#             X_c[nc]=X_c[nc-1]*(1+self.sigma*dW_c[nc-1])
#         for nf in range(1,2*Nsteps+1):
#             X_f[nf]=X_f[nf-1]*(1+self.sigma*dW_f[nf-1])    
        
#         return X_c[-1],X_f[-1],dbb_c,dbb_f,y1_c
           
    


#     # Root solving procedure
 
#     #Now we set up the methods used for newton iteration
#     def dx(self,x,Nsteps):
#         P1_c,dP1_c,P1_f,dP1_f=self.f(x,Nsteps)
#         return abs(0-P1_c),abs(0-P1_f)

#     def f(self,y1_f,Nsteps):# need to check this
#         X_c,X_f,dbb_c,dbb_f,y1_c=self.stock_price_trajectory_1D_BS(y1_f,Nsteps) # right version
        
#         fi_f=1+(self.sigma/float(np.sqrt(self.T)))*y1_f*(self.dt_f)+self.sigma*dbb_f
#         fi_c=1+(self.sigma/float(np.sqrt(self.T)))*y1_c*(self.dt_c)+self.sigma*dbb_c

#         product_f=np.prod(fi_f)
#         product_c=np.prod(fi_c)

#         summation_f=np.sum(1/fi_f)
#         summation_c=np.sum(1/fi_c)

#         Py_f=product_f-(self.K/float(self.S0))
#         Py_c=product_c-(self.K/float(self.S0))


#         dPy_f=(self.sigma/float(np.sqrt(self.T)))*(self.dt_f)*product_f*summation_f
#         dPy_c=(self.sigma/float(np.sqrt(self.T)))*(self.dt_c)*product_c*summation_c


#         return Py_c,dPy_c,Py_f,dPy_f    
        
        
#     def newtons_method(self,x0,Nsteps,eps=1e-10):
#         delta_c,delta_f = self.dx(x0,Nsteps)
#         x0_c=x0
#         x0_f=x0

#         while delta_c > eps:
#             P_value_c,dP_c=self.f(x0_c,Nsteps)[0:2]
#             x0_c= x0_c - 0.1*P_value_c/dP_c
#             delta_c = self.dx(x0_c,Nsteps)[0]
#             #print(delta_c)
        
#         while delta_f > eps:
        
#             #(self.f(x0,y))
#             P_value_f,dP_f=self.f(x0_f,Nsteps)[2:]
#             x0_f = x0_f - 0.1*P_value_f/dP_f
#             delta_f = self.dx(x0_f,Nsteps)[1]  
            
#         return x0_c,x0_f    




#     def cartesian(self,arrays, out=None):
#         """
#         Generate a cartesian product of input arrays.

#         Parameters
#         ----------
#         arrays : list of array-like
#             1-D arrays to form the cartesian product of.
#         out : ndarray
#             Array to place the cartesian product in.

#         Returns
#         -------
#         out : ndarray
#             2-D array of shape (M, len(arrays)) containing cartesian products
#             formed of input arrays.

#         Examples
#         --------
#         >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
#         array([[1, 4, 6],
#                [1, 4, 7],
#                [1, 5, 6],
#                [1, 5, 7],
#                [2, 4, 6],
#                [2, 4, 7],
#                [2, 5, 6],
#                [2, 5, 7],
#                [3, 4, 6],
#                [3, 4, 7],
#                [3, 5, 6],
#                [3, 5, 7]])

#         """

#         arrays = [np.asarray(x) for x in arrays]
#         dtype = float

#         n = np.prod([x.size for x in arrays])
#         if out is None:
#             out = np.zeros([n, len(arrays)], dtype=dtype)

#         m = n / arrays[0].size
#         out[:,0] = np.repeat(arrays[0], m)
#         if arrays[1:]:
#             self.cartesian(arrays[1:], out=out[0:m,1:])
#             for j in xrange(1, arrays[0].size):
#                 out[j*m:(j+1)*m,1:] = out[0:m,1:]
#         return out               
    
    
#     # this function defines the payoff function used here
#     def payoff(self,x): 
#        g=(x-self.K)
#        g[g < 0] = 0
#        return g  

#     @staticmethod
#     def Init():
#         import sys #This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter
#         count = len(sys.argv)  #sys.argv is a list in Python, which contains the command-line arguments passed to the script. With the len(sys.argv) function you can count the number of arguments. 
#         #arr = (ct.c_char_p * len(sys.argv))()
#         arr = sys.argv







# def weak_convergence_rate_plotting():    
#     #num_cores = multiprocessing.cpu_count()
    
#     exact=  0.499
#     marker=['>', 'v', '^', 'o', '*','+','-',':']
#     ax = figure().gca()
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     # # feed parameters to the problem
#     Nsteps_arr=np.array([2,4,8,16,32,64])
#     dt_arr=1.0/(Nsteps_arr)
#     error=np.zeros(6)
#     val=np.zeros(6)
#     sample_95ci =np.zeros(6)
#     sample_max =np.zeros(6)
#     sample_min =np.zeros(6)

#     for i in range(0,6):
#         error_val=np.zeros(100)
#         values=np.zeros(100)
#         for j in range(100):

#             #andom.seed( j)
#             print(j)
#             prb = Problem_non_smooth_richardson_extrapolation(1,Nsteps_arr[i]) 
#             values[j]=prb.objfun(Nsteps_arr[i]) 
#             error_val[j]= np.abs(values[j]-exact)
        
#         error[i]=np.mean(error_val)
#         sample_95ci [i]= 1.96 * sem(error_val)
#         sample_min[i] = error[i] - sample_95ci[i]
#         sample_max[i] = error[i] + sample_95ci[i]
   
#     print(sample_max)    
#     print(sample_min)     
#     print(error)   
    
#     #z= np.polyfit(dt_arr,error, 1)
#     #fit=np.exp(z[0]*dt_arr)
    
   
    
#     plt.plot(dt_arr, error,linewidth=2.0,label='weak error' ,linestyle = '--',marker='v', hold=True) 
#     plt.plot(dt_arr, sample_min,linewidth=2.0,label='lower bound ' ,linestyle = '--',marker='<', hold=True) 
#     plt.plot(dt_arr, sample_max,linewidth=2.0,label='upper bound' ,linestyle = '--',marker='>', hold=True) 
#     #plt.plot(dt_arr, fit,linewidth=2.0,label=r'rate= %s' % z[0], linestyle = '--', marker='o') 
#     plt.plot(dt_arr, dt_arr,linewidth=2.0,label=r'rate= %s' % 1, linestyle = '--', marker='o') 

#     plt.yscale('log')
#     plt.xscale('log')
#     plt.xlabel(r'$\Delta t$',fontsize=14)
    
#     plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
#     plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
#     plt.legend(loc='upper left')
#     plt.savefig('./results/weak_convergence_order_1D_BS_binary.eps', format='eps', dpi=1000)     
    


# weak_convergence_rate_plotting()
