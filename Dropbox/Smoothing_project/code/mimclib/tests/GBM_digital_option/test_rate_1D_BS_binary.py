#!/usr/bin/env python

import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


import warnings
import os.path
import numpy as np
import time
import sys



class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    T=1.0                      # maturity
    sigma=None    # volatility
    N=3 # number of time steps which will be equal to the number of brownian bridge components (we set is a power of 2)
    d=None #power 2 number steps
   
#methods
    # this method initializes the class of basket 
    def __init__(self,coeff):
        self.random_gen = None or np.random
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        #self.sigma=np.random.uniform(0.3,0.4,1)  #volatility
        self.sigma=0.4
        self.dt=self.T/float(self.N+1) # time steps length
        self.d=int(np.log2(self.N+1)) #power 2 number steps



    def BeginRuns(self):
        self.elapsed_time=0.0

    def EndRuns(self):
        elapsed_time=self.elapsed_time;
        self.elapsed_time=0.0;
        return elapsed_time;

    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y):
        Y = np.array(Y)
        goal=self.objfun(Y);
        return goal

    # objfun:  beta #number of points in the first direction
    def objfun(self,y):
        start_time=time.time()
        y=y[0:self.N] 
        bar_z,dP=self.newtons_method(0,y)

        from scipy.stats import norm
        QoI=    norm.sf(bar_z)   
        #(1/(np.sqrt(self.T)*self.sigma*self.K))
      # (1-norm.cdf(bar_z))
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
        t=np.linspace(0, self.T, self.N+2)     
        h=self.N+1
        j_max=1
        bb= np.zeros(self.N+2)
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

       



    # This function simulates a 1D BS trajectory for stock price, it plays the role of f_1 in our notes
    def stock_price_trajectory_1D_BS(self,y1,y):
        bb=self.brownian_increments(y1,y)
        dW= [bb[i+1]-bb[i] for i in range(0,len(bb)-1)] 
        dbb=dW-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
      
        X=np.zeros(self.N+2) #here will store the BS trajectory
        X[0]=self.S0
        for n in range(1,self.N+2):
            X[n]=X[n-1]*(1+self.sigma*dW[n-1])
        return X[-1],dbb
        
        



        
    
    # this function defines the payoff function used here
       
    def payoff(self,x): 
       g=(x-self.K)
       #g=(x-self.K)
       g[g < 0] = 0
       return g  


    # Root solving procedure
 
    #Now we set up the methods used for newton iteration
    def dx(self,x,y):
        P1,dP1=self.f(x,y)
        return abs(0-P1)




    def f(self,y1,y):# need to check this
        X,dbb=self.stock_price_trajectory_1D_BS(y1,y) # right version
        fi=1+(self.sigma/float(np.sqrt(self.T)))*y1*(self.dt)+self.sigma*dbb
        product=np.prod(fi)
        summation=np.sum(1/fi)
        Py=product-(self.K/float(self.S0))
        dPy=(self.sigma/float(np.sqrt(self.T)))*(self.dt)*product*summation
        return Py,dPy    
        

       


    def newtons_method(self,x0,y,eps=1e-10):
        delta = self.dx(x0,y)
       
        while delta > eps:
            global DP
            #(self.f(x0,y))
            P_value,dP=self.f(x0,y)
            x0 = x0 - 0.1*P_value/dP
            DP=dP
            delta = self.dx(x0,y)
            
            
        return x0,DP
    

def knots_gaussian(n, mi, sigma):
    # [x,w]=KNOTS_GAUSSIAN(n,mi,sigma)
    #
    # calculates the collocation points (x)
    # and the weights (w) for the gaussian integration
    # w.r.t to the weight function
    # rho(x)=1/sqrt(2*pi*sigma) *exp( -(x-mi)^2 / (2*sigma^2) )
    # i.e. the density of a gaussian random variable
    # with mean mi and standard deviation sigma
    # ----------------------------------------------------
    # Sparse Grid Matlab Kit
    # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
    # See LICENSE.txt for license
    # ----------------------------------------------------
    if n == 1:
        # the point (traslated if needed)
        # the weight is 1:
        return [mi], [1]

    def coefherm(n):
        if n <= 1:
            raise Exception(' n must be > 1 ')
        a = np.zeros(n)
        b = np.zeros(n)
        b[0] = np.sqrt(np.pi)
        b[1:] = 0.5 * np.arange(1, n)
        return a, b

    # calculates the values of the recursive relation
    
    a, b = coefherm(n)
    
    # builds the matrix
    
    JacM = np.diag(a)+np.diag(np.sqrt(b[1:n[0]]), 1)+np.diag(np.sqrt(b[1:n[0]]), -1)
    # calculates points and weights from eigenvalues / eigenvectors of JacM
    [x, W] = np.linalg.eig(JacM)
    w = W[0, :]**2.
    ind = np.argsort(x)
    x = x[ind]
    w = w[ind]
    # modifies points according to mi, sigma (the weigths are unaffected)
    x = mi + np.sqrt(2) * sigma * x
    return x, w    

# this method gives the number of points of the quadrature given the degree
def lev2knots_doubling(i):
    # m = lev2knots_doubling(i)
    #
    # relation level / number of points:
    #    m = 2^{i-1}+1, for i>1
    #    m=1            for i=1
    #    m=0            for i=0
    #
    # i.e. m(i)=2*m(i-1)-1
    # ----------------------------------------------------
    # Sparse Grid Matlab Kit
    # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
    # See LICENSE.txt for license
    # ----------------------------------------------------
    i = np.array([i] if np.isscalar(i) else i, dtype=np.int)
    
    m = 2 ** (i-1)+1
    m[i==1] = 1
    m[i==0] = 0
    return m

def cartesian(arrays, out=None):
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
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out               
#     # fnknots  gives two arrays first one for quadrature points and the second one for weights
fnKnots= lambda beta: knots_gaussian(lev2knots_doubling(1+beta),   0, 1.0)  



def first_difference_rate_plotting():       
    # # feed parameters to the problem
    prb = Problem(1) 
    marker=['>', 'v', '^', 'o']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for d1 in range(0,3):
        print(d1)
        mylist=[]
        bias=np.zeros(4)
        points=np.zeros(4)
        indices=np.zeros(4,dtype=int)
        fixed_points=fnKnots(0)[0]
        mylist[:prb.N]=[fixed_points for i in range(0,prb.N)]
        j=0
        for pts in range(2,6):
            fine_ind_points=fnKnots(pts)[0]
            fine_ind_weights=fnKnots(pts)[1]
            mylist[d1]=fine_ind_points
            fine_points=cartesian(mylist)
            
            fine_values=[prb.SolveFor(fine_points[i]) for i in range(0,lev2knots_doubling(1+pts))]
            #(fine_values)
            QoI_fine=fine_ind_weights.dot(fine_values)
            print('f=',QoI_fine)
            

            coarse_ind_points=fnKnots(pts-1)[0]
            coarse_ind_weights=fnKnots(pts-1)[1]
            mylist[d1]=coarse_ind_points
            coarse_points=cartesian(mylist)
            coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts))]
            QoI_coarse=coarse_ind_weights.dot(coarse_values)
            print('c=',QoI_coarse)
            bias[j]=np.abs(QoI_fine-QoI_coarse)
            points[j]=lev2knots_doubling(1+pts)
            indices[j]=pts

            j=j+1

      
        QoI_beta=np.zeros(prb.N,dtype=int)
        QoI_beta[d1]=1

        z= np.polyfit(indices, np.log(bias), 1)
        fit=np.exp(z[0]*indices)

        
        plt.plot(indices, bias,linewidth=2.0,label=r'$\bar{\beta}=$ %s' % np.array_str(QoI_beta),linestyle = '--', marker=marker[d1],hold=True) 
        plt.plot(indices, fit,linewidth=2.0,label=r'rate= %s' % z[0], linestyle = '--', marker='o') 
        #plt.plot(points, 0.001/points,'r',linewidth=2.0,label='order 1') 
        #plt.plot(points, 0.001/(points**2),'g',linewidth=2.0, label='order 2')  
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('k',fontsize=14)
        plt.ylabel(r'$\mid \Delta E_{\mathbf{1}+k \bar{\beta}} \mid $',fontsize=14)  
     
    plt.legend(loc='upper right')
    plt.savefig('./results/first_difference_1D_BS_16steps_binary_opt.eps', format='eps', dpi=1000)



def mixed_difference_order2_rate_plotting(d):       
    # # feed parameters to the problem
    prb = Problem(1) 
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for k in range(0,3):  
        if k==d:
            print('Hello')
            continue
        else:    
            mylist=[]
            mylist_weight=[]
            bias=np.zeros(4)
            points=np.zeros(4)
            indices=np.zeros(4,dtype=int)
            fixed_points=fnKnots(0)[0]
            fixed_weight=fnKnots(0)[1]
            mylist[:prb.N]=[fixed_points for i in range(0,prb.N)]
            mylist_weight[:prb.N]=[fixed_weight for i in range(0,prb.N)]
            j=0
            for pts in range(2,6):
                fine_ind_points=fnKnots(pts)[0]
                fine_ind_weights=fnKnots(pts)[1]
                coarse_ind_points=fnKnots(pts-1)[0]
                coarse_ind_weights=fnKnots(pts-1)[1]

                #block1
                mylist[d]=fine_ind_points
                mylist[k]=fine_ind_points
                mylist_weight[d]=fine_ind_weights
                mylist_weight[k]=fine_ind_weights
                fine_points=cartesian(mylist)
                fine_weights=cartesian(mylist_weight)
                weights_ff=np.asarray([np.prod(fine_weights[i]) for i in range (0,len(fine_weights))])
                fine_values=[prb.SolveFor(fine_points[i]) for i in range(0,(lev2knots_doubling(1+pts)*lev2knots_doubling(1+pts)))]
                QoI_fine_fine=weights_ff.dot(fine_values)
                print('ff=',QoI_fine_fine)
                
                
            
                #block2
                mylist[d]=coarse_ind_points
                mylist[k]=coarse_ind_points
                mylist_weight[d]=coarse_ind_weights
                mylist_weight[k]=coarse_ind_weights
                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_cc=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,(lev2knots_doubling(pts)*lev2knots_doubling(pts)))]
                QoI_coarse_coarse=weights_cc.dot(coarse_values)
                print('cc=',QoI_coarse_coarse)

                 #block3

            
                mylist[d]=fine_ind_points
                mylist[k]=coarse_ind_points
                mylist_weight[d]=fine_ind_weights
                mylist_weight[k]=coarse_ind_weights
                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_fc=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts+1)*lev2knots_doubling(pts))]
                QoI_fine_coarse=weights_fc.dot(coarse_values)
                print('fc=',QoI_fine_coarse)


                #block4

            
                mylist[k]=fine_ind_points
                mylist[d]=coarse_ind_points
                mylist_weight[k]=fine_ind_weights
                mylist_weight[d]=coarse_ind_weights
                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_cf=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts+1)*lev2knots_doubling(pts))]
                QoI_coarse_fine=weights_cf.dot(coarse_values)
                print('cf=',QoI_coarse_fine)




                bias[j]=np.abs(QoI_fine_fine+ QoI_coarse_coarse - QoI_fine_coarse - QoI_coarse_fine   )
                points[j]=lev2knots_doubling(1+pts)
                print(points)
                indices[j]=pts

                j=j+1

            
        QoI_beta=np.zeros(prb.N,dtype=int)
        QoI_beta[d]=1
        QoI_beta[k]=1
        
        z= np.polyfit(indices, np.log(bias), 1)
        fit=np.exp(z[0]*indices)

        
        plt.plot(indices, bias,linewidth=2.0,label=r'$\bar{\beta}=$ %s' % np.array_str(QoI_beta),linestyle = '--', marker=marker[d]) 
        plt.plot(indices, fit,linewidth=2.0,label=r'rate= %s' % z[0], linestyle = '--', marker='o') 
         
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('k',fontsize=14)
        plt.ylabel(r'$\mid \Delta E_{\mathbf{1}+k \bar{\beta}} \mid $',fontsize=14)  
    plt.legend(loc='upper right')
    plt.savefig('./results/mixed_difference_order2_1D_BS_16steps_binary_opt.eps', format='eps', dpi=1000)       
    
    



def mixed_difference_order3_rate_plotting(d,d2):       
    # # feed parameters to the problem
    prb = Problem(1) 
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #d=0
    for k in range(0,5):  
        if (k==d) or (k==d2):
            print('Hello')
            continue
        else:    
            mylist=[]
            mylist_weight=[]
            bias=np.zeros(3)
            points=np.zeros(3)
            indices=np.zeros(3,dtype=int)
            fixed_points=fnKnots(0)[0]
            fixed_weight=fnKnots(0)[1]
            mylist[:prb.N]=[fixed_points for i in range(0,prb.N)]
            mylist_weight[:prb.N]=[fixed_weight for i in range(0,prb.N)]
            j=0
            for pts in range(2,5):
                fine_ind_points=fnKnots(pts)[0]
                fine_ind_weights=fnKnots(pts)[1]
                coarse_ind_points=fnKnots(pts-1)[0]
                coarse_ind_weights=fnKnots(pts-1)[1]

                #block1
                mylist[d]=fine_ind_points
                mylist[d2]=fine_ind_points
                mylist[k]=fine_ind_points

                mylist_weight[d]=fine_ind_weights
                mylist_weight[d2]=fine_ind_weights
                mylist_weight[k]=fine_ind_weights

                fine_points=cartesian(mylist)
                #print(fine_points)
                fine_weights=cartesian(mylist_weight)
                weights_fff=np.asarray([np.prod(fine_weights[i]) for i in range (0,len(fine_weights))])
                fine_values=[prb.SolveFor(fine_points[i]) for i in range(0,(lev2knots_doubling(1+pts)*lev2knots_doubling(1+pts)*lev2knots_doubling(1+pts)))]
                QoI_fff=weights_fff.dot(fine_values)
                print('fff=',QoI_fff)
            
                
                #block2
                mylist[d]=coarse_ind_points
                mylist[d2]=coarse_ind_points
                mylist[k]=coarse_ind_points

                mylist_weight[d]=coarse_ind_weights
                mylist_weight[d2]=coarse_ind_weights
                mylist_weight[k]=coarse_ind_weights

                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_ccc=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,(lev2knots_doubling(pts)*lev2knots_doubling(pts)*lev2knots_doubling(pts)))]
                QoI_ccc=weights_ccc.dot(coarse_values)
                print('ccc=',QoI_ccc)

                 #block3

            
                mylist[d]=fine_ind_points
                mylist[k]=coarse_ind_points
                mylist[d2]=coarse_ind_points

                mylist_weight[d]=fine_ind_weights
                mylist_weight[k]=coarse_ind_weights
                mylist_weight[d2]=coarse_ind_weights

                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_fcc=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts+1)*lev2knots_doubling(pts)*lev2knots_doubling(pts))]
                QoI_fcc=weights_fcc.dot(coarse_values)
                print('fcc=',QoI_fcc)


                #block4

            
                mylist[k]=fine_ind_points
                mylist[d2]=fine_ind_points
                mylist[d]=coarse_ind_points

                mylist_weight[k]=fine_ind_weights
                mylist_weight[d2]=fine_ind_weights
                mylist_weight[d]=coarse_ind_weights

                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_cff=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts+1)*lev2knots_doubling(pts+1)*lev2knots_doubling(pts))]
                QoI_cff=weights_cff.dot(coarse_values)
                print('cff=',QoI_cff)


                

                #block5

            
                mylist[d]=fine_ind_points
                mylist[k]=coarse_ind_points
                mylist[d2]=fine_ind_points

                mylist_weight[d]=fine_ind_weights
                mylist_weight[k]=coarse_ind_weights
                mylist_weight[d2]=fine_ind_weights

                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_fcf=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts+1)*lev2knots_doubling(pts+1)*lev2knots_doubling(pts))]
                QoI_fcf=weights_fcf.dot(coarse_values)
                print('fcf=',QoI_fcf)


                #block6

            
                mylist[k]=fine_ind_points
                mylist[d2]=coarse_ind_points
                mylist[d]=coarse_ind_points

                mylist_weight[k]=fine_ind_weights
                mylist_weight[d2]=coarse_ind_weights
                mylist_weight[d]=coarse_ind_weights

                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_cfc=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts+1)*lev2knots_doubling(pts)*lev2knots_doubling(pts))]
                QoI_cfc=weights_cfc.dot(coarse_values)
                print('cfc=',QoI_cfc)

                

                #block7

            
                mylist[d]=coarse_ind_points
                mylist[k]=coarse_ind_points
                mylist[d2]=fine_ind_points

                mylist_weight[d]=coarse_ind_weights
                mylist_weight[k]=coarse_ind_weights
                mylist_weight[d2]=fine_ind_weights

                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_ccf=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts+1)*lev2knots_doubling(pts)*lev2knots_doubling(pts))]
                QoI_ccf=weights_ccf.dot(coarse_values)
                print('ccf=',QoI_ccf)


                #block8

            
                mylist[k]=fine_ind_points
                mylist[d2]=coarse_ind_points
                mylist[d]=fine_ind_points

                mylist_weight[k]=fine_ind_weights
                mylist_weight[d2]=coarse_ind_weights
                mylist_weight[d]=fine_ind_weights

                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                weights_ffc=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts+1)*lev2knots_doubling(pts+1)*lev2knots_doubling(pts))]
                QoI_ffc=weights_ffc.dot(coarse_values)
                print('ffc=',QoI_ffc)



                bias[j]=np.abs(QoI_fff+ QoI_ccf+QoI_cfc+QoI_fcc   - QoI_ffc-QoI_fcf-QoI_cff - QoI_ccc  )
                points[j]=lev2knots_doubling(1+pts)
                indices[j]=pts

                j=j+1

            
        QoI_beta=np.zeros(prb.N,dtype=int)
        QoI_beta[d]=1
        QoI_beta[k]=1
        QoI_beta[d2]=1
        
        plt.plot(indices, bias,linewidth=2.0,label=r'$\bar{\beta}=$ %s' % np.array_str(QoI_beta),linestyle = '--', marker=marker[d]) 
        #plt.plot(points, 0.001/points,'r',linewidth=2.0,label='order 1') 
        #plt.plot(points, 0.001/(points**2),'g',linewidth=2.0, label='order 2')  
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('k',fontsize=14)
        plt.ylabel(r'$\mid \Delta E_{\mathbf{1}+k \bar{\beta}} \mid $',fontsize=14)  
    plt.legend(loc='upper right')
    plt.savefig('./results/mixed_difference_order3_1D_BS_binary_opt.eps', format='eps', dpi=1000)  
    

first_difference_rate_plotting()
mixed_difference_order2_rate_plotting(0)
#mixed_difference_order3_rate_plotting(0,1)
