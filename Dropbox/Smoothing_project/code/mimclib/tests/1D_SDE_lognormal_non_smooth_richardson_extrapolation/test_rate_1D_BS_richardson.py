#!/usr/bin/env python

import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



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
    #N=4    # number of time steps which will be equal to the number of brownian bridge components (we set is a power of 2)
    N=4    # discretization resolution
    d=None
    dt=None
   
#methods
    # this method initializes the class of basket 
    def __init__(self,coeff):
        self.random_gen = None or np.random
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        #self.sigma=np.random.uniform(0.3,0.4,1)  #volatility
        self.sigma=0.4
        self.dt=self.T/float(self.N) # time steps length
        self.d=int(np.log2(self.N)) #power 2 number steps

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
        beta=10
        deg_leg = 2**3

        n=1
        d = np.array( [[0] * (n + 1)] * (n + 1), float )
     
        #finer level points 
        y_aux_f=y[0:2*self.N-1]

        #coarser level points
        y_aux_c=y_aux_f[0:self.N-1]

        #kink points by newton method
        bar_z_f=self.newtons_method(0,y_aux_f,2*self.N)
        bar_z_c=self.newtons_method(0,y_aux_c,self.N)

        z_min=np.minimum(bar_z_f,bar_z_c)
        z_max=np.maximum(bar_z_f,bar_z_c)
       
        # finer level QoI
        yknots_right_f=np.polynomial.laguerre.laggauss(beta)
        yknots_left_f=yknots_right_f
       
        #left_side finer
        mylist_left_f=[]
        mylist_left_f.append(yknots_left_f[0])
        mylist_left_f[1:]=[np.array(y_aux_f[i]) for i in range(0,len(y_aux_f))]
        points_left_f=self.cartesian(mylist_left_f)
        x_l_f=np.asarray([self.stock_price_trajectory_1D_BS(z_min-points_left_f[i,0],points_left_f[i,1:],2*self.N)[0]  for i in range(0,len(yknots_left_f[0]))])
        QoI_left_f= yknots_left_f[1].dot(self.payoff(x_l_f)*((1/np.sqrt(2 * np.pi)) * np.exp(-((z_min-points_left_f[:,0])**2)/2)* np.exp(points_left_f[:,0])))
        
        #right_side finer
        mylist_right_f=[]
        mylist_right_f.append(yknots_right_f[0])
        mylist_right_f[1:]=[np.array(y_aux_f[i]) for i in range(0,len(y_aux_f))]
        points_right_f=self.cartesian(mylist_right_f)
        x_r_f=np.asarray([self.stock_price_trajectory_1D_BS(points_right_f[i,0]+z_max,points_right_f[i,1:],2*self.N)[0] for i in range(0,len(yknots_right_f[0]))])
        QoI_right_f= yknots_right_f[1].dot(self.payoff(x_r_f)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right_f[:,0]+z_max)**2)/2)* np.exp(points_right_f[:,0]))
        
        # #middle_side finer
        # # Gauss-Legendre (default interval is [-1, 1])
        # yknots_middle = np.polynomial.legendre.leggauss(deg_leg)
        # # Translate x values from the interval [-1, 1] to [a, b]
        # yknots_middle_f= 0.5*(yknots_middle[0] + 1)*(z_max - z_min) + z_min

        # mylist_middle_f=[]
        # mylist_middle_f.append(yknots_middle_f)
        # mylist_middle_f[1:]=[np.array(y_aux_f[i]) for i in range(0,len(y_aux_f))]
        # points_middle_f=self.cartesian(mylist_middle_f)
        # x_mid_f=np.asarray([self.stock_price_trajectory_1D_BS(points_middle_f[i,0],points_middle_f[i,1:],2*self.N)[0] for i in range(0,len(yknots_middle_f))])
        # QoI_middle_f= (yknots_middle[1].dot(self.payoff(x_mid_f))) * 0.5*(z_max - z_min)
        # # final finer QoI
        #d[1,0] =QoI_left_f+QoI_right_f+QoI_middle_f
        d[1,0] =QoI_left_f+QoI_right_f


       # coarser level
        yknots_right_c=yknots_right_f
        yknots_left_c=yknots_left_f
        
        #left_side coarser
        mylist_left_c=[]
        mylist_left_c.append(yknots_left_c[0])
        mylist_left_c[1:]=[np.array(y_aux_c[i]) for i in range(0,len(y_aux_c))]
        points_left_c=self.cartesian(mylist_left_c)
        x_l_c=np.asarray([self.stock_price_trajectory_1D_BS(z_min-points_left_c[i,0],points_left_c[i,1:],self.N)[0]  for i in range(0,len(yknots_left_c[0]))])
        QoI_left_c= yknots_left_c[1].dot(self.payoff(x_l_c)*((1/np.sqrt(2 * np.pi)) * np.exp(-((z_min-points_left_c[:,0])**2)/2)* np.exp(points_left_c[:,0])))
        
        #right_side coarser
        mylist_right_c=[]
        mylist_right_c.append(yknots_right_c[0])
        mylist_right_c[1:]=[np.array(y_aux_c[i]) for i in range(0,len(y_aux_c))]
        points_right_c=self.cartesian(mylist_right_c)
        
        #option 1
        #x_r_c=np.asarray([self.stock_price_trajectory_1D_BS(((points_right_c[i,0]+y_aux_f[-1])/np.sqrt(2))+z_max,points_right_c[i,1:],self.N)[0] for i in range(0,len(yknots_right_c[0]))])
        #QoI_right_c= yknots_right_c[1].dot(self.payoff(x_r_c)*(1/np.sqrt(2 * np.pi)) * np.exp(-((((points_right_c[i,0]+y_aux_f[-1])/np.sqrt(2))+z_max)**2)/2)* np.exp(((points_right_c[i,0]+y_aux_f[-1])/np.sqrt(2))))
        #option 2
        x_r_c=np.asarray([self.stock_price_trajectory_1D_BS(points_right_c[i,0]+z_max,points_right_c[i,1:],self.N)[0] for i in range(0,len(yknots_right_c[0]))])
        QoI_right_c= yknots_right_c[1].dot(self.payoff(x_r_c)*(1/np.sqrt(2 * np.pi)) * np.exp(-(( points_right_c[:,0]+z_max)**2)/2)* np.exp(points_right_c[:,0]))
              
        # #middle_side coarser
        # # Gauss-Legendre (default interval is [-1, 1])
        # yknots_middle_c = yknots_middle_f
      
        # mylist_middle_c=[]
        # mylist_middle_c.append(yknots_middle_c)
        # mylist_middle_c[1:]=[np.array(y_aux_c[i]) for i in range(0,len(y_aux_c))]
        # points_middle_c=self.cartesian(mylist_middle_c)

        # #option 1
        # #x_mid_c=np.asarray([self.stock_price_trajectory_1D_BS(((points_middle_c[i,0]+y_aux_f[-1])/np.sqrt(2)) ,points_middle_c[i,1:],self.N)[0] for i in range(0,len(yknots_middle_c))])
        # #option 2
        # x_mid_c=np.asarray([self.stock_price_trajectory_1D_BS(points_middle_c[i,0] ,points_middle_c[i,1:],self.N)[0] for i in range(0,len(yknots_middle_c))])
        # QoI_middle_c= (yknots_middle[1].dot(self.payoff(x_mid_c))) * 0.5*(z_max - z_min)
        # # final coarser QoI
        #d[0,0] =QoI_left_c+QoI_right_c+QoI_middle_c
        d[0,0] =QoI_left_c+QoI_right_c

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

        dbb=dW-(dt_s/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        return X[-1],dbb
        
    # this function defines the payoff function used here
    def payoff(self,x): 
       
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
        mylist_weights=[]
        bias=np.zeros(4)
        points=np.zeros(4)
        indices=np.zeros(4,dtype=int)
        fixed_points=fnKnots(0)[0]
        fixed_weights=fnKnots(0)[1]
        mylist[:2*prb.N-1]=[fixed_points for i in range(0,2*prb.N-1)]
        mylist_weights[:2*prb.N-1]=[fixed_weights for i in range(0,2*prb.N-1)]
        j=0
        for pts in range(2,6):
            fine_ind_points=fnKnots(pts)[0]
            fine_ind_weights=fnKnots(pts)[1]
            mylist[d1]=fine_ind_points
            mylist_weights[d1]=fine_ind_weights

            fine_points=cartesian(mylist)
            fine_weights=cartesian(mylist_weights)

            fine_weights_prod=np.asarray([np.prod(fine_weights[i]) for i in range (0,len(fine_weights))])
            
            fine_values=[prb.SolveFor(fine_points[i]) for i in range(0,len(fine_points))]
            
            QoI_fine=fine_weights_prod.dot(fine_values)
            print('f=',QoI_fine)
            

            coarse_ind_points=fnKnots(pts-1)[0]
            coarse_ind_weights=fnKnots(pts-1)[1]
            mylist[d1]=coarse_ind_points
            mylist_weights[d1]=coarse_ind_weights
            coarse_points=cartesian(mylist)
            coarse_weights=cartesian(mylist_weights)
            coarse_weights_prod=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])

            coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,len(coarse_points))]
            QoI_coarse=coarse_weights_prod.dot(coarse_values)
            print('c=',QoI_coarse)

            bias[j]=np.abs(QoI_fine-QoI_coarse)
            points[j]=lev2knots_doubling(1+pts)
            indices[j]=pts

            j=j+1


      
        QoI_beta=np.zeros(2*prb.N-1,dtype=int)
        QoI_beta[d1]=1


        z = np.polyfit(indices, np.log(bias), 1)
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
    plt.savefig('./results/first_difference_1D_BS_richardson_4_8_no_middle.eps', format='eps', dpi=1000)





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
            mylist[:2*prb.N-1]=[fixed_points for i in range(0,2*prb.N-1)]
            mylist_weight[:2*prb.N-1]=[fixed_weight for i in range(0,2*prb.N-1)]
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
                fine_values=[prb.SolveFor(fine_points[i]) for i in range(0,len(fine_points))]
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
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,len(coarse_points))]
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
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,len(coarse_points))]
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
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,len(coarse_points))]
                QoI_coarse_fine=weights_cf.dot(coarse_values)
                print('cf=',QoI_coarse_fine)

                bias[j]=np.abs(QoI_fine_fine+ QoI_coarse_coarse - QoI_fine_coarse - QoI_coarse_fine   )
                points[j]=lev2knots_doubling(1+pts)
                print(points)
                indices[j]=pts

                j=j+1


   
    
        
    

        QoI_beta=np.zeros(2*prb.N-1,dtype=int)
        QoI_beta[d]=1
        QoI_beta[k]=1
           


        z = np.polyfit(indices, np.log(bias), 1)
        fit=np.exp(z[0]*indices)       
                
        plt.plot(indices, bias,linewidth=2.0,label=r'$\bar{\beta}=$ %s' % np.array_str(QoI_beta),linestyle = '--', marker=marker[d]) 
        plt.plot(indices, fit,linewidth=2.0,label=r'rate= %s' % z[0], linestyle = '--', marker='o') 
      
         #plt.plot(points, 0.001/points,'r',linewidth=2.0,label='order 1') 
        #plt.plot(points, 0.001/(points**2),'g',linewidth=2.0, label='order 2')  
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('k',fontsize=14)
        plt.ylabel(r'$\mid \Delta E_{\mathbf{1}+k \bar{\beta}} \mid $',fontsize=14)  
    plt.legend(loc='upper right')
    #plt.savefig('./results/mixed_difference_order2_spread_option_cut_off_1_rho_05.eps', format='eps', dpi=1000)     
    plt.savefig('./results/mixed_difference_order2_BS_richardson_4_8_no_middle.eps', format='eps', dpi=1000)       
    
    


def mixed_difference_order2_rate_plotting_surface(d):       
    # # feed parameters to the problem
    prb = Problem(1) 
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    k=1
    mylist=[]
    mylist_weight=[]
    bias=np.zeros((5,5))
    points1=np.zeros(5)
    points2=np.zeros(5)
    indices1=np.zeros(5,dtype=int)
    indices2=np.zeros(5,dtype=int)
    fixed_points=fnKnots(0)[0]
    fixed_weight=fnKnots(0)[1]
    mylist[:2*prb.N-1]=[fixed_points for i in range(0,2*prb.N-1)]
    mylist_weight[:2*prb.N-1]=[fixed_weight for i in range(0,2*prb.N-1)]
    

    for pts1 in range(1,6):
        print(pts1)
        for pts2 in range (1,6): 
            print(pts2)
            if pts1==pts2:
                fine_ind_points=fnKnots(pts1)[0]
                fine_ind_weights=fnKnots(pts1)[1]
                coarse_ind_points=fnKnots(pts1-1)[0]
                coarse_ind_weights=fnKnots(pts1-1)[1]

                #block1
                mylist[d]=fine_ind_points
                mylist[k]=fine_ind_points
                mylist_weight[d]=fine_ind_weights
                mylist_weight[k]=fine_ind_weights
                fine_points=cartesian(mylist)
                fine_weights=cartesian(mylist_weight)

                weights_ff=np.asarray([np.prod(fine_weights[i]) for i in range (0,len(fine_weights))])
                fine_values=[prb.SolveFor(fine_points[i]) for i in range(0,len(fine_points))]
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
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,len(coarse_points))]
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
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,len(coarse_points))]
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
                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,len(coarse_points))]
                QoI_coarse_fine=weights_cf.dot(coarse_values)
                print('cf=',QoI_coarse_fine)

                bias[pts1-1,pts1-1]=np.abs(QoI_fine_fine+ QoI_coarse_coarse - QoI_fine_coarse - QoI_coarse_fine   )
                points1[pts1-1]=lev2knots_doubling(1+pts1)
                points2[pts2-1]=lev2knots_doubling(1+pts2)
                print(points1)
                print(points2)
                indices1[pts1-1]=pts1
                indices1[pts2-1]=pts2

                
            else:

                fine_ind_points_d=fnKnots(pts1)[0]
                fine_ind_weights_d=fnKnots(pts1)[1]
                mylist[d]=fine_ind_points_d
                mylist_weight[d]=fine_ind_weights_d


                fine_ind_points_k=fnKnots(pts2)[0]
                fine_ind_weights_k=fnKnots(pts2)[1]
                mylist[k]=fine_ind_points_k
                mylist_weight[k]=fine_ind_weights_k



                fine_points=cartesian(mylist)
                fine_weights=cartesian(mylist_weight)

                fine_weights_prod=np.asarray([np.prod(fine_weights[i]) for i in range (0,len(fine_weights))])
                
                fine_values=[prb.SolveFor(fine_points[i]) for i in range(0,len(fine_points))]
                
                QoI_fine=fine_weights_prod.dot(fine_values)
                print('f=',QoI_fine)
                

                coarse_ind_points_d=fnKnots(pts1-1)[0]
                coarse_ind_weights_d=fnKnots(pts1-1)[1]
                mylist[d]=coarse_ind_points_d
                mylist_weight[d]=coarse_ind_weights_d



                coarse_ind_points_k=fnKnots(pts2-1)[0]
                coarse_ind_weights_k=fnKnots(pts2-1)[1]
                mylist[k]=coarse_ind_points_k
                mylist_weight[k]=coarse_ind_weights_k





                coarse_points=cartesian(mylist)
                coarse_weights=cartesian(mylist_weight)
                coarse_weights_prod=np.asarray([np.prod(coarse_weights[i]) for i in range (0,len(coarse_weights))])

                coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,len(coarse_points))]
                QoI_coarse=coarse_weights_prod.dot(coarse_values)
                print('c=',QoI_coarse)

                bias[pts1-1,pts2-1]=np.abs(QoI_fine-QoI_coarse)
                points1[pts1-1]=lev2knots_doubling(1+pts1)
                points2[pts2-1]=lev2knots_doubling(1+pts2)
                indices1[pts1-1]=pts1
                indices2[pts1-1]=pts2



           
        
            
        

    
       
    
    print(indices1)
    print(indices2)
    print(bias)
    print(np.log(bias))
    
  

    fig = plt.figure(figsize=(14,6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = Axes3D(fig)

    p = ax.plot_surface(indices1, indices2, np.log(bias), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(p, shrink=0.5,aspect=5)
     # Customize the z axis.
    ax.set_xlim(0,5 )
    ax.set_ylim(0,5 )    
    
    ax.set_xlabel(r'$l_1$',  fontsize=20, fontweight='bold')
    ax.set_ylabel(r'$l_2$', fontsize=20, fontweight='bold')
    ax.set_zlabel(r'$log(\Delta)$', fontsize=20, fontweight='bold')


    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    

     # Set rotation angle to 30 degrees
    ax.view_init(azim=0, elev=90)
    plt.savefig('./results/mixed_difference_order2_BS_richardson_4_8_no_middle_surface_1.eps', format='eps', dpi=1000)  


    fig = plt.figure(figsize=(14,6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = Axes3D(fig)

    p = ax.plot_surface(indices1, indices2, np.log(bias), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(p, shrink=0.5)
     # Customize the z axis.
    ax.set_xlim(0,5 )
    ax.set_ylim(0,5 )    
    
    ax.set_xlabel(r'$l_1$',  fontsize=20, fontweight='bold')
    ax.set_ylabel(r'$l_2$', fontsize=20, fontweight='bold')
    ax.set_zlabel(r'$log(\Delta)$', fontsize=20, fontweight='bold')

     # Set rotation angle to 30 degrees
    ax.view_init(0, 30)
    plt.savefig('./results/mixed_difference_order2_BS_richardson_4_8_no_middle_surface_2.eps', format='eps', dpi=1000)  






    fig = plt.figure(figsize=(14,6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = Axes3D(fig)

    p = ax.plot_surface(indices1, indices2, np.log(bias), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(p, shrink=0.5)
     # Customize the z axis.
    ax.set_xlim(0,5 )
    ax.set_ylim(0,5 )    
    
    ax.set_xlabel(r'$l_1$',  fontsize=20, fontweight='bold')
    ax.set_ylabel(r'$l_2$', fontsize=20, fontweight='bold')
    ax.set_zlabel(r'$log(\Delta)$', fontsize=20, fontweight='bold')

     # Set rotation angle to 30 degrees
    ax.view_init(0, 60)
    plt.savefig('./results/mixed_difference_order2_BS_richardson_4_8_no_middle_surface_3.eps', format='eps', dpi=1000)     





    fig = plt.figure(figsize=(14,6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = Axes3D(fig)

    p = ax.plot_surface(indices1, indices2, np.log(bias), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(p, shrink=0.5)
     # Customize the z axis.
    ax.set_xlim(0,5 )
    ax.set_ylim(0,5 )    
    
    ax.set_xlabel(r'$l_1$',  fontsize=20, fontweight='bold')
    ax.set_ylabel(r'$l_2$', fontsize=20, fontweight='bold')
    ax.set_zlabel(r'$log(\Delta)$', fontsize=20, fontweight='bold')

     # Set rotation angle to 30 degrees
    ax.view_init(0, 90)
    plt.savefig('./results/mixed_difference_order2_BS_richardson_4_8_no_middle_surface_4.eps', format='eps', dpi=1000)       





    fig = plt.figure(figsize=(14,6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = Axes3D(fig)

    p = ax.plot_surface(indices1, indices2, np.log(bias), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(p, shrink=0.5)
     # Customize the z axis.
    ax.set_xlim(0,5 )
    ax.set_ylim(0,5 )    
    
    ax.set_xlabel(r'$l_1$',  fontsize=20, fontweight='bold')
    ax.set_ylabel(r'$l_2$', fontsize=20, fontweight='bold')
    ax.set_zlabel(r'$log(\Delta)$', fontsize=20, fontweight='bold')

     # Set rotation angle to 30 degrees
    ax.view_init(0, 45)
    plt.savefig('./results/mixed_difference_order2_BS_richardson_4_8_no_middle_surface_5.eps', format='eps', dpi=1000) 







    fig = plt.figure(figsize=(14,6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = Axes3D(fig)

    p = ax.plot_surface(indices1, indices2, np.log(bias), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(p, shrink=0.5)
     # Customize the z axis.
    ax.set_xlim(0,5 )
    ax.set_ylim(0,5 )    
    
    ax.set_xlabel(r'$l_1$',  fontsize=20, fontweight='bold')
    ax.set_ylabel(r'$l_2$', fontsize=20, fontweight='bold')
    ax.set_zlabel(r'$log(\Delta)$', fontsize=20, fontweight='bold')

     # Set rotation angle to 30 degrees
    ax.view_init(0, 120)
    plt.savefig('./results/mixed_difference_order2_BS_richardson_4_8_no_middle_surface_5.eps', format='eps', dpi=1000)            







    fig = plt.figure(figsize=(14,6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = Axes3D(fig)

    p = ax.plot_surface(indices1, indices2, np.log(bias), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(p, shrink=0.5)
     # Customize the z axis.
    ax.set_xlim(0,5 )
    ax.set_ylim(0,5 )    
    
    ax.set_xlabel(r'$l_1$',  fontsize=20, fontweight='bold')
    ax.set_ylabel(r'$l_2$', fontsize=20, fontweight='bold')
    ax.set_zlabel(r'$log(\Delta)$', fontsize=20, fontweight='bold')

     # Set rotation angle to 30 degrees
    ax.view_init(0, 150)
    plt.savefig('./results/mixed_difference_order2_BS_richardson_4_8_no_middle_surface_6.eps', format='eps', dpi=1000)    





    fig = plt.figure(figsize=(14,6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = Axes3D(fig)

    p = ax.plot_surface(indices1, indices2, np.log(bias), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(p, shrink=0.5)
     # Customize the z axis.
    ax.set_xlim(0,5 )
    ax.set_ylim(0,5 )    
    
    ax.set_xlabel(r'$l_1$',  fontsize=20, fontweight='bold')
    ax.set_ylabel(r'$l_2$', fontsize=20, fontweight='bold')
    ax.set_zlabel(r'$log(\Delta)$', fontsize=20, fontweight='bold')

     # Set rotation angle to 30 degrees
    ax.view_init(0, 180)
    plt.savefig('./results/mixed_difference_order2_BS_richardson_4_8_no_middle_surface_7.eps', format='eps', dpi=1000)    



    

#first_difference_rate_plotting()

mixed_difference_order2_rate_plotting_surface(0)
#mixed_difference_order3_rate_plotting(0,1)