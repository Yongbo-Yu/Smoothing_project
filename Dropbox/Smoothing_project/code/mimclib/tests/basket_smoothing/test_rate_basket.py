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
    basket_d=6      # number of assets in the basket
    c= (1/float(basket_d))*np.ones(basket_d)     # weigths
    sigma=None    # vector of volatilities
    K=None         # Strike price
    T=1.0                      # maturity
    w=None                    # new weights
    rho=None                  #correlation matrix
    Sigma=None                # Covariance matrix

#methods
    # this method initializes the class of basket 
    def __init__(self,coeff):
        self.random_gen = None or np.random
        #self.S0=np.random.uniform(8,20,self.basket_d) # vector of initial stock prices
        self.S0=20*np.ones(self.basket_d) 
        #self.sigma=np.random.uniform(0.3,0.4,self.basket_d) #vector of volatilities
        self.sigma=0.4*np.ones(self.basket_d) 
        self.K= coeff*self.c.dot(self.S0)                           # Strike price and coeff determine if we have in/at/out the money option
        self.w=self.c*self.S0* np.exp(-0.5*pow(self.sigma,2)*self.T)    #new weights
        #self.rho=self.correlation()                                                  #correlation matrx
        #self.rho=np.array([[1 , 0.3,  0.3],[ 0.3,  1 , 0.3],[0.3 , 0.3 ,1,0.3,  0.3] ,[0.3 , 0.3 ,0.3, 1, 0.3] ,[0.3 , 0.3 ,0.3,  0.3,1] ])
    
        from scipy.linalg import toeplitz
        self.rho=toeplitz([1,0.3,0.3,0.3,0.3,0.3])
        self.Sigma=np.zeros((self.basket_d,self.basket_d))
        for i in range(0,self.basket_d):
                for j in range(i,self.basket_d):
                      self.Sigma[i,j]=self.sigma[i]*self.sigma[j]*self.rho[i,j]*self.T
        self.Sigma=self.Sigma+np.transpose(self.Sigma)-np.diag(np.diag(self.Sigma))
       
    
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

    # objfun
    def objfun(self,y):
        start_time=time.time();
        lambda_vect,V=self.factorization()
        A=np.diag(np.sqrt(lambda_vect[1:]))
        z=A.dot(y) 
        QoI=self.Call_BS(self.h(z,V,self.w)*np.exp(((lambda_vect[0])/2)),self.K,np.sqrt(lambda_vect[0])) 
        #QoI=self.Call_BS(self.h(z,V,self.w)*np.exp(((lambda_vect[0])/2)),self.K,np.sqrt(lambda_vect[0])) *np.exp(-y.dot(y)/2)*(1/np.sqrt( 2*np.pi))
        elapsed_time_qoi=time.time()-start_time;
        self.elapsed_time=self.elapsed_time+elapsed_time_qoi;
        return QoI

    # this function computes the price of European call option BS model
    def Call_BS(self,S0,K,sigma):
        #here we assume that maturity T=1 and interest rate r=0
        from scipy.stats import norm
        d1=(1/sigma)*(np.log(S0/K)+((sigma**2)/2))
        d2=(1/sigma)*(np.log(S0/K)-((sigma**2)/2))
        price=S0*norm.cdf(d1)-K*norm.cdf(d2)
        return price
    
    # This function computes the transformation operation
    def h(self,y_bar, V,w):
        V_1=np.array(V[:,1:])
        
        transformed_result=np.exp(np.transpose(V_1.dot(y_bar))).dot(w)
        return transformed_result    

# this provides the lammdas and the V but still to be fixed
    def factorization(self):
        from numpy import linalg as LA
        bold_1=np.ones(self.basket_d)
        v=LA.inv(self.Sigma).dot(bold_1)
        tilde_Sigma=self.Sigma- (np.outer(bold_1,bold_1)/float(bold_1.dot(v))) 
        lambda_vect=np.zeros(self.basket_d)
        V=np.zeros((self.basket_d,self.basket_d))
        tilde_lambda_vect, tilde_V = LA.eig(tilde_Sigma)
        idx = tilde_lambda_vect.argsort()[::-1]   
        tilde_lambda_vect = tilde_lambda_vect[idx]
        tilde_V = tilde_V[:,idx]
        lambda_vect[0]=1/float(bold_1.dot(v))
        #print (lambda_vect[0])
        lambda_vect[1:]=tilde_lambda_vect[:-1]
        V[:,0]=bold_1
        V[:,1:]=tilde_V[:,:-1]
        return lambda_vect,V

   # this function generates the correlation matrix rho ( think it is Ok but as I increase d the elements in the diagonal are not exactly 1, I need to check that more)
    def correlation(self):
        x = np.random.uniform(0.8,1,self.basket_d-1)
        cp=[np.prod(x[:i]) for i in range(1,self.basket_d) ]
        diag_terms=np.zeros(self.basket_d)
        diag_terms[0]=1
        diag_terms[1:]=[np.sqrt(1-(x[i]**2)) for i in range(0,self.basket_d-1)]
        tau=np.eye(self.basket_d)
        for i in range(0,self.basket_d):
                tau[i+1:,i]=cp[:self.basket_d-1-i]
        tau=tau.dot(np.diag(diag_terms))
        rho=tau.dot(np.transpose(tau))
        return rho
       

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

    #def GetDim(self):
     #   return 3

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




def knots_CC(nn, x_a, x_b, whichrho='prob'):
    # [x,w] = KNOTS_CC(nn,x_a,x_b)
    #
    # calculates the collocation points (x)
    # and the weights (w) for the Clenshaw-Curtis integration formula
    # w.r.t to the weight function rho(x)=1/(b-a)
    # i.e. the density of a uniform random variable
    # with range going from x=a to x=b.
    #
    # [x,w] = KNOTS_CC(nn,x_a,x_b,'prob')
    #
    # is the same as [x,w] = KNOTS_CC(nn,x_a,x_b) above
    #
    # [x,w]=[x,w] = KNOTS_CC(nn,x_a,x_b,'nonprob')
    #
    # calculates the collocation points (x)
    # and the weights (w) for the Clenshaw-Curtis integration formula
    # w.r.t to the weight function rho(x)=1
    # ----------------------------------------------------
    # Sparse Grid Matlab Kit
    # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
    # See LICENSE.txt for license
    # ----------------------------------------------------

    if nn == 1:
        x = np.array([(x_a+x_b)/2.])
        wt = np.array([1])
    elif nn % 2 == 0:
        raise Exception('error in knots_CC: Clenshaw-Curtis formula \n \
use only odd number of points')
    else:
        n = nn-1
        N = np.arange(1, n, 2)
        l = end_N = N.shape[0]
        m = n-l
        v0 = np.concatenate((2./N/(N-2.), [1./N[end_N-1]], np.zeros(m)))
        end_v0 = v0.shape[0]
        v2 = -v0[0:end_v0-1] - v0[end_v0-1:0:-1]

        g0 = -np.ones(n)
        g0[l] = g0[l]+n
        g0[m] = g0[m]+n
        g = g0/(n**2 - 1 + n % 2)

        wcc = np.real(np.fft.ifft(v2+g))
        wt = np.concatenate((wcc, [wcc[0]])) / 2.

        x = np.cos(np.arange(0, n+1) * np.pi / n)
        x = ((x_b-x_a)/2.)*x + (x_a+x_b)/2.

    if whichrho == 'nonprob':
        w = (x_b-x_a)*wt
    elif whichrho == 'prob':
        w = wt
    else:
        raise Exception('4th input not recognized')
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
#fnKnots= lambda beta: knots_CC(lev2knots_doubling(1+beta), -(10**3), (10**3), whichrho='nonprob')
def first_difference_rate_plotting():       
    # # feed parameters to the problem
    prb = Problem(1) 
    
    marker=['>', 'v', '^', 'o', '*']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for d in range(0,prb.basket_d-1):
        mylist=[]
        bias=np.zeros(4)
        points=np.zeros(4)
        indices=np.zeros(4,dtype=int)
        fixed_points=fnKnots(0)[0]
        mylist[:d-1]=[fixed_points for i in range(0,d)]
        mylist[d+1:prb.basket_d-1]=[fixed_points for i in range(d+1,prb.basket_d)]
        j=0
        for pts in range(2,6):
            fine_ind_points=fnKnots(pts)[0]
            print(fine_ind_points)
            fine_ind_weights=fnKnots(pts)[1]
            
            mylist[d]=fine_ind_points
            fine_points=cartesian(mylist)
            fine_values=[prb.SolveFor(fine_points[i]) for i in range(0,lev2knots_doubling(1+pts))]
            QoI_fine=fine_ind_weights.dot(fine_values)
            print('f=',QoI_fine)
            

            coarse_ind_points=fnKnots(pts-1)[0]
            coarse_ind_weights=fnKnots(pts-1)[1]
            mylist[d]=coarse_ind_points
            coarse_points=cartesian(mylist)
            coarse_values=[prb.SolveFor(coarse_points[i]) for i in range(0,lev2knots_doubling(pts))]
            QoI_coarse=coarse_ind_weights.dot(coarse_values)
            print('c=',QoI_coarse)
            bias[j]=np.abs(QoI_fine-QoI_coarse)
            points[j]=lev2knots_doubling(1+pts)
            indices[j]=pts

            j=j+1

      
        QoI_beta=np.zeros(prb.basket_d-1,dtype=int)
        QoI_beta[d]=1
        
        plt.plot(indices, bias,linewidth=2.0,label=r'$\bar{\beta}=$ %s' % np.array_str(QoI_beta),linestyle = '--', marker=marker[d]) 
        #plt.plot(points, 0.001/points,'r',linewidth=2.0,label='order 1') 
        #plt.plot(points, 0.001/(points**2),'g',linewidth=2.0, label='order 2')  
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('k',fontsize=14)
        plt.ylabel(r'$\mid \Delta E_{\mathbf{1}+k \bar{\beta}} \mid $',fontsize=14)  
    plt.legend(loc='upper right')
    plt.savefig('./results/first_difference_basket.eps', format='eps', dpi=1000)

def mixed_difference_order2_rate_plotting(d):       
    # # feed parameters to the problem
    prb = Problem(1) 
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #d=0
    for k in range(0,prb.basket_d-1):  
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
            mylist[:d-1]=[fixed_points for i in range(0,d) ]
            mylist[d+1:prb.basket_d-1]=[fixed_points for i in range(d+1,prb.basket_d)]
            mylist_weight[:d-1]=[fixed_weight for i in range(0,d) ]
            mylist_weight[d+1:prb.basket_d-1]=[fixed_weight for i in range(d+1,prb.basket_d) ]
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
                indices[j]=pts

                j=j+1

            
        QoI_beta=np.zeros(prb.basket_d-1,dtype=int)
        QoI_beta[d]=1
        QoI_beta[k]=1
        
        plt.plot(indices, bias,linewidth=2.0,label=r'$\bar{\beta}=$ %s' % np.array_str(QoI_beta),linestyle = '--', marker=marker[d]) 
        #plt.plot(points, 0.001/points,'r',linewidth=2.0,label='order 1') 
        #plt.plot(points, 0.001/(points**2),'g',linewidth=2.0, label='order 2')  
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('k',fontsize=14)
        plt.ylabel(r'$\mid \Delta E_{\mathbf{1}+k \bar{\beta}} \mid $',fontsize=14)  
    plt.legend(loc='upper right')
    plt.savefig('./results/mixed_difference_order2_basket.eps', format='eps', dpi=1000)    
def mixed_difference_order3_rate_plotting(d,d2):       
    # # feed parameters to the problem
    prb = Problem(1) 
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #d=0
    for k in range(0,prb.basket_d-1):  
        if (k==d) or (k==d2):
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
            mylist=[fixed_points for i in range(0,prb.basket_d-1) ]
            mylist_weight=[fixed_weight for i in range(0,prb.basket_d-1) ]
            j=0
            for pts in range(2,6):
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

            
        QoI_beta=np.zeros(prb.basket_d-1,dtype=int)
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
    plt.savefig('./results/mixed_difference_order3_basket.eps', format='eps', dpi=1000)        
first_difference_rate_plotting()  
#mixed_difference_order2_rate_plotting(2)
#mixed_difference_order3_rate_plotting(0,2)
# plt.figure()
# plt.plot(points, elapsed_time,'bs',linewidth=2.0, label='runnig time') 
# plt.plot(points, points,'r',linewidth=2.0,label='order 1') 
# plt.yscale('log')
# plt.xscale('log')
# plt.legend(loc='upper right')
# plt.xlabel('Number of quadrature points',fontsize=14)
# plt.ylabel('running_time',fontsize=14)  
# plt.savefig('./results/work_basket.eps', format='eps', dpi=1000)