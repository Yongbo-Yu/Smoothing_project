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


import scipy.stats as ss
import fftw3
import RBergomi
from RBergomi import *





class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    N=2 # Number of time steps N, discretization resolution
  
    # for the values of below paramters, we need to see the paper as well check with Christian 
    x=0.235**2;   # this will provide the set of xi parameter values 
    HIn=Vector(1)    # this will provide the set of H parameter values
    HIn[0]=0.43
    e=Vector(1)    # This will provide the set of eta paramter values
    e[0]=1.9
    r=Vector(1)   # this will provide the set of rho paramter values
    r[0]=-0.9
    T=Vector(1)     # this will provide the set of T(time to maturity) parameter value
    T[0]=1.0
    k=Vector(1)     # this will provide the set of K (strike ) paramter value
    #k[0]=np.exp(-4)
    k[0]=1.2
    #y1perp = Vector(N)
    MIn=1        # number of samples M (I think we do not need this paramter here by default in our case it should be =1)

  
#methods
    # this method initializes 
    def __init__(self,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
         #Here we need to use the C++ code to compute the payoff 
        self.z=RBergomi.RBergomiST( self.x,  self.HIn, self.e,  self.r,  self.T, self.k,  self.N, self.MIn)

        self.dt=self.T[0]/float(self.N) # time steps length
        self.d=int(np.log2(self.N)) #power 2 number steps
       # for i in range(0,self.N):
        #    self.y1perp[i] = np.random.normal(loc=0.0, scale=1.0)

       

      
        

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
        goal=self.objfun(Y);
        return goal


     # objfun: 
    def objfun(self,y):
        
        start_time=time.time()
        y1=y[0:self.N]
        y1perp=y[self.N:2*self.N] # second way
        


        # bb1=self.brownian_increments(y1)
        # W1= [bb1[i+1]-bb1[i] for i in range(0,len(bb1)-1)] 


        # bb1perp=self.brownian_increments(y1perp)
        # W1perp= [bb1perp[i+1]-bb1perp[i] for i in range(0,len(bb1perp)-1)] 



        #QoI=self.z.ComputePayoffRT_single(y1,self.y1perp); # this is the computed payoff
        QoI=self.z.ComputePayoffRT_single(y1,y1perp); #second

        elapsed_time_qoi=time.time()-start_time;
        self.elapsed_time=self.elapsed_time+elapsed_time_qoi

        return QoI

    


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


fnKnots= lambda beta: knots_gaussian(lev2knots_doubling(1+beta),   0, 1.0)  
prb = Problem() 
pts=4
fixed_points=fnKnots(pts)[0]
fixed_weight=fnKnots(pts)[1]
mylist=[]
mylist_weight=[]
mylist[:2*prb.N]=[fixed_points for i in range(0,2*prb.N)]# second way
mylist_weight[:2*prb.N]=[fixed_weight for i in range(0,2*prb.N)] #second way

points=cartesian(mylist)
#print(points)
weights=cartesian(mylist_weight)
#print(weights)

weights_f=np.asarray([np.prod(weights[i]) for i in range (0,len(weights))])
#print weights_f
values_f=[prb.SolveFor(points[i]) for i in range(0,(lev2knots_doubling(1+pts)*lev2knots_doubling(1+pts)*lev2knots_doubling(1+pts)*lev2knots_doubling(1+pts)))]
#print values_f
QoI=weights_f.dot(values_f)
print(QoI )

