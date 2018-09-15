#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm, ticker, colors
from matplotlib.ticker import MaxNLocator


import time
import scipy.stats as ss

import fftw3
import RBergomi
from RBergomi import *
import mimclib.misc as misc
import numdifftools as nd


class Problem_measure_change(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    N=4 # Number of time steps N, discretization resolution
  
    # for the values of below paramters, we need to see the paper as well check with Christian 
    x=0.235**2;   # this will provide the set of xi parameter values 
    #x=0.1;
    HIn=Vector(1)    # this will provide the set of H parameter values
    HIn[0]=0.43
    #HIn[0]=0.02
    e=Vector(1)    # This will provide the set of eta paramter values
    e[0]=1.9
    #e[0]=0.4
    r=Vector(1)   # this will provide the set of rho paramter values
    r[0]=-0.9
    #r[0]=-0.7
    T=Vector(1)     # this will provide the set of T(time to maturity) parameter value
    T[0]=1.0
    k=Vector(1)     # this will provide the set of K (strike ) paramter value
    #k[0]=1
    k[0]=1
   # y1perp = Vector(N)
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

        


    

    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y):
        Y = np.array(Y)
        goal=self.objfun(self.nelem,Y);
        return goal

     

    def fun_1D(self,y11):
        y12=np.zeros_like(y11)
    
        #y=np.array([y12,y12,y13,y14,y21,y22,y23,y11])   # 4 steps
        # y22=np.zeros_like(y11)
        # y21=np.zeros_like(y11)
        # y12=np.zeros_like(y11)
        # y=np.array([y12,y12,y11,y12])  # 2 steps

        y=np.array([y12,y12,y12,y12,y12,y12,y12,y11,y12,y12,y12,y12,y12,y12,y12,y12])   # 8 steps
        #hierarchical
        yperp_1=y[self.N+1:2*self.N]
        yperp1=y[self.N]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y[1:self.N]
        y1=y[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        
       
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y))) #  computed payoff including weights
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp)) #   computed payoff without weights
        return QoI    



    def fun(self,y11,y12):
        # y21=np.zeros_like(y11)
        # y22=np.zeros_like(y11)
        y13=np.zeros_like(y11)
        y14=np.zeros_like(y11)
        y21=np.zeros_like(y11)
        y22=np.zeros_like(y11)
        y23=np.zeros_like(y11)
        y24=np.zeros_like(y11)
        #y=np.array([y11,y12,y21,y22])
        y=np.array([y11,y12,y13,y14,y21,y22,y23,y24]) 


        #hierarchical
        yperp_1=y[self.N+1:2*self.N]
        yperp1=y[self.N]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y[1:self.N]
        y1=y[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y))) #  computed payoff including weights
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp)) #   computed payoff without weights
       
        return QoI    
         
  

    

    def brownian_increments(self,y1,y):
        t=np.linspace(0, self.T, self.N+1)     
        h=self.N
        j_max=1
        bb= np.zeros(self.N+1)
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


#1D_plots

# prb=Problem_measure_change()
# x1= np.linspace(-10, 10, 100)
# Z=np.zeros((100))

# for i1 in range(0,100):
#         Z[i1]=prb.fun_1D(x1[i1])

# print Z         
# print(np.max(Z))         
# fig = plt.figure(figsize=(14,6))

# plt.plot(x1,Z)

# plt.xlabel(r'$W_1^8$',  fontsize=20, fontweight='bold')
# plt.ylabel(r'$I(W_1^8)$', fontsize=20, fontweight='bold')

# plt.savefig('./results/Bergomi_integrand_K_1_H_007_W18_N_8.pdf', format='pdf', dpi=600) 


# #2D_plots
prb=Problem_measure_change()
x1= np.linspace(-20, 20, 100)
x2= np.linspace(-20,20, 100)
# x3= np.linspace(-10000, 10000, 5)
# x4= np.linspace(-10000, 10000, 5)
# X1,X2,X3,X4 = np.meshgrid(x1,x2,x3,x4)

X1,X2 = np.meshgrid(x1,x2)

#Z=np.zeros((5,5,5,5))
Z=np.zeros((100,100))
# for i1 in range(0,5):
#     for i2 in range(0,5):
#     	for i3 in range(0,5):
#     		for i4 in range(0,5):
#         		Z[i1,i2,i3,i4]=prb.fun(x1[i1],x2[i2],x3[i3],x4[i4])
for i1 in range(0,100):
    for i2 in range(0,100):
        Z[i1,i2]=prb.fun(x1[i1],x2[i2])
         
         

fig = plt.figure(figsize=(14,6))



print Z
print np.amax(Z)
from numpy import unravel_index
print (unravel_index(Z.argmax(), Z.shape))

print(x1[0])

print(x2[99])
print prb.fun(x1[0],x2[99])


#print np.argmax(Z, axis=0, axis=1)






#plotting contours
# cp = plt.contourf(X1, X2, Z)
# plt.colorbar(cp)
# plt.xlabel(r'$W_1^1$',  fontsize=20, fontweight='bold')
# plt.ylabel(r'$W_1^2$', fontsize=20, fontweight='bold')



#plotting the surface
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax = Axes3D(fig)
p = ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)
# Get current rotation angle
print ax.azim


ax.set_zlabel(r'$I(W_1^1,W_1^2)$', fontsize=20, fontweight='bold')
ax.set_xlabel(r'$W_1^1$',  fontsize=20, fontweight='bold')
ax.set_ylabel(r'$W_1^2$', fontsize=20, fontweight='bold')
# ax.set_zlabel(r'$I(W_2^3,W_2^4)$', fontsize=20, fontweight='bold')

# Set rotation angle to 30 degrees
#ax.view_init(60, 60)


for ii in xrange(0,360,40):
        ax.view_init(elev=10., azim=ii)
        plt.savefig("./results/Bergomi_integrand_contours_K_1_H_043_W1_1_2_N_4_without_weights_2_%d.pdf" % ii, dpi=600)
#plt.savefig('./results/Bergomi_integrand_contours_K_1_H_007_W1_1_2_N_4_without_weights_5.pdf', format='pdf', dpi=600)    






# def knots_gaussian(n, mi, sigma):
#     # [x,w]=KNOTS_GAUSSIAN(n,mi,sigma)
#     #
#     # calculates the collocation points (x)
#     # and the weights (w) for the gaussian integration
#     # w.r.t to the weight function
#     # rho(x)=1/sqrt(2*pi*sigma) *exp( -(x-mi)^2 / (2*sigma^2) )
#     # i.e. the density of a gaussian random variable
#     # with mean mi and standard deviation sigma
#     # ----------------------------------------------------
#     # Sparse Grid Matlab Kit
#     # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
#     # See LICENSE.txt for license
#     # ----------------------------------------------------
#     if n == 1:
#         # the point (traslated if needed)
#         # the weight is 1:
#         return [mi], [1]

#     def coefherm(n):
#         if n <= 1:
#             raise Exception(' n must be > 1 ')
#         a = np.zeros(n)
#         b = np.zeros(n)
#         b[0] = np.sqrt(np.pi)
#         b[1:] = 0.5 * np.arange(1, n)
#         return a, b

#     # calculates the values of the recursive relation
    
#     a, b = coefherm(n)
    
#     # builds the matrix
    
#     JacM = np.diag(a)+np.diag(np.sqrt(b[1:n[0]]), 1)+np.diag(np.sqrt(b[1:n[0]]), -1)
#     # calculates points and weights from eigenvalues / eigenvectors of JacM
#     [x, W] = np.linalg.eig(JacM)
#     w = W[0, :]**2.
#     ind = np.argsort(x)
#     x = x[ind]
#     w = w[ind]
#     # modifies points according to mi, sigma (the weigths are unaffected)
#     x = mi + np.sqrt(2) * sigma * x
#     return x, w    

# # this method gives the number of points of the quadrature given the degree
# def lev2knots_doubling(i):
#     # m = lev2knots_doubling(i)
#     #
#     # relation level / number of points:
#     #    m = 2^{i-1}+1, for i>1
#     #    m=1            for i=1
#     #    m=0            for i=0
#     #
#     # i.e. m(i)=2*m(i-1)-1
#     # ----------------------------------------------------
#     # Sparse Grid Matlab Kit
#     # Copyright (c) 2009-2014 L. Tamellini, F. Nobile
#     # See LICENSE.txt for license
#     # ----------------------------------------------------
#     i = np.array([i] if np.isscalar(i) else i, dtype=np.int)
    
#     m = 2 ** (i-1)+1
#     m[i==1] = 1
#     m[i==0] = 0
#     return m

# def cartesian(arrays, out=None):
#     """
#     Generate a cartesian product of input arrays.

#     Parameters
#     ----------
#     arrays : list of array-like
#         1-D arrays to form the cartesian product of.
#     out : ndarray
#         Array to place the cartesian product in.

#     Returns
#     -------
#     out : ndarray
#         2-D array of shape (M, len(arrays)) containing cartesian products
#         formed of input arrays.

#     Examples
#     --------
#     >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
#     array([[1, 4, 6],
#            [1, 4, 7],
#            [1, 5, 6],
#            [1, 5, 7],
#            [2, 4, 6],
#            [2, 4, 7],
#            [2, 5, 6],
#            [2, 5, 7],
#            [3, 4, 6],
#            [3, 4, 7],
#            [3, 5, 6],
#            [3, 5, 7]])

#     """

#     arrays = [np.asarray(x) for x in arrays]
#     dtype = float

#     n = np.prod([x.size for x in arrays])
#     if out is None:
#         out = np.zeros([n, len(arrays)], dtype=dtype)

#     m = n / arrays[0].size
#     out[:,0] = np.repeat(arrays[0], m)
#     if arrays[1:]:
#         cartesian(arrays[1:], out=out[0:m,1:])
#         for j in xrange(1, arrays[0].size):
#             out[j*m:(j+1)*m,1:] = out[0:m,1:]
#     return out   




     
# fnKnots= lambda beta: knots_gaussian(lev2knots_doubling(1+beta),   0, 1.0)  
# def order2_pts_integrand_isolines_plotting():       
#     # # feed parameters to the problem
#     prb = Problem_measure_change() 
#     marker=['>', 'v', '^', 'o', '*','+','-',':']
#     ax = figure().gca()
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     d=0
#     k=1  
#     mylist=[]
#     mylist_weight=[]
#     fixed_points=fnKnots(5)[0]
#     fixed_weights=fnKnots(5)[1]
#     mylist[:2]=[fixed_points for i in range(0,2)]
#     mylist_weight[:2]=[fixed_weights for i in range(0,2)]
#     bias=np.zeros(6)
#     points=np.zeros(6)
#     indices=np.zeros(6,dtype=int)
#     j=0
#     for pts in range(5,6):
#         fine_ind_points=fnKnots(pts)[0]
#         print fine_ind_points
#         #block1
#         mylist[d]=fine_ind_points
#         mylist[k]=fine_ind_points
        
#         fine_points=cartesian(mylist)
#         print 'hi'
#         print fine_points
        
            
#         plt.plot(fine_points[:,0],fine_points[:,1],'*',color='green',label='herm pts')
    
#         plt.legend(loc='upper right')

#     x= np.linspace(5, 15, 10**2)
#     y= np.linspace(-5, 5, 10**2)
#     X,Y = np.meshgrid(x, y)
#     Z=np.zeros((10**2,10**2))          
#     for j in range(0,10**2):
#         for i in range(0,10**2):
#            Z[i,j]=prb.fun(x[j],y[i])
#            #print Z[i,j]         

    
#     cp = plt.contourf(X, Y, Z)
#     plt.colorbar(cp)
#     plt.xlabel(r'$W_1^1$',  fontsize=20, fontweight='bold')
#     plt.ylabel(r'$W_1^2$', fontsize=20, fontweight='bold')
#     plt.savefig('./results/gauss_herm_pts_full_isolines_rBergomi_K_1_H_007_W1_1_2_N_4.eps', format='eps', dpi=1000) 
# order2_pts_integrand_isolines_plotting()    