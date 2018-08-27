#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm, ticker


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
    N=8 # Number of time steps N, discretization resolution
  
    # for the values of below paramters, we need to see the paper as well check with Christian 
    x=0.235**2;   # this will provide the set of xi parameter values 
    #x=0.00001;
    HIn=Vector(1)    # this will provide the set of H parameter values
    HIn[0]=0.07
    e=Vector(1)    # This will provide the set of eta paramter values
    e[0]=1.9
    r=Vector(1)   # this will provide the set of rho paramter values
    r[0]=-0.9
    #r[0]=0
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
        
        #QoI=self.z.ComputePayoffRT_single(y1,self.y1perp); # this is the computed payoff
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y)))
       
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
        y=np.array([y13,y14,y11,y12,y21,y22,y23,y24]) 
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
        

        #QoI=self.z.ComputePayoffRT_single(y1,self.y1perp); # this is the computed payoff
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y)))
       
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

prb=Problem_measure_change()
x1= np.linspace(-10, 10, 100)
Z=np.zeros((100))

for i1 in range(0,100):
        Z[i1]=prb.fun_1D(x1[i1])

print Z         
print(np.max(Z))         
fig = plt.figure(figsize=(14,6))

plt.plot(x1,Z)

plt.xlabel(r'$W_1^8$',  fontsize=20, fontweight='bold')
plt.ylabel(r'$I(W_1^8)$', fontsize=20, fontweight='bold')

plt.savefig('./results/Bergomi_integrand_K_1_H_007_W18_N_8.pdf', format='pdf', dpi=600) 


# #2D_plots
# prb=Problem_measure_change()
# x1= np.linspace(-5, 5, 100)
# x2= np.linspace(-5, 5, 100)
# # x3= np.linspace(-10000, 10000, 5)
# # x4= np.linspace(-10000, 10000, 5)
# # X1,X2,X3,X4 = np.meshgrid(x1,x2,x3,x4)

# X1,X2 = np.meshgrid(x1,x2)
# #Z=np.zeros((5,5,5,5))
# Z=np.zeros((100,100))
# # for i1 in range(0,5):
# #     for i2 in range(0,5):
# #     	for i3 in range(0,5):
# #     		for i4 in range(0,5):
# #         		Z[i1,i2,i3,i4]=prb.fun(x1[i1],x2[i2],x3[i3],x4[i4])
# for i1 in range(0,100):
#     for i2 in range(0,100):
#         Z[i1,i2]=prb.fun(x1[i1],x2[i2])
         
         

# fig = plt.figure(figsize=(14,6))

# # ax = fig.add_subplot(1, 2, 1, projection='3d')
# # ax = Axes3D(fig)

# print Z
# print np.amax(Z)

# #plotting the surface
# # p = ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # cb = fig.colorbar(p, shrink=0.5)

# #plotting contours
# cp = plt.contourf(X1, X2, Z)
# plt.colorbar(cp)
# # Get current rotation angle
# #print ax.azim

# plt.xlabel(r'$W_1^3$',  fontsize=20, fontweight='bold')
# plt.ylabel(r'$W_1^4$', fontsize=20, fontweight='bold')
# #ax.set_zlabel(r'$I(W_2^3,W_2^4)$', fontsize=20, fontweight='bold')
# # ax.set_xlabel(r'$W_2^3$',  fontsize=20, fontweight='bold')
# # ax.set_ylabel(r'$W_2^4$', fontsize=20, fontweight='bold')
# # ax.set_zlabel(r'$I(W_2^3,W_2^4)$', fontsize=20, fontweight='bold')

# # Set rotation angle to 30 degrees
# #ax.view_init(30, 30)
# #ax.view_init(azim=0)
# plt.savefig('./results/Bergomi_integrand_contours_K_1_H_007_W1_3_4_N_4.pdf', format='pdf', dpi=600)    