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

import numdifftools as nd


class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    T=1.0                      # maturity
    sigma=None    # volatility
    N=2  # number of time steps which will be equal to the number of brownian bridge components (we set is a power of 2)
    d=None
    dt=None

    #exact=13.0847 #  S_0=K=100, T=10, r=0,rho=-0.9, v_0=0.04, theta=0.04, xi=1,\kapp=0.5
    exact=7.5789 #  S_0=K=100, T=1, r=0,rho=-0.9, v_0=0.04, theta=0.04, xi=0.5,\kapp=5 (satisfies Feller condition)
    yknots_right=[]
    yknots_left=[]


#methods
    # this method initializes 
    def __init__(self,coeff,nested=False):
        self.nested = nested

        self.random_gen = None or np.random
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        
        self.rho=-0.9

        #self.kappa= 0.5
        self.kappa= 5.0
        self.theta=0.04

        #self.xi=1
        self.xi=0.5
        self.v0=0.04
        
       # self.K= coeff*self.S0   
        self.dt=self.T/float(self.N) # time steps length
        self.d=int(np.log2(self.N)) #power 2 number steps

        # For less than 185 points
        # beta=32
        # self.yknots_right=np.polynomial.laguerre.laggauss(beta)
      
        # For more than 185 points
        #beta=512
        from Parser import Parser
        fx = open('lag_512_x.txt', 'r')
        Element_properties_x = Parser('./lag_512_x.txt')
        Element_properties_x.parse_file(fx.read(),'\n')
        x=np.array([float(i) for i in Element_properties_x.element_list])
       
        Element_properties_x.close_file()   
        fw = open('lag_512_w.txt', 'r')
        Element_properties_w = Parser('./lag_512_w.txt')
        Element_properties_w.parse_file(fw.read(),'\n')
        w=np.array([float(i) for i in Element_properties_w.element_list])
   
        Element_properties_x.close_file()   
        self.yknots_right.append(x[:360])
        self.yknots_right.append(w[:360])
     
       
        self.yknots_left=self.yknots_right
        

    

     

    def fun_1D(self,y11):
        y12=np.zeros_like(y11)
    
      
        y=np.array([y12,y12,y11])  # 2 steps

        #y=np.array([y12,y12,y12,y12,y12,y12,y12,y11,y12,y12,y12,y12,y12,y12,y12,y12])   # 8 steps
        #y=np.array([y12,y12,y13,y14,y21,y22,y23,y11])   # 4 steps
        # y22=np.zeros_like(y11)
        # y21=np.zeros_like(y11)
        # y12=np.zeros_like(y11)


        y1=y[0:self.N] # this points are related to the volatility path

        y2=[self.N]
        y2[0]=0.0
        y2[1:]=y[self.N:]

        #y_s= self.rho *np.array(y1)+ np.sqrt(1-self.rho**2) * np.array(y2) 
        #ys=y_s[1:]

        y2s=y2[1:]
        
        # step 2: computing the location of the kink
        #bar_z=self.newtons_method(y_s[0],ys,y1[0],y1[1:self.N])
        bar_z=self.newtons_method(y2[0],y2s,y1[0],y1[1:self.N])
        #bar_z=0
        
        # step 3: performing the pre-intgeration step wrt kink point
    
        mylist_left=[]
        mylist_left.append(self.yknots_left[0])
        mylist_left[1:]=[np.array(y2s[i]) for i in range(0,len(y2s))]
        points_left=self.cartesian(mylist_left)

        x_l=np.asarray([self.stock_price_trajectory_1D_heston(bar_z-points_left[i,0],points_left[i,1:],y1[0],y1[1:self.N])[0]  for i in range(0,len(self.yknots_left[0]))])

       
        QoI_left= self.yknots_left[1].dot(self.payoff(x_l)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z-points_left[:,0])**2)/2)* np.exp(points_left[:,0])))


        mylist_right=[]
        mylist_right.append(self.yknots_right[0])
        mylist_right[1:]=[np.array(y2s[i]) for i in range(0,len(y2s))]
        points_right=self.cartesian(mylist_right)
        x_r=np.asarray([self.stock_price_trajectory_1D_heston(points_right[i,0]+bar_z,points_right[i,1:],y1[0],y1[1:self.N])[0] for i in range(0,len(self.yknots_right[0]))])
        QoI_right= self.yknots_right[1].dot(self.payoff(x_r)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right[:,0]+bar_z)**2)/2)* np.exp(points_right[:,0]))

        QoI=QoI_left+QoI_right
                
        return QoI    
  


    def fun(self,y11,y12):
        y21=np.zeros_like(y11)
        y=np.array([y21,y11,y12])
        # y13=np.zeros_like(y11)
        # y14=np.zeros_like(y11)
        # y21=np.zeros_like(y11)
        # y22=np.zeros_like(y11)
        # y23=np.zeros_like(y11)
        # y24=np.zeros_like(y11)
       
        #y=np.array([y11,y12,y13,y14,y21,y22,y23,y24]) 


        y1=y[0:self.N] # this points are related to the volatility path

        y2=[self.N]
        y2[0]=0.0
        y2[1:]=y[self.N:]

        #y_s= self.rho *np.array(y1)+ np.sqrt(1-self.rho**2) * np.array(y2) 
        #ys=y_s[1:]

        y2s=y2[1:]
        
        # step 2: computing the location of the kink
        #bar_z=self.newtons_method(y_s[0],ys,y1[0],y1[1:self.N])
        bar_z=self.newtons_method(y2[0],y2s,y1[0],y1[1:self.N])
        #bar_z=0
        
        # step 3: performing the pre-intgeration step wrt kink point
    
        mylist_left=[]
        mylist_left.append(self.yknots_left[0])
        mylist_left[1:]=[np.array(y2s[i]) for i in range(0,len(y2s))]
        points_left=self.cartesian(mylist_left)

        x_l=np.asarray([self.stock_price_trajectory_1D_heston(bar_z-points_left[i,0],points_left[i,1:],y1[0],y1[1:self.N])[0]  for i in range(0,len(self.yknots_left[0]))])

       
        QoI_left= self.yknots_left[1].dot(self.payoff(x_l)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z-points_left[:,0])**2)/2)* np.exp(points_left[:,0])))


        mylist_right=[]
        mylist_right.append(self.yknots_right[0])
        mylist_right[1:]=[np.array(y2s[i]) for i in range(0,len(y2s))]
        points_right=self.cartesian(mylist_right)
        x_r=np.asarray([self.stock_price_trajectory_1D_heston(points_right[i,0]+bar_z,points_right[i,1:],y1[0],y1[1:self.N])[0] for i in range(0,len(self.yknots_right[0]))])
        QoI_right= self.yknots_right[1].dot(self.payoff(x_r)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right[:,0]+bar_z)**2)/2)* np.exp(points_right[:,0]))

        QoI=QoI_left+QoI_right
        return QoI    
         
  

    

    # This function implements the brownian bridge construction in 1 D, the argument y here represent independent  multivariate gaussian  rdv  given by
    #mean = np.zeros(self.N)
    #covariance= np.identity(self.N)
    #y = np.random.multivariate_normal(mean, covariance)
    #y1: is the first direction given by W(T)/sqrt(T)/   #This function gives  the brownian motion increments built from Brownian bridge construction: # the composition of  BB and brownian_increments give us
    # the function \phi in our notes(discussion)
        
    def brownian_increments(self,y1,y):
        t=np.linspace(0, self.T, self.N+1)     
        h=self.N
        j_max=1
        bb= np.zeros((1,self.N+1))
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

    

    # This function simulates a 1D heston trajectory for stock price and volatility paths
    def stock_price_trajectory_1D_heston(self,y1,y,yv1,yv):
        bb=self.brownian_increments(y1,y)
        dW= [bb[0,i+1]-bb[0,i] for i in range(0,self.N)] 
       # # # non hierarhcical
       #  dW=[]
       #  dW.append(y1)
       #  dW[1:]=[np.array(y[i]) for i in range(0,len(y))]
       #  dW=np.array(dW)*np.sqrt(self.dt)
        
    
        # #  hierarhcical
        bb_v=self.brownian_increments(yv1,yv)
        dW_v= [bb_v[0,i+1]-bb_v[0,i] for i in range(0,self.N)] 

        # # non hierarhcical
        # dW_v=[]
        # dW_v.append(yv1)
        # dW_v[1:]=[np.array(yv[i]) for i in range(0,len(yv))]
        # dW_v=np.array(dW_v)*np.sqrt(self.dt)
        

        
        dW_s= self.rho *np.array(dW_v)+ np.sqrt(1-self.rho**2) * np.array(dW)
        y1s= self.rho *yv1 + np.sqrt(1-self.rho**2) * y1


        #option1 
        # dbb1=dW-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        # dbbv=dW_v*np.sqrt(self.dt) -(self.dt/np.sqrt(self.T))*yv1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        # dbb_s= self.rho *np.array(dbbv) + np.sqrt(1-self.rho**2) * np.array(dbb1)
        # #option2
        dbb_s=dW_s-(self.dt/np.sqrt(self.T))*y1s



        X=np.zeros(self.N+1) #here will store the asset trajectory
        V=np.zeros(self.N+1) #here will store the  volatility trajectory

        X[0]=self.S0
        V[0]=self.v0
        
        
        for n in range(1,self.N+1):
            X[n]=X[n-1]*(1+np.sqrt(V[n-1])*dW_s[n-1])
            V[n]=V[n-1]- self.kappa *self.dt* max(V[n-1],0)+ self.xi *np.sqrt(max(V[n-1],0))*dW_v[n-1]+ self.kappa*self.theta*self.dt
            V[n]=max(V[n],0)
            
        return X[-1],dbb_s,V
       
    # this function defines the payoff function used here
    def payoff(self,x): 

       g=(x-self.K)
       g[g < 0] = 0
       return g  


    # Root solving procedure
 
    #Now we set up the methods used for newton iteration
    def dx(self,x,y,yv1,yv):
        P1,dP1=self.f(x,y,yv1,yv)
        return abs(0-P1)

  
    def f(self,y1,y,yv1,yv):
        X,dbb,V=self.stock_price_trajectory_1D_heston(y1,y,yv1,yv) # right version
        fi=np.zeros((1,len(dbb)))
        

        y1s= self.rho *yv1 + np.sqrt(1-self.rho**2) * y1
        
        fi=1+(np.sqrt(V[0:self.N])/float(np.sqrt(self.T)))*y1s*(self.dt) +(np.sqrt(V[0:self.N]))*dbb
        product=np.prod(fi)
        Py=product-(self.K/float(self.S0))


        summation=np.sum(np.sqrt(V[0:self.N])/fi)
        dPy=(1/float(np.sqrt(self.T)))*(self.dt)*product*summation
       # print dPy
        #dPy=1.0
        return Py,dPy    
        

                    
  
    def newtons_method(self,x0,y,yv1,yv,eps=1e-10):
        delta = self.dx(x0,y,yv1,yv)
        while delta > eps:
    
            P_value,dP=self.f(x0,y,yv1,yv)
            x0 = x0 - 0.1*P_value/dP
            delta = self.dx(x0,y,yv1,yv) 
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
    

    



#1D_plots

prb=Problem(1)
x1= np.linspace(-20, 20, 100)
Z=np.zeros((100))

for i1 in range(0,100):
        Z[i1]=prb.fun_1D(x1[i1])

print Z         
print(np.max(Z))         
fig = plt.figure(figsize=(14,6))

plt.plot(x1,Z)

plt.xlabel(r'$y_3$',  fontsize=20, fontweight='bold')
plt.ylabel(r'$I(y_3)$', fontsize=20, fontweight='bold')

plt.savefig('./results/heston_2steps_hierarchical_y3.pdf', format='pdf', dpi=600) 


# # #2D_plots
# prb=Problem(1)
# x1= np.linspace(-20, 20, 50)
# x2= np.linspace(-20,20, 50)
# # x3= np.linspace(-10000, 10000, 5)
# # x4= np.linspace(-10000, 10000, 5)
# # X1,X2,X3,X4 = np.meshgrid(x1,x2,x3,x4)

# X1,X2 = np.meshgrid(x1,x2)

# #Z=np.zeros((5,5,5,5))
# Z=np.zeros((50,50))
# # for i1 in range(0,5):
# #     for i2 in range(0,5):
# #     	for i3 in range(0,5):
# #     		for i4 in range(0,5):
# #         		Z[i1,i2,i3,i4]=prb.fun(x1[i1],x2[i2],x3[i3],x4[i4])
# for i1 in range(0,50):
#     for i2 in range(0,50):
#         Z[i1,i2]=prb.fun(x1[i1],x2[i2])
         
         

# fig = plt.figure(figsize=(14,6))



# print Z
# print np.amax(Z)
# from numpy import unravel_index
# print (unravel_index(Z.argmax(), Z.shape))



# #print np.argmax(Z, axis=0, axis=1)


# #plotting contours
# # cp = plt.contourf(X1, X2, Z)
# # plt.colorbar(cp)
# # plt.xlabel(r'$W_1^1$',  fontsize=20, fontweight='bold')
# # plt.ylabel(r'$W_1^2$', fontsize=20, fontweight='bold')



# #plotting the surface
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax = Axes3D(fig)
# p = ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# cb = fig.colorbar(p, shrink=0.5)
# # Get current rotation angle
# print ax.azim


# ax.set_zlabel(r'$I(y_2,y_3)$', fontsize=20, fontweight='bold')
# ax.set_xlabel(r'$y_2$',  fontsize=20, fontweight='bold')
# ax.set_ylabel(r'$y_3$', fontsize=20, fontweight='bold')
# # ax.set_zlabel(r'$I(W_2^3,W_2^4)$', fontsize=20, fontweight='bold')

# # Set rotation angle to 30 degrees
# #ax.view_init(60, 60)


# for ii in xrange(0,360,40):
#         ax.view_init(elev=10., azim=ii)
#         plt.savefig("./results/heston_2steps_non_hierarchical_y2_3_%d.pdf" % ii, dpi=600)


