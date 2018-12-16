#!/usr/bin/env python

# In this file, we plot the first and second differences for the  rBergomi integrand  without Richardson extrapolation without
# doing the partial change of measure



#modules used

#plotting modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



import numpy as np


#Rbergomi modules
import fftw3
import RBergomi
from RBergomi import *

#module used to compute the hessian
import numdifftools as nd





class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    N=4 # Number of time steps N, discretization resolution
  
    # for the values of below paramters, we need to see the paper as well check with Christian 
    x=0.235**2;   # this will provide the set of xi parameter values 
    #x=10**(-5)
    HIn=Vector(1)    # this will provide the set of H parameter values
    HIn[0]=0.07
    e=Vector(1)    # This will provide the set of eta paramter values
    e[0]=1.9
    r=Vector(1)   # this will provide the set of rho paramter values
    r[0]=-0.9
    T=Vector(1)     # this will provide the set of T(time to maturity) parameter value
    T[0]=1.0
    k=Vector(1)     # this will provide the set of K (strike ) paramter value
    #k[0]=np.exp(-4)
    k[0]=1
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


        self.Hfun = nd.Hessian(self.fun) 
        self.z_bar=np.ones((self.N))
        

        from scipy.optimize import minimize    
        mini = minimize(self.fun2, self.z_bar, method='Nelder-Mead', options={'xatol': 1e-5, 'disp': False})
        self.z_bar1=mini.x

        

        Hfun_mode=np.linalg.inv(-self.Hfun(self.z_bar1))
        

        #  # Using Cholesky decomposition
        #self.L=np.linalg.cholesky(Hfun_mode)
     
        # #using Spectral decompositon
        e_vals, e_vecs = np.linalg.eig(Hfun_mode)
        self.L=e_vecs.dot(np.diag(np.sqrt(e_vals)))

    

    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y):
        Y = np.array(Y)
        goal=self.objfun(Y);
        return goal



    def fun(self,y11):        
        yperp=np.zeros((self.N))
        y=np.array([y11,yperp]).reshape(2*self.N)
        #hierarchical
        yperp_1=yperp[1:self.N]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y11[1:self.N]
        y1=y11[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        QoI=np.log((QoI))
     
        return QoI    
    
    def fun2(self,y11):
        yperp=np.zeros((self.N))
        y=np.array([y11,yperp]).reshape(2*self.N)
        
        #hierarchical
        yperp_1=yperp[1:self.N]
        yperp1=yperp[0]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

        #hierarchical way
        y_1=y11[1:self.N]
        y1=y11[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        

        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N))*(np.exp(-0.5*y.dot(y)))
        #QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*((2*np.pi)**(-self.N/2))*(np.exp(-0.5*y11.dot(y11)))
        #print y
        

        

        #print QoI
        QoI=-np.log(QoI)
        
        return QoI  


     # objfun: 
    def objfun(self,y):

    
        #hermite with importance sampling
       
        bar_y=np.sqrt(2)*np.dot(self.L,y[0:self.N])+self.z_bar1
        yperp=y[self.N:2*self.N]
        #hierarchical
        yperp_1=y[self.N+1:2*self.N]
        yperp1=y[self.N]
        bbperp=self.brownian_increments(yperp1,yperp_1)
        W1perp= [(bbperp[i+1]-bbperp[i]) *np.sqrt(self.N) for i in range(0,len(bbperp)-1)]

       
        #hierarchical way
        y_1=bar_y[1:self.N]
        y1=bar_y[0]
        bb=self.brownian_increments(y1,y_1)
        W1= [(bb[i+1]-bb[i]) *np.sqrt(self.N) for i in range(0,len(bb)-1)]
        
        cst=np.linalg.det(self.L)*(2**(self.N/2))*(np.exp(0.5*y[0:self.N].dot(y[0:self.N])))
        QoI=(self.z.ComputePayoffRT_single(W1,W1perp))*(np.exp(-0.5*bar_y.dot(bar_y)))*cst
        

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
    prb = Problem() 
    marker=['>', 'v', '^', 'o', '*','+','>','v']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for d1 in range(0,8,2):
        print(d1)
        mylist=[]
        bias=np.zeros(5)
        points=np.zeros(5)
        indices=np.zeros(5,dtype=int)
        fixed_points=fnKnots(0)[0]
        #mylist[:prb.N]=[fixed_points for i in range(0,prb.N)]
        mylist[:2*prb.N]=[fixed_points for i in range(0,2*prb.N)] #second way
        j=0
        for pts in range(2,7):
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

      
       # QoI_beta=np.zeros(prb.N,dtype=int)
        QoI_beta=np.zeros(2*prb.N,dtype=int) #second way
        QoI_beta[d1]=1

        z= np.polyfit(indices, np.log(bias), 1)
        fit=np.exp(z[0]*indices)

        
        plt.plot(indices, bias,linewidth=2.0,label=r'$\bar{\beta}=$ %s' % np.array_str(QoI_beta),linestyle = '--', marker=marker[d1/4],hold=True) 
        #plt.plot(indices, bias,linewidth=2.0,linestyle = '--', marker=marker[d1/4],hold=True) 
        plt.plot(indices, fit,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--', marker='o') 
        #plt.plot(points, 0.001/points,'r',linewidth=2.0,label='order 1') 
        #plt.plot(points, 0.001/(points**2),'g',linewidth=2.0, label='order 2')  
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('k',fontsize=14)
        plt.ylabel(r'$\mid \Delta E_{\mathbf{1}+k \bar{\beta}} \mid $',fontsize=14)  
     
    plt.legend(loc='lower left')
    plt.savefig('./results/first_difference_rbergomi_8steps_H_007_K_1_totally_hierarch_with_rate_W1_change_measure_part_spec.eps', format='eps', dpi=1000)



def mixed_difference_order2_rate_plotting(d):       
    # # feed parameters to the problem
    prb = Problem() 
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for k in range(0,4,1):  
        if k==d:
            print('Hello')
            continue
        else:    
            mylist=[]
            mylist_weight=[]
            bias=np.zeros(6)
            points=np.zeros(6)
            indices=np.zeros(6,dtype=int)
            fixed_points=fnKnots(0)[0]
            fixed_weight=fnKnots(0)[1]
            # mylist[:prb.N]=[fixed_points for i in range(0,prb.N)]
            # mylist_weight[:prb.N]=[fixed_weight for i in range(0,prb.N)]
            mylist[:2*prb.N]=[fixed_points for i in range(0,2*prb.N)]# second way
            mylist_weight[:2*prb.N]=[fixed_weight for i in range(0,2*prb.N)] #second way
            j=0
            for pts in range(2,8):
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

            
       # QoI_beta=np.zeros(prb.N,dtype=int)
        QoI_beta=np.zeros(2*prb.N,dtype=int) #second way
        QoI_beta[d]=1
        QoI_beta[k]=1
        
        z= np.polyfit(indices, np.log(bias), 1)
        fit=np.exp(z[0]*indices)

        
        plt.plot(indices, bias,linewidth=2.0,label=r'$\bar{\beta}=$ %s' % np.array_str(QoI_beta),linestyle = '--', marker=marker[d/4]) 
        #plt.plot(indices, bias,linewidth=2.0,linestyle = '--', marker=marker[d/4]) 
        plt.plot(indices, fit*100,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--', marker='o') 
         
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('k',fontsize=14)
        plt.ylabel(r'$\mid \Delta E_{\mathbf{1}+k \bar{\beta}} \mid $',fontsize=14)  
    plt.legend(loc='lower left')
    plt.savefig('./results/mixed_difference_order2_rbergomi_4steps_H_007_K_1_totally_hierarch_with_rate_W1_change_measure_part_spec.eps', format='eps', dpi=1000)       
    
    


#first_difference_rate_plotting()
mixed_difference_order2_rate_plotting(0)