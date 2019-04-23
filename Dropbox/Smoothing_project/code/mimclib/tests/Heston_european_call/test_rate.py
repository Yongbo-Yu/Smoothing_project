#!/usr/bin/env python


# In this file, we plot the first and second differences for the  Heston single call integrand  without Richardson extrapolation 


#modules used

#plotting modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


import numpy as np
np.set_printoptions(threshold=np.nan)
from numpy import unravel_index


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
        

     


    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y):
        Y = np.array(Y)
        goal=self.objfun(Y);
        return goal


     # objfun:  beta #number of points in the first direction
    def objfun(self,y):        

        # step 1 # get the two partitions of coordinates y_1 for the volatility path  and y_s for  the asset path  
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
        # bb_v=self.brownian_increments(yv1,yv)
        # dW_v= [bb_v[0,i+1]-bb_v[0,i] for i in range(0,self.N)] 

        # # non hierarhcical
        dW_v=[]
        dW_v.append(yv1)
        dW_v[1:]=[np.array(yv[i]) for i in range(0,len(yv))]
        dW_v=np.array(dW_v)*np.sqrt(self.dt)
        

        
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
    

    


def knots_gaussian(n, mi, sigma):
    # [x,w]=KNOTS_GAUSSIAN(n,mi,sigma)
    #
    # calculates the collocation points (x)
    # and the weights (w) for the gaussian integration
    # w.r.t to the weight function
    # rho(x)=1/sqrt(2*pi*sigma) *exp( -(x-mi)^2 / (2*sigma^2) )
    # i.e. the density of a gaussian random variable
    # wimth mean mi and standard deviation sigma
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
    
    #m = 4* (i-1)+1
    m = 2** (i-1)+1
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


# this function computes and plots the first differences

def first_difference_rate_plotting():       
    # # feed parameters to the problem
    prb = Problem(1) 
    marker=['>', 'v', '^', 'o', '*','+','>','v']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for d1 in range(0,7,1):
        print(d1)
        mylist=[]
        bias=np.zeros(5)
        points=np.zeros(5)
        indices=np.zeros(5,dtype=int)
        fixed_points=fnKnots(0)[0]
        mylist[:2*prb.N-1]=[fixed_points for i in range(0,2*prb.N-1)] #second way
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
        QoI_beta=np.zeros(2*prb.N-1,dtype=int) #second way
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
     
    plt.legend(loc='upper right')
    plt.savefig('./results/first_difference_heston_4steps_non_hierarchical.eps', format='eps', dpi=1000)



def mixed_difference_order2_rate_plotting(d):       
    # # feed parameters to the problem
    prb = Problem(1) 
    marker=['>', 'v', '^', 'o', '*','+','-',':']
    ax = figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for k in range(0,7,1):  
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
        
            mylist[:2*prb.N-1]=[fixed_points for i in range(0,2*prb.N-1)]# second way
            mylist_weight[:2*prb.N-1]=[fixed_weight for i in range(0,2*prb.N-1)] #second way
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


                # if pts>5:
                    
                #     weights_cc[np.multiply(coarse_values,weights_cc)>0.001]=0

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

                # if pts>5:
                    
                #     weights_fc[np.multiply(coarse_values,weights_fc)>0.001]=0


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
                # if pts>5:
                #     weights_cf[np.multiply(coarse_values,weights_cf)>0.004]=0


                QoI_coarse_fine=weights_cf.dot(coarse_values)
                print('cf=',QoI_coarse_fine)




                bias[j]=np.abs(QoI_fine_fine+ QoI_coarse_coarse - QoI_fine_coarse - QoI_coarse_fine   )
                points[j]=lev2knots_doubling(1+pts)
                print(points)
                indices[j]=pts

                j=j+1

            
       # QoI_beta=np.zeros(prb.N,dtype=int)
        QoI_beta=np.zeros(2*prb.N-1,dtype=int) #second way
        QoI_beta[d]=1
        QoI_beta[k]=1
        
        z= np.polyfit(indices, np.log(bias), 1)
        fit=np.exp(z[0]*indices)

        
        plt.plot(indices, bias,linewidth=2.0,label=r'$\bar{\beta}=$ %s' % np.array_str(QoI_beta),linestyle = '--', marker=marker[d/4]) 
        #plt.plot(indices, bias,linewidth=2.0,linestyle = '--', marker=marker[d/4]) 
        plt.plot(indices, fit,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--', marker='o') 
         
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('k',fontsize=14)
        plt.ylabel(r'$\mid \Delta E_{\mathbf{1}+k \bar{\beta}} \mid $',fontsize=14)  
    plt.legend(loc='lower left')
    plt.savefig('./results/mixed_difference_order2_heston_4steps_non_hierarchical.eps', format='eps', dpi=1000)       
    
    


first_difference_rate_plotting()
mixed_difference_order2_rate_plotting(0)
