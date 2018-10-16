import numpy as np
import os
import sys
import time

class Problem(object):
# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    basket_d=3       # number of assets in the basket
    c= (1/float(basket_d))*np.ones(basket_d)     # weigths
    sigma=None    # vector of volatilities
    K=None         # Strike price
    T=1.0                      # maturity
    w=None                    # new weights
    rho=None                  #correlation matrix
    Sigma=None                # Covariance matrix
    nelem=None;               # discretization

#methods
    # this method initializes the class of basket 
    def __init__(self,params,coeff, nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random
        #self.S0=np.random.uniform(8,20,self.basket_d) # vector of initial stock prices
        self.S0=20*np.ones(self.basket_d) 
        #self.sigma=np.random.uniform(0.3,0.4,self.basket_d) #vector of volatilities
        self.sigma=0.4*np.ones(self.basket_d) 
        self.K= coeff*self.c.dot(self.S0)                           # Strike price and coeff determine if we have in/at/out the money option
        self.w=self.c*self.S0    #new weights
        #self.rho=self.correlation()                                                  #correlation matrx
        #self.rho=np.array([[1 , 0.3,  0.3],[ 0.3,  1 , 0.3],[0.3 , 0.3 ,1] ])
        #from numpy import concatenate, zeros
        #from scipy.linalg import toeplitz
        #self.rho=toeplitz([1,0.8,0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.8, 0.8,0.8,0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.8])
        from scipy.linalg import toeplitz
        #self.rho=toeplitz([1,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3])
        self.rho=toeplitz([1,0.3,0.3])
        self.Sigma=np.zeros((self.basket_d,self.basket_d))
        for i in range(0,self.basket_d):
                for j in range(i,self.basket_d):
                      self.Sigma[i,j]=self.sigma[i]*self.sigma[j]*self.rho[i,j]*self.T
        self.Sigma=self.Sigma+np.transpose(self.Sigma)-np.diag(np.diag(self.Sigma))
       
    
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
        goal=self.objfun(self.nelem,Y);
        return goal

    # objfun
    def objfun(self,nelem,y):
        start_time=time.time();
        
        #preparing point for Laguerre
        beta=8
        yknots_right=np.polynomial.laguerre.laggauss(beta)
        yknots_left=yknots_right


        
        lambda_vect,V=self.factorization()

        bar_y=stock_price_trajectory_basket_BS(y)
        
        bar_z=np.log(K/float(self.h(bar_y,V,self.w)))


        
       



        mylist_left=[]
        mylist_left.append(yknots_left[0])
        mylist_left[1:]=[np.array(y[i]) for i in range(0,len(y))]
        points_left=self.cartesian(mylist_left)
        x_l= np.exp()
        QoI_left= yknots_left[1].dot(self.payoff(x_l)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_z-points_left[:,0])**2)/2)* np.exp(points_left[:,0])))

        mylist_right=[]
        mylist_right.append(yknots_right[0])
        mylist_right[1:]=[np.array(y[i]) for i in range(0,len(y))]
        points_right=self.cartesian(mylist_right)
        x_r=np.asarray([self.stock_price_trajectory_1D_BS(points_right[i,0]+bar_z,points_right[i,1:])[0] for i in range(0,len(yknots_right[0]))])
        QoI_right= yknots_right[1].dot(self.payoff(x_r)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right[:,0]+bar_z)**2)/2)* np.exp(points_right[:,0]))

        QoI=QoI_left+QoI_right



        elapsed_time_qoi=time.time()-start_time;
        self.elapsed_time=self.elapsed_time+elapsed_time_qoi;


       
        return QoI

    




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


      # This function simulates a basket BS trajectory for stock price, it plays the role of f_1 in our notes
    def stock_price_trajectory_basket_BS(self,y):

        for i in range (basket_d-1):
            bb[i,:]=self.brownian_increments(y[i*self.N],y[i*self.N+1])

        
        for i in range (basket_d-1):

            dW[i,:]= [bb[i,j+1]-bb[i,j]  for j in range(0,self.N)] 
    



        # construct the correlated  brownian bridge increments
        lower_triang_cholesky = np.linalg.cholesky(self.rho)
     
        dW=np.dot(lower_triang_cholesky,dW)  
          
          for i in range (basket_d-1):

            dWbb[i,:]= dW[i,:]-(self.dt/np.sqrt(self.T))*y[i*self.N] 
          


        X=np.zeros((self.basket_d-1,self.N+1)) #here will store the BS trajectory
      
        X[:,0]=self.S0
        for n in range(1,self.N+1):
            for i in  range (basket_d-1):
                X[i,n]=X[i,n-1]*(1+self.sigma[i]*dW[i,n-1])
       
      
        
        return X[:,-1]


      # this function defines the payoff function used here
    def payoff(self,x): 
       #print(x)
       g=(x-self.K)
       g[g < 0] = 0
       return g      
    
    # This function computes the transformation operation
    def h(self,y_bar, V,w):
        V_1=np.array(V[:,1:])
        transformed_result=np.exp(V_1.dot(y_bar)).dot(w)
        return transformed_result    

# this provides the lammdas and the V 
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

    def GetDim(self):
        return 0
