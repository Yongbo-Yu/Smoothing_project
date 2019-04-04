import numpy as np
import os
import sys
import time



 
class Problem(object):
# attributes
    random_gen=None;
    elapsed_time=0.0;
    
    S0=None     # vector of initial stock prices
    basket_d=4   # number of assets in the basket
    c= (1/float(basket_d))*np.ones(basket_d)     # weigths
    N=4
    sigma=None    # vector of volatilities
    K=None         # Strike price
    T=1.0                      # maturity
    rho=None                  #correlation matrix
    nelem=None;               # discretization
    exact=11.04 # 4-d, sigma=0.4, S_0=K=100, T=1, r=0,rho=0.3
    yknots_right=[]
    yknots_left=[]
    


#methods
    # this method initializes the class of basket 
    def __init__(self,params, nested=False):
        self.nested = nested
        self.params = params

        self.random_gen = None or np.random
        
        self.S0=100*np.ones(self.basket_d) 
       
        self.sigma=0.4*np.ones(self.basket_d) #vector of volatilities
     
        self.K= 100                        # Strike price and coeff determine if we have in/at/out the money option
        
        # correltion matrix
        from scipy.linalg import toeplitz  
        self.rho=toeplitz([1,0.3,0.3,0.3])

        # defining the transformation matrix
        self.A=self.rotation_matrix()
       
        
        self.A_inv=np.transpose(self.A) # since A is  a rotation matrix than A^{-1}=A^T
        


        self.dt=self.T/float(self.N) # time steps length
        self.d=int(np.log2(self.N)) #power 2 number steps


        idx=[]
        for i in range(0,self.basket_d*self.N,self.N):
            idx.append(i)
        
        
        self.idxc=np.setdiff1d(range(0,self.basket_d*self.N),idx)
        
        # For less than 185 points
        # beta=128
        # self.yknots_right=np.polynomial.laguerre.laggauss(beta)
      
        # # For more than 185 points
        beta=512
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
 
     # objfun:  beta #number of points in the first direction
    def objfun(self,nelem,y):
       
        
        
        
        yy=[self.basket_d*self.N]
        yy[0]=0.0
        yy[1:]=y    
       
         
        # step 1 # get the two partitions of coordinates \mathbf{Z}_1 and \mathbf{Z}_{-1} for y which is a vector of N \times basket_d
        z1=np.array(yy[0:-1:self.N]) # getting \mathbf{Z}_1 

    
        

        
                
        z__1=np.array(yy)[self.idxc]
        
        # step 2: doing the rotation from  \mathbf{Z}_1  to \mathbf{Y}_1
        y1=np.dot(self.A,z1.transpose()) # getting \mathbf{Y}_1 by rotation using matrix A (to be defined)
        y__1=y1[1:]        # getting \mathbf{Y}_{-1}
    

        # step 3: computing the location of the kink
        bar_y1=self.newtons_method(y1[0],y__1,z__1,z1) 

        y1[0]=bar_y1
        
        z=self.A_inv.dot(y1)
        z1[0]=z[0]
        y1=np.dot(self.A,z1.transpose())
        y__1=y1[1:]   


        # step 4: performing the pre-intgeration step wrt kink point
       
        
        mylist_left_y=[]
        mylist_left_z=[]



        
        # y11_left=np.zeros(len(yknots_left[0]))
       
        # for i in range(0,len(yknots_left[0])):
        #    y1[0]=bar_y1-yknots_left[0][i]
        #    z=self.A_inv.dot(y1)
        #    z1[0]=z[0]
        #     y1=np.dot(self.A,z1.transpose())
        #     y11_left[i]=y1[1:]   


        mylist_left_y.append(self.yknots_left[0])
    
        mylist_left_y[1:]=[np.array(y__1[i]) for i in range(0,len(y__1))]

        mylist_left_z=[np.array(z__1[i]) for i in range(0,len(z__1))]
        mylist_left=mylist_left_y+mylist_left_z

        points_left=self.cartesian(mylist_left)
        
        # to be updated   (we start with the case d=2)  
        x_l=np.asarray([self.stock_price_trajectory_basket_BS(bar_y1-points_left[i,0],points_left[i,4:self.N+3]\
                                                             ,points_left[i,1],points_left[i,self.N+3:2*self.N+2]\
                                                             ,points_left[i,2], points_left[i,2*self.N+2:3*self.N+1]
                                                             ,points_left[i,3], points_left[i,3*self.N+1:4*self.N])[0]  for i in range(0,len(self.yknots_left[0]))])
   
        QoI_left= self.yknots_left[1].dot(self.payoff(x_l)*((1/np.sqrt(2 * np.pi)) * np.exp(-((bar_y1-points_left[:,0])**2)/2)* np.exp(points_left[:,0])))

        mylist_right_y=[]
        mylist_right_z=[]

       #y11_right=np.zeros(len(yknots_right[0]))
       # for i in range(0,len(yknots_right[0])):
        #    y1[0]=bar_y1+yknots_right[0][i]
         #   z=self.A_inv.dot(y1)
          #  z1[0]=z[0]
            #y1=np.dot(self.A,z1.transpose())
           # y11_right[i]=y1[1:]   

        mylist_right_y.append(self.yknots_right[0])
        mylist_right_y[1:]=[np.array(y__1[i]) for i in range(0,len(y__1))]
        mylist_right_z=[np.array(z__1[i]) for i in range(0,len(z__1))]
        mylist_right=mylist_right_y+mylist_right_z
        points_right=self.cartesian(mylist_right)

        # to be updated    (we start with the case d=2)  
        x_r=np.asarray([self.stock_price_trajectory_basket_BS(points_right[i,0]+bar_y1,points_right[i,4:self.N+3]\
                                                             ,points_right[i,1],points_right[i,self.N+3:2*self.N+2]\
                                                             ,points_right[i,2], points_right[i,2*self.N+2:3*self.N+1]
                                                             ,points_right[i,3], points_right[i,3*self.N+1:4*self.N])[0]  for i in range(0,len(self.yknots_right[0]))])
        QoI_right= self.yknots_right[1].dot(self.payoff(x_r)*(1/np.sqrt(2 * np.pi)) * np.exp(-((points_right[:,0]+bar_y1)**2)/2)* np.exp(points_right[:,0]))

        
        QoI=QoI_left+QoI_right
    
  
        return QoI


    # This function creates the desired rotation matrix A (orthonormal transformation)
    def rotation_matrix(self):   
        X1=(1/np.sqrt(self.basket_d))*np.ones((1,self.basket_d))
        A=np.eye(self.basket_d,self.basket_d)   
        A[0,:]=X1
        A=A.transpose()

        def normalize(v):
            return v / np.sqrt(v.dot(v))
        # Gram-shmidt procedure
        n = len(A)
        A[:, 0] = normalize(A[:, 0])


        for i in range(1, n):
            Ai = A[:, i]
            for j in range(0, i):
                Aj = A[:, j]
                t = Ai.dot(Aj)
                Ai = Ai - t * Aj
            A[:, i] = normalize(Ai)

        return A.transpose()


 
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
     
     
    def stock_price_trajectory_basket_BS(self,y1,yvec_1,y2,yvec_2,y3,yvec_3,y4,yvec_4):

        y=np.array([y1,y2,y3,y4])
        z=self.A_inv.dot(y)

        #building the brownian bridge increments
        bb1=self.brownian_increments(z[0],yvec_1)
        bb2=self.brownian_increments(z[1],yvec_2)
        bb3=self.brownian_increments(z[2],yvec_3)
        bb4=self.brownian_increments(z[3],yvec_4)

        dW1= [bb1[0,i+1]-bb1[0,i]  for i in range(0,self.N)] 
        dW2= [bb2[0,i+1]-bb2[0,i] for i in range(0,self.N)] 
        dW3= [bb3[0,i+1]-bb3[0,i]  for i in range(0,self.N)] 
        dW4= [bb4[0,i+1]-bb4[0,i] for i in range(0,self.N)] 

        dW=np.array([dW1 ,dW2,dW3,dW4])

        
          # construct the correlated  brownian bridge increments
        lower_triang_cholesky = np.linalg.cholesky(self.rho)
     
        dW=np.dot(lower_triang_cholesky,dW)  


    
        dW1=dW[0,:]
        dW2=dW[1,:]
        dW3=dW[2,:]
        dW4=dW[3,:]

        
        # brownian bridge increments dbbs_i (used later for the location of the kink point)
         
        dbb1=dW1-(self.dt/np.sqrt(self.T))*z[0] 
        dbb2=dW2-(self.dt/np.sqrt(self.T))*z[1] 
        dbb3=dW3-(self.dt/np.sqrt(self.T))*z[2] 
        dbb4=dW4-(self.dt/np.sqrt(self.T))*z[3] 
       
        #y__1=y[1:]

        X=np.zeros((self.basket_d,self.N+1)) #here will store the BS trajectory
      
        X[:,0]=self.S0
        for n in range(1,self.N+1):
            #X[0,n]=X[0,n-1]*(1+self.sigma[0]*((self.dt/float(np.sqrt(self.T)))*(self.A_inv[0,0]*y1+ self.A_inv[0,1:].dot(y__1)) +  dbb1[n-1] ))  
            #X[1,n]=X[1,n-1]*(1+self.sigma[1]*((self.dt/float(np.sqrt(self.T)))*(self.A_inv[1,0]*y1+ self.A_inv[1,1:].dot(y__1)) +  dbb2[n-1] ) ) 
            #X[2,n]=X[2,n-1]*(1+self.sigma[2]*((self.dt/float(np.sqrt(self.T)))*(self.A_inv[2,0]*y1+ self.A_inv[2,1:].dot(y__1)) +  dbb3[n-1] ) )  
            #X[3,n]=X[3,n-1]*(1+self.sigma[3]*((self.dt/float(np.sqrt(self.T)))*(self.A_inv[3,0]*y1+ self.A_inv[3,1:].dot(y__1)) +  dbb4[n-1] ) )   
            X[0,n]=X[0,n-1]*(1+self.sigma[0]*dW1[n-1])
            X[1,n]=X[1,n-1]*(1+self.sigma[1]*dW2[n-1])
            X[2,n]=X[2,n-1]*(1+self.sigma[2]*dW3[n-1])
            X[3,n]=X[3,n-1]*(1+self.sigma[3]*dW4[n-1])

      
        return X[:,-1],dbb1,dbb2,dbb3,dbb4
         
         
      
         # this function defines the payoff function used here
    def payoff(self,x): 
       #print(x)
       g=(x.dot(self.c)-self.K)
       
       g[np.where(g<0)]=0
       
       return g
         

       # Root solving procedure
      # Now we set up the methods used for newton iteration
    def dx(self,x,y__1,z__1):
        P1,dP1=self.f(x,y__1,z__1)
        return abs(0-P1)

    def f(self,y1,y__1,z__1):# need to check this for case d=2, N=2 and then we can extend
       
        y2=y__1[0] 
        y3=y__1[1] 
        y4=y__1[2] 
       
        yvec_1=z__1[0:self.N-1]
        yvec_2=z__1[self.N-1:2*self.N-2]
        yvec_3=z__1[2*self.N-2:3*self.N-3]
        yvec_4=z__1[3*self.N-3:]

        X,dbb1,dbb2,dbb3,dbb4=self.stock_price_trajectory_basket_BS(y1,yvec_1,y2,yvec_2,y3,yvec_3,y4,yvec_4)

        gi=np.zeros((self.basket_d,len(dbb1)))
        product=np.zeros(self.basket_d)
        summation=np.zeros(self.basket_d)
        Py=np.zeros(self.basket_d)
        dPy=np.zeros(self.basket_d)

        gi[0,:]=  1+(self.sigma[0]/float(np.sqrt(self.T)))*(self.A_inv[0,0]*y1*(self.dt)+ self.A_inv[0,1:].dot(y__1)*(self.dt))+self.sigma[0]*dbb1
        gi[1,:]=  1+(self.sigma[1]/float(np.sqrt(self.T)))*(self.A_inv[1,0]*y1*(self.dt)+ self.A_inv[1,1:].dot(y__1)*(self.dt))+self.sigma[1]*dbb2
        gi[2,:]=  1+(self.sigma[2]/float(np.sqrt(self.T)))*(self.A_inv[2,0]*y1*(self.dt)+ self.A_inv[2,1:].dot(y__1)*(self.dt))+self.sigma[2]*dbb3
        gi[3,:]=  1+(self.sigma[3]/float(np.sqrt(self.T)))*(self.A_inv[3,0]*y1*(self.dt)+ self.A_inv[3,1:].dot(y__1)*(self.dt))+self.sigma[3]*dbb4
        

        product[0]=np.prod(gi[0,:])
        product[1]=np.prod(gi[1,:])
        product[2]=np.prod(gi[2,:])
        product[3]=np.prod(gi[3,:])

        Py=(self.S0[0]*self.c[0]*product[0]+ self.S0[1]*self.c[1]*product[1]+self.S0[2]*self.c[2]*product[2]+self.S0[3]*self.c[3]*product[3])-self.K    

        summation[0]=np.sum(1/gi[0,:])
        summation[1]=np.sum(1/gi[1,:])
        summation[2]=np.sum(1/gi[2,:])
        summation[3]=np.sum(1/gi[3,:])

        
        dPy[0]= (self.S0[0]*self.c[0]* self.A_inv[0,0]*self.sigma[0]/float(np.sqrt(self.T)))*(self.dt)*product[0]*summation[0]
        dPy[1]=  (self.S0[1]*self.c[1]* self.A_inv[1,0]*self.sigma[1]/float(np.sqrt(self.T)))*(self.dt)*product[1]*summation[1]
        dPy[2]= (self.S0[2]*self.c[2]* self.A_inv[2,0]*self.sigma[2]/float(np.sqrt(self.T)))*(self.dt)*product[2]*summation[2]
        dPy[3]=  (self.S0[3]*self.c[3]* self.A_inv[3,0]*self.sigma[3]/float(np.sqrt(self.T)))*(self.dt)*product[3]*summation[3]

        dP=dPy[0]+dPy[1]+dPy[2]+dPy[3]
        return Py,dP    
    

        
    def newtons_method(self,x0,y__1,z__1,z1,eps=1e-10):
        
        delta= self.dx(x0,y__1,z__1)

        while delta > eps:
        
            #(self.f(x0,y))
            P_value,dP=self.f(x0,y__1,z__1)
            x0 = x0 - 0.1*P_value/dP
          
            #y=np.array([x0,y__1[0],y__1[1],y__1[2]])
        
       #     z=self.A_inv.dot(y)
        #    z1[0]=z[0]
         #   y=np.dot(self.A,z1.transpose())
          #  y__1=y[1:]
            delta = self.dx(x0,y__1,z__1)    
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
