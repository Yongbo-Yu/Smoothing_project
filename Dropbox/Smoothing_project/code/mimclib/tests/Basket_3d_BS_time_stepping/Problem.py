import numpy as np
import os
import sys
import time



class Problem(object):
# attributes
    random_gen=None;
    elapsed_time=0.0;
    N=2
    S0=None     # vector of initial stock prices
    basket_d=3     # number of assets in the basket
    c= (1/float(basket_d))*np.ones(basket_d)     # weigths
    sigma=None    # vector of volatilities
    K=None         # Strike price
    T=1.0                      # maturity
   
    rho=None                  #correlation matrix
  
    nelem=None;               # discretization
    
    exact=  11.704640   # 3-d, sigma=0.4, S_0=K=100, T=1, r=0,rho=0.3


#methods
    # this method initializes the class of basket 
    def __init__(self,params, nested=False):
        self.nested = nested
        self.params = params
        self.random_gen = None or np.random
        
        self.S0=100*np.ones(self.basket_d) 
       
        self.sigma=0.4*np.ones(self.basket_d) #vector of volatilities
     
        self.K= 100                        # Strike price and coeff determine if we have in/at/out the money option
        
        #self.rho=self.correlation()                                                  #correlation matrx
        #self.rho=np.array([[1 , 0.3,  0.3],[ 0.3,  1 , 0.3],[0.3 , 0.3 ,1] ])
        #from numpy import concatenate, zeros
        #from scipy.linalg import toeplitz
        #self.rho=toeplitz([1,0.8,0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.8, 0.8,0.8,0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.8])
        from scipy.linalg import toeplitz
        #self.rho=toeplitz([1,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3])
        self.rho=toeplitz([1,0.3,0.3])
        # self.Sigma=np.zeros((self.basket_d,self.basket_d))
        # for i in range(0,self.basket_d):
        #         for j in range(i,self.basket_d):
        #               self.Sigma[i,j]=self.sigma[i]*self.sigma[j]*self.rho[i,j]*self.T
        # self.Sigma=self.Sigma+np.transpose(self.Sigma)-np.diag(np.diag(self.Sigma))
       

        self.dt=self.T/float(self.N) # time steps length
        self.d=int(np.log2(self.N)) #power 2 number steps
    
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
      
        
        # step 1

       

        y3_1=y[2*(self.N-1):self.basket_d*(self.N-1)]

        y2_1=y[self.N-1:2*(self.N-1)]
        

        y1_1=y[0:self.N-1]
        
       
        #step2
       

        bar_z=self.newtons_method(np.zeros(self.basket_d),y1_1,y2_1,y3_1)
        

        #step3


        beta=8
        yknots_right=np.polynomial.laguerre.laggauss(beta)
        yknots_left=yknots_right

     
       ############################################################

        mylist_left1=[]
        mylist_left1.append(yknots_left[0])
        mylist_left1[1:]=[np.array(y1_1[i]) for i in range(0,len(y1_1))]

        mylist_left2=[]
        mylist_left2.append(yknots_left[0])
        mylist_left2[1:]=[np.array(y2_1[i]) for i in range(0,len(y2_1))]

        mylist_left3=[]
        mylist_left3.append(yknots_left[0])
        mylist_left3[1:]=[np.array(y3_1[i]) for i in range(0,len(y3_1))]

      

       ############################################################
        mylist_right1=[]
        mylist_right1.append(yknots_right[0])
        mylist_right1[1:]=[np.array(y1_1[i]) for i in range(0,len(y1_1))]

        mylist_right2=[]
        mylist_right2.append(yknots_right[0])
        mylist_right2[1:]=[np.array(y2_1[i]) for i in range(0,len(y2_1))]

        mylist_right3=[]
        mylist_right3.append(yknots_right[0])
        mylist_right3[1:]=[np.array(y3_1[i]) for i in range(0,len(y3_1))]

   
        ############################################################

        #fist quadrant (r,r,r)
        
        mylist_r_r_r=mylist_right1+mylist_right2+mylist_right3
        weights_list_r_r_r=[yknots_right[1],yknots_right[1],yknots_right[1]]


        points_r_r_r=self.cartesian(mylist_r_r_r)
        weights_r_r_r_aux=self.cartesian(weights_list_r_r_r)
        weights_r_r_r=np.prod(weights_r_r_r_aux,1)


        x_rrr=np.asarray([self.stock_price_trajectory_basket_BS(points_r_r_r[i,0]+bar_z[0],points_r_r_r[i,1:]\
                                                                 ,points_r_r_r[i,self.N] +bar_z[1],points_r_r_r[i,self.N+1:]\
                                                                 ,points_r_r_r[i,2*self.N]+bar_z[2],points_r_r_r[i,2*self.N+1:])[0] \
                                                                  for i in range(0,len(yknots_right[0])*len(yknots_right[0])*len(yknots_right[0]))])

        pay_rrr=np.asarray( [self.payoff(x_rrr[i,:])for i in range(0,len(yknots_right[0])*len(yknots_right[0])*len(yknots_right[0]))])

        pay_rrr=np.multiply(pay_rrr,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_r_r_r[:,0]+bar_z[0])**2)/2) \
                                                * np.exp(points_r_r_r[:,0]) \
                                                * np.exp(-((points_r_r_r[:,self.N]+bar_z[1])**2)/2)* np.exp(points_r_r_r[:,self.N])\
                                                * np.exp(-((points_r_r_r[:,2*self.N]+bar_z[2])**2)/2)* np.exp(points_r_r_r[:,2*self.N])))

        
        QoI_rrr= weights_r_r_r.dot(pay_rrr)

        ###############################
        #second quadrant (r,r,l)

        mylist_r_r_l=mylist_right1+mylist_right2+mylist_left3
        weights_list_r_r_l=[yknots_right[1],yknots_right[1],yknots_left[1]]

        points_r_r_l=self.cartesian(mylist_r_r_l)
        weights_r_r_l_aux=self.cartesian(weights_list_r_r_l)
        weights_r_r_l=np.prod(weights_r_r_l_aux,1)


        x_rrl=np.asarray([self.stock_price_trajectory_basket_BS(points_r_r_l[i,0]+bar_z[0],points_r_r_l[i,1:]\
                                                                ,points_r_r_l[i,self.N]+bar_z[1],points_r_r_l[i,self.N+1:]\
                                                                ,bar_z[2]-points_r_r_l[i,2*self.N],points_r_r_l[i,2*self.N+1:])[0] \

                                                                for i in range(0,len(yknots_right[0])*len(yknots_right[0])*len(yknots_left[0]))])
        
        pay_rrl=np.asarray( [self.payoff(x_rrl[i,:])for i in range(0,len(yknots_right[0])*len(yknots_right[0])*len(yknots_left[0]) )])

        pay_rrl=np.multiply(pay_rrl,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_r_r_l[:,0]+bar_z[0])**2)/2)\
                                                            * np.exp(points_r_r_l[:,0])\
                                                            * np.exp(-((points_r_r_l[:,self.N]+bar_z[1])**2)/2)* np.exp(points_r_r_l[:,self.N])\
                                                            * np.exp(-((bar_z[2]-points_r_r_l[:,2*self.N])**2)/2)* np.exp(points_r_r_l[:,2*self.N]) ))

        QoI_rrl= weights_r_r_l.dot(pay_rrl)


        ###############################
        #3rd quadrant (r,l,l)

        mylist_r_l_l=mylist_right1+mylist_left2+mylist_left3
        weights_list_r_l_l=[yknots_right[1],yknots_left[1],yknots_left[1]]

        points_r_l_l=self.cartesian(mylist_r_l_l)
        weights_r_l_l_aux=self.cartesian(weights_list_r_l_l)
        weights_r_l_l=np.prod(weights_r_l_l_aux,1)

        x_rll=np.asarray([self.stock_price_trajectory_basket_BS(points_r_l_l[i,0]+bar_z[0],points_r_l_l[i,1:] \
                                                                ,bar_z[1]-points_r_l_l[i,self.N],points_r_l_l[i,self.N+1:]\
                                                                ,bar_z[2]-points_r_l_l[i,2*self.N],points_r_l_l[i,2*self.N+1:])[0]\
                                                                for i in range(0,len(yknots_right[0])*len(yknots_left[0])\
                                                                                                    *len(yknots_left[0]))])


        pay_rll=np.asarray( [self.payoff(x_rll[i,:])for i in range(0,len(yknots_right[0])*len(yknots_left[0])*len(yknots_left[0])    )])
        

        pay_rll=np.multiply(pay_rll,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_r_l_l[:,0]+bar_z[0])**2)/2) \
            * np.exp(points_r_l_l[:,0]) \
            * np.exp(-((bar_z[1]-points_r_l_l[:,self.N])**2)/2)* np.exp(points_r_l_l[:,self.N]) \
            * np.exp(-((bar_z[2]-points_r_l_l[:,2*self.N])**2)/2)* np.exp(points_r_l_l[:,2*self.N])  ))


        QoI_rll= weights_r_l_l.dot(pay_rll)

        ###############################
        #4th quadrant (l,l,l)

        mylist_l_l_l=mylist_left1+mylist_left2+mylist_left3
        weights_list_l_l_l=[yknots_left[1],yknots_left[1],yknots_left[1]]

        points_l_l_l=self.cartesian(mylist_l_l_l)
        weights_l_l_l_aux=self.cartesian(weights_list_l_l_l)
        weights_l_l_l=np.prod(weights_l_l_l_aux,1)

        x_lll=np.asarray([self.stock_price_trajectory_basket_BS(bar_z[0]-points_l_l_l[i,0],points_l_l_l[i,1:]\
                                                                ,bar_z[1]-points_l_l_l[i,self.N],points_l_l_l[i,self.N+1:]\
                                                                ,bar_z[2]-points_l_l_l[i,2*self.N],points_l_l_l[i,2*self.N+1:] )[0]\
                                                                for i in range(0,len(yknots_left[0])*len(yknots_left[0])*len(yknots_left[0]))])


        pay_lll=np.asarray( [self.payoff(x_lll[i,:])for i in range(0,len(yknots_left[0])*len(yknots_left[0])*len(yknots_left[0]) )])
        

        pay_lll=np.multiply(pay_lll,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z[0]-points_l_l_l[:,0])**2)/2) \
            * np.exp(points_l_l_l[:,0]) \
            * np.exp(-((bar_z[1]-points_l_l_l[:,self.N])**2)/2)* np.exp(points_l_l_l[:,self.N]) \
            * np.exp(-((bar_z[2]-points_l_l_l[:,2*self.N])**2)/2)* np.exp(points_l_l_l[:,2*self.N]) ))


        QoI_lll= weights_l_l_l.dot(pay_lll)
        
        

         ###############################
        #5th quadrant (l,l,r)


        mylist_l_l_r=mylist_left1+mylist_left2+mylist_right3
        weights_list_l_l_r=[yknots_left[1],yknots_left[1],yknots_right[1]]

        points_l_l_r=self.cartesian(mylist_l_l_r)
        weights_l_l_r_aux=self.cartesian(weights_list_l_l_r)
        weights_l_l_r=np.prod(weights_l_l_r_aux,1)


        x_llr=np.asarray([self.stock_price_trajectory_basket_BS(bar_z[0]-points_l_l_r[i,0],points_l_l_r[i,1:]\
                                                                ,bar_z[1]-points_l_l_r[i,self.N],points_l_l_r[i,self.N+1:]\
                                                                ,points_l_l_r[i,2*self.N]+bar_z[2],points_l_l_r[i,1+2*self.N:])[0]\
                                                                for i in range(0,len(yknots_left[0])*len(yknots_left[0])\
                                                                                                        *len(yknots_right[0]))])

        

        pay_llr=np.asarray( [self.payoff(x_llr[i,:])for i in range(0,len(yknots_left[0])*len(yknots_left[0])*len(yknots_right[0]) )])




        pay_llr=np.multiply(pay_llr,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z[0]-points_l_l_r[:,0])**2)/2) \
            * np.exp(points_l_l_r[:,0]) \
            * np.exp(-((bar_z[1]-points_l_l_r[:,self.N])**2)/2)* np.exp(points_l_l_r[:,self.N]) \
            * np.exp(-((points_l_l_r[i,2*self.N]+bar_z[2])**2)/2)* np.exp(points_l_l_r[:,2*self.N])  ))   
        


        QoI_llr= weights_l_l_r.dot(pay_llr)
        
          ###############################
          #6th quadrant (l,r,r)

        mylist_l_r_r=mylist_left1+mylist_right2+mylist_right3
        weights_list_l_r_r=[yknots_left[1],yknots_right[1],yknots_right[1]]


        points_l_r_r=self.cartesian(mylist_l_r_r)
        weights_l_r_r_aux=self.cartesian(weights_list_l_r_r)
        weights_l_r_r=np.prod(weights_l_r_r_aux,1)


        x_lrr=np.asarray([self.stock_price_trajectory_basket_BS(bar_z[0]-points_l_r_r[i,0],points_l_r_r[i,1:]\
                                                                ,points_l_r_r[i,self.N]+bar_z[1],points_l_r_r[i,1+self.N:]\
                                                                ,points_l_r_r[i,2*self.N]+bar_z[2],points_l_r_r[i,1+2*self.N:])[0]\
                                                                for i in range(0,len(yknots_left[0])*len(yknots_right[0])\
                                                                                                        *len(yknots_right[0]))])
           
        
        pay_lrr=np.asarray( [self.payoff(x_lrr[i,:])for i in range(0,len(yknots_left[0])*len(yknots_right[0])*len(yknots_right[0]) )])


        
        pay_lrr=np.multiply(pay_lrr,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z[0]-points_l_r_r[:,0])**2)/2) \
            * np.exp(points_l_r_r[:,0]) \
            * np.exp(-((points_l_r_r[i,self.N]+bar_z[1])**2)/2)* np.exp(points_l_r_r[:,self.N]) \
            * np.exp(-((points_l_r_r[i,2*self.N]+bar_z[2])**2)/2)* np.exp(points_l_r_r[:,2*self.N]) ))  

        QoI_lrr= weights_l_r_r.dot(pay_lrr)

         

        ###############################
          #7th quadrant (l,r,l)


        mylist_l_r_l=mylist_left1+mylist_right2+mylist_left3
        weights_list_l_r_l=[yknots_left[1],yknots_right[1],yknots_left[1]] 

        points_l_r_l=self.cartesian(mylist_l_r_l)
        weights_l_r_l_aux=self.cartesian(weights_list_l_r_l)
        weights_l_r_l=np.prod(weights_l_r_l_aux,1)


        x_lrl=np.asarray([self.stock_price_trajectory_basket_BS(bar_z[0]-points_l_r_l[i,0],points_l_r_l[i,1:]\
                                                                ,points_l_r_l[i,self.N]+bar_z[1],points_l_r_l[i,1+self.N:]\
                                                                ,bar_z[2]-points_l_r_l[i,2*self.N],points_l_r_l[i,1+2*self.N:])[0]\
                                                                for i in range(0,len(yknots_left[0])*len(yknots_right[0])\
                                                                                                        *len(yknots_left[0]))])
        
        pay_lrl=np.asarray( [self.payoff(x_lrl[i,:])for i in range(0,len(yknots_left[0])*len(yknots_right[0])*len(yknots_left[0]))])
       


        pay_lrl=np.multiply(pay_lrl,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((bar_z[0]-points_l_r_l[:,0])**2)/2) \
            * np.exp(points_l_r_l[:,0]) \
            * np.exp(-((points_l_r_l[i,self.N]+bar_z[1])**2)/2)* np.exp(points_l_r_l[:,self.N]) \
            * np.exp(-((bar_z[2]-points_l_r_l[:,2*self.N])**2)/2)* np.exp(points_l_r_l[:,2*self.N])  ))     



        QoI_lrl= weights_l_r_l.dot(pay_lrl)


       
        ###############################
          #15th quadrant (r,l,r)

        mylist_r_l_r=mylist_right1+mylist_left2+mylist_right3
        weights_list_r_l_r=[yknots_right[1],yknots_left[1],yknots_right[1]]   

        points_r_l_r=self.cartesian(mylist_r_l_r)
        weights_r_l_r_aux=self.cartesian(weights_list_r_l_r)
        weights_r_l_r=np.prod(weights_r_l_r_aux,1) 

        
        x_rlr=np.asarray([self.stock_price_trajectory_basket_BS(points_r_l_r[i,0]+bar_z[0],points_r_l_r[i,1:]\
                                                                ,bar_z[1]-points_r_l_r[i,self.N],points_r_l_r[i,1+self.N:]\
                                                                ,points_r_l_r[i,2*self.N]+bar_z[2],points_r_l_r[i,1+2*self.N:])[0] \
                                                                for i in range(0,len(yknots_right[0])*len(yknots_left[0])\
                                                                                                        *len(yknots_right[0]))])




        pay_rlr=np.asarray( [self.payoff(x_rlr[i,:])for i in range(0,len(yknots_right[0])*len(yknots_left[0])*len(yknots_right[0]))])

        


        pay_rlr=np.multiply(pay_rlr,(((1/np.sqrt(2 * np.pi))**self.basket_d) * np.exp(-((points_r_l_r[i,0]+bar_z[0])**2)/2) \
            * np.exp(points_r_l_r[:,0]) \
            * np.exp(-((bar_z[1]-points_r_l_r[:,self.N])**2)/2)* np.exp(points_r_l_r[:,self.N]) \
            * np.exp(-((points_r_l_r[i,2*self.N]+bar_z[2])**2)/2)* np.exp(points_r_l_r[:,2*self.N])  ))


        QoI_rlr= weights_r_l_r.dot(pay_rlr)


        

       
        ###############################
        #step 4
        
        QoI_l=QoI_lll+QoI_llr+QoI_lrr+QoI_lrl
        QoI_r=QoI_rrr+QoI_rrl+QoI_rll+QoI_rlr
        QoI=(QoI_l+QoI_r)/self.exact



        
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
    def stock_price_trajectory_basket_BS(self,y1,yvec_1,y2,yvec_2,y3,yvec_3):
        bb1=self.brownian_increments(y1,yvec_1)
        bb2=self.brownian_increments(y2,yvec_2)
        bb3=self.brownian_increments(y3,yvec_3)
       

    

        dW1= [bb1[0,i+1]-bb1[0,i]  for i in range(0,self.N)] 
        dW2= [bb2[0,i+1]-bb2[0,i] for i in range(0,self.N)] 
        dW3= [bb3[0,i+1]-bb3[0,i] for i in range(0,self.N)] 
       

 

        dW=np.array([dW1 ,dW2,dW3 ])

        # construct the correlated  brownian bridge increments
        lower_triang_cholesky = np.linalg.cholesky(self.rho)
     
        dW=np.dot(lower_triang_cholesky,dW)  
          
    
       
        dW1=dW[0,:]
        dW2=dW[1,:]
        dW3=dW[2,:]
      

 

        dbb1=dW1-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        dbb2=dW2-(self.dt/np.sqrt(self.T))*y2 # brownian bridge increments dbb_i (used later for the location of the kink point)
        dbb3=dW3-(self.dt/np.sqrt(self.T))*y3 
       
          


        X=np.zeros((self.basket_d,self.N+1)) #here will store the BS trajectory
      
        X[:,0]=self.S0
        for n in range(1,self.N+1):
            X[0,n]=X[0,n-1]*(1+self.sigma[0]*dW[0,n-1])
            X[1,n]=X[1,n-1]*(1+self.sigma[1]*dW[1,n-1])
            X[2,n]=X[2,n-1]*(1+self.sigma[2]*dW[2,n-1])
     

      
        
        return X[:,-1],dbb1,dbb2,dbb3
       


        # this function defines the payoff function used here
    def payoff(self,x): 
       #print(x)
       g=(self.c.dot(x)-self.K)
   
       if g>0:
          return g
       else:
          return 0
     
   


    
    


       # Root solving procedure
 
    #Now we set up the methods used for newton iteration


       #Now we set up the methods used for newton iteration
    # def dx(self,x,yvec_1,yvec_2,j):
    #     #print x
    #     if j==1:
    #         P1,dP1=self.f(x[0],yvec_1,x[1],yvec_2,1)
    #         return np.abs(0-P1)
    #     else:
        
    #         P1,dP1=self.f(x[0],yvec_1,x[1],yvec_2,2)
    #         return np.abs(0-P1)    

    def dx(self,x,yvec_1,yvec_2,yvec_3):
        #print x
      
            P1,dP1=self.f(x[0],yvec_1,x[1],yvec_2,x[2],yvec_3)
            
       
            return np.abs(0-P1[0])  ,  np.abs(0-P1[1]) , np.abs(0-P1[2]) 


    # def f(self,y1,yvec_1,y2,yvec_2,j):# need to check this
    #     X,dbb1,dbb2=self.stock_price_trajectory_basket_BS(y1,yvec_1,y2,yvec_2) 
    #     fi=np.zeros((self.basket_d,len(dbb1)))
    #     # product=np.zeros(self.basket_d)
    #     # summation=np.zeros(self.basket_d)
    #     # Py=np.zeros(self.basket_d)
    #     # dPy=np.zeros(self.basket_d)
    #     if j==1:
    #         fi=  1+(self.sigma[0]/float(np.sqrt(self.T)))*y1*(self.dt)+self.sigma[0]*dbb1
    #         product=np.prod(fi)
            
    #         Py=product-(self.K/(float(self.S0[0]*self.c[0])))
    #         summation=np.sum(1/fi)
    #         dPy=  (self.sigma[0]/float(np.sqrt(self.T)))*(self.dt)*product*summation
    #     else:    
        
    #         fi=  1+(self.sigma[1]/float(np.sqrt(self.T)))*y2*(self.dt)+self.sigma[1]*dbb2
    #         product=np.prod(fi)
    #         Py=product-(self.K/(float(self.S0[1]*self.c[1])))
    #         summation=np.sum(1/fi)
    #         dPy=  (self.sigma[1]/float(np.sqrt(self.T)))*(self.dt)*product*summation
    #     return Py,dPy    
    


    def f(self,y1,yvec_1,y2,yvec_2,y3,yvec_3):# need to check this
        X,dbb1,dbb2,dbb3=self.stock_price_trajectory_basket_BS(y1,yvec_1,y2,yvec_2,y3,yvec_3) 
        fi=np.zeros((self.basket_d,len(dbb1)))
        product=np.zeros(self.basket_d)
        summation=np.zeros(self.basket_d)
        Py=np.zeros(self.basket_d)
        dPy=np.zeros(self.basket_d)

        fi[0,:]=  1+(self.sigma[0]/float(np.sqrt(self.T)))*y1*(self.dt)+self.sigma[0]*dbb1
        fi[1,:]=  1+(self.sigma[1]/float(np.sqrt(self.T)))*y2*(self.dt)+self.sigma[1]*dbb2
        fi[2,:]=  1+(self.sigma[2]/float(np.sqrt(self.T)))*y3*(self.dt)+self.sigma[2]*dbb3
 
        
        product[0]=np.prod(fi[0,:])
        product[1]=np.prod(fi[1,:])
        product[2]=np.prod(fi[2,:])
       

        Py[0]=product[0]-(self.K/(float(self.S0[0]*self.c[0]*self.basket_d)))
        Py[1]=product[1]-(self.K/(float(self.S0[1]*self.c[1]*self.basket_d)))
        Py[2]=product[2]-(self.K/(float(self.S0[2]*self.c[2]*self.basket_d)))
       

       

        summation[0]=np.sum(1/fi[0,:])
        summation[1]=np.sum(1/fi[1,:])
        summation[2]=np.sum(1/fi[2,:])
    

        
        dPy[0]= (self.sigma[0]/float(np.sqrt(self.T)))*(self.dt)*product[0]*summation[0]
        dPy[1]=  (self.sigma[1]/float(np.sqrt(self.T)))*(self.dt)*product[1]*summation[1]
        dPy[2]= (self.sigma[2]/float(np.sqrt(self.T)))*(self.dt)*product[2]*summation[2]
       

        return Py,dPy    
    

        
    def newtons_method(self,x0,yvec_1,yvec_2,yvec_3,eps=1e-10):
        
        delta1, delta2,delta3 = self.dx(x0,yvec_1,yvec_2,yvec_3)


      
        while (delta1 > eps) | (delta2 > eps)| (delta3 > eps) :
        
            P_value,dP=self.f(x0[0],yvec_1,x0[1],yvec_2,x0[2],yvec_3)
       
            x0[0] = x0[0] - 0.1*P_value[0]/dP[0]
            x0[1] = x0[1] - 0.1*P_value[1]/dP[1]
            x0[2] = x0[2] - 0.1*P_value[2]/dP[2]
            

            delta1, delta2,delta3  = self.dx(x0,yvec_1,yvec_2,yvec_3)

        return x0     

    # def newtons_method(self,x0,yvec_1,yvec_2,eps=1e-5):
        
    #     delta1= self.dx(x0,yvec_1,yvec_2,1)
    #     delta2= self.dx(x0,yvec_1,yvec_2,2)
        
      
    #     while (delta1 > eps): 
        
    #         P_value,dP=self.f(x0[0],yvec_1,x0[1],yvec_2,1)
       
    #         x0[0] = x0[0] - 0.1*P_value/dP
           
    #         delta1= self.dx(x0,yvec_1,yvec_2,1)

    #     while (delta2 > eps): 
        
    #         P_value,dP=self.f(x0[1],yvec_1,x0[1],yvec_2,2)
       
    #         x0[1] = x0[1] - 0.1*P_value/dP

    #         delta2 = self.dx(x0 ,yvec_1,yvec_2,2)    

    #     return x0   




    




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
