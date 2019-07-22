import numpy as np
import time
 
import random
 

 
import pathos.multiprocessing as mp
import pathos.pools as pp

#plotting modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
 
class Problem(object):

# attributes
    random_gen=None;
    elapsed_time=0.0;
    S0=None     # vector of initial stock prices
    K=None         # Strike price
    #T=10.0                      # maturity
    T=1.0                      
    sigma=None    # volatility
    d=None
    dt=None

    #exact=6.332542 #  S_0=K=100, T=1, r=0,rho=-0.9, v_0=0.04, theta=0.0025, xi=0.1,\kapp=1  
    exact=10.86117 #  S_0=K=100, T=1, r=0,rho=-0.3, v_0=0.09, theta=0.09, xi=1,\kapp=2.7778 set 4
    yknots_right=[]
    yknots_left=[]


#methods
    # this method initializes 
    def __init__(self,coeff,Nsteps,nested=False):
        self.nested = nested
        self.random_gen = None or np.random
        self.S0=100
        self.K= coeff*self.S0        # Strike price and coeff determine if we have in/at/out the money option
        
        self.rho=-0.3
        self.kappa=2.7778
        self.xi=1.0
        self.v0=0.09
        self.theta=0.09
        #self.theta=(self.xi**2)/(4*self.kappa)
        
        # paramters for the bessel process
        self.beta=self.xi/float(2)
        self.alpha=-self.kappa/float(2)

       # self.K= coeff*self.S0   
        self.dt=self.T/float(Nsteps) # time steps length
        self.d=int(np.log2(Nsteps)) #power 2 number steps

        # For less than 185 points
        #beta=32
        #self.yknots_right=np.polynomial.laguerre.laggauss(beta)
      
        # For more than 185 points
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
        


     # objfun:  beta #number of points in the first direction
    def objfun(self,Nsteps):

        mean = np.zeros(2*Nsteps)
        covariance= np.identity(2*Nsteps)
        y = np.random.multivariate_normal(mean, covariance)   
       
        

        # step 1 # get the two partitions of coordinates y_1 for the volatility path  and y_s for  the asset path  
        y1=y[0:Nsteps] # this points are related to the volatility path

        y2=[Nsteps]
      
        y2=y[Nsteps:]

        y2s=y2[1:]


        
   
     

        x=self.stock_price_trajectory_1D_heston(y2[0],y2[1:],y1[0],y1[1:Nsteps],Nsteps)  
        

                
        return x


    # This function implements the brownian bridge construction in 1 D, the argument y here represent independent  multivariate gaussian  rdv  given by
    #mean = np.zeros(self.N)
    #covariance= np.identity(self.N)
    #y = np.random.multivariate_normal(mean, covariance)
    #y1: is the first direction given by W(T)/sqrt(T)/   #This function gives  the brownian motion increments built from Brownian bridge construction: # the composition of  BB and brownian_increments give us
    # the function \phi in our notes(discussion)
        
    def brownian_increments(self,y1,y,Nsteps):
        t=np.linspace(0, self.T, Nsteps+1)     
        h=Nsteps
        j_max=1
        bb= np.zeros((1,Nsteps+1))
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
    def stock_price_trajectory_1D_heston(self,y1,y,yv1,yv,Nsteps):
        bb=self.brownian_increments(y1,y,Nsteps)
        dW= [bb[0,i+1]-bb[0,i] for i in range(0,Nsteps)] 
    
        #  hierarhcical
       # bb_v=self.brownian_increments(yv1,yv)
        #dW_v= [bb_v[0,i+1]-bb_v[0,i] for i in range(0,self.N)] 

        # # non hierarhcical
        dW_v=[]
        dW_v.append(yv1)
        dW_v[1:]=[np.array(yv[i]) for i in range(0,len(yv))]
        dW_v=np.array(dW_v)
        

        
        dW_s= self.rho *np.array(dW_v)*np.sqrt(self.dt) + np.sqrt(1-self.rho**2) * np.array(dW)
        y1s= self.rho *yv1 + np.sqrt(1-self.rho**2) * y1


        #option1 
        # dbb1=dW-(self.dt/np.sqrt(self.T))*y1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        # dbbv=dW_v*np.sqrt(self.dt) -(self.dt/np.sqrt(self.T))*yv1 # brownian bridge increments dbb_i (used later for the location of the kink point)
        # dbb_s= self.rho *np.array(dbbv) + np.sqrt(1-self.rho**2) * np.array(dbb1)
        # #option2
        dbb_s=dW_s-(self.dt/np.sqrt(self.T))*y1s



        X=np.zeros(Nsteps+1) #here will store the asset trajectory
        X_v=np.zeros(Nsteps+1) #here will store the  Bessel process trajectory
        V=np.zeros(Nsteps+1) #here will store the  volatility trajectory

        X[0]=self.S0
        V[0]=self.v0
        X_v[0]=np.sqrt(self.v0)
        
        
        for n in range(1,Nsteps+1):
            X[n]=X[n-1]*(1+np.sqrt(V[n-1])*dW_s[n-1])
            X_v[n]=X_v[n-1]*(1+self.alpha*self.dt)+self.beta*dW_v[n-1]*np.sqrt(self.dt)
            V[n]=X_v[n]**2
            
        return V
       
                 
 
 
def trajectory_plotting():    
 
   
        

      
        # # feed parameters to the problem
        Nsteps_arr=np.array([4])
        dt_arr=1.0/(Nsteps_arr)
  
        
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        values=np.zeros((10**2,Nsteps_arr[0]+1))
        for i in range(0,1):
            print i
            start_time=time.time()

            prb = Problem(1,Nsteps_arr[i]) 
            t=np.linspace(0, prb.T, Nsteps_arr[0]+1)    

            for j in range(1*(10**2)):
                        
                values[j,:]=prb.objfun(Nsteps_arr[i])
                plt.plot( t,values[j,:],linewidth=2.0,linestyle = '--',hold=True) 
            # # #prb = Problem(Nsteps_arr[i]) 
            # def processInput(j):
            #     return prb.objfun(Nsteps_arr[i])/float(exact)
 
            
            # p =  pp.ProcessPool(num_cores)  # Processing Pool with four processors
            
            # values= p.map(processInput, range(((1*(10**1))))  )

        
        
        #plt.xscale('log')
        plt.xlabel('time',fontsize=14)
        plt.ylabel('V',fontsize=14)  
        # plt.legend(loc='upper right')
        plt.savefig('./results/paths_smooth_vol_scheme_set3_N4_V.eps', format='eps', dpi=1000) 

 
 
          
      
 
        
        
trajectory_plotting()   