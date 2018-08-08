import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

# This function implements the brownian bridge construction in 1 D
def BB(T,d):
  steps=pow(2,d)
  dt = T/float(steps)
  t=np.linspace(0, T, steps+1)  
  
  mean = np.zeros(steps)
  covariance= np.identity(steps)
  y = np.random.multivariate_normal(mean, covariance)
  h=steps
  j_max=1
  bb= np.zeros(steps+1)
  bb[h]=np.sqrt(T)*y[h-1]
  for k in range(1,d+1):
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
  return steps,bb



      
    

# This function simulates a 1D BS trajectory for stock price  
def stock_price_trajectory_1D_BS(T,d,X0,sigma):
    steps,bb=BB(T,d)
    print(bb)
    print(bb[-1])
    dW= [bb[i+1]-bb[i] for i in range(len(bb)-1)]
    print(dW)
    #dW=[x - bb[i - 1] for i, x in enumerate(bb)][1:] #brownian motion increments dW_i
    dbb=dW-(1/float(steps))*bb[-1] # brownian bridge increments dbb_i (used later for the location of the kink point)
    print(dbb)
    X=np.zeros(steps+1) #here will store the BS trajectory
    X[0]=X0
    for n in range(1,steps+1):
        X[n]=X[n-1]*(1+sigma*dW[n-1])
    return X[-1],dbb

# This function gives the exact location of the kink for 1D BS
def exact_location_kink_continuous_1D_BS(K,X0,T,sigma):
    y=(np.log(K/float(X0))+(float(T*pow(sigma,2))/2))*(1/(np.sqrt(T)*sigma))
    print(y)
    return y
       
#Now we set up the methods used for newton iteration
def dx(x,T,d,X0,sigma,K):
    P,dP=f(x,T,d,X0,sigma,K)
    return abs(0-P)
 
def newtons_method(x0,eps,K,X0,T,sigma,d):
    delta = dx(x0,T,d,X0,sigma,K)
    while delta > eps:
        P,dP=f(x0,T,d,X0,sigma,K)
        print(P)
        x0 = x0 - P/dP
        delta = dx(x0,T,d,X0,sigma,K)
    print 'Root is at: ', x0
    print 'f(x) at root is: ', P
    return x0

def f(y,T,d,X0,sigma,K):
    X,dbb=stock_price_trajectory_1D_BS(T,d,X0,sigma)
    print(dbb)
    fi=1+(sigma/float(np.sqrt(T)))*y*(T/float(pow(2,d)))+sigma*dbb
    product=np.prod(fi)
    summation=np.sum(1/fi)
    Py=product-(K/float(X0))
    dPy=(sigma/float(np.sqrt(T)))*(T/float(pow(2,d)))*product*summation
    return Py,dPy

#this function defines the BS call payoff
def payoff(K,T,d,X0,sigma):
  x=stock_price_trajectory_1D_BS(T,d,X0,sigma)[0]
  
  g=max(x-K,0)
  return g    
#def kink_location_1D_BS_plot(T,d,X0,sigma,K):
 #   y= np.arange(-10, 10, 0.5)
#    P=np.zeros(np.size(y))
   # for i in range(0,np.size(y)):
    #     Pi,dPi=f(y[i],T,d,X0,sigma,K)
     #    P[i]=Pi 
    #plt.plot(y, P,linewidth=2.0) 
    #plt.xlabel('y',fontsize=14)
    #plt.ylabel('P',fontsize=14)  
    #plt.title('kink location plot for 1D BS',fontsize=14)
#kink_location_1D_BS_plot(10,7,1,0.1,5)   
def error_kink_location_1D_BS_plot(x0,eps,K,X0,T,sigma):
    y= exact_location_kink_continuous_1D_BS(K,X0,T,sigma) 
    print y
    i=-1
    ny=np.zeros(6)
    err=np.zeros(6)
    steps=np.zeros(6)
    for d in range(2,8):
         i=i+1
         ny[i]=newtons_method(x0,eps,K,X0,T,sigma,d)
         err[i]=np.abs(ny[i]-y)
         steps[i]= pow(2,d)
    plt.figure()
    plt.plot(steps, err,'bs',linewidth=2.0) 
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$T/\Delta$ t',fontsize=14)
    plt.ylabel('Error',fontsize=14)  
    plt.title('Newton error for the kink location for 1D BS (K=%s,$X_0$=%s,T=%s,$\sigma$=%s)'%(K,X0,T,sigma),fontsize=14)  
    if X0>K:
        plt.savefig('./results/kink_location_1D_BS_in_the_money.eps', format='eps', dpi=1000)
    elif X0<K:
        plt.savefig('./results/kink_location_1D_BS_out_the_money.eps', format='eps', dpi=1000)
    else:
        plt.savefig('./results/kink_location_1D_BS_at_the_money.eps', format='eps', dpi=1000)
    
#error_kink_location_1D_BS_plot(0, 1e-5,5,1,10,0.1)
#error_kink_location_1D_BS_plot(0, 1e-5,5,5,10,0.1)
#error_kink_location_1D_BS_plot(0, 1e-5,5,10,10,0.1)
 
#z=np.zeros(1)
#for k in range(1,2):
 #  z[k-1]=newtons_method(0,1e-2,100,100,1,0.4,2)
#print(np.mean(z))
print(exact_location_kink_continuous_1D_BS(100,100,1,0.4))