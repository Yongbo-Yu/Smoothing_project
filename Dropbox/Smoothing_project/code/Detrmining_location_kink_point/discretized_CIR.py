import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

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
  return dt,steps,bb

# This function simulates a 1D CIR trajectory for stock price  using full truncation scheme
def stock_price_trajectory_1D_CIR_full_truncation(T,d,X0,sigma,a,b):
    dt,steps,bb=BB(T,d)
    dW=[x - bb[i - 1] for i, x in enumerate(bb)][1:] #brownian motion increments dW_i
    dbb=dW-(1/float(steps))*bb[-1] # brownian bridge increments dbb_i (used later for the location of the kink point)
    X=np.zeros(steps+1) #here will store the CIR trajectory
    X[0]=X0
    for n in range(1,steps+1):
        X[n]=X[n-1]*(1-a *dt)+ sigma *np.sqrt(np.max(X[n-1],0))*dW[n-1]+a*b*dt
    t=np.linspace(0, T, steps+1)     
    plt.figure()
    plt.plot(t,X,linewidth=2.0) 
    plt.xlabel('t',fontsize=14)
    plt.ylabel('CIR process',fontsize=14)  
    plt.title('1D CIR process simulated by full truncation FE ($X_0$=%s,T=%s,$\sigma$=%s,$a$=%s,$b$=%s)'%(X0,T,sigma,a,b),fontsize=14)  
    plt.savefig('./results/1D_CIR_FE.eps', format='eps', dpi=1000)    
    return X
    
stock_price_trajectory_1D_CIR_full_truncation(1,9,0.04,1,0.5,0.04)

# This function simulates a 1D CIR trajectory for stock price  using full reflection scheme
#def stock_price_trajectory_1D_CIR_reflection(T,d,X0,sigma,a,b):
#    dt,steps,bb=BB(T,d)
#    dW=[x - bb[i - 1] for i, x in enumerate(bb)][1:] #brownian motion increments dW_i
#    dbb=dW-(1/float(steps))*bb[-1] # brownian bridge increments dbb_i (used later for the location of the kink point)
#    X=np.zeros(steps+1) #here will store the CIR trajectory
#    X[0]=X0
#    for n in range(1,steps+1):
#        X[n]=np.abs(X[n-1]*(1-a *dt)+ sigma *np.sqrt(X[n-1])*dW[n-1]+a*b*dt)
#    return X


#Now we set up the methods used for newton iteration



def f_full_truncation(y,T,d,X0,sigma,a,b,K):
    dt,steps,bb=BB(T,d)
    dW=[x - bb[i - 1] for i, x in enumerate(bb)][1:] #brownian motion increments dW_i
    dbb=dW-(1/float(steps))*bb[-1] # brownian bridge increments dbb_i 
    f_i=np.zeros(steps+1) 
    df_i=np.zeros(steps+1) 
    f_i[0]=X0
    df_i[0]=0
    for n in range(1,steps+1):
        f_i[n]=f_i[n-1]*(1-a *dt)+ sigma *np.sqrt(np.max(f_i[n-1],0))*((y/float(np.sqrt(T)))* dt+dbb[n-1])+a*b*dt
        df_i[n]=df_i[n-1]*(1-a *dt)+ sigma *np.sqrt(np.max(f_i[n-1],0))*(dt/float(np.sqrt(T)))+ sigma *(((y/float(np.sqrt(T)))* dt)+dbb[n-1]) *(df_i[n-1]/float(2))*np.sqrt(np.max(f_i[n-1],0))
    Py=f_i[-1]-(K/float(X0))
    dPy=df_i[-1]
    return Py,dPy
def kink_location_1D_CIR_plot(T,d,X0,sigma,a,b,K):
    y= np.arange(50, 200, 0.5)
    P=np.zeros(np.size(y))
    for i in range(0,np.size(y)):
         Pi,dPi=f_full_truncation(y[i],T,d,X0,sigma,a,b,K)
         P[i]=Pi 
    plt.plot(y, P,linewidth=2.0) 
    plt.xlabel('y',fontsize=14)
    plt.ylabel('P',fontsize=14)  
    plt.title('kink location plot for 1D CIR',fontsize=14)
kink_location_1D_CIR_plot(1,8,0.04,0.1,0.5,0.04,1)    
def dx(x,T,d,X0,sigma,a,b,K):
    P,dP=f_full_truncation(x,T,d,X0,sigma,a,b,K)
    return abs(0-P)
 
    
def newtons_method(x0,eps,K,X0,T,sigma,a,b,d):
    delta = dx(x0,T,d,X0,sigma,a,b,K)
    while delta > eps:
        P,dP=f_full_truncation(x0,T,d,X0,sigma,a,b,K)
        print P,dP
        x0 = x0 - 100 *P/dP
        print x0
        delta = dx(x0,T,d,X0,sigma,a,b,K)
    print 'Root is at: ', x0
    print 'f(x) at root is: ', P

#X1=stock_price_trajectory_1D_CIR_full_truncation(1,8,0.04,0.2,0.5,0.04)
#newtons_method(80, 1e-2,1,0.04,1,0.1,0.5,0.04,8)