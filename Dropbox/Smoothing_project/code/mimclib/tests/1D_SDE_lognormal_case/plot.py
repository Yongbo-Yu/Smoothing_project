

import numpy as np
import time
import scipy.stats as ss

import random



import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

from joblib import Parallel, delayed
import multiprocessing

z=np.zeros(2)
z3=np.zeros(2)    
z_diff= np.zeros(2)   
Nsteps_arr=np.array([2,4,8,16])
dt_arr=1.0/(Nsteps_arr)

error=[0.02044204, 0.01586867, 0.00911185, 0.00490212]


Lb=[0.01956894, 0.01494066, 0.00833106, 0.00430642]
Ub=[0.02131514, 0.01679669, 0.00989265, 0.00549782]
 

error_diff=[0.00457337, 0.00675682, 0.00420974]
Ub_diff=[0.00584645, 0.00796899, 0.00518842]
Lb_diff=[0.00330028,0.00554465, 0.00323105]

z[0]=0.698055154393



z= np.polyfit(np.log(dt_arr), np.log(error), 1)
fit=np.exp(z[0]*np.log(dt_arr))
     


z3=np.zeros(2)
z3[0]=1.0
z3[1]=np.log(error[0]*2)
fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])






z_diff[0]=0.059763851122




z_diff= np.polyfit(np.log(dt_arr[0:3]), np.log(error_diff), 1)
fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:3]))


z3diff=np.zeros(2)
z3diff[0]=1.0
z3diff[1]=np.log(error_diff[0]*2)
fit3diff=np.exp(z3diff[0]*np.log(dt_arr[0:3])+z3diff[1])

fig = plt.figure()

plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
plt.plot(dt_arr, Lb,linewidth=2.0,label='Lb' ,linestyle = '--', hold=True) 
plt.plot(dt_arr, Ub,linewidth=2.0,label='Ub' ,linestyle = '--', hold=True) 
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\Delta t$',fontsize=14)

plt.plot(dt_arr, fit,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--', marker='o')
plt.plot(dt_arr, fit3,linewidth=2.0,label=r'rate= %s' % format(z3[0]  , '.2f'), linestyle = '--', marker='o')


plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper left')
plt.savefig('./results/weak_convergence_order_call_option_relative.eps', format='eps', dpi=1000)  

fig = plt.figure()
plt.plot(dt_arr[0:3], error_diff,linewidth=2.0,label='weak_error' ,linestyle = '--',marker='>', hold=True) 
plt.plot(dt_arr[0:3], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = '--', hold=True) 
plt.plot(dt_arr[0:3], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = '--', hold=True) 
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\Delta t$',fontsize=14)

plt.plot(dt_arr[0:3], fit_diff,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--', marker='o')
plt.plot(dt_arr[0:3], fit3diff,linewidth=2.0,label=r'rate= %s' % format(z3diff[0]  , '.2f'), linestyle = '--', marker='o')
plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper left')
plt.savefig('./results/weak_convergence_order_differences_call_option_relative.eps', format='eps', dpi=1000)  
