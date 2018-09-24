
import numpy as np
import time
import scipy.stats as ss

import random

#from joblib import Parallel, delayed
#import multiprocessing
from numba import autojit, prange

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


import fftw3
import RBergomi
from RBergomi import *
import mimclib.misc as misc
exact=0.2407117  #exact value of K=0.8, H=0.02_xi_01_eta_0_4_r__07


marker=['>', 'v', '^', 'o', '*','+','-',':']
ax = figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# # feed parameters to the problem
Nsteps_arr=np.array([2,4,8,16])
dt_arr=1.0/(Nsteps_arr)
error_diff=np.zeros(3)
stand_diff=np.zeros(3)
error=np.zeros(4)
stand=np.zeros(4)
elapsed_time_qoi=np.zeros(4)
Ub=np.zeros(4)
Lb=np.zeros(4)
Ub_diff=np.zeros(3)
Lb_diff=np.zeros(3)
values=np.array([(0.242008/float(exact))-1, (0.241559/float(exact))-1, (0.241409/float(exact))-1, (0.2412931/float(exact))-1])
error=np.abs(values)


stand=np.array([3.4e-04,3.4e-04,3.3e-04,3.3e-04])   
Ub=error+1.96*stand
Lb=error-1.96*stand
differences= [values[i]-values[i+1] for i in range(0,3)]
error_diff=np.abs(differences)
print error_diff 
stand_diff=np.array([3.4e-04,3.4e-04,3.3e-04])         
Ub_diff=error_diff+1.96*stand_diff
Lb_diff=error_diff-1.96*stand_diff
print Ub_diff
print Lb_diff 
  
 
z= np.polyfit(np.log(dt_arr), np.log(error), 1)
fit=np.exp(z[0]*np.log(dt_arr))
print z[0]

z_diff= np.polyfit(np.log(dt_arr[0:3]), np.log(error_diff), 1)
fit_diff=np.exp(z_diff[0]*np.log(dt_arr[0:3]))
print z_diff[0]     




z3=np.zeros(4)
z3[0]=1.0
z3[1]=np.log(error[0])
fit3=np.exp(z3[0]*np.log(dt_arr)+z3[1])


z3diff=np.zeros(3)
z3diff[0]=1.0
z3diff[1]=np.log(error_diff[0])
fit3diff=np.exp(z3diff[0]*np.log(dt_arr[0:3])+z3diff[1])

fig = plt.figure()

plt.plot(dt_arr, error,linewidth=2.0,label='weak_error' , marker='>',hold=True) 
plt.plot(dt_arr, Lb,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
plt.plot(dt_arr, Ub,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\Delta t$',fontsize=14)

plt.plot(dt_arr, fit*10,linewidth=2.0,label=r'rate= %s' % format(z[0]  , '.2f'), linestyle = '--')
plt.plot(dt_arr, fit3*10,linewidth=2.0,label=r'rate= %s' % format(z3[0]  , '.2f'), linestyle = '--')


plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X) \mid $',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper left')
plt.savefig('./results/weak_convergence_order_Bergomi_H_002_K_08_M_5_10_6_CI_relative.eps', format='eps', dpi=1000)  

fig = plt.figure()
plt.plot(dt_arr[0:3], error_diff,linewidth=2.0,label='weak_error' , marker='>', hold=True) 
plt.plot(dt_arr[0:3], Lb_diff,linewidth=2.0,label='Lb' ,linestyle = ':', hold=True) 
plt.plot(dt_arr[0:3], Ub_diff,linewidth=2.0,label='Ub' ,linestyle = ':', hold=True) 
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\Delta t$',fontsize=14)

plt.plot(dt_arr[0:3], fit_diff*10,linewidth=2.0,label=r'rate= %s' % format(z_diff[0]  , '.2f'), linestyle = '--')
plt.plot(dt_arr[0:3], fit3diff*10,linewidth=2.0,label=r'rate= %s' % format(z3diff[0]  , '.2f'), linestyle = '--')
plt.ylabel(r'$\mid  g(X_{\Delta t})-  g(X_{\Delta t/2}) \mid $',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper left')
plt.savefig('./results/weak_convergence_order_differences_Bergomi_H_002_K_08_M_5_10_6_CI_relative.eps', format='eps', dpi=1000)  




