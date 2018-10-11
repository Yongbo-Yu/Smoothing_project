
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


dt_arr=np.array([0.5, 0.1,  0.05, 0.01, 0.001,0.0001,0.00001])

#Case  non richardson
quad_err_N_2=np.array([0.0015,0.0015,0.0015,0.0027,0.0027,0.0027,0.0027])
quad_err_N_4=np.array([0.0108,4.4e-04,6.2e-04,6.0e-04,6.0e-04,6.2e-04])
quad_err_N_8=np.array([ 0.0035,0.0020,0.0019, 0.0019])
quad_err_N_16=np.array([0.0015,2.9e-04])












fig = plt.figure()

plt.plot(dt_arr, quad_err_N_2,linewidth=2.0,label=r'$N=2$' , marker='>',hold=True) 
plt.plot(dt_arr[0:6], quad_err_N_4,linewidth=2.0,label=r'$N=4$' , marker='o',hold=True) 
plt.plot(dt_arr[0:4], quad_err_N_8,linewidth=2.0,label=r'$N=8$'  , marker='v',hold=True) 
plt.plot(dt_arr[0:2], quad_err_N_16,linewidth=2.0,label=r'$N=16$'  , marker='*',hold=True) 

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$TOL_{MISC}$',fontsize=14)




plt.ylabel('relative Quadrature error',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper left')
plt.savefig('./results/relative_quad_error_wrt_MISC_TOL_non_rich.eps', format='eps', dpi=1000)  

