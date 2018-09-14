
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


dt_arr=np.array([0.5, 0.1,  0.05, 0.01, 0.001])

#Case  non richardson
quad_err_N_2=np.array([0.0007,0.0007,0.0007,0.0007,0.0007])
quad_err_N_4=np.array([0.0018,0.0018,0.0018,0.0003,0.0003])
quad_err_N_8=np.array([ 6.3e-05,0.0002,0.0002 ,0.0002,0.0002 ])
quad_err_N_16=np.array([0.0004,0.0001,0.0001,0.0001])



# #Case  with richardson
# quad_err_N_2=np.array([2.5e-04,2.5e-04,2.5e-04,2.5e-04,2.5e-04])
# quad_err_N_4=np.array([4.4e-04,4.4e-04,2.1e-04,2.1e-04,1.5e-04])
# quad_err_N_8=np.array([ 7.6e-05,1.1e-04,9.5e-05,7.6e-05,9.5e-05])
# quad_err_N_16=np.array([3.4e-04,2.3e-04,2.1e-04,1.7e-04])









fig = plt.figure()

plt.plot(dt_arr, quad_err_N_2,linewidth=2.0,label=r'$N=2$' , marker='>',hold=True) 
plt.plot(dt_arr[0:5], quad_err_N_4,linewidth=2.0,label=r'$N=4$' , marker='o',hold=True) 
plt.plot(dt_arr[0:5], quad_err_N_8,linewidth=2.0,label=r'$N=8$'  , marker='v',hold=True) 
plt.plot(dt_arr[0:4], quad_err_N_16,linewidth=2.0,label=r'$N=16$'  , marker='*',hold=True) 

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$TOL_{MISC}$',fontsize=14)




plt.ylabel('relative Quadrature error',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper left')
plt.savefig('./results/relative_quad_error_wrt_MISC_TOL_non_rich.eps', format='eps', dpi=1000)  

