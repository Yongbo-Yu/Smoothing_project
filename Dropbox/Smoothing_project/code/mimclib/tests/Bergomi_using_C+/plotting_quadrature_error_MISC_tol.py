
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


dt_arr=np.array([0.5, 0.1,  0.05, 0.01, 0.001, 0.0001])

# #Case for parameter set 1 non richardson
# quad_err_N_2=np.array([ 0.1180, 0.1180, 0.1180,0.0295, 0.0295 , 0.0295])
# quad_err_N_4=np.array([0.0449, 0.0449, 0.0477, 0.0211, 0.0112,0.0112 ])
# quad_err_N_8=np.array([0.0449, 0.0772, 0.0379, 0.0309, 0.0042,0.0084   ])
# quad_err_N_16=np.array([0.0197, 0.0492, 0.0800, 0.0070, 0.0070 ])

#Case for parameter set 1 with richardson
quad_err_N_2=np.array([ 0.1685,  0.1685, 0.1685,0.00001, 0.0183, 0.0183 ])
quad_err_N_4=np.array([0.0435, 0.0435, 0.1109,0.0407,0.0197,0.0154])
quad_err_N_8=np.array([  0.0197, 0.0899, 0.0730,0.0337, 0.0014,0.0014])
quad_err_N_16=np.array([0.0014, 0.0674,0.0632, 0.0014,0.0112  ])


# #Case for parameter set 2 non richardson
# quad_err_N_2=np.array([ 0.2071, 0.2071, 0.2071, 0.0745,0.0379,0.0354])
# quad_err_N_4=np.array([0.1048,0.1048, 0.0997, 0.0177,0.0303])
# quad_err_N_8=np.array([ 0.1313, 0.1490,0.0126, 0.0063,0.0076 ])
# quad_err_N_16=np.array([0.1490, 0.0631,0.0417 ])




##Case for parameter set 2 with richardson
# quad_err_N_2=np.array([0.3914, 0.3914, 0.3914,0.1174,0.0593,0.0657])
# quad_err_N_4=np.array([0.1187, 0.1187,0.2146,0.0013,0.0240])
# quad_err_N_8=np.array([ 0.1540, 0.1843, 0.0050, 0.0013])
# quad_err_N_16=np.array([0.1654, 0.0410])



##Case for parameter set 3 non richardson
# quad_err_N_2=np.array([0.0605,0.0605,0.0605,0.0338,0.0107, 0.0124])
# quad_err_N_4=np.array([0.0787, 0.0787,0.0889,0.0160,0.0142])
# quad_err_N_8=np.array([ 0.0743,0.0507,0.0205,0.0027 ])
# quad_err_N_16=np.array([0.0831,0.0231,0.0245])



# ##Case for parameter set 3 with richardson
# quad_err_N_2=np.array([])
# quad_err_N_4=np.array([])
# quad_err_N_8=np.array([])
# quad_err_N_16=np.array([])

		

	




fig = plt.figure()

plt.plot(dt_arr, quad_err_N_2,linewidth=2.0,label=r'$N=2$' , marker='>',hold=True) 
plt.plot(dt_arr[0:6], quad_err_N_4,linewidth=2.0,label=r'$N=4$' , marker='o',hold=True) 
plt.plot(dt_arr[0:6], quad_err_N_8,linewidth=2.0,label=r'$N=8$'  , marker='v',hold=True) 
plt.plot(dt_arr[0:5], quad_err_N_16,linewidth=2.0,label=r'$N=16$'  , marker='*',hold=True) 

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$TOL_{MISC}$',fontsize=14)




plt.ylabel('relative Quadrature error',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='lower left')
plt.savefig('./results/relative_quad_error_wrt_MISC_TOL_set1_with_rich.eps', format='eps', dpi=1000)  

