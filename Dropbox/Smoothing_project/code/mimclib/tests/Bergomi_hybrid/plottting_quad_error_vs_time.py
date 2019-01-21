import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



# #Case for parameter set 1 non richardson
# quad_err_N_2=np.array([ 0.1180, 0.1180, 0.1180,0.0295, 0.0295 , 0.0295])
# misc_time_N_2=np.array([ 0.1,0.1,0.1,0.2,2,6])
# quad_err_N_4=np.array([0.0449, 0.0449, 0.0477, 0.0211, 0.0112,0.0112 ])
# misc_time_N_4=np.array([0.1,0.1,0.3, 1, 11,96])
# quad_err_N_8=np.array([0.0449, 0.0772, 0.0379, 0.0309, 0.0042,0.0084  ])
# misc_time_N_8=np.array([0.2,0.6,3 ,9,243,5760])
# quad_err_N_16=np.array([0.0197, 0.0492, 0.0800, 0.0070, 0.0070 ])
# misc_time_N_16=np.array([ 0.4,6,14,215,4650])



# #Case for parameter set 1 with richardson
# quad_err_N_2=np.array([ 0.1685,  0.1685, 0.1685,0.00001, 0.0183, 0.0183 ])
# misc_time_N_2=np.array([ 0.1,0.1,0.1,1,4,7])
# quad_err_N_4=np.array([0.0435, 0.0435, 0.1109,0.0407,0.0197,0.0154])
# misc_time_N_4=np.array([ 0.1,0.1,0.4,2,12,191])
# quad_err_N_8=np.array([  0.0197, 0.0899, 0.0730,0.0337, 0.0014,0.0014])
# misc_time_N_8=np.array([0.2,0.6,2 ,18,664,7650])
# quad_err_N_16=np.array([0.0014, 0.0674,0.0632, 0.0014,0.0112 ])
# misc_time_N_16=np.array([0.5,8,38,490,54065])




# #Case for parameter set 2 non richardson
# quad_err_N_2=np.array([ 0.2071, 0.2071, 0.2071, 0.0745,0.0379,0.0354])
# misc_time_N_2=np.array([ 0.08,0.08,0.08,0.45,7,63])
# quad_err_N_4=np.array([0.1048,0.1048, 0.0997, 0.0177,0.0303])
# misc_time_N_4=np.array([ 0.13,0.13,0.3,6,350])
# quad_err_N_8=np.array([ 0.1313, 0.1490,0.0126, 0.0063,0.0076 ])
# misc_time_N_8=np.array([0.2 ,1,10,800,5370])
# quad_err_N_16=np.array([0.1490, 0.0631,0.0417 ])
# misc_time_N_16=np.array([ 0.5,220,5600])


	

# #Case for parameter set 2 with richardson
# quad_err_N_2=np.array([0.3914, 0.3914, 0.3914,0.1174,0.0593,0.0657])
# misc_time_N_2=np.array([0.1,0.1,0.1,2,16,180])
# quad_err_N_4=np.array([0.1187, 0.1187,0.2146,0.0013,0.0240])
# misc_time_N_4=np.array([ 0.13,0.13,0.5,9,2460])
# quad_err_N_8=np.array([ 0.1540, 0.1843, 0.0050, 0.0013])
# misc_time_N_8=np.array([0.2,1.3,34,3450])
# quad_err_N_16=np.array([0.1654, 0.0410])
# misc_time_N_16=np.array([0.5,1198 ])

		
		
	

# #Case for parameter set 3 non richardson
# quad_err_N_2=np.array([0.0605,0.0605,0.0605,0.0338,0.0107, 0.0124])
# misc_time_N_2=np.array([ 0.1,0.1,0.1,1,20,78])
# quad_err_N_4=np.array([0.0787, 0.0787,0.0889,0.0160,0.0142])
# misc_time_N_4=np.array([ 0.1,0.1,0.4,9,1760])
# quad_err_N_8=np.array([ 0.0743,0.0507,0.0205,0.0027 ])
# misc_time_N_8=np.array([ 0.2,4,11,1400])
# quad_err_N_16=np.array([0.0831,0.0231,0.0245])
# misc_time_N_16=np.array([2,640,1256])




##Case for parameter set 3 with richardson
# quad_err_N_2=np.array([0.0858,0.0858,0.0858,0.0040,0.0138,0.0124])
# misc_time_N_2=np.array([ 0.1,0.1,0.1,3,30,234])
# quad_err_N_4=np.array([0.1009,0.1076,0.0098,0.0076,	0.0151])
# misc_time_N_4=np.array([ 0.1,0.2,2,24,9510])
# quad_err_N_8=np.array([0.0800,0.0480,0.0307,0.0022])
# misc_time_N_8=np.array([ 0.2,5,19,1900])
# quad_err_N_16=np.array([0.0778, 0.0396])
# misc_time_N_16=np.array([ 7,1375])
		


# ##Case for parameter set 4 non richardson
# quad_err_N_2=np.array([2.3039,2.3039,2.3039,0.0302, 0.0805, 0.0201])
# misc_time_N_2=np.array([ 0.1,0.1,0.1,0.8,4,80])
# quad_err_N_4=np.array([2.2838,2.2838,2.2838,1.7807,0.0704])
# misc_time_N_4=np.array([ 0.1,0.1,0.1,0.5,2368])
# quad_err_N_8=np.array([2.0826,2.0826,2.0826,2.0826,0.0020])
# misc_time_N_8=np.array([ 0.2,0.2,0.2,0.2,6750])



# ##Case for parameter set 4 with richardson
# quad_err_N_2=np.array([4.7386,4.7386,4.7386,   0.6540, 0.0704 ,0.0101 ])
# misc_time_N_2=np.array([ 0.1,0.1,0.1,2,12,159])
# quad_err_N_4=np.array([0.6238,0.6238,0.6238, 2.1630,0.0805])
# misc_time_N_4=np.array([ 0.1,0.1,0.1,0.9,3652])
# quad_err_N_8=np.array([ 1.7506,1.7506,1.7506,0.3119])
# misc_time_N_8=np.array([ 0.2,0.2,0.2,1640])
# quad_err_N_16=np.array([])
	

##Case for parameter set 5 non richardson
quad_err_N_2=np.array([0.0088,0.0088,0.0088,0.0088,0.0016,0.0008 ])
misc_time_N_2=np.array([ 0.1,0.1,0.1,0.1,0.5,1])
quad_err_N_4=np.array([0.0144,0.0144,0.0144,0.0088,0.0016,0.0008 ])
misc_time_N_4=np.array([ 0.1,0.1,0.1,0.5,3,6])
quad_err_N_8=np.array([0.0176,0.0176,0.0176,0.0040,0.00005,0.0008])
misc_time_N_8=np.array([ 0.2,0.2,0.2,8,24,140])
quad_err_N_16=np.array([0.0176,0.0160,0.0064,0.0008,0.00005])
misc_time_N_16=np.array([ 0.4,0.8,22,92,2226])


fig = plt.figure()

plt.plot(quad_err_N_2,misc_time_N_2,linewidth=2.0,label=r'$N=2$' , marker='>',hold=True) 
plt.plot(quad_err_N_4,misc_time_N_4,linewidth=2.0,label=r'$N=4$' , marker='o',hold=True) 
plt.plot(quad_err_N_8,misc_time_N_8,linewidth=2.0,label=r'$N=8$'  , marker='v',hold=True) 
plt.plot(quad_err_N_16,misc_time_N_16,linewidth=2.0,label=r'$N=16$'  , marker='*',hold=True) 

plt.yscale('log')
plt.xscale('log')
plt.xlabel('relative Quadrature error',fontsize=14)

plt.ylabel('CPU time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='lower left')
plt.savefig('./results/relative_quad_error_wrt_time_set4_with_rich.eps', format='eps', dpi=1000)  