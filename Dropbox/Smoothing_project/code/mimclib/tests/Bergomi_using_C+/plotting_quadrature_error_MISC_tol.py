
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


dt_arr=np.array([0.5, 0.1,  0.05, 0.01, 0.001,0.0001])

# #Case for parameter set 1 non richardson
# quad_err_N_2=np.array([ 0.0868, 0.0868,0.0868,0.0017, 0.0017 , 0.0017,0.0017])
# quad_err_N_4=np.array([0.0563, 0.0563, 0.0591, 0.0324,1.4e-04,1.4e-04,1.4e-04 ])
# quad_err_N_8=np.array([0.0563, 0.0681, 0.0288, 0.0218, 0.0049,0.0021,6.9e-04  ])
# quad_err_N_16=np.array([0.0197, 0.0492, 0.0800, 0.0070, 0.0070 ])



# #Case for parameter set 1 non richardson+linear
# quad_err_N_2=np.array([ 0.1095 ,0.1095,0.1095,0.0056, 0.0011 , 2.8e-04 ])
# quad_err_N_4=np.array([0.0731, 0.0731, 0.0591, 0.0268,1.4e-04,1.4e-04 ])
# quad_err_N_8=np.array([0.0485, 0.0863, 0.0288, 0.0162, 0.0035,0.0035 ])
# quad_err_N_16=np.array([0.0323, 0.0267, 0.0674, 0.0042, 0.0042 ])

# #Case for parameter set 1 with richardson
# quad_err_N_2=np.array([ 0.1685,  0.1685, 0.1685,0.00001, 0.0183, 0.0183 ])
# quad_err_N_4=np.array([0.0435, 0.0435, 0.1109,0.0407,0.0197,0.0154])
# quad_err_N_8=np.array([  0.0197, 0.0899, 0.0730,0.0337, 0.0014,0.0014])
# quad_err_N_16=np.array([0.0014, 0.0674,0.0632, 0.0014,0.0112  ])



# # #Case for parameter set 1 with richardson(level2)
# quad_err_N_2=np.array([  0.0239,0.0576, 0.1755,0.0435,0.0169,	1.4e-04,	1.4e-04 ])
# quad_err_N_4=np.array([0.0126,0.0520, 0.1011,0.0267,0.0028,0.0028])




# #Case for parameter set 2 non richardson
# quad_err_N_2=np.array([ 0.1717, 0.1717, 0.1717, 0.0391,0.0023,2.5e-04])
# quad_err_N_4=np.array([ 0.1345, 0.1345, 0.1294, 0.0120,6.3e-04,6.3e-04])
# quad_err_N_8=np.array([ 0.1442, 0.1619,2.5e-04, 0.0066,0.0053 ])
# quad_err_N_16=np.array([0.1490, 0.0631,0.0417 ])



#Case for parameter set 2 non richardson+ linear hierarchy
quad_err_N_2=np.array([  0.1525,  0.1525, 0.1525, 0.1247,0.0288,2.5e-04])
quad_err_N_4=np.array([ 0.1231, 0.1231, 0.1686, 6.3e-04,6.3e-04 ,06.3e-04])
quad_err_N_8=np.array([ 0.1353, 0.1555, 0.0823, 0.0053,0.0053 ])
quad_err_N_16=np.array([0.1414, 0.0429,0.0101,0.0101 ])


# #Case for parameter set 2 with richardson
# quad_err_N_2=np.array([0.3914, 0.3914, 0.3914,0.1174,0.0593,0.0593,0.0657])
# quad_err_N_4=np.array([0.1187, 0.1187,0.2146,0.0013,0.0240,0.0202 ])
# quad_err_N_8=np.array([ 0.1540, 0.1843, 0.0050, 0.0013])
# quad_err_N_16=np.array([0.1654, 0.0410])


# #Case for parameter set 2 with richardson+ linear hierarchy
# quad_err_N_2=np.array([0.3687, 0.3687, 0.3687,0.1212,0.0694,0.0694])
# quad_err_N_4=np.array([0.1136, 0.1136,0.1641,0.0101,0.0101 ])
# quad_err_N_8=np.array([ 0.1477, 0.1288, 0.0101,0.0101 ])
# quad_err_N_16=np.array([0.1591, 0.0518,0.0518])




# # #Case for parameter set 2 with richardson level2
# quad_err_N_4=np.array([ 0.1629,0.1502,0.0278,0.0012])
# quad_err_N_2=np.array([0.0152,0.0152,0.2462,0.0669,	0.0215,0.0013])


# # #Case for parameter set 2 with richardson level2,linear
# quad_err_N_2=np.array([ 0.0177,0.0177,0.2525,0.0278,0.0278,0.0088])
# quad_err_N_4=np.array([0.1553,0.1073,0.0038,0.0038 ])


# #Case for parameter set 3 non richardson
# quad_err_N_2=np.array([0.0471,0.0471,0.0471,0.0204,0.0027,  9.3e-04, 9.3e-04])
# quad_err_N_4=np.array([0.0666, 0.0666,	0.0768, 0.0039, 0.0021])
# quad_err_N_8=np.array([ 0.0702,0.0466, 0.0164,0.0027])
# quad_err_N_16=np.array([0.0811,0.0211,0.0224])



# ##Case for parameter set 3 with richardson
# quad_err_N_2=np.array([0.0858,0.0858,0.0858,0.0040,0.0138,0.0124])
# quad_err_N_4=np.array([0.1009,0.1076,0.0098,0.0076,	0.0151])
# quad_err_N_8=np.array([0.0800,0.0480,0.0307,0.0022])
# quad_err_N_16=np.array([0.0778, 0.0396])


# ##Case for parameter set 4 non richardson
# quad_err_N_2=np.array([2.3039,2.3039,2.3039,0.0302, 0.3421,0.0805, 0.0201])
# quad_err_N_4=np.array([2.3240,2.3240,2.3240,1.8210,0.0050,0.0050])
# quad_err_N_8=np.array([2.0826,2.0826,2.0826,2.0826,0.0020])
# quad_err_N_16=np.array([])
		
# ##Case for parameter set 4 with richardson
# quad_err_N_2=np.array([4.7386,4.7386,4.7386,   0.6540, 0.0704 ,0.0101 ])
# quad_err_N_4=np.array([0.6238,0.6238,0.6238, 2.1630,0.0805])
# quad_err_N_8=np.array([ 1.7506,1.7506,1.7506,0.3119])
# quad_err_N_16=np.array([1.3381,1.3381,1.3381,1.3381])



# #Case for parameter set 5 non richardson
# quad_err_N_2=np.array([0.0088,0.0088,0.0088,0.0088,0.0016,0.0008,0.0008 ])
# quad_err_N_4=np.array([0.0144,0.0144,0.0144,0.0088,0.0016,0.0008,0.0008 ])
# quad_err_N_8=np.array([0.0176,0.0176,0.0176,0.0040,0.0008,0.0008,0.0008])
# quad_err_N_16=np.array([0.0176,0.0160,0.0064,0.0008,0.00005])



# ##Case for parameter set 5 with richardson
# quad_err_N_2=np.array([ 0.0184,0.0184,0.0184,  0.0136,0.0008,0.0008,0.0008])
# quad_err_N_4=np.array([ 0.0248,0.0248,0.0248,0.0168, 0.0032, 0.0032, 0.0032])
# quad_err_N_8=np.array([ 0.0216, 0.0216, 0.0216,  0.0032, 0.0016,	0.0016,	0.0016])
# quad_err_N_16=np.array([ 0.0200, 0.0176,	0.0064, 0.0008,0.0008])



# #Case for parameter set 6 non richardson
# quad_err_N_2=np.array([0.0029,0.0029,0.0029,0.0029,0.0029,0.0004,0.0004 ])
# quad_err_N_4=np.array([  0.0054,  0.0054,  0.0054,  0.0054,0.0021,2.1e-04,2.1e-04])
# quad_err_N_8=np.array([  0.0046,  0.0046,  0.0046,0.0021,1.2e-04,1.2e-04,1.2e-04])
# quad_err_N_16=np.array([   0.0071,   0.0066,	0.0033,8.3e-05,8.3e-05])




# # #Case for parameter set 7 non richardson
# quad_err_N_2=np.array([0.0264,0.0264,0.0264,0.0264,5.3e-04,5.3e-04,5.3e-04])
# quad_err_N_4=np.array([  0.0406,  0.0406,  0.0406,  0.0406,3.5e-04,3.5e-04,3.5e-04])
# quad_err_N_8=np.array([  0.0491,  0.0491, 0.0491,0.0021,3.3e-04,3.3e-04,3.3e-04])
# quad_err_N_16=np.array([   0.0524,   0.0524,	0.0331,0.0065,5.3e-04])




fig = plt.figure()

plt.plot(dt_arr, quad_err_N_2,linewidth=2.0,label=r'$N=2$' , marker='>',hold=True) 
plt.plot(dt_arr[0:6], quad_err_N_4,linewidth=2.0,label=r'$N=4$' , marker='o',hold=True) 
plt.plot(dt_arr[0:5], quad_err_N_8,linewidth=2.0,label=r'$N=8$'  , marker='v',hold=True) 
plt.plot(dt_arr[0:4], quad_err_N_16,linewidth=2.0,label=r'$N=16$'  , marker='*',hold=True) 

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$TOL_{MISC}$',fontsize=14)



plt.ylabel('relative Quadrature error',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper left')
plt.savefig('./results/relative_quad_error_wrt_MISC_TOL_set2_non_rich_linear.eps', format='eps', dpi=1000)  

