import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



# # # # # #Case for parameter set 5, non rich
 # MC_err=np.array([ 0.035,0.016,0.007,0.002])
 # MC_time=4*np.array([  0.15,1.6,16.5,494])

 # MC_err_extrapol=np.array([ 0.035,0.016,0.007,0.002,0.001])

 # z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
 # fit_MC=np.exp(z_MC[0]*np.log(MC_err_extrapol))
 # print z_MC[0]

 # MISC_err=np.array([0.03,0.017,0.009,0.001 ])
 # MISC_time=np.array([ 0.1,0.5,3,92])

 # z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
 # fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
 # print z_MISC[0]



 # QMC_err=np.array([0.04,   0.017 ,  0.008 ,0.002 ])
 # QMC_time=np.array([ 0.3, 0.7 , 3.25, 27])

 # z_QMC= np.polyfit(np.log(QMC_err), np.log(QMC_time), 1)
 # fit_QMC=np.exp(z_QMC[0]*np.log(QMC_err))
 # print z_QMC[0]


# QMC_err_non_smooth=np.array([0.036,   0.0175 ,  0.009 ,0.0024 ])
# QMC_time_non_smooth=np.array([ 0.9, 1.8 , 7, 55])
#
# z_QMC_non_smooth= np.polyfit(np.log(QMC_err_non_smooth), np.log(QMC_time_non_smooth), 1)
# fit_QMC_non_smooth=np.exp(z_QMC_non_smooth[0]*np.log(QMC_err_non_smooth))
# print z_QMC_non_smooth[0]
# # # #Case for parameter set 5, with rich
# MC_rich_err=np.array([ 0.006,0.0025,0.0013 ])
# MC_rich_time=np.array([45, 438,2240  ])
#
# z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
# fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
# print z_MC_rich[0]
#
# MISC_rich_err=np.array([0.027,0.004,0.002 ])
# MISC_rich_time=np.array([ 0.15,10,112])
#
# z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
# fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
# print z_MISC_rich[0]
##############################################################################
# # #Case for parameter set 2, non rich
# MC_err=np.array([ 0.6472,0.2932,0.1487])
# MC_time=np.array([1.3,4.7,37.5])

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

# MISC_err=np.array([0.5389,0.2928,0.1608])
# MISC_time=np.array([ 7,350,800])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]



# # #Case for parameter set 2, with rich
# MC_rich_err=np.array([ 1.0187,0.0926,0.0162  ])
# MC_rich_time=np.array([ 16,20,2780])

# z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
# fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
# print z_MC_rich[0]

# MISC_rich_err=np.array([1.0187,0.0926,0.0162 ])
# MISC_rich_time=np.array([ 16,2460,3450])

# z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
# fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
# print z_MISC_rich[0]


##############################################################################
# # # #Case for parameter set 1, non rich

# MC_err=np.array([0.5159,0.2938,0.1555])
# MC_time=np.array([ 50,344,637])

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

# MISC_err=np.array([0.5159,0.2934,0.1558])
# MISC_time=np.array([ 0.2,11,5760])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]

# # # # #Case for parameter set 1, with rich
# MC_rich_err=np.array([ 0.7561,0.0758,0.0141 ])
# MC_rich_time=np.array([  34.7,37,532])

# z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
# fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
# print z_MC_rich[0]

# MISC_rich_err=np.array([0.7561,0.0758,0.0141])
# MISC_rich_time=np.array([ 4,12,520])

# z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
# fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
# print z_MISC_rich[0]
##############################################################################



# # # #Case for parameter set 1, non rich+linear

# MC_err=np.array([0.5159,0.2938,0.1584])
# MC_time=np.array([ 50,344,637])

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

# MISC_err=np.array([0.5153,0.2934,0.1586])
# MISC_time=np.array([ 0.2,11,5760])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]


# # # # # #Case for parameter set 1, with rich+linear


# MISC_rich_err=np.array([0.7561,0.0758,0.0141])
# MISC_rich_time=np.array([ 4,12,520])

# z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
# fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
# print z_MISC_rich[0]


# # # # # #Case for parameter set 1, with rich(level2)+linear


# MISC_rich_2_err=np.array([0.1628,0.0052])
# MISC_rich_2_time=np.array([ 5,64])

# z_MISC_rich_2= np.polyfit(np.log(MISC_rich_2_err), np.log(MISC_rich_2_time), 1)
# fit_MISC_rich_2=np.exp(z_MISC_rich_2[0]*np.log(MISC_rich_2_err))
# print z_MISC_rich_2[0]


##############################################################################
# # # #Case for parameter set 6, non rich

# MC_err=np.array([0.01,0.008,0.006,0.004])  
	
# MC_time=4*np.array([ 1,3,10,40])  
# #MC_err_extrapol=np.array([0.01,0.008,0.006,0.004,0.002])  	
		

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

# MISC_err=np.array([0.008,0.006,0.004,0.002])
# MISC_time=np.array([ 0.1,1,6,112])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]



# QMC_err=np.array([0.015,   0.008 ,  0.0066 ,0.004 ])
# QMC_time=np.array([ 0.65, 1.4 , 3.25, 7.5])

# z_QMC= np.polyfit(np.log(QMC_err), np.log(QMC_time), 1)
# fit_QMC=np.exp(z_QMC[0]*np.log(QMC_err))
# print z_QMC[0]

# QMC_err_non_smooth=np.array([0.014,   0.009 ,  0.0062 ,0.0035 ])
# QMC_time_non_smooth=np.array([ 0.9, 1.8 , 3.6, 15])

# z_QMC_non_smooth= np.polyfit(np.log(QMC_err_non_smooth), np.log(QMC_time_non_smooth), 1)
# fit_QMC_non_smooth=np.exp(z_QMC_non_smooth[0]*np.log(QMC_err_non_smooth))
# print z_QMC_non_smooth[0]


##############################################################################

# # #Case for parameter set 1, with rich
# MC_rich_err=np.array([ 0.7561,0.0715,0.0141 ])
# MC_rich_time=np.array([  34.7,37,532])

# z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
# fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
# print z_MC_rich[0]

# MISC_rich_err=np.array([0.7561,0.0715,0.0141])
# MISC_rich_time=np.array([ 4,191,664])

# z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
# fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
# print z_MISC_rich[0]



##############################################################################
# # # #Case for parameter set 7, non rich

# MC_err=np.array([0.14,0.07,0.04,0.02])           
   
# MC_time=4*np.array([ 0.02,0.15,1.4,10])  


# #MC_err_extrapol=np.array([0.14,0.07,0.04,0.02])           
       

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

# MISC_err=np.array([0.09,0.07,0.03,0.02])
# MISC_time=np.array([ 0.1,0.1,4, 8])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]


# QMC_err=np.array([0.155,   0.07 ,  0.039 ,0.02])
# QMC_time=np.array([ 0.17, 0.35 , 1.6, 4])

# z_QMC= np.polyfit(np.log(QMC_err), np.log(QMC_time), 1)
# fit_QMC=np.exp(z_QMC[0]*np.log(QMC_err))
# print z_QMC[0]

# QMC_err_non_smooth=np.array([0.145,   0.055 ,  0.038 ,0.021 ])
# QMC_time_non_smooth=np.array([ 0.45, 1.8 , 3.6, 8])

# z_QMC_non_smooth= np.polyfit(np.log(QMC_err_non_smooth), np.log(QMC_time_non_smooth), 1)
# fit_QMC_non_smooth=np.exp(z_QMC_non_smooth[0]*np.log(QMC_err_non_smooth))
# print z_QMC_non_smooth[0]


##############################################################################
# # # #Case for parameter set 2, non rich+linear

# MC_err=np.array([    1.05, 0.59, 0.31,0.14,0.08,0.04,0.02, 0.008 ])
# MC_time=4*np.array([ 0.001, 0.003, 0.02,0.2,1,18,135,15380])

# MC_err_extrapol=np.array([ 1.05, 0.59, 0.31,0.14,0.08,0.04,0.02,0.008 ])
# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err_extrapol))
# print z_MC[0]


# MISC_err=np.array([0.69,  0.42,0.29,0.16,0.08 ])
# MISC_time=np.array([ 0.08, 0.13 , 5 ,333, 1602])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]


# complexity QMC
# QMC_err=np.array([1.2,   0.665 ,  0.355,0.17, 0.09, 0.05,0.025,0.0125, 0.006])
# QMC_time=np.array([ 0.022, 0.045, 0.1, 0.26, 1.4, 6.5, 30,  130, 500])

# z_QMC= np.polyfit(np.log(QMC_err), np.log(QMC_time), 1)
# fit_QMC=np.exp(z_QMC[0]*np.log(QMC_err))
# print z_QMC[0]

# # # # #Case for parameter set 2, with rich+linear
MC_rich_err=np.array([ 1.88,0.13,0.032,0.0045  ])
MC_rich_time=4* np.array([ 0.0003,3,38,1100])


MC_rich_err_extrapol=np.array([ 1.88,0.14,0.03 ,0.01,0.005 ])

z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err_extrapol))
print z_MC_rich[0]

# MISC_rich_err=np.array([1.33,0.18,0.08,0.0250, 0.0059 ])
# MISC_rich_time=np.array([ 0.1,0.2,6,37,1045])

# z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
# fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
# print z_MISC_rich[0]


QMC_rich_err=np.array([  1.87,0.16  ,0.033,0.004  ])
QMC_rich_time=np.array([ 0.02, 2, 18, 333 ])

z_QMC_rich= np.polyfit(np.log(QMC_rich_err), np.log(QMC_rich_time), 1)
fit_QMC_rich=np.exp(z_QMC_rich[0]*np.log(QMC_rich_err))
print z_QMC_rich[0]

# # # #Case for parameter set 2, with rich(level2)+linear

# MC_rich_2_err=np.array([ 0.45,0.012 ])
# MC_rich_2_time=4*np.array([ 0.2,690])


# MC_rich_2_err_extrapol=np.array([ 0.45,0.012 ,0.01,0.005])

# z_MC_rich_2= np.polyfit(np.log(MC_rich_2_err), np.log(MC_rich_2_time), 1)
# fit_MC_rich_2=np.exp(z_MC_rich_2[0]*np.log(MC_rich_2_err_extrapol))
# print z_MC_rich_2[0]

MISC_rich_2_err=np.array([0.49, 0.113,0.009])
MISC_rich_2_time=np.array([ 0.5,2,74])

z_MISC_rich_2= np.polyfit(np.log(MISC_rich_2_err), np.log(MISC_rich_2_time), 1)
fit_MISC_rich_2=np.exp(z_MISC_rich_2[0]*np.log(MISC_rich_2_err))
print z_MISC_rich_2[0]



# QMC_rich_2_err=np.array([  0.5,  0.014])
# QMC_rich_2_time=np.array([ 0.5, 175 ])

# z_QMC_rich_2= np.polyfit(np.log(QMC_rich_2_err), np.log(QMC_rich_2_time), 1)
# fit_QMC_rich_2=np.exp(z_QMC_rich_2[0]*np.log(QMC_rich_2_err))
# print z_QMC_rich_2[0]
#############################################################################

# # #Case for parameter set 3, non rich

# MC_err=np.array([0.1118])       	
	
# MC_time=np.array([ 188])  
		

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

# MISC_err=np.array([0.1119,])
# MISC_time=np.array([ 25])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]


##############################################################################



fig = plt.figure()
plt.rcParams.update({'font.size': 14})

# plt.plot(MC_err,MC_time,linewidth=2.0,label='MC' , marker='>',hold=True,color='b') 
# plt.plot(MC_err, fit_MC*0.0001,linewidth=2.0,label=r'slope= %s' % format(z_MC[0]  , '.2f'), linestyle = '--', color = 'b')
#plt.plot(MC_err_extrapol, fit_MC*0.001,linewidth=1.0,label=r'slope= %s' % format(z_MC[0] , '.2f'), linestyle = '--', color = 'b')

# plt.plot(MISC_err,MISC_time,linewidth=2.0,label='ASGQ'  , marker='*',hold=True,color='b') 
# plt.plot(MISC_err, fit_MISC*0.001,linewidth=2.0,label=r'slope= %s' % format(z_MISC[0]   , '.2f'), linestyle = '--', color='b')

# plt.plot(QMC_err,QMC_time,linewidth=2.0,label='QMC'  , marker='o',hold=True,color='b')
# plt.plot(QMC_err, fit_QMC*0.01,linewidth=2.0,label=r'slope= %s' % format(z_QMC[0]   , '.2f'), linestyle = '--', color='b')

#plt.plot(QMC_err_non_smooth,QMC_time_non_smooth,linewidth=2.0,label='QMC+non smooth'  , marker='o',hold=True,color='m')
#plt.plot(QMC_err_non_smooth, fit_QMC_non_smooth*0.05,linewidth=2.0,label=r'slope= %s' % format(z_QMC_non_smooth[0]   , '.2f'), linestyle = '--', color='m')


plt.plot(MC_rich_err,MC_rich_time,linewidth=2.0,label='MC+Rich(level 1)' , marker='>',hold=True, color='b') 
plt.plot(MC_rich_err_extrapol, fit_MC_rich*0.02,linewidth=2.0,label=r'slope= %s' % format(z_MC_rich[0]  , '.2f'), linestyle = '--', color='b')

# plt.plot(MISC_rich_err,MISC_rich_time,linewidth=2.0,label='ASGQ+Rich(level 1)'  , marker='*',hold=True, color='r') 
# plt.plot(MISC_rich_err, fit_MISC_rich*0.1,linewidth=2.0,label=r'slope= %s' % format(z_MISC_rich[0]  , '.2f'), linestyle = '--',color='r')

plt.plot(QMC_rich_err,QMC_rich_time,linewidth=2.0,label='QMC+Rich(level 1)'  , marker='o',hold=True, color='r') 
plt.plot(QMC_rich_err, fit_QMC_rich*0.1,linewidth=2.0,label=r'slope= %s' % format(z_QMC_rich[0]  , '.2f'), linestyle = '--',color='r')

# plt.plot(MC_rich_2_err,MC_rich_2_time,linewidth=2.0,label='MC+Rich(level 2)' , marker='>',hold=True,color='g') 
# plt.plot(MC_rich_2_err_extrapol, fit_MC_rich_2*0.3,linewidth=1.0,label=r'slope= %s' % format(z_MC_rich_2[0]  , '.2f'), linestyle = '--',color='g')

plt.plot(MISC_rich_2_err,MISC_rich_2_time,linewidth=2.0,label='ASGQ+Rich(level 2)'  , marker='*',hold=True,color='g' ) 
plt.plot(MISC_rich_2_err, fit_MISC_rich_2*0.1,linewidth=2.0,label=r'slope= %s' % format(z_MISC_rich_2[0]  , '.2f'), linestyle = '--',color='g')


# plt.plot(QMC_rich_2_err,QMC_rich_2_time,linewidth=2.0,label='QMC+Rich(level 2)'  , marker='o',hold=True,color='g' )
#plt.plot(QMC_rich_2_err, fit_QMC_rich_2*0.1,linewidth=1.0,label=r'slope= %s' % format(z_QMC_rich_2[0]  , '.2f'), linestyle = '--',color='g')

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Error')

plt.ylabel('CPU time') 
#plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='lower left', fontsize=13)
plt.savefig('./results/error_vs_time_set2_full_comparison.eps', format='eps', dpi=1000)  
