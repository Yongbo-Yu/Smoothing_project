import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



#Case for parameter set 5, non rich
# MC_err=np.array([ 0.0179,0.0083,0.0047,0.0013])
# MC_time=np.array([  122,  211, 427, 766])

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# # print z_MC[0]

# MISC_err=np.array([0.0182,0.0086,0.0050,0.0016 ])
# MISC_time=np.array([ 1,6,24,92])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]

# # # #Case for parameter set 5, with rich
# MC_rich_err=np.array([ 0.0073,0.0025,0.0013 ])
# MC_rich_time=np.array([45, 438,2240  ])

# z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
# fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
# print z_MC_rich[0]

# MISC_rich_err=np.array([0.0057,0.0025,0.0013 ])
# MISC_rich_time=np.array([ 3.5,34,112])

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
# # #Case for parameter set 6, non rich

# MC_err=np.array([0.0057,0.0038,0.0032,0.0027])  
	
# MC_time=np.array([ 141,246,461,820])  
		
		

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

# MISC_err=np.array([0.0058,0.0037,0.0030,0.0025])
# MISC_time=np.array([ 1,6,27,112])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]

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
# # #Case for parameter set 7, non rich

MC_err=np.array([0.0657,0.0337,0.0209,0.0136])       	
	
MC_time=np.array([ 154,229,420,938])  
		

z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
fit_MC=np.exp(z_MC[0]*np.log(MC_err))
print z_MC[0]

MISC_err=np.array([0.0655,0.0334,0.0205,0.0135])
MISC_time=np.array([ 0.7,4,26,1984])

z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
print z_MISC[0]



##############################################################################
# # # #Case for parameter set 2, non rich+linear
# MISC_err=np.array([0.2928,0.1595])
# MISC_time=np.array([ 5,333])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]

# # # # #Case for parameter set 2, with rich+linear
# # MC_rich_err=np.array([ 1.0288,0.0787,0.0250  ])
# # MC_rich_time=np.array([ 12,113,130])

# # z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
# # fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
# # print z_MC_rich[0]

# MISC_rich_err=np.array([1.0288,0.0787,0.0250 ])
# MISC_rich_time=np.array([ 3.5,6,37])

# z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
# fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
# print z_MISC_rich[0]

# # # # #Case for parameter set 2, with rich(level2)+linear


# MISC_rich_2_err=np.array([0.2689,0.0096])
# MISC_rich_2_time=np.array([ 9,74])

# z_MISC_rich_2= np.polyfit(np.log(MISC_rich_2_err), np.log(MISC_rich_2_time), 1)
# fit_MISC_rich_2=np.exp(z_MISC_rich_2[0]*np.log(MISC_rich_2_err))
# print z_MISC_rich_2[0]

##############################################################################

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

plt.plot(MC_err,MC_time,linewidth=2.0,label='MC' , marker='>',hold=True) 
plt.plot(MC_err, fit_MC*10,linewidth=2.0,label=r'rate= %s' % format(z_MC[0]  , '.2f'), linestyle = '--')

plt.plot(MISC_err,MISC_time,linewidth=2.0,label='MISC'  , marker='v',hold=True) 
plt.plot(MISC_err, fit_MISC*0.000001,linewidth=2.0,label=r'rate= %s' % format(z_MISC[0]  , '.2f'), linestyle = '--')

# plt.plot(MC_rich_err,MC_rich_time,linewidth=2.0,label='MC+Rich' , marker='>',hold=True) 
# plt.plot(MC_rich_err, fit_MC_rich*10,linewidth=2.0,label=r'rate= %s' % format(z_MC_rich[0]  , '.2f'), linestyle = '--')

# plt.plot(MISC_rich_err,MISC_rich_time,linewidth=2.0,label='MISC+Rich(level 1)'  , marker='v',hold=True) 
# plt.plot(MISC_rich_err, fit_MISC_rich*0.0005,linewidth=2.0,label=r'rate= %s' % format(z_MISC_rich[0]  , '.2f'), linestyle = '--')

# plt.plot(MISC_rich_2_err,MISC_rich_2_time,linewidth=2.0,label='MISC+Rich(level 2)'  , marker='v',hold=True) 
# plt.plot(MISC_rich_2_err, fit_MISC_rich_2*10,linewidth=2.0,label=r'rate= %s' % format(z_MISC_rich_2[0]  , '.2f'), linestyle = '--')

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Error',fontsize=14)

plt.ylabel('CPU time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper right')
plt.savefig('./results/error_vs_time_set7.eps', format='eps', dpi=1000)  
