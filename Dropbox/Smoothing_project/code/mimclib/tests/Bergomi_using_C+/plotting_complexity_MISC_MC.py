import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



# #Case for parameter set 5, non rich
# MC_err=np.array([ 0.0182,0.0086,0.0050,0.0016])
# MC_time=np.array([  682,  721,  814, 1125])

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

MISC_err=np.array([0.0182,0.0086,0.0050,0.0016 ])
MISC_time=np.array([ 1,6,80,92])

z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
print z_MISC[0]

# # #Case for parameter set 5, with rich
# MC_rich_err=np.array([ 0.0073,0.0025,0.0013 ])
# MC_rich_time=np.array([45, 438,2240  ])

# z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
# fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
# print z_MC_rich[0]

MISC_rich_err=np.array([0.0073,0.0025,0.0013 ])
MISC_rich_time=np.array([ 11,34,112])

z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
print z_MISC_rich[0]
##############################################################################
# # #Case for parameter set 2, non rich
# MC_err=np.array([ 0.6472,0.2932,0.1487])
# MC_time=np.array([1.3,4.7,37.5])

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

# MISC_err=np.array([0.6472,0.2932,0.1487])
# MISC_time=np.array([ 0.45,350,5370])

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
# # #Case for parameter set 1, non rich

# MC_err=np.array([0.5120,0.3154,0.1550,0.0847])
# MC_time=np.array([ 3.4,10,13.4,18])

# z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
# fit_MC=np.exp(z_MC[0]*np.log(MC_err))
# print z_MC[0]

# MISC_err=np.array([0.5120,0.3154,0.1550,0.0847])
# MISC_time=np.array([ 0.2,11,5760,215])

# z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
# fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
# print z_MISC[0]

# #Case for parameter set 1, with rich
MC_rich_err=np.array([ 0.7561,0.0715,0.0141 ])
MC_rich_time=np.array([  34.7,37,532])

z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
print z_MC_rich[0]

MISC_rich_err=np.array([0.7561,0.0715,0.0141])
MISC_rich_time=np.array([ 4,191,664])

z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
print z_MISC_rich[0]






fig = plt.figure()

# plt.plot(MC_err,MC_time,linewidth=2.0,label='MC' , marker='>',hold=True) 
# plt.plot(MC_err, fit_MC*1000,linewidth=2.0,label=r'rate= %s' % format(z_MC[0]  , '.2f'), linestyle = '--')

# plt.plot(MISC_err,MISC_time,linewidth=2.0,label='MISC'  , marker='v',hold=True) 
# plt.plot(MISC_err, fit_MISC*10,linewidth=2.0,label=r'rate= %s' % format(z_MISC[0]  , '.2f'), linestyle = '--')

plt.plot(MC_rich_err,MC_rich_time,linewidth=2.0,label='MC+Rich' , marker='>',hold=True) 
plt.plot(MC_rich_err, fit_MC_rich*10,linewidth=2.0,label=r'rate= %s' % format(z_MC_rich[0]  , '.2f'), linestyle = '--')

plt.plot(MISC_rich_err,MISC_rich_time,linewidth=2.0,label='MISC+Rich'  , marker='v',hold=True) 
plt.plot(MISC_rich_err, fit_MISC_rich*10,linewidth=2.0,label=r'rate= %s' % format(z_MISC_rich[0]  , '.2f'), linestyle = '--')

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Error',fontsize=14)

plt.ylabel('CPU time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper right')
plt.savefig('./results/error_vs_time_set1_rich.eps', format='eps', dpi=1000)  
