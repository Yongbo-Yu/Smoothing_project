import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



# #case non rich
MC_normal_err=np.array([0.1,0.05,0.02])


MC_normal_time=np.array([0.3,3,29])

z_MC_normal= np.polyfit(np.log(MC_normal_err), np.log(MC_normal_time), 1)
fit_MC_normal=np.exp(z_MC_normal[0]*np.log(MC_normal_err))
print z_MC_normal[0]




MC_err=np.array([0.08,0.04,0.02])

MC_time=np.array([0.2, 1, 9])

z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
fit_MC=np.exp(z_MC[0]*np.log(MC_err))
print z_MC[0]



MISC_err=np.array([ 0.05,0.02,0.01])
MISC_time=np.array([0.8, 2,9])

z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
print z_MISC[0]


# #case with rich

# MC_normal_rich_err=np.array([ 0.0116,0.0073,0.0029])


# MC_normal_rich_time=np.array([12,23,147 ])

# z_MC_normal_rich= np.polyfit(np.log(MC_normal_rich_err), np.log(MC_normal_rich_time), 1)
# fit_MC_normal_rich=np.exp(z_MC_normal_rich[0]*np.log(MC_normal_rich_err))
# print z_MC_normal_rich[0]



# MC_rich_err=np.array([0.0111,0.0070,0.0035])

# MC_rich_time=np.array([48,62,122])


# z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
# fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
# print z_MC_rich[0]


# MISC_rich_err=np.array([ 0.0115,0.0072,0.0032])
# MISC_rich_time=np.array([4,15,42])

# z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
# fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
# print z_MISC_rich[0]



fig = plt.figure()

plt.plot(MC_err,MC_time,linewidth=2.0,label='MC+root finding' , marker='>',hold=True) 
plt.plot(MC_err, fit_MC/1000,linewidth=2.0,label=r'rate= %s' % format(z_MC[0]  , '.2f'), linestyle = '--')

plt.plot(MC_normal_err,MC_normal_time,linewidth=2.0,label='MC' , marker='>',hold=True) 
plt.plot(MC_normal_err, fit_MC_normal/1000,linewidth=2.0,label=r'rate= %s' % format(z_MC_normal[0]  , '.2f'), linestyle = '--')

plt.plot(MISC_err,MISC_time,linewidth=2.0,label='MISC'  , marker='v',hold=True) 
plt.plot(MISC_err, fit_MISC/100,linewidth=2.0,label=r'rate= %s' % format(z_MISC[0]  , '.2f'), linestyle = '--')



# plt.plot(MC_rich_err,MC_rich_time,linewidth=2.0,label='MC+root_finding+Rich' , marker='o',hold=True) 
# plt.plot(MC_rich_err, fit_MC_rich*10,linewidth=2.0,label=r'rate= %s' % format(z_MC_rich[0]  , '.2f'), linestyle = '--')
# plt.plot(MC_normal_rich_err,MC_normal_rich_time,linewidth=2.0,label='MC' , marker='>',hold=True) 
# plt.plot(MC_normal_rich_err, fit_MC_normal_rich*10,linewidth=2.0,label=r'rate= %s' % format(z_MC_normal_rich[0]  , '.2f'), linestyle = '--')
# plt.plot(MISC_rich_err,MISC_rich_time,linewidth=2.0,label='MISC+Rich'  , marker='*',hold=True) 
# plt.plot(MISC_rich_err, fit_MISC_rich*10,linewidth=2.0,label=r'rate= %s' % format(z_MISC_rich[0]  , '.2f'), linestyle = '--')

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Error',fontsize=14)

plt.ylabel('CPU time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper right')
plt.savefig('./results/error_vs_time.eps', format='eps', dpi=1000)  
