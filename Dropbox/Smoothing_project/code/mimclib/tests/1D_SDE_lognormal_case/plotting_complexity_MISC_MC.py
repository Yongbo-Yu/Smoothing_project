import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



# case non richardson


MC_normal_err=np.array([0.0223,0.0159,0.0095,0.0046])


MC_normal_time=np.array([1450,9990,32790,158108 ])


z_MC_normal= np.polyfit(np.log(MC_normal_err), np.log(MC_normal_time), 1)
fit_MC_normal=np.exp(z_MC_normal[0]*np.log(MC_normal_err))
print z_MC_normal[0]

MC_err=np.array([0.0223,0.0159,0.0095,0.0046])
MC_time=np.array([1328,8140,21400, 70200])


MISC_err=np.array([ 0.0223,0.0159,0.0095,0.0046])
MISC_time=np.array([ 0.3,6,58,656])

z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
fit_MC=np.exp(z_MC[0]*np.log(MC_err))
print z_MC[0]

z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
print z_MISC[0]


#case with richardson

# MC_rich_err=np.array([0.0364,0.0113, 0.0030,0.0008])
# MC_rich_time=np.array([ 194, 394,516, 725])
MISC_rich_err=np.array([ 0.0357,0.0109,0.0025,0.0006])
MISC_rich_time=np.array([ 0.3,4,56,713])


# z_MC_rich= np.polyfit(np.log(MC_rich_err), np.log(MC_rich_time), 1)
# fit_MC_rich=np.exp(z_MC_rich[0]*np.log(MC_rich_err))
# print z_MC_rich[0]


z_MISC_rich= np.polyfit(np.log(MISC_rich_err), np.log(MISC_rich_time), 1)
fit_MISC_rich=np.exp(z_MISC_rich[0]*np.log(MISC_rich_err))
print z_MISC_rich[0]


fig = plt.figure()

plt.plot(MC_err,MC_time,linewidth=2.0,label='MC+root finding' , marker='>',hold=True) 
plt.plot(MC_err, fit_MC*10,linewidth=2.0,label=r'rate= %s' % format(z_MC[0]  , '.2f'), linestyle = '--')

plt.plot(MC_normal_err,MC_normal_time,linewidth=2.0,label='MC' , marker='>',hold=True) 
plt.plot(MC_normal_err, fit_MC_normal*10,linewidth=2.0,label=r'rate= %s' % format(z_MC_normal[0]  , '.2f'), linestyle = '--')


plt.plot(MISC_err,MISC_time,linewidth=2.0,label='MISC'  , marker='v',hold=True) 
plt.plot(MISC_err, fit_MISC*10,linewidth=2.0,label=r'rate= %s' % format(z_MISC[0]  , '.2f'), linestyle = '--')

# plt.plot(MC_rich_err,MC_rich_time,linewidth=2.0,label='MC+Rich' , marker='o',hold=True) 
#plt.plot(MC_rich_err, fit_MC_rich*10,linewidth=2.0,label=r'rate= %s' % format(z_MC_rich[0]  , '.2f'), linestyle = '--')
# plt.plot(MISC_rich_err,MISC_rich_time,linewidth=2.0,label='MISC+Rich'  , marker='*',hold=True) 
# plt.plot(MISC_rich_err, fit_MISC_rich*10,linewidth=2.0,label=r'rate= %s' % format(z_MISC_rich[0]  , '.2f'), linestyle = '--')


plt.yscale('log')
plt.xscale('log')
plt.xlabel('Error',fontsize=14)

plt.ylabel('CPU time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper right')
plt.savefig('./results/error_vs_time.eps', format='eps', dpi=1000)  
