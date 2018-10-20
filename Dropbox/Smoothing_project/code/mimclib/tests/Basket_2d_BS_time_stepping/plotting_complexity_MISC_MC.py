import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



# case non richardson


MC_normal_err=np.array([0.0119,0.0042,0.0029])


MC_normal_time=np.array([164,451,1112 ])


z_MC_normal= np.polyfit(np.log(MC_normal_err), np.log(MC_normal_time), 1)
fit_MC_normal=np.exp(z_MC_normal[0]*np.log(MC_normal_err))
print z_MC_normal[0]

MC_err=np.array([0.0120,0.0042,0.0029])
MC_time=np.array([557, 1218,1749])

z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
fit_MC=np.exp(z_MC[0]*np.log(MC_err))
print z_MC[0]

MISC_err=np.array([ 0.0120,0.0081,0.0043,0.0029])
MISC_time=np.array([ 4,51,238,1161])


z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
print z_MISC[0]





fig = plt.figure()

plt.plot(MC_err,MC_time,linewidth=2.0,label='MC+root finding' , marker='>',hold=True) 
plt.plot(MC_err, fit_MC*100,linewidth=2.0,label=r'rate= %s' % format(z_MC[0]  , '.2f'), linestyle = '--')

plt.plot(MC_normal_err,MC_normal_time,linewidth=2.0,label='MC' , marker='>',hold=True) 
plt.plot(MC_normal_err, fit_MC_normal*100,linewidth=2.0,label=r'rate= %s' % format(z_MC_normal[0]  , '.2f'), linestyle = '--')


plt.plot(MISC_err,MISC_time,linewidth=2.0,label='MISC'  , marker='v',hold=True) 
plt.plot(MISC_err, fit_MISC*10,linewidth=2.0,label=r'rate= %s' % format(z_MISC[0]  , '.2f'), linestyle = '--')




plt.yscale('log')
plt.xscale('log')
plt.xlabel('Error',fontsize=14)

plt.ylabel('CPU time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper right')
plt.savefig('./results/error_vs_time.eps', format='eps', dpi=1000)  