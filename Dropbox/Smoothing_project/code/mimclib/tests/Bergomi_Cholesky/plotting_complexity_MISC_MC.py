import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


##############################################################################
##############################################################################
# # # #Case for parameter set 2, non rich

#MC_err=np.array([   ])
#MC_time=np.array([ , , ,,])

#MC_err_extrapol=np.array([ 1.05, 0.59, 0.31,0.14,0.08,0.04,0.02,0.01,0.005 ])
#z_MC= np.polyfit(np.log(MC_err), np.log(MC_time), 1)
#fit_MC=np.exp(z_MC[0]*np.log(MC_err_extrapol))
#print z_MC[0]


MISC_err=np.array([0.057, 0.016 ,0.028,0.029 ])
MISC_time=np.array([ 0.3, 2.5 , 15 ,2040])

z_MISC= np.polyfit(np.log(MISC_err), np.log(MISC_time), 1)
fit_MISC=np.exp(z_MISC[0]*np.log(MISC_err))
print z_MISC[0]



#############################################################################



fig = plt.figure()

plt.plot(MC_err,MC_time,linewidth=2.0,label='MC' , marker='>',hold=True) 
#plt.plot(MC_err_extrapol, fit_MC*0.001,linewidth=2.0,label=r'slope= %s' % format(z_MC[0]  , '.2f'), linestyle = '--')
plt.plot(MC_err_extrapol, fit_MC*0.001,linewidth=1.0,label=r'slope= %s' % format(-3.0 , '.2f'), linestyle = '--', color = 'b')

plt.plot(MISC_err,MISC_time,linewidth=2.0,label='MISC'  , marker='v',hold=True,color='r') 
#plt.plot(MISC_err, fit_MISC*0.00000000001,linewidth=2.0,label=r'slope= %s' % format(-5.0  , '.2f'), linestyle = '--', color='r')


plt.yscale('log')
plt.xscale('log')
plt.xlabel('Error',fontsize=14)

plt.ylabel('CPU time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='upper right', fontsize=10.5)
plt.savefig('./results/error_vs_time_set2.eps', format='eps', dpi=1000)  
