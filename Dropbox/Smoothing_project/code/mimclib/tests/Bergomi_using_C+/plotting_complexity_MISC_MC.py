import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



# #Case for parameter set 5
MC_err=np.array([ 0.0180,0.0084,0.0048,0.0014])
MC_time=np.array([ 88,96,112,156])
MISC_err=np.array([0.0182,0.0086,0.0042,0.0016 ])
MISC_time=np.array([ 1,6,24,92])










fig = plt.figure()

plt.plot(MC_err,MC_time,linewidth=2.0,label='MC' , marker='>',hold=True) 
plt.plot(MISC_err,MISC_time,linewidth=2.0,label='MISC'  , marker='v',hold=True) 


plt.yscale('log')
plt.xscale('log')
plt.xlabel('Error',fontsize=14)

plt.ylabel('CPU time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='lower left')
plt.savefig('./results/error_vs_time_set5.eps', format='eps', dpi=1000)  
