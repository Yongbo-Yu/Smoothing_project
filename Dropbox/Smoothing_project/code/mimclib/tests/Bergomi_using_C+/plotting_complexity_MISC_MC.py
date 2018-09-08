import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator



# #Case for parameter set 1 
MC_err=np.array([ ])
MC_time=np.array([ ])
MC_rich_err=np.array([ ])
MC_rich_time=np.array([ ])

MISC_err=np.array([ ])
MISC_time=np.array([ ])
MISC_rich_err=np.array([ ])
MISC_rich_time=np.array([ ])










fig = plt.figure()

plt.plot(MC_time,MC_err,linewidth=2.0,label=r'$N=2$' , marker='>',hold=True) 
plt.plot(MC_rich_time,MC_rich_err,linewidth=2.0,label=r'$N=4$' , marker='o',hold=True) 
plt.plot(MISC_time,MISC_err,linewidth=2.0,label=r'$N=8$'  , marker='v',hold=True) 
plt.plot(MISC_rich_time,MISC_rich_err,linewidth=2.0,label=r'$N=16$'  , marker='*',hold=True) 

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Error',fontsize=14)

plt.ylabel('CPU time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
plt.legend(loc='lower left')
plt.savefig('./results/error_vs_time_set1.eps', format='eps', dpi=1000)  
