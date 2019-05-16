import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

x=[2,30,46,66,137,441,1800]
y=[0.1625,0.0098,0.026,0.0468,0.0639,0.0451,0.0232]
#bound1=0.05*np.ones((np.size(x)))
bound2=0.025*np.ones((np.size(x)))

plt.plot(x, y,linewidth=2.0, label=r'$\mathcal{E}_Q$', linestyle = '--',marker='>', hold=True) 
#plt.plot(x, bound1,linewidth=2.0,label=r'$TOL_Q$ = %s' % 0.05)
plt.plot(x, bound2,linewidth=2.0,label=r'$TOL_Q$ = %s' % 0.025)
plt.yscale('log')
plt.xscale('log')

plt.xlabel(r'$W$',fontsize=14)
plt.ylabel(r'$\mathcal{E}_Q$',fontsize=14) 

plt.legend(loc='lower left')
plt.savefig('./results/complexity_MISC_Bergomi_N_8_level2_rich_bias_0025.eps', format='eps', dpi=1000) 