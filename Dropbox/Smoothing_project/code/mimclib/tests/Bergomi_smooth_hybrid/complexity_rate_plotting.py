
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import numpy as np


y_MC= np.array([2380,615,19,2])
y_rich_level_2 = np.array([ 40, 40,20,15])
y_rich_level_1= np.array([ 35, 2.5,2, 0.13])
y= np.array([ 900, 30,21,0.4])
x = np.array([ 5*10**(-3),10**(-2),5*10**(-2),10**(-1) ])



z= np.polyfit(np.log(x), np.log(y),1)
fit=np.exp(z[0]*np.log(x))
print z[0]

z_l1= np.polyfit(np.log(x), np.log(y_rich_level_1),1)
fit_l1=np.exp(z_l1[0]*np.log(x))
print z_l1[0]


z_l2= np.polyfit(np.log(x), np.log(y_rich_level_2),1)
fit_l2=np.exp(z_l2[0]*np.log(x))*100
print z_l2[0]


z_MC= np.polyfit(np.log(x), np.log(y_MC),1)
fit_MC=np.exp(z_MC[0]*np.log(x))*100
print z_MC[0]

fig = plt.figure()


plt.plot(x, y,linewidth=2.0,label='MISC' ,linestyle = '--',marker='>', hold=True) 
plt.plot(x, fit,linewidth=2.0,label=r'rate = %s' % format(z[0]  , '.2f'), marker='o')

plt.plot(x, y_rich_level_1,linewidth=2.0,label='MISC+Rich(level 1)' ,linestyle = '--',marker='>', hold=True) 
plt.plot(x, fit_l1,linewidth=2.0,label=r'rate = %s' % format(z_l1[0]  , '.2f'), marker='o')

plt.plot(x, y_rich_level_2,linewidth=2.0,label='MISC+Rich(level 2)' ,linestyle = '--',marker='>', hold=True) 
plt.plot(x, fit_l2,linewidth=2.0,label=r'rate = %s' % format(z_l2[0]  , '.2f'), marker='o')

plt.plot(x, y_MC,linewidth=2.0,label='MC' ,linestyle = '--',marker='>', hold=True) 
plt.plot(x, fit_MC,linewidth=2.0,label=r'rate = %s' % format(z_MC[0]  , '.2f'), marker='o')

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$TOL$',fontsize=14)


plt.ylabel('Average time',fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
params = {'legend.fontsize': 10,
          'legend.handlelength': 1}
plt.rcParams.update(params)
plt.legend(loc='lower left')
plt.savefig('./results/complexity_rate_Bergomi_2.eps', format='eps', dpi=1000) 