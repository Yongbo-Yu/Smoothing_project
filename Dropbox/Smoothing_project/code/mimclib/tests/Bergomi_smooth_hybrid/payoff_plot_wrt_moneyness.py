import fftw3
import RBergomi
from RBergomi import *
import numpy as np

import matplotlib.pyplot as plt

k_gird=np.linspace(0.8,1.3,10)
k=Vector(1)
HIn=Vector(1)


e=Vector(1)
r=Vector(1)
t=Vector(1)
HIn[0]=0.43

e[0]=1.9
r[0]=-0.9
t[0]=1.0

	

x=0.235**2 
NIn=16
MIn=1


W1 = Vector(NIn)
W2 = Vector(NIn)
for i in range(0,NIn):
	W1[i] = np.random.normal(loc=0.0, scale=1.0)
	W2[i] = np.random.normal(loc=0.0, scale=1.0)

payoff=np.zeros(10)
for i in range(0,10):
	k[0]=k_gird[i]
	z=RBergomi.RBergomiST( x,  HIn, e,  r,  t,k ,  NIn, MIn)
	payoff[i]=z.ComputePayoffRT_single(W1,W2)
print(payoff)
plt.plot(k_gird,payoff ,linewidth=2.0,label='rBergomi payoff' ,linestyle = '--',marker='>', hold=True) 
plt.xlabel(r'$K$',fontsize=14)
        #ax.axis([0.1, 0.6, 0.0001, 0.1])
plt.ylabel('rBergomi payoff' ,fontsize=14) 
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.22, right=0.96, top=0.96)
        #plt.subplot_tool()
plt.legend(loc='upper right')
plt.savefig('./results/rBergomi_payoff_16steps_wrt_monyeness_2_H043.eps', format='eps', dpi=1000)  
