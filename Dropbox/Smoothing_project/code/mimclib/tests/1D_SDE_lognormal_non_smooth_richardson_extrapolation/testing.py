
import numpy as np
y = np.array([52,52,53, 52,42])
x = np.array([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1)])
# plt.yscale('log')
# plt.xscale('log')
z = np.polyfit(np.log(x), np.log(y), 1)
fit=-z[0]*x
print(z)