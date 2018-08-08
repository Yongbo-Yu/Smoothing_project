import numpy as np
y = np.array([ 0.0073,0.0118,0.0192])
x = np.array([ 1/16.0, 1/8.0,1/4.0])

z = np.polyfit(np.log(x), np.log(y), 1)
#z=np.zeros(2)
#z[0]=3
#z[1]= -1.9379
#z[1]=-1.5001
#fit=z[0]*np.log(x)+z[1]

print(z)
#print(x)
#print(np.exp(fit))