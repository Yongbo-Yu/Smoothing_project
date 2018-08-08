

#Import required libraries, classes and functions
import os
import numpy as np
from matplotlib import pyplot as plt
from rbergomi import rBergomi
from utils import bsinv
vec_bsinv = np.vectorize(bsinv)


#Create instance of the rBergomi class with $n$ steps per year, $N$ paths, maximum maturity $T$ and roughness index $a$


rB = rBergomi(n = 4, N = 1, T = 1.0, a = -0.43)

#Fix the generator's seed for replicable results
np.random.seed(0)



#Generate required Brownian increments

dW1 = rB.dW1()
print dW1

#print(np.random.multivariate_normal(rB.e, rB.c,(1,4)))


rng = np.random.randn(1, 2,4)
#print rng
rng=rB.L.dot(rng)
print(rng.transpose())


dW2 = rB.dW2()









#Construct the Volterra process
Y = rB.Y(dW1)


#Correlate the orthogonal increments, using $\rho$

dB = rB.dB(dW1, dW2, rho = -0.9)

#Construct the variance process, using $\xi$ and $\eta$
V = rB.V(Y, xi = 0.235**2, eta = 1.9)

#Finally construct the price process
S = rB.S(V, dB)



#replicate implied volatiliies shared by Bennedsen, Lunde and Pakkanen. Fix the log-strike range, $k$

#k = np.arange( -0.2231, 0.1, 0.1)

k=[-0.2231, 0 ,   0.1823]

#Compute call payoffs, prices, and implied volatilities,



ST = S[:,-1][:,np.newaxis]
K = np.exp(k)[np.newaxis,:]
print(K)
call_payoffs = np.maximum(ST - K,0)
call_prices = np.mean(call_payoffs, axis = 0)[:,np.newaxis]

call_prices_std=np.std(call_payoffs, axis = 0)/np.sqrt(np.shape(call_payoffs)[0])

print(call_prices)
print(call_prices_std)




