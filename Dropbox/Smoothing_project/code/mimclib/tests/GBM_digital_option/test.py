import random
from math import exp, sqrt

def gbm(S, v, r, T):
    return S * exp((r - 0.5 * v**2) * T + v * sqrt(T) * random.gauss(0,1.0))

def binary_call_payoff(K, S_T):
    if S_T >= K:
        return 1.0
    else:
        return 0.0

# parameters
S = 100.0 # asset price
v = 0.2 # vol of 40%
r = 0.05 # rate of 1%
maturity = 1.0
K = 100.0 # ATM strike
simulations = 50000
payoffs = 0.0

# run simultaion
for i in xrange(simulations):
    S_T = gbm(S, v, r, maturity)
    payoffs += binary_call_payoff(K, S_T)

# find prices
option_price = exp(-r * maturity) * (payoffs / float(simulations))

print 'Price: %.8f' % option_price



#Binary options can also be priced using the traditional Black Scholes model, using the following formula  
#Where N is the cumulative normal distribution function, and d2 is given by the standard Black Scholes formula.
from scipy.stats import norm
from math import exp, log, sqrt
S, K, v, r, T = 100.0, 100.0, 0.2, 0.05, 1.0
d2 = (log(S/K) + (r - 0.5 * v**2) * T) / v*sqrt(T)
print exp(-r * T) * norm.cdf(d2)
