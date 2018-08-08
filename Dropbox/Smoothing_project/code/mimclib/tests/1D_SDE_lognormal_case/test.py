import datetime
from random import gauss
import numpy as np

def generate_asset_price(S,v,r,T):
    return S * np.exp((r - 0.5 * v**2) * T + v * np.sqrt(T) * gauss(0,1.0))

def call_payoff(S_T,K):
    return max(0.0,S_T-K)

S = 100.0 # underlying price
v = 0.4 # vol of 40%
r = 0.0 # rate of 0.0%
T = 1.0
K = 100.0
simulations = 100000000
payoffs = []
discount_factor = np.exp(-r * T)

for i in xrange(simulations):
    S_T = generate_asset_price(S,v,r,T)
    payoffs.append(
        call_payoff(S_T, K)
    )

price = discount_factor * (sum(payoffs) / float(simulations))
print 'Price: %.4f' % price