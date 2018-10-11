import numpy as np
from scipy import stats
 
class BlackScholes:
    @staticmethod
    def Generate_Asset(S_0,R,T,Vol,X):
        return S_0*np.exp((R-Vol**2/2)*T + Vol*np.sqrt(T)*X)
 
class Monte_Carlo:
    @staticmethod
    def __Get_Correlated_Brownian(nb_assets,nb_simulation,correlation_matrix):
    # """Function that returns a matrix with all the correlated brownian for all the simulations by proceeding a Cholesky decomposition"""
        X = np.random.randn(nb_simulation,nb_assets)
        lower_triang_cholesky = np.linalg.cholesky(correlation_matrix)
        for i in range(nb_simulation):
            X[i,:]=np.dot(lower_triang_cholesky,X[i,:])  #np.dot perform a matrix product
        return X
 
    @staticmethod
    def Get_Basket_Call_Price(starting_asset_values,correlation_matrix,asset_vol,maturity,nb_simulation,risk_free_rate,weights,strike):
        nb_assets = len(starting_asset_values)
 
        #Generate independant random variable:
        X = Monte_Carlo.__Get_Correlated_Brownian(nb_assets,nb_simulation,correlation_matrix)
 
        Final_Stock_values = BlackScholes.Generate_Asset(starting_asset_values[:],risk_free_rate,maturity,asset_vol[:],X[:])
 
        #print(Final_Stock_values[:])
        #print(weights)
        #print(Final_Stock_values[:]*weights)
        #print(np.sum(Final_Stock_values[:]*weights,axis=1))
        #print(np.maximum(np.sum(Final_Stock_values[:]*weights,axis=1)-strike,0))
 
        Payoffs = np.maximum(np.sum(Final_Stock_values[:]*weights,axis=1)-strike,0)
        return np.mean(Payoffs)*np.exp(-risk_free_rate*maturity), np.std(Payoffs*np.exp(-risk_free_rate*maturity))/float(np.sqrt(1*(10**8)))
 

 
option_parameters_1 = {
    'starting_asset_values' : np.array([100,100]),
    'correlation_matrix':[[1,0.3],[0.3,1]],
    'asset_vol' : np.array([0.2,0.2]),
    'maturity' : 1,
    'nb_simulation' : 5*(10**8),
    'risk_free_rate' : 0.00,
    'weights' : np.array([0.5,0.5]),
    'strike' : 100
}
 
option_parameters_2 = {
    'starting_asset_values' : np.array([100,100,100]),
    'correlation_matrix':[[1,0.3,0.3],[0.3,1,0.3],[0.3,0.3,1]],
    'asset_vol' : np.array([0.4,0.4,0.4]),
    'maturity' : 1,
    'nb_simulation' : 1*(10**8),
    'risk_free_rate' : 0.00,
    'weights' : np.array([1/3.0,1/3.0,1/3.]),
    'strike' : 100
}
 

#Example on a basket option with 3 stocks :
basket_option_price = Monte_Carlo.Get_Basket_Call_Price(**option_parameters_2)
 
print("Basket Option Price on 3 stocks : {0}".format(basket_option_price))
