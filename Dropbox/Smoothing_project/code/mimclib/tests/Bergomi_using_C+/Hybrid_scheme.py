'''
Copyright (c) 2018 Jens Nierste

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import math
import numexpr as ne
import numpy as np
import time
import tables
import pyfftw
import mpmath
from numpy.random import multivariate_normal

def gen_Z(W, W_orth, p, N, steps):
    """
   gen_Z generates Z = p * W + sqrt( 1 - p^2 ) * W_orth

   Input parameters:
      Number W - brownian motion with cov matrix.
      Number W_orth - standard brownian motion independent of W.
      Number p - the correlation parameter.
      N - the total number of simulations.
      steps - the number of steps in each simulation.

   Output: 
        Z - the corralated brownain motion
    """
    return ne.evaluate('p*W + sqrt(1-p**2)*W_orth', optimization='moderate')

def b_opt(k, a):
    """
    Calculates the optimal b for the hybrid scheme. 
    The optimal b minimizes the error of the hybrid scheme.

    Input parameters:
        k - the indice of the b
        a - the parameter alpha element of (-0.5,0.5) \ {0} 

    Ouput:
        the optimal b for the given k and alpha (a)

    Defined by
    Bennedsen, Lunde, Pakkanen
    Hybrid scheme for Brownian semistationary processes
    Finance Stoch (2017) 21:931-965
    page 941
    """
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


def cov_matrix(k, a, n):
    """
    Calculates the covariance matrix for the hybrid scheme, 
    as definied on page 946.
    
    Input parameters:
        k - the parameter kappa of the hybrid scheme. (page 945)
        a - the parameter alpha of the hybrid scheme.
        n - the parameter n of the hybrid scheme, steps per time unit.
           n is equal to steps.

    Output:
     A covariance matrix.

    Defined by
    Bennedsen, Lunde, Pakkanen
    Hybrid scheme for Brownian semistationary processes
    Finance Stoch (2017) 21:931-965
    page 945-946
    """
    cov = np.zeros((k + 1, k + 1))
    cov[0, 0] = 1. / n
    for j in range(1, k + 1):
        cov[0, j] = (j ** (a + 1) - (j - 1) ** (a + 1)) / ((a + 1) * n ** (a + 1))
        cov[j, 0] = cov[0, j]
        cov[j, j] = (j ** (2 * a + 1) - (j - 1) ** (2 * a + 1)) / ((2 * a + 1) * n ** (2 * a + 1))
        for m in range(1, k + 1):
            if (j < m):
                cov[j, m] = (1 / ((a + 1) * n ** (2 * a + 1))) * (
                        j ** (a + 1) * m ** a * mpmath.hyp2f1(-a, 1, a + 2, (j * 1.0) / (m * 1.0)) - (j - 1) ** (
                        a + 1).real * (m - 1) ** a * mpmath.hyp2f1(-a, 1, a + 2, (j * 1.0 - 1) / (m * 1.0 - 1)).real)
            if (j > m and m != 0):
                    # need to swap j,m because of the loop iterrations
                    cov[j, m] = cov[m, j]     
    return cov

def gen_W(k, a, N, steps):
    """
    Generates the multivariate normal distributed random variables needed for the
    hybrid scheme. Therefore the random variables are created with the
    covariance matrix given by the function cov_matrix.

    Input parameters:
        k - the parameter kappa of the hybrid scheme.
        a - the parameter alpha of the hybrid scheme.
        N - the number of total simulations.
        steps, the number of steps in each simulation.

    Output:
        N(0,cov_matrix) multivariate normal variables.
    """
    return multivariate_normal(np.zeros(k + 1), cov_matrix(k, a, steps), (N, 2*steps), check_valid='ignore')

def gen_W_orth(N, steps, dt):
    """
    Generates a standard Brownian motion independent of W,
    which is needed for the calculation of Z in the function gen_Z.

    Input parameters:
        N - the total number of simulations.
        steps - the number of steps in each simulation.
        dt - the size of each individual step (stepsize).

    Output:
        Standard Brownian motion
    """
    return math.sqrt(dt)*np.random.randn(N, steps)


def gen_Y(kappa, W, a, N, steps, qmc):
    """
    Calculate Y of the hybrid scheme for the rBergomi model
    as on page 951. Y is defined on page 944 and 946
    in "Hybrid scheme for Brownian semistationary processes".

    Input parameters:
        kappa, the parameter kappa of the hybrid scheme.
        W - multivariate normal distributed random variables
            with covariance matrix cov_matrix.
        a - the parameter alpha of the rBergomi scheme.
        N - the total number of simulations.
        steps - the number of steps in each simulation.

    Output:
        Y as defined on page 951 for the rBergomi model.

    Defined by
    Bennedsen, Lunde, Pakkanen
    Hybrid scheme for Brownian semistationary processes
    Finance Stoch (2017) 21:931-965
    """

    G = np.zeros(steps)
    for i in range(kappa, steps):
        G[i] = (b_opt(i+1, a) / steps) ** a

    X1 = np.zeros((N,steps))
    for i in range(steps):
        for r in range(1,k+1):
            X1[:,i] += W[:,i + steps - r, r] 
        
    pyfftw.interfaces.cache.enable()
    nthreads = multiprocessing.cpu_count()
    Xi = W[:,:steps,0]
    shape = 2*steps + 1
    fft_G_obj = pyfftw.builders.fft(G, n = shape, threads=nthreads,overwrite_input=True, planner_effort='FFTW_ESTIMATE', auto_align_input = False, auto_contiguous = False)
    fft_W_obj = pyfftw.builders.fft(Xi,n= shape, threads=nthreads,overwrite_input=True, planner_effort='FFTW_ESTIMATE' ,auto_align_input = False, auto_contiguous = False)
    ifft_obj = pyfftw.builders.ifft(fft_W_obj.get_output_array(),n=shape, threads=nthreads,overwrite_input=True, planner_effort='FFTW_ESTIMATE',auto_align_input = False, auto_contiguous = False)
    fft_padded_G = fft_G_obj(G)
    fft_padded_W = fft_W_obj(Xi)
    convolution = ifft_obj(fft_padded_G * fft_padded_W)[:,:steps]
    convolution = convolution.real
    convolution[:,:kappa] = 0
        
    return ne.evaluate('sqrt(2.*a+1.)*(X1 + convolution)', optimization='moderate')

def gen_V(Y, t, alpha, eta, xi):
    """
    Calculate the spot variance process v of the rBergomi model,
    see page 951 of the hybrid scheme paper for the formula.

    Input parameters:
        Y - the parameter calculated by the hybrid scheme, see page 951,
            output of the function Y(kappa,W,a,eta,N,steps)
        t - t element of [0,T]
        alpha - the parameter alpha of the rBergomi model
        eta - the parameter eta of the rBergomi model -- look up what eta stands for
        xi - the parameter xi of the rBergomi model -- look up what xi stands for

    Output:
        v as defined on page 951 hybrid scheme paper
    """
    t_part = t[1:]
    return ne.evaluate('xi * exp( eta * Y - 0.5 * eta**2 * t_part**( 2 * alpha + 1))', optimization='moderate')
  
def gen_S(V, Z, S0, dt, N, steps):
    """
    Calculates the value of the underlying S in the rBergomi model.

    Input parameters:
        V - the spot variance process of the rBergomi model,
            see page 951 of the hybrid scheme paper
            output of the function gen_V(Y, t, alpha, eta, xi)
        Z - a standard brownin motion with Z = p * W + sqrt(1 - p^2 ) * W_orth
            output of the function gen_Z(W,W_orth,p)
        S0 - the price of the underlying at t = 0
        dt -  the size of each step (stepzise)

    Output:
        the value of the underlying S in the rBergomi model for each t
    """
    start = time.time()
    V_part = V[:,:-1]
    Z_part = Z[:,1:]
    integral = np.cumsum(ne.evaluate('sqrt(V_part) * Z_part - 0.5 * V_part * dt', optimization='moderate'), axis=1)
    S = np.zeros((N, steps))
    S[:, 0] = S0
    S[:, 1:] = ne.evaluate('S0*exp(integral)', optimization='moderate')
    print('Zeit für gen_S = ', time.time()-start)
    return S

def run_simulation(N, steps, T, k, S0, a, eta, xi, p, m):
    """
    Runs the simulation of S and returns the values for ST.

    Input parameters:
        N - number of simulations
        steps - numbers of steps in the grid
        dt - size of the grid steps
        T - maturity
        k - parameter kappa of the hybrid scheme
        S0 - the value of the underlying at time 0
        alpha - the parameter alpha of the rBergomi model
        eta - the parameter eta of the rBergomi model -- look up what eta stands for
        xi - the parameter xi of the rBergomi model -- look up what xi stands for
        p - the correlation of W and W_orth, for the generation of Z
        qmc - boolean, if True, then a qmc run is made, if False a normal
              mc simulation is done

    Output:
        ST values for the underlying.
    """
    dt = 1.0 / steps
    t = np.linspace(0, 1,  steps + 1)
    start1 = time.time()
    W = gen_W(k, a, N, steps)
    print('Zeit für gen_W = ', time.time() - start1)
    start2 = time.time()
    W_orth = gen_W_orth(N, steps, dt)
    print('Zeit für W_orth = ', time.time() - start2)
    start3 = time.time()
    Z = gen_Z(W[:, :steps, 0], W_orth, p, N, steps)
    print('Zeit für gen_Z = ', time.time() - start3)
    start4 = time.time()
    Y = gen_Y(k, W, a, N, steps, qmc)
    print('Zeit für gen_Y = ', time.time() - start4)
    start5 = time.time()
    V = gen_V(Y, t, a, eta, xi)
    print('Zeit für gen_V = ', time.time() - start5)
    return  gen_S(V, Z, S0, dt, N, steps)[:,steps-1]


if __name__ == '__main__':
    # Initializierung der Werte
    # Anzahl der Simulationen
    N = 2**12
    # m für 2^m
    m = 14
    # Anzahl der Schritte pro Zeiteinheit, im Artikel definiert als n
    steps = 2 ** m
    # Schrittgroesse
    dt = 1.0 / steps
    # Laufzeit/Maturity -- hat Einfluss auf die Gesamtanzahl der Schritte durch n*T, aber ich
    # denke ich werde das ignorieren und nur mit T = 1 arbeiten, d.h.
    # der Wert ist nur von theoretischem Interesse
    # darüber wird die größe des grids festgelegt
    T = 1
    # S0 the start price of the underlying
    S0 = 1
    # Kappa
    k = 3
    # Alpha
    a = -0.43
    # eta
    eta = 1.9
    # xi
    xi = 0.235 ** 2
    # rho the correlation for Z
    p = -0.9
    #set number of repeats
    

    '''
    For loop  which writes files to the hard drive.
    Writes the values for ST of each simulation run in a .hdf5 file,
    if the files shouldn't get saved just run the function
    run_simulation outside the for loop. run_simulation
    returns the values of ST.
    '''
    for repeats in [r]:
        #save the parameters of the simulation in an array called parameters
        parameters = np.array([N*repeats, steps, m,  
                        T, dt, S0, k, a, eta,
                        xi, p])
        #set desired path, where the simulation resutls shoud get saved
        #set values for the hdf file
        '''
        insert the filename, where the hdf. files should get saved
        '''
        fileName = f"*******"     
        shape_carray = (N*repeats, m)
        atom = tables.Float64Atom()
        hdf5 = tables.open_file(fileName, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        #array where the simulation data gets saved
        simulation_data = hdf5.create_carray(hdf5.root, 'simulation', atom, shape_carray, filters=filters)
        #array where the parameters get saved
        parameters_data = hdf5.create_carray(hdf5.root, 'parameters',atom,  (1,11), filters=filters)
        #array where the time information gets saved
        time_date = hdf5.create_carray(hdf5.root, 'time', atom, (m,4), filters = filters)
        parameters_data[0,:] = parameters
        for i in range(1,m+1):
            start2 = time.time()
            for r in range(repeats):
                print('steps = ', 2**i, ' noch ', repeats - r, ' Wiederholungen', 'rho = ', p, 'kappa = ', k)
                simulation_data[N*r:N*(r+1), i-1] = run_simulation(N,2**i,T,k,S0,a,eta,xi,p,m)
            time_date[i-1,:] = np.array([N*repeats,2**i, k, time.time()-start2])
    hdf5.close()

   
        
 
    
    

