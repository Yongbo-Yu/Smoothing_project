import warnings
import os.path
import numpy as np
import mimclib.test
import mimclib.misc as misc
from mimclib import setutil_richardson
from mimclib import mimc
import argparse

warnings.filterwarnings("error")
warnings.filterwarnings("always", category=mimclib.test.ArgumentWarning)



class MyRun:

    #extrapolate_s_dims =2
   
    #Returns  work estimate of lvls    
    def workModel(self, run, lvls):
        mat = lvls.to_dense_matrix()
        gamma = np.hstack((run.params.gamma, np.ones(mat.shape[1]-len(run.params.gamma))))
        beta = np.hstack((run.params.beta, 2*np.ones(mat.shape[1]-len(run.params.gamma))))
        return np.prod(beta**(mat*gamma), axis=1)


    def initRun(self, run):
        self.prev_val = 0
        self.prb = Problem_non_smooth_richardson_extrapolation_basket(run.params) 
        #self.extrapolate_s_dims = self.prb.N -1
        self.extrapolate_s_dims = 2*self.prb.basket_d*(self.prb.N-1) #SET TO SAME AS INPUT DIMENSION
        

        
    ## PROBLEM SPECIFIC
        # fnknots  gives two arrays first one for quadrature points and the second one for weights
        fnKnots= lambda beta: misc.knots_gaussian(misc.lev2knots_doubling(1+beta),  0, 1) # the standard deviation is coming from 
        #the distribution of the brownian bridges increments
        
    ##
        # Construct MiscSampler class with the following parameters
        #self.misc = misc.MISCSampler(d=0, fnKnots=fnKnots, min_dim=self.prb.N-1)
       
        self.misc = misc.MISCSampler(d=0, fnKnots=fnKnots, min_dim=2*self.prb.basket_d*(self.prb.N-1))
        

        #feed parameters to the problem
        

        # those are the error and work contribution used to compute the profit
        self.d_err_rates = 2.*np.log(run.params.beta);#*np.minimum(1, run.params.qoi_df_nu / run.params.qoi_dim)
        self.d_work_rates = np.log(run.params.beta) * run.params.gamma;
        

        #Returns  work estimate of lvls  and optimal indices of levels using profit rule
        run.setFunctions(ExtendLvls=lambda lvls, r=run: self.extendLvls(run, lvls), 
                                WorkModel=lambda lvls, r=run: self.workModel(run, lvls))
        return

    
    def mySampleQoI(self, run, inds, M):
        # import os
        # import psutil
        # process = psutil.Process(os.getpid())
        # mem = psutil.virtual_memory()
        # print(mem)
        return self.misc.sample(inds, M, fnSample=self.solveFor_seq)


       # this defines the stochastic field used in solveAtpoints in sample in mimc, arrY: plays the role of qudrature points after transformation as in sf in solveAtpoints
    def solveFor_seq(self, alpha, arrY):
        #print(arrY)
        output = np.zeros(len(arrY))
        self.prb.BeginRuns(alpha, np.max([len(Y) for Y in arrY]))
        for i, Y in enumerate(arrY):
            output[i] = self.prb.SolveFor(np.array(Y)) # this computes the value of the objective function (given by  objfun) at quad points
        
        self.prb.EndRuns()# gives the elapsed time
        return output    
   
    

    def extendLvls(self, run, lvls):
        if len(lvls) == 0:
            # First run, add min_lvls on each dimension
            d = 0 + self.extrapolate_s_dims
            eye = np.eye(d, dtype=int)
            new_lvls = np.vstack([np.zeros(d, dtype=int)] +
                             [i*eye for i in range(1, run.params.min_lvls)])
            lvls.add_from_list(new_lvls)
            return

        import time
        tStart = time.time()
        #print(run.params.min_dim)
        # estimate rates
        self.d_err_rates, \
            s_fit_rates = misc.estimate_misc_error_rates(d=0,
                                                         lvls=lvls,
                                                         errs=run.last_itr.calcDeltaEl(),
                                                         d_err_rates=self.d_err_rates,
                                                         #lev2knots=lambda beta:misc.lev2knots_doubling(1+beta)
                                                         #my corrected version(otherwise with the previous version I get an error I need to check that)
                                                         lev2knots=lambda beta:misc.lev2knots_doubling(1+beta))
        #################### extrapolate error rates
        if s_fit_rates is not None:
            valid = np.nonzero(s_fit_rates > 1e-15)[0]  # rates that are negative or close to zero are not accurate.

            N = len(s_fit_rates) + self.extrapolate_s_dims
            k_of_N = self.transNK(1 , N)
        
            K = np.max(k_of_N)
            

            c = np.polyfit(np.log(1+k_of_N[valid]), s_fit_rates[valid], 1)
            k_rates_stoch = c[0]*np.log(1+np.arange(0, K+1)) + c[1]
            s_err_rates = np.maximum(k_rates_stoch[k_of_N[:N]],
                                     np.min(s_fit_rates[valid]))
            s_err_rates[valid] = s_fit_rates[valid]  # The fitted rates should remain the same
        else:
            s_err_rates = []

        tEnd_rates = time.time() - tStart
        
        ######### Update
        tStart = time.time()
        self.profCalc = setutil_richardson.MISCProfCalculator(self.d_err_rates +
                                                   self.d_work_rates,
                                                   s_err_rates)
        mimc.extend_prof_lvls(lvls, self.profCalc, run.params.min_lvls)
        #print(lvls)

    def transNK(self, d, N, problem_arg=0):  # what is d here ???
        # return np.arange(0, N), np.arange(0, N)
        # Each ind has 2*|ind|_0 samples
        indSet = setutil_richardson.GenTDSet(d, N, base=0)
        N_per_ind = 2**np.sum(indSet!=0, axis=1)
        
        if problem_arg == 1:
            N_per_ind[1:] /= 2    
        _, k_ind = np.unique(np.sum(indSet, axis=1), return_inverse=True)
        k_of_N = np.repeat(k_ind, N_per_ind.astype(np.int))[:N]
    
        # N_of_k = [j+np.arange(0, i, dtype=np.uint) for i, j in
        #           zip(N_per_ind, np.hstack((np.array([0],
        #                                              dtype=np.uint),
        #                                     np.cumsum(N_per_ind)[:np.max(k_of_N)])))]
        return k_of_N



    def addExtraArguments(self, parser):
	    pass


if __name__ == "__main__":
    from Problem_non_smooth_richardson_extrapolation_basket import Problem_non_smooth_richardson_extrapolation_basket

    Problem_non_smooth_richardson_extrapolation_basket.Init()
    import mimclib.test
    run = MyRun()
    mimclib.test.RunStandardTest(fnSampleLvl=run.mySampleQoI,
                                 fnAddExtraArgs=run.addExtraArguments,
                                 fnInit=run.initRun)# initialize the run
    Problem_non_smooth_richardson_extrapolation_basket.Final()