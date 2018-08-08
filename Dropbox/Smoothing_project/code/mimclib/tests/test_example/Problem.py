import numpy as np
import os
import sys
import time

class Problem(object):
# attributes
    nelem=None;
    random_gen=None;
    elapsed_time=0.0;
#methods
    # this method initializes the class
    def __init__(self, params, nested=False):
        self.nested = nested
        self.params = params

        self.random_gen = None or np.random
    # this method returns the mesh size used in each dimension (physical dimension)
    def BeginRuns(self, ind, N):
	self.elapsed_time=0.0;
        self.nelem = np.array(self.params.h0inv * self.params.beta**(np.array(ind)), dtype=np.uint32)
        if self.nested:
            self.nelem -= 1
        assert(len(self.nelem) == self.GetDim())
        return self.nelem

    def EndRuns(self):
	elapsed_time=self.elapsed_time;
	self.elapsed_time=0.0;
	return elapsed_time;
    # this computes the value of the objective function (given by  objfun) at quad points
    def SolveFor(self, Y):
       Y = np.array(Y)
       goal=self.objfun(self.nelem,Y)
       return goal

    # objfun
    def objfun(self,nelem,y):
	  #  y=y[0:10];
    	start_time=time.time();
        #print(y)
    	QoI=1.0/(1.0+0.01*sum(y))+1/sum(nelem**2.0);
    	elapsed_time_qoi=time.time()-start_time;
    	self.elapsed_time=self.elapsed_time+elapsed_time_qoi;
	return QoI


    def Quit(self):
	pass

    def __exit__(self, type, value, traceback):
        pass

    def __enter__(self):
        return self

    @staticmethod
    def Init():
        import sys 
        count = len(sys.argv)   
        #arr = (ct.c_char_p * len(sys.argv))()
        arr = sys.argv

    @staticmethod
    def Final():
	pass

    def GetDim(self):
        return 3;
