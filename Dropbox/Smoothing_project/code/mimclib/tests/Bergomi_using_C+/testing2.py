import pathos.multiprocessing as mp
import pickle
import numpy as np
# p = mp.Pool(4)  # Processing Pool with four processors
# results=p.map(lambda x: x**2, range(10))
# print results

import fftw3
import RBergomi
from RBergomi import *




# class PickalableSWIG:

#     def __setstate__(self, state):
#         self.__init__(*state['args'])

#     def __getstate__(self):
#         return {'args': self.args}

# class PickalableRBergomi(RBergomi, PickalableSWIG):

#     def __init__(self, *args):
#         self.args = args
#         print args
#         RBergomi.__init__(self)





pickle.loads(pickle.dumps(RBergomi()))  