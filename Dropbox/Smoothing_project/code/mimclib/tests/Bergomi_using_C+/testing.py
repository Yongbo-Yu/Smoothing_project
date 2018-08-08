from joblib import Parallel, delayed
import multiprocessing
    
# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = range(10**4) 


num_cores = multiprocessing.cpu_count()
print num_cores
    
values[:,i] = Parallel(n_jobs=num_cores)(delayed(processInput)() for j in inputs)                