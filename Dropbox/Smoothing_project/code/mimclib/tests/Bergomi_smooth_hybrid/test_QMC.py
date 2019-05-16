#!/usr/bin/env python
from numpy import *
from timeit import default_timer as timer
import sys
from latticeseq_b2 import latticeseq_b2 # Python point generator
def g(x, c, b):
	"""Our function g, which accepts x as a [sz1-by-...-by-s]
	array with s the number of dimensions in a single vector."""
	s = size(x, -1)
	return exp(c * inner(arange(1, s+1, dtype='d')**-b, x))
def gexact(s, c, b):
	a = c*arange(1, s+1, dtype='d')**-b
	return prod(expm1(a) / a)
# parameters for our function
s = 100
c = 1
b = 2
exact = gexact(s, c, b)
# random shifting
M = 2**3
random.seed(1) # Mersenne Twister
shifts = random.rand(M, s)
# QMC generator (truncated to correct number of dimensions)
latgen = latticeseq_b2('exod2_base2_m20_CKN.txt', s=s)
acc = zeros((M,)) # accumulator for each shift
nprev = 0
mmin = 4; mmax = 16
for m in range(mmin, mmax+1):
	N = 2**m
	n = N / M
	start = timer()
	for k in range(nprev, n):
		x = latgen.next() # next point, evaluate in all shifts:
		acc += [ g((x+shift) % 1, c, b) for shift in shifts ]
	nprev = n
	Q = mean(acc/n)
	stdQ = std(acc/n) / sqrt(M)
	end = timer()
	print "QMC_Q = %g (error=%g, std=%g, N=%d) in %f sec" % \
	(Q, abs(Q-exact), stdQ, N, end-start)
	sys.stderr.write("%d\t%g\t%g\t%g\t%g\n" % \
	(N, Q, stdQ, abs(Q-exact), end-start))