import numpy as np
d=2
X1=(1/np.sqrt(d))*np.ones((1,d))
A=np.eye(d,d)	
A[0,:]=X1
A=A.transpose()
print A



def normalize(v):
    return v / np.sqrt(v.dot(v))

n = len(A)

A[:, 0] = normalize(A[:, 0])


for i in range(1, n):
    Ai = A[:, i]
    for j in range(0, i):
        Aj = A[:, j]
        t = Ai.dot(Aj)
        Ai = Ai - t * Aj
    A[:, i] = normalize(Ai)
print A.transpose()
print A.dot(A.transpose())
z=[1,2,3,4,6]
print z

