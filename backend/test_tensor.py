# from CHAINER import Variable, tensor, np, cp

from TORCH import tensor, np, Variable


A = tensor(np.random.randn(4, 2), device=0)

print(A.shape, A.ndim, A.dtype)

print(A+A)
print(A+5)
print(5+A)
print(A)
print(-A)

print(5*A)
print(A*5)

B = A/5.
C = 5./A

print(B*C)

print(A)

print(abs(A))

print(A**3)

print(A==B, A!=B, A>=B, A<=B, A>B, A<B)

print(type(A==B))

import copy
print(copy.copy(A))
print(copy.deepcopy(A))

print(A.astype('int16'))

A = Variable(tensor(np.random.randn(4, 2), device=0), device=0)

print(A.shape, A.ndim, A.dtype)

print(A+5)
print(5+A)
print(A)
print(-A)

print(5*A)
print(A*5)

B = A/5.
C = 5./A

print(B*C)

print(A)

print(abs(A))

print(A**3)

import copy
print(copy.copy(A))
print(copy.deepcopy(A))

print(A.astype('int16'))