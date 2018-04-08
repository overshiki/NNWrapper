import torch
import numpy as np 

A = np.random.randn(10)
A = torch.from_numpy(A)
print(A)
print(type(A))

print(isinstance(A, torch._TensorBase))

print(torch.cuda.device_count())

# print(A.requires_grad)
B = A.cuda()

C = A.cuda()

D = B+C

# print(D.shape, D.ndim, D.dtype, D.size)
print(D.type())
E = D.float()
print(E.type())

F = E.clone()

G = F+1

print(type(G))

H = torch.autograd.Variable(G)