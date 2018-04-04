from CHAINER import tensor, np, cp


A = tensor(np.random.randn(4, 2), device=0)

print(A.shape(), A.ndim())

# B = tensor(np.random.randn(4, 2), device=0)

# C = np.random.randn(4, 2)

# D = cp.asarray(np.random.randn(4, 2))

# E = 5

# print(A+B, A-B, A*B, A**B, A>B, A>=B, A<B, A<=B, A==B, A!=B, A/B, A//B)

# print("#"*164)

# print(A+C, A-C, A*C, A**C, A>C, A>=C, A<C, A<=C, A==C, A!=C, A/C, A//C)

# print("#"*164)

# print(A+D, A-D, A*D, A**D, A>D, A>=D, A<D, A<=D, A==D, A!=D, A/D, A//D)

# print("#"*164)

# print(A+E, A-E, A*E, A**E, A>E, A>=E, A<E, A<=E, A==E, A!=E, A/E, A//E)