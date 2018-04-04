from .. import node, Graph, np, Variable

class Linear(Graph):
	'''
	TODO: register
	'''
	def __init__(self, n_in, n_hidden, device=0):
		'''
		linear = Ax+b

		could be: param1->n_in, param2->n_hidden
		or: 	  param1->weights, param2->bias
		'''
		super().__init__(device=device)

		self.n_in, self.n_hidden = n_in, n_hidden

		weights = self.op.asarray(np.random.randn(self.n_hidden, self.n_in))
		bias = self.op.asarray(np.random.randn(self.n_hidden))

		self.register_weights('weights', weights)
		self.register_weights('bias', bias)


	def forward(self, x):

		weights = self.get_weights('weights')
		bias = self.get_weights('bias')
		# N*P*S*k
		if(weights.shape[0]==1):
			x = x*weights
			# N*P*S
			x = x.sum(axis=-1)

		else:
			X = []
			for index in range(weights.shape[0]):
				_x =  x*weights[index]
				_x = _x.sum(axis=-1)
				X.append(_x)
			x = self.op.stack(X, axis=-1)

		x = x+bias
		return Variable(x)
