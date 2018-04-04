from .. import F, L, np, cp, Variable, operation, device_guard, Graph, GraphList

class Linear(Graph):
	'''
	TODO: consider a better to wrap a link into node

	we considering not directly registrate any weights, but just treat self.layer as child. we then handle this link type child in base
	'''
	def __init__(self, n_in, n_hidden, device=0):
		'''
		linear = Ax+b
		'''
		super().__init__(device=device)
		self.link = True

		self.n_in, self.n_hidden = n_in, n_hidden

		# with self.guard():
			# self.layer = L.Linear(self.n_in, self.n_hidden)
			# if(self.device!=-1):
			# 	self.layer.to_gpu(device=self.device)


		layer = L.Linear(self.n_in, self.n_hidden)
		if(self.device!=-1):
			layer.to_gpu(device=self.device)
		self.add_node('layer', layer)


	def forward(self, x):
		# x = chainer.Variable(x.ndarray.astype(cp.float32))
		x = F.cast(x, 'float32')
		shape = list(x.shape)
		shape.pop(-1)
		x = F.reshape(x, (-1, self.n_in))
		x = self.layer(x)
		if(self.n_hidden==1):
			x = F.reshape(x, shape)
		else:
			shape.append(self.n_hidden)
			x = F.reshape(x, shape)
		return Variable(x, device=self.device)
