from .. import Graph, Variable

class conv2d(Graph):
	def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, device=0):
		'''
		convolution2d
		'''
		super().__init__(device=device)
		self.link = True

		self.in_channels, self.out_channels, self.ksize, self.stride, self.pad, self.nobias = in_channels, out_channels, ksize, stride, pad, nobias

		with self.guard():
			self.layer = L.Convolution2D(self.in_channels, self.out_channels, ksize=self.ksize, stride=self.stride, pad=self.pad, nobias=self.nobias)
			if(self.device!=-1):
				self.layer.to_gpu(device=self.device)

		for name, params in self.layer.namedparams():
			if name=="/b":
				self.register_weights('bias', params)
			elif name=="/W":
				self.register_weights('weights', params)


	def forward(self, x):
		# x = chainer.Variable(x.ndarray.astype(cp.float32))
		x = F.cast(x, 'float32')
		# shape = list(x.shape)
		# shape.pop(-1)
		# x = F.reshape(x, (-1, self.n_in))
		# x = self.layer(x)
		# if(self.n_hidden==1):
		# 	x = F.reshape(x, shape)
		# else:
		# 	shape.append(self.n_hidden)
		# 	x = F.reshape(x, shape)
		x = self.layer(x)
		return Variable(x)