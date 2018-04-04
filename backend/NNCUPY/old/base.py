import cupy as cp
import numpy as np
from NN.NNWrapper import Variable
from NN.NNWrapper import operation
op = operation()


class base:
	def __init__(self):
		self.device = None

	def gpu(self):
		return cp.cuda.Device(self.device)

	def guard(self):
		return cp.cuda.Device(self.device)

	def register_weights(self, ndarray):
		with self.gpu():
			var = Variable(cp.asarray(ndarray))
		return var

	def forward(self, x):
		pass

	def __call__(self, *x):
		with self.gpu():
			x = tuple(map(lambda i:Variable(i), x))
			return self.forward(*x)

class Sigmoid(base):
	def __init__(self, device=0):
		'''
		sigmoid = 1./(1+exp(-x))
		'''
		super().__init__()
		self.device = device

	def forward(self, x):
		x = (op.exp(x*(-1.)) + 1.) ** -1
		return x


class Linear(base):
	'''
	TODO: register
	'''
	def __init__(self, param1, param2, device=0):
		'''
		linear = Ax+b

		could be: param1->n_in, param2->n_hidden
		or: 	  param1->weights, param2->bias
		'''
		super().__init__()
		self.device = device
		self.weights = None
		self.bias = None

		if(type(param1)==int):
			self.type = 'init'
			self.n_in, self.n_hidden = param1, param2
		else:
			self.type = 'load'
			self.weights, self.bias = param1, param2
			self.n_in, self.n_hidden = self.weights.shape[0], self.weights.shape[1]


		with self.gpu():
			if(self.weights is None):
				self.weights = cp.asarray(np.random.randn(self.n_hidden, self.n_in))
				self.bias = cp.asarray(np.random.randn(self.n_hidden))
			else:
				self.weights = cp.asarray(self.weights).transpose([1,0])
				self.bias = cp.asarray(self.bias)


	def forward(self, x):
		# N*P*S*k
		if(self.weights.shape[0]==1):
			x = x*self.weights
			# N*P*S
			x = x.sum(axis=-1)

		else:
			X = []
			for index in range(self.weights.shape[0]):
				_x =  x*self.weights[index]
				_x = _x.sum(axis=-1)
				X.append(_x)
			x = op.stack(X, axis=-1)

		x = x+self.bias
		return x

if __name__ == '__main__':
	sigmoid = Sigmoid(device=1)

	with cp.cuda.Device(1):
		x = cp.array([[1,2,3,4],[4,5,6,7]]).astype(cp.float32)

	print(type(x))
	x = sigmoid(x)
	print(type(x))

	linear = Linear(4, 10, device=1)
	_x = linear(x)
	print(_x)