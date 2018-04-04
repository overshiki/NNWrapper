import chainer
try:
	import cupy as cp
except:
	pass

from NN.NNWrapper import operation

class Variable():
	def __init__(self, x, device=0):
		self.ndarray = None
		self.chainer = None
		self.grad = None
		self.device = device
		self.op = operation.operation(device=self.device)

		# print("type(x): ", type(x), "self.__class__: ", self.__class__)
		if type(x)==self.__class__:
			self.ndarray = x.ndarray
			self.chainer = x.chainer
		elif type(x)==chainer.variable.Variable:
			self.chainer = x
			self.ndarray = x._data[0]
		elif type(x)==self.op.arrayType:
			self.ndarray = x 
			self.chainer = chainer.Variable(x)
		elif type(x)==chainer.Parameter:
			self.chainer = x
			self.ndarray = x._data[0]

		self.shape = self.ndarray.shape
		self.ndim = self.ndarray.ndim
		self._update_grad()

	def _update_grad(self):
		if(self.chainer._grad_var is None):
			self.grad = None
		else: 
			self.grad = self.chainer._grad_var._data[0]

	def _grad(self):
		self._update_grad()
		return self.grad

	def cast(self, dtype):
		self.ndarray = self.ndarray.astype(dtype)
		self.chainer = chainer.functions.cast(self.chainer, dtype)

	def __repr__(self):
		return self.ndarray.__str__()
	def __str__(self):
		return self.ndarray.__str__()


	def backward(self):
		self.chainer.backward()
		self._update_grad()

	def reshape(self, *x):
		return Variable(self.ndarray.reshape(*x))

	def sum(self, *x, **kwarg):
		return Variable(self.ndarray.sum(*x, **kwarg))

	def max(self, *x, **kwarg):
		return Variable(self.ndarray.max(*x, **kwarg))

	def transpose(self, *x):
		return Variable(self.ndarray.transpose(*x))

	def __getitem__(self, item):
		 return Variable(self.ndarray[item])

	def __len__(self):
		return len(self.ndarray)

	def __mul__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray * other.ndarray) #use ndarray because chainer does not support broadcast
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray * other)
		elif(type(other)==chainer.Variable):
			return Variable(self.chainer * other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __add__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray + other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray + other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __sub__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray - other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray - other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __truediv__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray / other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray / other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __floordiv__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray // other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray // other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))


	def __lt__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray < other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray < other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __le__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray <= other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray <= other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __eq__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray == other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray == other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __ne__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray != other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray != other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __gt__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray > other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray > other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __ge__(self, other):
		if(type(other)==self.__class__):
			return Variable(self.ndarray >= other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray >= other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __pow__(self, other, *args):
		if(type(other)==self.__class__):
			return Variable(self.ndarray ** other.ndarray)
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			return Variable(self.ndarray ** other)
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))



if __name__ == '__main__':
	pass