

import numpy as np
from .variable import Variable

class empty:
	def __init__(self, ):
		self.core = self
		self.ndarray = self


try:
	import cupy as cp
except:
	print("no cupy avaliable on device")
	cp = empty()
	pass


class operation:
	def __init__(self, device=0):
		self.device = device
		if self.device==-1:
			self.op = np 
			self.arrayType = np.ndarray
		else:
			self.op = cp
			self.arrayType = cp.core.ndarray

	def unskin(self, x):
		_x = []
		for i in x:
			if(type(i)==Variable):
				_x.append(i.ndarray)
			elif(type(i)==cp.core.ndarray):
				_x.append(i)
			else:
				_x.append(i)
		return _x

	def concatenate(self, x, **kwargs):
		_x = self.unskin(x)
		return Variable(self.op.concatenate(_x, **kwargs))

	def argmax(self, *x, **kwargs):
		_x = self.unskin(x)
		return Variable(self.op.argmax(*_x, **kwargs))

	def expand_dims(self, *x, **kwargs):
		_x = self.unskin(x)
		return Variable(self.op.expand_dims(*_x, **kwargs))

	def where(self, *x, **kwargs):
		_x = self.unskin(x)
		return self.op.where(*_x, **kwargs)

	def stack(self, x, **kwargs):
		_x = self.unskin(x)
		return Variable(self.op.stack(_x, **kwargs))

	def array(self, x):
		return Variable(self.op.array(x))


	def exp(self, *x):
		_x = self.unskin(x)
		return Variable(self.op.exp(*_x))

	def arange(self, *x):
		return Variable(self.op.arange(*x))

	def minimum(self, *x):
		_x = self.unskin(x)
		return Variable(self.op.minimum(*_x))

	def equal(self, *x):
		_x = self.unskin(x)
		return Variable(self.op.equal(*_x))		

	def cumsum(self, *x, **kwargs):
		_x = self.unskin(x)
		return Variable(self.op.cumsum(*_x, **kwargs))

	def asarray(self, x):
		if(type(x)==Variable):
			return x
		elif(type(x)==self.op.core.ndarray):
			return Variable(x)
		else:
			return Variable(self.op.asarray(x))

	def asnumpy(self, x):
		if(type(x)==Variable):
			return self.op.asnumpy(x.ndarray)
		elif(type(x)==self.op.core.ndarray):
			return self.op.asnumpy(x)


if __name__ == '__main__':
	tensor = cp.array([[1,2,3,4], [1,2,3,4]])
	var = Variable(tensor)
	print(type(var)==Variable, type(var.ndarray))

	op = operation()
	_list = [tensor]*5
	result = op.concatenate(_list, axis=0)
	print(result.shape)

	_list = [tensor]*5
	result = op.concatenate(_list, axis=1)
	print(result.shape)

	_list = [var]*5
	result = op.concatenate(_list, axis=0)
	print(result.shape)