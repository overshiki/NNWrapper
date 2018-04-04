from . import np, cp
import chainer
from copy import copy, deepcopy

class Variable():
	def __init__(self, x, device=0):
		# self.chainer = None
		self.device = device
		self.op = operation(device=self.device)
		self.guard = device_guard(device=self.device)

		with self.guard:
			if type(x)==self.__class__:
				self._ndarray = x._ndarray

			elif type(x)==self.op.arrayType:
				self._ndarray = x 

			elif type(x)==self.op.counterType:
				self._ndarray = self.op.array_to_device(x)

			else:
				raise ValueError("input type is neither Variable, nor op.arrayType: {}, nor op.counterType: {}, but {}".format(self.op.arrayType, self.op.counterType, type(x)))

		self.shape = self._ndarray.shape
		self.ndim = self._ndarray.ndim
		# self._update_grad() #not implemented in NNCUPY


	def ndarray(self):
		return self._ndarray


	def to_device(self, device):
		self.device = device 
		self.op = operation(device=self.device)
		self.guard = device_guard(device=self.device)
		self.ndarray = self.op.array_to_device(self.ndarray())		

	def cpu(self, delete=False):
		reValue = self.op.asnumpy(self.ndarray())
		if delete==True:
			del self._ndarray
		return reValue

	# def _update_grad(self):
	# 	raise NotImplementedError("for NNCUPY, no grad is generated")

	# def _grad(self):
	# 	raise NotImplementedError("for NNCUPY, no grad is generated")

	def cast(self, dtype):
		self._ndarray = self._ndarray.astype(dtype)

	def __repr__(self):
		return self.ndarray().__str__()
	def __str__(self):
		return self.ndarray().__str__()


	def backward(self):
		raise NotImplementedError("for NNCUPY, no backward is implemented")

	def reshape(self, *x):
		return Variable(self.ndarray().reshape(*x), device=self.device)

	def sum(self, *x, **kwarg):
		return Variable(self.ndarray().sum(*x, **kwarg), device=self.device)

	def max(self, *x, **kwarg):
		return Variable(self.ndarray().max(*x, **kwarg), device=self.device)

	def transpose(self, *x):
		return Variable(self.ndarray().transpose(*x), device=self.device)

	def __getitem__(self, item):
		with self.guard:
			data = self.ndarray()
			return Variable(data[item], device=self.device)


	def __copy__(self):
		result = Variable(self.ndarray().copy(), device=self.device)
		return result

	def __deepcopy__(self, memo):
		result = Variable(self.ndarray().copy(), device=self.device)
		memo[id(self)] = result
		return result

	def __len__(self):
		return len(self.ndarray())

	def __mul__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() * other.ndarray(), device=self.device) #use ndarray because chainer does not support broadcast
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() * other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __add__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() + other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() + other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __sub__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() - other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() - other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __truediv__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() / other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() / other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __floordiv__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() // other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() // other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))


	def __lt__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() < other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() < other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __le__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() <= other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() <= other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __eq__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() == other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() == other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __ne__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() != other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() != other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __gt__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() > other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() > other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __ge__(self, other):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() >= other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() >= other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))

	def __pow__(self, other, *args):
		with self.guard:
			if(type(other)==self.__class__):
				return Variable(self.ndarray() ** other.ndarray(), device=self.device)
			elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
				return Variable(self.ndarray() ** other, device=self.device)
			else:
				raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))


class device_guard:
	def __init__(self, device=0):
		self.device = device
		if(self.device!=-1):
			self.guard = cp.cuda.Device(self.device)

	def __enter__(self):
		if(self.device!=-1):
			return self.guard.__enter__()

	def __exit__(self, *args):
		if(self.device!=-1):
			self.guard.__exit__()


class operation:
	def __init__(self, device=0):
		self.device = device
		if self.device==-1:
			self.run = np 
			self.arrayType = np.ndarray
			self.counterType = cp.core.ndarray
		else:
			self.run = cp
			self.arrayType = cp.core.ndarray
			self.counterType = np.ndarray

		self.newaxis = self.run.newaxis
		self.guard = device_guard(device=self.device)

	def __str__(self):
		return "operation for device: {}, of arrayType should be: {}".format(self.device, self.arrayType)


	def array_to_device(self, array):
		if self.device==-1:
			if(type(array)!=self.arrayType):
				return array.get()
		else:
			if(type(array)!=self.arrayType):
				with self.guard:
					return self.run.asarray(array)
			#TODO, considering array in different gpus
		return array


	def unskin_element(self, x):
		if(type(x)==Variable):
			_x = x.ndarray()
		elif(type(x)==self.arrayType):
			#TODO, considering array in different gpus
			_x = x
		elif(type(x)==self.counterType):
			_x = self.array_to_device(x)
		else:
			_x = x
		return _x


	def unskin(self, x):
		_x = []
		for i in x:
			_x.append(self.unskin_element(i))
		return _x

	def concatenate(self, x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.concatenate(_x, **kwargs), device=self.device)

	def argmax(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.argmax(*_x, **kwargs), device=self.device)

	def expand_dims(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.expand_dims(*_x, **kwargs), device=self.device)

	def where(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return self.run.where(*_x, **kwargs)

	def stack(self, x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.stack(_x, **kwargs), device=self.device)

	def array(self, x):
		with self.guard:
			return Variable(self.run.array(self.unskin_element(x)), device=self.device)

	def exp(self, *x):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.exp(*_x), device=self.device)

	def arange(self, *x):
		with self.guard:
			return Variable(self.run.arange(*x), device=self.device)

	def minimum(self, *x):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.minimum(*_x), device=self.device)

	def equal(self, *x):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.equal(*_x), device=self.device)		

	def cumsum(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.cumsum(*_x, **kwargs), device=self.device)

	def asarray(self, x):
		with self.guard:
			if(type(x)==Variable):
				return x
			elif(type(x)==self.run.core.ndarray):
				return Variable(x, device=self.device)
			else:
				return Variable(self.run.asarray(x), device=self.device)

	def asnumpy(self, x):
		with self.guard:
			if(type(x)==Variable):
				return self.run.asnumpy(x.ndarray)
			elif(type(x)==self.run.core.ndarray):
				return self.run.asnumpy(x)
