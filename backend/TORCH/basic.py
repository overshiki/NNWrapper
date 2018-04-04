from . import np, cp
import chainer
import chainer.functions as F

class VarBase:
	def __init__(self, device=0):
		self.device = device
		self.op = operation(device=self.device)
		self.guard = device_guard(device=self.device)
		self.var = None


	def ndarray(self):
		return self.var._data[0]


	def to_device(self, device):
		self.device = device 
		self.op = operation(device=self.device)
		self.guard = device_guard(device=self.device)
		self.var.to_gpu(self.device)
		# self.ndarray = self.var._data[0]		

	def cast(self, dtype):
		self.var = chainer.functions.cast(self.var, dtype)

	def __repr__(self):
		return self.ndarray().__str__()
	def __str__(self):
		return self.ndarray().__str__()

	def backward(self):
		self.var.backward()
		# self._update_grad()

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

	def __len__(self):
		return len(self.ndarray())

	def math_prepare(self, other):
		use_chainer = False
		if(type(other)==self.__class__):
			var, other = self.ndarray(), other.ndarray()
		elif((type(other)==self.op.arrayType) or (type(other)==int) or (type(other)==float)):
			var = self.ndarray()
		elif(type(other)==chainer.Variable):
			var = self.var
			use_chainer = True

		elif(type(other)==chainer.Parameter):
			var = self.var 
			use_chainer = True
		else:
			raise ValueError("input other is not Variable nor self.op.arrayType, but to be: {}".format(type(other)))
		return use_chainer, var, other

	def __mul__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var * other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var * other, device=self.device)

	def __add__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var + other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var + other, device=self.device)

	def __sub__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var - other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var - other, device=self.device)

	def __truediv__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var / other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var / other, device=self.device)

	def __floordiv__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var // other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var // other, device=self.device)


	def __lt__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var < other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var < other, device=self.device)

	def __le__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var <= other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var <= other, device=self.device)

	def __eq__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var == other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var == other, device=self.device)

	def __ne__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var != other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var != other, device=self.device)

	def __gt__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var > other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var > other, device=self.device)

	def __ge__(self, other):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var >= other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var >= other, device=self.device)

	def __pow__(self, other, *args):
		use_chainer, var, other = self.math_prepare(other)
		with self.guard:
			if(use_chainer==False):
				#use cupy array, where broadcast is automatically supported
				return Variable(var ** other, device=self.device)
			else:
				#use chainer, where manually do the broadcast
				var, other = F.broadcast(var, other)
				return Variable(var ** other, device=self.device)

	def __copy__(self):
		result = Variable(self.ndarray().copy(), device=self.device)
		return result

	def __deepcopy__(self, memo):
		result = Variable(self.ndarray().copy(), device=self.device)
		memo[id(self)] = result
		return result

	def astype(self, _type):
		with self.guard:
			return Variable(self.ndarray().astype(_type), device=self.device)





class Variable(VarBase):
	'''
	basically, we want Variable and Parameter to be as transparancy as possible, so that just as a switcher node for type transformation
	'''
	def __init__(self, x, device=0):
		super().__init__(device=device)

		if issubclass(type(x), Variable):
			self.var = x.var
		elif type(x)==chainer.variable.Variable:
			self.var = x
		elif type(x)==self.op.arrayType:
			self.var = chainer.Variable(x)
		elif type(x)==self.op.counterType:
			self.var = chainer.Variable(self.op.array_to_device(x))
		elif type(x)==chainer.Parameter:
			self.var = x
		else:
			raise ValueError("input type is neither {}, chainer.variable.Variable, nor op.arrayType: {}, nor chainer.Parameter, but {}".format(self.__class__, self.op.arrayType, type(x)))
		#--> this will just bring any input into chainer.Variable type


		self.shape = self.var._data[0].shape
		self.ndim = self.var._data[0].ndim





class Parameter(VarBase):
	def __init__(self, x, device=0):
		super().__init__(device=device)

		if issubclass(type(x), Parameter):
			self.var = x.var
		elif type(x)==chainer.variable.Variable:
			self.var = chainer.Parameter(x.array)
		elif type(x)==self.op.arrayType:
			self.var = chainer.Parameter(x)
		elif type(x)==self.op.counterType:
			self.var = chainer.Parameter(self.op.array_to_device(x))
		elif type(x)==chainer.Parameter:
			self.var = x
		else:
			raise ValueError("input type is neither {}, chainer.variable.Variable, nor op.arrayType: {}, nor chainer.Parameter, but {}".format(self.__class__, self.op.arrayType, type(x)))

		self.shape = self.var._data[0].shape
		self.ndim = self.var._data[0].ndim


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
