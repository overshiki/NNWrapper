from . import np, torch
import copy


r"""it seems pytorch now deprecated Variable and use tensor to handle backward method instead
however, in our current wrapping implementation, we just use autograd.Variable 
TODO: check tensor autograd method in future document of pytorch
"""

class VarBase:
	def __init__(self, device=0):
		self.device = device
		self.var = None
		self.varType = None

	def new(self, *args, **kwargs):
		r"""Constructs a new variable of the same data type as :attr:`self` variable.
		"""
		return self.__class__(*args, **kwargs)

	@property
	def ndarray(self):
		r'''for safety consideration, only set getter for ndarray, which means no access to changing the value of self.var._data[0]
		'''
		print(type(self.var.data))
		print(type(tensor(self.var.data)))
		return tensor(self.var.data, device=self.device)

	def numpy(self):
		if self.device==-1:
			return self.var.data.numpy()
		else:
			return self.var.data.cpu().numpy()

	@property
	def shape(self):
		return self.ndarray.shape

	@property
	def ndim(self):
		return self.ndarray.ndim

	@property
	def dtype(self):
		return self.ndarray.dtype

	def to_device(self, device):
		self.device = device 
		self.var = self.var.cuda(self.device)
		#TODO: implement cpu transfer


	def cast(self, dtype):
		raise NotImplementedError("cast method for torch.autograd.Variable is not wrapped here, in the future, it will be directly reflected in torch.tensor class")

	def __repr__(self):
		return self.ndarray.__str__()
	def __str__(self):
		return self.ndarray.__str__()

	def backward(self):
		self.var.backward()

	def reshape(self, *x):
		return self.new(self.var.reshape(*x), device=self.device)

	def transpose(self, x):
		var = self.var 
		pairs = []
		_x = copy.deepcopy(x)
		for index, i in enumerate(x):
			if index!=i and index!=_x[index]:
				pairs.append([index, i])
				_x[index] = index 
				_x[i] = i
		for pair in pairs:
			var = var.transpose(*pair)
		return self.new(var, device=self.device)

	def sum(self, *x, **kwarg):
		raise NotImplementedError()

	def max(self, axis=0):
		return self.new(torch.max(self.var, dim=axis)[0], device=self.device)

	def argmax(self, axis=0):
		return self.new(torch.max(self.var, dim=axis)[1], device=self.device)

	def __getitem__(self, item):
		return self.new(self.var[item], device=self.device)

	def __len__(self):
		return len(self.var)

	#following the pytorch document, we see Variable object actually returns tensor object. We treat self.var to actually be tensor
	def __copy__(self):
		result = self.new(self.var.clone(), device=self.device)
		return result

	def __deepcopy__(self, memo):
		result = self.new(self.var.clone(), device=self.device)
		memo[id(self)] = result
		return result

	def astype(self, dtype):
		if dtype=='float64':
			return self.new(self.var.double(), device=self.device)
		elif dtype=='float32':
			return self.new(self.var.float(), device=self.device)
		elif dtype=='float16':
			return self.new(self.var.half(), device=self.device)
		elif dtype=='int64':
			return self.new(self.var.long(), device=self.device)
		elif dtype=='int32':
			return self.new(self.var.int(), device=self.device)
		elif dtype=='int16':
			return self.new(self.var.short(), device=self.device)
		elif dtype=='int8':
			return self.new(self.var.char(), device=self.device)
		elif dtype=='bool':
			return self.new(self.var.byte(), device=self.device)
		else:
			raise TypeError("unsupported dtype {}".format(dtype))





class device_guard:
	r"""empty device_guard class, compatible with other backend.
	Temperaly
	"""
	def __init__(self, device=0):
		pass
	# 	self.device = device
	# 	if(self.device!=-1):
	# 		self.guard = cp.cuda.Device(self.device)

	def __enter__(self):
		pass
	# 	if(self.device!=-1):
	# 		return self.guard.__enter__()

	def __exit__(self, *args):
		pass
	# 	if(self.device!=-1):
	# 		self.guard.__exit__()

	# def use(self):
	# 	if(self.device!=-1):
	# 		self.guard.use()

# def device_guard_decorator(fun):
# 	def guard_fun(self, *x, **kwargs):
# 		with self.guard:
# 			return fun(self, *x, **kwargs)
# 	return guard_fun


class TensorBase:
	r"""a wrapper to basic tensor type for all neural network framework like: numpy and cupy in chainer, tensor in pytorch
	"""
	def __init__(self, device=0):
		self.device = device
		# self.guard = device_guard(device=self.device)

		self.data = None

	# def guard(self):
	# 	return self.guard

	def new(self, *args, **kwargs):
		r"""Constructs a new variable of the same data type as :attr:`self` variable.
		"""
		return self.__class__(*args, **kwargs)

	@property
	def ndarray(self):
		r'''for safety consideration, only set getter for ndarray, which means no access to changing the value of self._data
		'''
		return self.data

	def numpy(self):
		if self.device!=-1:
			return self.data.cpu().numpy()
		else:
			return self.data.numpy()

	@property
	def var(self):
		return self.data

	def to_device(self, device):
		r"""transfer tensor from self.device to device, and then set self.device to be new device
		if one of self.device and device is -1, then a cupy->numpy or numpy->cupy transfer is handled
		"""
		if device>-1:
			#while target device is not cpu, this include from cpu to gpu, and from gpu to gpu
			self.data = self.data.cuda(device)
		elif device==-1 and self.device>-1:
			self.data = self.data.cpu()
		elif device==-1 and self.device==-1:
			pass
		else:
			raise ValueError("current device is {}, while target device is {}".format(self.device, device))

		self.device = device 


	@property
	def shape(self):
		return self.ndarray.shape

	@property
	def ndim(self):
		return self.ndarray.dim()

	@property
	def dtype(self):
		return self.ndarray.type()
		#TODO: transfer pytorch tensor type to numpy-like dtype expression

	@property
	def size(self):
		return self.ndarray.size()
		#TODO: transfer pytorch tensor size to numpy-like size expression

	def __repr__(self):
		return self.ndarray.__str__()+" on device {}".format(self.device)
	def __str__(self):
		return self.__repr__()

	def cast(self, dtype):
		#transfer pytorch cast method into dtype-based method
		if dtype=='float64':
			self.data = self.ndarray.double()
		elif dtype=='float32':
			self.data = self.ndarray.float()
		elif dtype=='float16':
			self.data = self.ndarray.half()
		elif dtype=='int64':
			self.data = self.ndarray.long()
		elif dtype=='int32':
			self.data = self.ndarray.int()
		elif dtype=='int16':
			self.data = self.ndarray.short()
		elif dtype=='int8':
			self.data = self.ndarray.char()
		elif dtype=='bool':
			self.data = self.ndarray.byte()
		else:
			raise TypeError("unsupported dtype {}".format(dtype))


	def astype(self, dtype):
		if dtype=='float64':
			return self.new(self.ndarray.double(), device=self.device)
		elif dtype=='float32':
			return self.new(self.ndarray.float(), device=self.device)
		elif dtype=='float16':
			return self.new(self.ndarray.half(), device=self.device)
		elif dtype=='int64':
			return self.new(self.ndarray.long(), device=self.device)
		elif dtype=='int32':
			return self.new(self.ndarray.int(), device=self.device)
		elif dtype=='int16':
			return self.new(self.ndarray.short(), device=self.device)
		elif dtype=='int8':
			return self.new(self.ndarray.char(), device=self.device)
		elif dtype=='bool':
			return self.new(self.ndarray.byte(), device=self.device)
		else:
			raise TypeError("unsupported dtype {}".format(dtype))

	def reshape(self, *x):
		return self.new(self.ndarray.reshape(*x), device=self.device)

	# @device_guard_decorator
	def sum(self, *x, axis=0):
		return self.new(self.ndarray.sum(*x, dim=axis), device=self.device)

	# @device_guard_decorator
	def max(self, *x, **kwarg):
		return self.new(self.ndarray.max(*x, **kwarg), device=self.device)

	def transpose(self, *x):
		return self.new(self.ndarray.transpose(*x), device=self.device)

	def __getitem__(self, item):
		return self.new(self.ndarray[item], device=self.device)

	def __len__(self):
		return len(self.ndarray)

	def __copy__(self):
		result = self.new(self.ndarray.clone(), device=self.device)
		return result

	def __deepcopy__(self, memo):
		result = self.new(self.ndarray.clone(), device=self.device)
		memo[id(self)] = result
		return result



class Variable(VarBase):
	r'''variable wrapper for all kinds of neural network framework, like in chainer, it wraps chainer.Variable object, in pytorch, it wraps torch.autograd.Variable object
	for simplicty and transparency, only tensor data type or chainer.Variable is accepted as input
	'''
	def __init__(self, x, device=0):
		super().__init__(device=device)

		if isinstance(x, tensor):
			if self.device != x.device:
				x.to_device(self.device)
			self.var = torch.autograd.Variable(x.ndarray)

		elif isinstance(x, torch.autograd.Variable):
			#TODO:
			# if self.device != x.device:
			# 	x = x.to_gpu(self.device)
			self.var = x
		elif isinstance(x, VarBase):
			if self.device != x.device:
				x = x.to_device(self.device)
			self.var = x.var
		else:
			raise ValueError("input type is neither tensor, chainer.Variable, nor chainer.Parameter, but {}".format(type(x)))
		self.varType = torch.autograd.Variable


#TODO:
class Parameter(VarBase):
	r'''parameter wrapper for all kinds of neural network framework, like in chainer, it wraps chainer.Parameter object, in pytorch, it wraps torch.autograd.Parameter object
	for simplicty and transparency, only tensor data type or chainer.Parameter is accepted as input
	'''
	def __init__(self, x, device=0):
		super().__init__(device=device)

		if isinstance(x, tensor):
			if self.device != x.device:
				x.to_device(self.device)
			self.var = torch.nn.Parameter(x.ndarray)

		elif isinstance(x, torch.nn.Parameter):
			self.var = x
		else:
			raise ValueError("input type is neither tensor, nor torch.nn.Parameter, but {}".format(type(x)))
		self.varType = torch.nn.Parameter


class tensor(TensorBase):
	r"""implementation of tensor data structure

    .. note::
        The following operators are defined for variable(s).
        * Indexing: ``a[slices]`` (:meth:`__getitem__`)
        * Addition: ``a + b`` (:meth:`__add__`, :meth:`__radd__`)
        * Subtraction: ``a - b`` (:meth:`__sub__`, :meth:`__rsub__`)
        * Multiplication: ``a * b`` (:meth:`__mul__`, :meth:`__rmul__`)
        * Division: ``a / b`` (:meth:`__div__`, :meth:`__rdiv__`, \
                               :meth:`__truediv__`, :meth:`__rtruediv__`)
        * Floor Division: ``a // b`` (:meth:`__floordiv__`, \
                                      :meth:`__rfloordiv__`)
        * Exponentiation: ``a ** b`` (:meth:`__pow__`, :meth:`__rpow__`)
        * Matirx Multiplication: ``a @ b`` (:meth:`__matmul__`, \
                                            :meth:`__rmatmul__`)
        * Negation (Arithmetic): ``- a`` (:meth:`__neg__`)

    Args:
        x torch.tensor: Initial data array.
		device (int): device index for gpu and cpu(-1)
		TODO: support dtype check for list, so that enable list as input
	"""
	def __init__(self, x, device=0):
		super().__init__(device=device)
		if self.device>-1:
			if isinstance(x, torch._TensorBase):
				if self.device>-1:
					self.data = x.cuda(self.device) 
				else:
					self.data = x
				#TODO: support for cuda.tensor to tensor
			elif isinstance(x, np.ndarray):
				if self.device>-1:
					self.data = torch.from_numpy(x).cuda(self.device) 
				else:
					self.data = torch.from_numpy(x)
			elif isinstance(x, list):
				x = np.array(x)
				if self.device>-1:
					self.data = torch.from_numpy(x)
					print("self.data.type before", self.data.type())
					self.data = self.data.cuda(self.device)	
					print("self.data.type", self.data.type())				
				else:
					self.data = torch.from_numpy(x)
				#TODO: more dtype support for list
			else:
				raise TypeError("input x is neither numpy ndarray nor tensor, nor list, but {}".format(type(x)))

		else:
			raise ValueError("invalid device setting, current device is {}".format(self.device))