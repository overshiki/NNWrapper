import torch
import numpy as np

class device_guard:
	r"""a with guard for device selection
	also provide option to permantely change default device to the selected one
	"""
	# def __init__(self, device=0):
	# 	self.device = device
	# 	if(self.device!=-1):
	# 		self.guard = cp.cuda.Device(self.device)

	# def __enter__(self):
	# 	if(self.device!=-1):
	# 		return self.guard.__enter__()

	# def __exit__(self, *args):
	# 	if(self.device!=-1):
	# 		self.guard.__exit__()

	# def use(self):
	# 	if(self.device!=-1):
	# 		self.guard.use()
	pass

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
	def sum(self, *x, **kwarg):
		return self.new(self.ndarray.sum(*x, **kwarg), device=self.device)

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
			else:
				raise TypeError("input x is neither numpy ndarray nor tensor, but {}".format(type(x)))

		else:
			raise ValueError("invalid device setting, current device is {}".format(self.device))


# TensorBase.device_guard_decorator = device_guard_decorator