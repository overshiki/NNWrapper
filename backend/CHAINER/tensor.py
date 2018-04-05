from . import np, cp
import chainer
import chainer.functions as F


class device_guard:
	r"""a with guard for device selection
	also provide option to permantely change default device to the selected one
	"""
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

	def use(self):
		if(self.device!=-1):
			self.guard.use()


class TensorBase:
	r"""a wrapper to basic tensor type for all neural network framework like: numpy and cupy in chainer, tensor in pytorch
	"""
	def __init__(self, device=0):
		self.device = device
		# self.op = operation(device=self.device)
		self.guard = device_guard(device=self.device)
		self.guard.use()
		#TODO: use decorator for the guard
		#use the selected device as default device

		self.data = None

	def guard(self):
		return self.guard

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
			with device_guard(device):
				self.data = cp.asarray(self.data)
		elif device==-1 and self.device>-1:
			self.data = self.data.get()
		elif device==-1 and self.device==-1:
			pass
		else:
			raise ValueError("current device is {}, while target device is {}".format(self.device, device))

		self.device = device 
		self.op = operation(device=self.device)

		self.guard = device_guard(device=self.device)

	@property
	def shape(self):
		return self.data.shape

	@property
	def ndim(self):
		return self.data.ndim

	@property
	def dtype(self):
		return self.data.dtype

	@property
	def size(self):
		return self.data.size

	def __repr__(self):
		return self.ndarray.__str__()+" on device {}".format(self.device)
	def __str__(self):
		return self.__repr__()

	def cast(self, dtype):
		self.data = self.ndarray.astype(dtype)

	def astype(self, dtype):
		return tensor(self.ndarray.astype(_type), device=self.device)

	def reshape(self, *x):
		return tensor(self.ndarray.reshape(*x), device=self.device)

	def sum(self, *x, **kwarg):
		return tensor(self.ndarray.sum(*x, **kwarg), device=self.device)

	def max(self, *x, **kwarg):
		return tensor(self.ndarray.max(*x, **kwarg), device=self.device)

	def transpose(self, *x):
		return tensor(self.ndarray.transpose(*x), device=self.device)

	def __getitem__(self, item):
		return tensor(self.ndarray[item], device=self.device)

	def __len__(self):
		return len(self.ndarray)

	def __copy__(self):
		result = tensor(self.ndarray.copy(), device=self.device)
		return result

	def __deepcopy__(self, memo):
		result = tensor(self.ndarray.copy(), device=self.device)
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
        x (numpy.ndarray or cupy.ndarray): Initial data array.
		device (int): device index for gpu and cpu(-1)
	"""
	def __init__(self, x, device=0):
		super().__init__(device=device)
		if not isinstance(x, (np.ndarray, cp.ndarray, list)):
			raise TypeError("input x is neither numpy ndarray nor cupy ndarray, nor list, but {}".format(type(x)))
		if self.device>-1:
			self.data = cp.asarray(x)
		elif self.device==-1:
			if isinstance(x, np.ndarray):
				self.data = x
			elif isinstance(x, cp.ndarray):
				self.data = x.get()
		else:
			raise ValueError("invalid device setting, current device is {}".format(self.device))

