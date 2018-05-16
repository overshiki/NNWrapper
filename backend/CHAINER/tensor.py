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

def device_guard_decorator(fun):
	def guard_fun(self, *x, **kwargs):
		with self.guard:
			return fun(self, *x, **kwargs)
	return guard_fun
	#TODO: replace 'with' closure with decorator totally, since it is more compacitble with code for other backend


class TensorBase:
	r"""a wrapper to basic tensor type for all neural network framework like: numpy and cupy in chainer, tensor in pytorch
	"""
	def __init__(self, device=0):
		self.device = device
		# self.op = operation(device=self.device)
		self.guard = device_guard(device=self.device)
		# self.guard.use()
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
		# self.op = operation(device=self.device)

		self.guard = device_guard(device=self.device)

	@property
	def shape(self):
		return self.ndarray.shape

	@property
	def ndim(self):
		return self.ndarray.ndim

	@property
	def dtype(self):
		return self.ndarray.dtype

	@property
	def size(self):
		return self.ndarray.size

	def __repr__(self):
		return self.ndarray.__str__()+" on device {}".format(self.device)
	def __str__(self):
		return self.__repr__()

	@device_guard_decorator
	def cast(self, dtype):
		self.data = self.data.astype(dtype)

	@device_guard_decorator
	def astype(self, dtype):
		return self.new(self.ndarray.astype(_type), device=self.device)

	def reshape(self, *x):
		return self.new(self.ndarray.reshape(*x), device=self.device)

	@device_guard_decorator
	def sum(self, *x, **kwarg):
		return self.new(self.ndarray.sum(*x, **kwarg), device=self.device)

	@device_guard_decorator
	def max(self, *x, **kwarg):
		return self.new(self.ndarray.max(*x, **kwarg), device=self.device)

	def transpose(self, *x):
		return self.new(self.ndarray.transpose(*x), device=self.device)

	def __getitem__(self, item):
		return self.new(self.ndarray[item], device=self.device)

	def __len__(self):
		return len(self.ndarray)

	def __copy__(self):
		result = self.new(self.ndarray.copy(), device=self.device)
		return result

	def __deepcopy__(self, memo):
		result = self.new(self.ndarray.copy(), device=self.device)
		memo[id(self)] = result
		return result




TensorBase.device_guard_decorator = device_guard_decorator