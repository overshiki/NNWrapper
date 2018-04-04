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
		#use the selected device as default device

		self.data = None

	def guard(self):
		return self.guard

	def ndarray(self):
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


	def shape(self):
		return self.data.shape

	def ndim(self):
		return self.data.ndim

	def dtype(self):
		return self.data.dtype

	def __repr__(self):
		return self.ndarray().__str__()+" on device {}".format(self.device)
	def __str__(self):
		return self.__repr__()

	def cast(self, dtype):
		self.data = self.ndarray().astype(dtype)

	def astype(self, dtype):
		return tensor(self.ndarray().astype(_type), device=self.device)

	def reshape(self, *x):
		return tensor(self.ndarray().reshape(*x), device=self.device)

	def sum(self, *x, **kwarg):
		return tensor(self.ndarray().sum(*x, **kwarg), device=self.device)

	def max(self, *x, **kwarg):
		return tensor(self.ndarray().max(*x, **kwarg), device=self.device)

	def transpose(self, *x):
		return tensor(self.ndarray().transpose(*x), device=self.device)

	def __getitem__(self, item):
		return tensor(self.ndarray()[item], device=self.device)

	def __len__(self):
		return len(self.ndarray())

	def type_check(self, other):
		number_check = False
		if isinstance(other, tensor):
			var = other
		elif isinstance(other, (np.ndarray, cp.ndarray)):
			var = tensor(other, device=self.device)
		elif isinstance(other, (int, float, complex, bool)):
			var = other
			number_check = True
		else:
			raise ValueError("input other is not numpy cupy ndarray nor tensor nor python variable, but to be: {}".format(type(other)))

		if number_check==False:
			if var.device!=self.device:
				var.to_device(self.device)
			return var.ndarray()
		else:
			return var

	def mul(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() * var, device=self.device)

	def __mul__(self, other):
		return self.mul(other)

	def add(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() + var, device=self.device)		

	def __add__(self, other):
		return self.add(other)

	def sub(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() - var, device=self.device)		

	def __sub__(self, other):
		return self.sub(other)

	def truediv(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() / var, device=self.device)

	def __truediv__(self, other):
		return self.truediv(other)

	def floordiv(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() // var, device=self.device)	

	def __floordiv__(self, other):
		return self.floordiv(other)


	def __lt__(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() < var, device=self.device)

	
	def __le__(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() <= var, device=self.device)
	

	def __eq__(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() == var, device=self.device)


	def __ne__(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() != var, device=self.device)

	def __gt__(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() > var, device=self.device)

	def __ge__(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() >= var, device=self.device)

	def pow(self, other):
		var = self.type_check(other)
		return tensor(self.ndarray() ** var, device=self.device)

	def __pow__(self, other):
		return self.pow(other)

	def __copy__(self):
		result = tensor(self.ndarray().copy(), device=self.device)
		return result

	def __deepcopy__(self, memo):
		result = tensor(self.ndarray().copy(), device=self.device)
		memo[id(self)] = result
		return result



class tensor(TensorBase):
	r"""implementation of tensor data structure
	"""
	def __init__(self, x, device=0):
		super().__init__(device=device)
		if not isinstance(type(x), np.ndarray) and isinstance(type(x), cp.ndarray):
			raise ValueError("input x is neither numpy ndarray nor cupy ndarray, but {}".format(type(x)))
		if self.device>-1:
			self.data = cp.asarray(x)
		elif self.device==-1:
			if isinstance(x, np.ndarray):
				self.data = x
			elif isinstance(x, cp.ndarray):
				self.data = x.get()
		else:
			raise ValueError("invalid device setting, current device is {}".format(self.device))

