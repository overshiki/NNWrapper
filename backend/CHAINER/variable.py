from . import np, cp, device_guard
import chainer
import chainer.functions as F

class VarBase:
	def __init__(self, device=0):
		self.device = device
		self.guard = device_guard(device=self.device)
		self.guard.use()
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
		return self.var._data[0]

	def numpy(self):
		if self.device==-1:
			return self.ndarray
		else:
			return self.ndarray.get()

	@property
	def shape(self):
		return self.var._data[0].shape

	@property
	def ndim(self):
		return self.var._data[0].ndim

	@property
	def dtype(self):
		return self.var._data[0].dtype

	def to_device(self, device):
		self.device = device 
		self.guard = device_guard(device=self.device)
		self.guard.use()
		self.var.to_gpu(self.device)


	def cast(self, dtype):
		self.var = chainer.functions.cast(self.var, dtype)

	def __repr__(self):
		return self.ndarray.__str__()
	def __str__(self):
		return self.ndarray.__str__()

	def backward(self):
		self.var.backward()
		# self._update_grad()

	def reshape(self, *x):
		return self.new(self.var.reshape(*x), device=self.device)

	def transpose(self, *x):
		return self.new(self.var.transpose(*x), device=self.device)

	def sum(self, **kwarg):
		return self.new(F.sum(self.var, **kwarg), device=self.device)

	def max(self, **kwarg):
		return self.new(F.max(self.var, **kwarg), device=self.device)

	def __getitem__(self, item):
		return self.new(self.var[item], device=self.device)

	def __len__(self):
		return len(self.var)

	def __copy__(self):
		result = self.new(self.ndarray.copy(), device=self.device)
		return result

	def __deepcopy__(self, memo):
		result = self.new(self.ndarray.copy(), device=self.device)
		memo[id(self)] = result
		return result

	def astype(self, _type):
		with self.guard:
			return self.new(self.ndarray.astype(_type), device=self.device)





