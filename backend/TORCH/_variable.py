from . import np, device_guard, tensor
import torch, copy


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

	def max(self, *x, **kwarg):
		raise NotImplementedError()

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



