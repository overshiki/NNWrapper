from . import np, device_guard, tensor
import torch


r"""it seems pytorch now deprecated Variable and use tensor to handle backward method instead
however, in our current wrapping implementation, we just use autograd.Variable 
TODO: check tensor autograd method in future document of pytorch
"""

class VarBase:
	def __init__(self, device=0):
		self.device = device
		self.var = None

	def new(self, *args, **kwargs):
		r"""Constructs a new variable of the same data type as :attr:`self` variable.
		"""
		return self.__class__(*args, **kwargs)

	@property
	def ndarray(self):
		r'''for safety consideration, only set getter for ndarray, which means no access to changing the value of self.var._data[0]
		'''
		return self.var._data[0]

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

	def transpose(self, *x):
		return self.new(self.var.reshape(*x), device=self.device)

	def sum(self, *x, **kwarg):
		raise NotImplementedError()

	def max(self, *x, **kwarg):
		raise NotImplementedError()

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

