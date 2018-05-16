from .variable import VarBase
from .tensor import TensorBase
from . import np, cp, chainer

class Variable(VarBase):
	r'''variable wrapper for all kinds of neural network framework, like in chainer, it wraps chainer.Variable object, in pytorch, it wraps torch.autograd.Variable object
	for simplicty and transparency, only tensor data type or chainer.Variable is accepted as input
	'''
	def __init__(self, x, device=0):
		super().__init__(device=device)

		if isinstance(x, tensor):
			if self.device != x.device:
				x.to_device(self.device)
			self.var = chainer.Variable(x.ndarray)

		elif isinstance(x, (chainer.Variable, chainer.Parameter)):
			#TODO:
			# if self.device != x.device:
			# 	x = x.to_gpu(self.device)
			self.var = x
		elif isinstance(x, VarBase):
			if self.device != x.device:
				x = x.to_device(self.device)
			self.var = x.var
		elif isinstance(x, np.ndarray):
			x = tensor(x, device=self.device)
			self.__init__(x, device=device)
		else:
			raise ValueError("input type is neither tensor, chainer.Variable, nor chainer.Parameter, but {}".format(type(x)))
		self.varType = chainer.Variable


class Parameter(VarBase):
	r'''parameter wrapper for all kinds of neural network framework, like in chainer, it wraps chainer.Parameter object, in pytorch, it wraps torch.autograd.Parameter object
	for simplicty and transparency, only tensor data type or chainer.Parameter is accepted as input
	'''
	def __init__(self, x, device=0):
		super().__init__(device=device)

		if isinstance(x, tensor):
			if self.device != x.device:
				x.to_device(self.device)
			self.var = chainer.Parameter(x.ndarray)

		elif isinstance(x, chainer.Parameter):
			self.var = x
		else:
			raise ValueError("input type is neither tensor, nor chainer.Parameter, but {}".format(type(x)))
		self.varType = chainer.Parameter

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
		with self.guard:
			if not isinstance(x, (np.ndarray, cp.ndarray, list, Variable)):
				raise TypeError("input x is neither numpy ndarray nor cupy ndarray, nor list, but {}".format(type(x)))

			if isinstance(x, Variable):
				self.data = x.ndarray
			else:
				if self.device>-1:
					self.data = cp.asarray(x)
				elif self.device==-1:
					if isinstance(x, np.ndarray):
						self.data = x
					elif isinstance(x, cp.ndarray):
						self.data = x.get()
				else:
					raise ValueError("invalid device setting, current device is {}".format(self.device))