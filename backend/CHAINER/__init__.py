import numpy as np
class empty:
	def __init__(self, ):
		self.core = self
		self.ndarray = self

try:
	import cupy as cp
except:
	print("no cupy avaliable on device")
	cp = empty()
	pass



from .tensor import device_guard, tensor, TensorBase

from .variable import Variable, Parameter, VarBase

from .operation import operation

from .pool import task_parallel

from .base import Graph, GraphList, F, L
from .wrapper import wrapper

from .nodes import *

from .math.tensor_math import install_variable_arithmetics as iva_tensor
from .math.variable_math import install_variable_arithmetics as iva_variable

iva_tensor()
iva_variable()
