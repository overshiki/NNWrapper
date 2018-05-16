import numpy as np

import chainer

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

from .loss import *
from .optimize import *

from ..base import base

from ..operation import _OperationBase

from .tensor import device_guard, TensorBase

from .variable import VarBase

from .basic import Variable, Parameter, tensor

from .operation import TensorOperation, VariableOperation

from .pool import task_parallel

from .base import Graph, GraphList, F, L
from .wrapper import wrapper

from .nodes import *

from .math.tensor_math import install_variable_arithmetics as iva_tensor
from .math.variable_math import install_variable_arithmetics as iva_variable

iva_tensor()
iva_variable()
