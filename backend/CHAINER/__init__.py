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



from .tensor import device_guard, tensor

from .basic import Variable, Parameter

from .operation import operation

from .pool import task_parallel

from .base import Graph, GraphList, F, L
from .wrapper import wrapper

from .nodes import *

