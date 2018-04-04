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


from .basic import Variable, operation, device_guard

from .pool import task_parallel

from .base import Graph, node, F
from .wrapper import wrapper

from .nodes.activation import Sigmoid
from .nodes.linear import Linear