# from NNWrapper.backend.CHAINER.nodes.linear import Linear
# from NNWrapper.backend.CHAINER import Graph, GraphList, np, cp, Variable, operation, device_guard, wrapper

from NNWrapper.backend.TORCH.nodes.linear import Linear
from NNWrapper.backend.TORCH import Graph, GraphList, np, cp, Variable, operation, device_guard, wrapper

import re
from timeit import default_timer as timer
from IO.basic import load_obj, save_obj, mkdir
from BASIC_LIST.basic import groupby

class model(node):
	def __init__(self, device=0):
		super().__init__(device=device)
		l = Linear(4, 10, device=0)
		self.add_node('l', l)
		# with self.gpu():
			# weigths = cp.asarray(np.random.randn(10,10,10))
		self.register_weights('weights', np.random.randn(10,10,10))

_model = model(device=0)


nodes = nodeList(*[_model, _model])
_str = str(nodes)
print(_str)