# from NNWrapper.backend.CHAINER.nodes.linear import Linear
# from NNWrapper.backend.CHAINER import Graph, GraphList, np, cp, Variable, operation, device_guard, wrapper

from NNWrapper.backend.TORCH.nodes.linear import Linear
from NNWrapper.backend.TORCH import Graph, GraphList, np, Variable, device_guard, tensor, TensorOperation, VariableOperation

import re
from timeit import default_timer as timer
from IO.basic import load_obj, save_obj, mkdir
from BASIC_LIST.basic import groupby

class model(Graph):
	def __init__(self, device=0):
		super().__init__(device=device)
		l = Linear(4, 10, device=0)
		self.add_node('l', l)
		self.register_weights('weights', np.random.randn(10,10,10))
		self.link = True

	def forward(self, x):
		x = self.l(x)
		return x

_model = model(device=0)


print(_model)


# x = tensor(np.random.randn(10, 4), device=0)
# x = _model(x)
# print(x.shape, type(x))

# op = TensorOperation(device=0)

# op = VariableOperation(device=0)

# y = op.concatenate([x,x], dim=0)
# print(y.shape, type(y))

for name, params in _model.namedparamsOut():
	print(name, type(params))

param_dict = _model.params_to_dict()
for key, param in param_dict.items():
	print(key, type(param))

_model.save_params("./params.pkl")

_model.params_from_dict("./params.pkl")
# for name, node in _model.namednodes():
# 	print(name, type(node))

# nodes = nodeList(*[_model, _model])
# _str = str(nodes)
# print(_str)

print("#"*100)

from NNWrapper.backend.CHAINER.nodes.linear import Linear
from NNWrapper.backend.CHAINER import Graph, GraphList, np, Variable, device_guard, tensor, TensorOperation, VariableOperation

class model(Graph):
	def __init__(self, device=0):
		super().__init__(device=device)
		l = Linear(4, 10, device=0)
		self.add_node('l', l)
		self.register_weights('weights', np.random.randn(10,10,10))
		self.link = True

	def forward(self, x):
		x = self.l(x)
		return x

_model = model(device=0)


print(_model)

for name, params in _model.namedparamsOut():
	print(name, type(params))

_model.save_params("./params.pkl")

_model.params_from_dict("./params.pkl")