# import chainer
# import chainer.functions as F
# import chainer.links as L

from . import np, cp, Variable, operation

import re, copy

'''
For register_weights function, we want to register the weights in a hierarchy order
'''

import chainer.functions as F
# class F:
# 	pass

class node:
	def __init__(self, device=0):
		self.device = device
		self.op = operation(device=self.device)

		self.weights_dict = {}
		self.subgraphs_dict = {}

		self.type = self.__class__.__name__
		self.name = self.__class__.__name__

	def register_weights(self, name, ndarray):
		self.weights_dict[name] = Variable(ndarray, device=self.device)

	def __str__(self):
		return "model for device: {}, model name: {}".format(self.device, self.name)

	def get_weights(self, name):
		return self.weights_dict[name]


	def param_to_dict(self, device=-1):
		'''
		the parameters will collect in heirrarchy, resulting a dict_in_dict structure
		'''
		op = operation(device=device)
		_dict = {}
		if(len(self.weights_dict)>0):
			temp_dict = {}
			for key in self.weights_dict.keys():
				temp_dict[key] = op.array_to_device(self.weights_dict[key].ndarray())
			_dict['_local_'+self.name] = temp_dict

		if(len(self.subgraphs_dict)>0):
			for key in self.subgraphs_dict.keys():
				_dict[key] = self.subgraphs_dict[key].param_to_dict(device=device)

		# #_dict is a linked dict
		# re_params = copy.deepcopy(_dict)

		# 
		# def Var_2_Array(params):
		# 	for key in params.keys():
		# 		if(type(params[key])==dict):
		# 			Var_2_Array(params[key])
		# 		else:
		# 			if(type(params[key])==Variable):
		# 				params[key] = op.array_to_device(params[key].ndarray())

		# Var_2_Array(re_params)

		return _dict


	def param_from_dict(self, param_dict):
		'''
		loading the parameters in heirarchy order
		'''

		# def Array_2_Var(params):
		# 	for key in params.keys():
		# 		if(type(params[key])==dict):
		# 			Array_2_Var(params[key])
		# 		else:
		# 			if(type(params[key])==Variable):
		# 				params[key] = Variable(self.op.array_to_device(params[key]))

		# Array_2_Var(param_dict)

		for key in param_dict.keys():
			if bool(re.search("_local_", key)):
				# print([type(param_dict[key][x]) for x in param_dict[key].keys()])
				# self.weights_dict = param_dict[key]
				for _key in param_dict[key].keys():
					self.weights_dict[_key] = Variable(self.op.array_to_device(param_dict[key][_key]))
			else:
				if key in self.subgraphs_dict.keys():
					self.subgraphs_dict[key].param_from_dict(param_dict[key])
				else:
					raise ValueError("key not exist in subgrahs in model: {}".format(self.name))

		self.update_params()


	def update_params(self):
		with self.guard():
			for key in self.weights_dict.keys():
				val = self.weights_dict[key]
				# print(key, val.shape, self.name)
				# print(type(val))
				setattr(self, key, val)


	def grad_to_dict(self):
		raise NotImplementedError("for NNCUPY, no grad is generated")

	def grad_add_from_dict(self, grad_dict):
		raise NotImplementedError("for NNCUPY, no grad is generated")

	def forward(self, *args, **kwargs):
		pass

	def __call__(self, *args, **kwargs):
		if self.device!=-1:
			with self.gpu():
				args = tuple(map(lambda i:Variable(i, device=self.device), args))
				return self.forward(*args, **kwargs)
		else:

			args = tuple(map(lambda i:Variable(i, device=self.device), args))
			return self.forward(*args, **kwargs)


	def to_gpu(self, device):
		'''
		TODO: with problem
		'''
		self.device = device 
		self.op = operation(device=self.device)

	def guard(self):
		return cp.cuda.Device(self.device)

	def gpu(self):
		return cp.cuda.Device(self.device)


class Graph(node):
	def __init__(self, device=0, link=False):
		super().__init__(device=device)
		self.link = link
		

	# def register_graphs(self, nodes_info):
	# 	'''
	# 	after this operation, all corresponding subgraphs, parameters are connected by self.subgraphs_dict and self.weights_dict, hierarchily
	# 	'''
	# 	for key, name, data in nodes_info:
	# 		if(key=='graph'):
	# 			self.subgraphs_dict[name] = data 
	# 			data.name = name
	# 		elif(key=='weights'):
	# 			self.register_weights(name, data)

	def register_graphs(self, nodes_info):
		'''
		after this operation, all corresponding subgraphs, parameters are connected by self.subgraphs_dict and self.weights_dict, hierarchily
		'''
		with self.guard():
			for key, name, data in nodes_info:
				if(key=='graph'):
					data.name = name
					setattr(self, name, data)
					self.subgraphs_dict[name] = getattr(self, name) 
					# print(type(data))
					# print(self._children)
					# self.add_link(data)
					# self._children.add(data)
				elif(key=='weights'):
					self.register_weights(name, data)

	def update_graphs(self):
		with self.guard():
			for key in self.subgraphs_dict.keys():
				graph = self.subgraphs_dict[key]
				graph.update_params()
				graph.update_graphs()

	def param_from_dict(self, param_dict):
		super().param_from_dict(param_dict)
		# print("update_graphs "+self.name)
		self.update_graphs()





if __name__ == '__main__':
	sigmoid = Sigmoid(device=1)

	print(sigmoid)
	print(sigmoid.name)

	# with cp.cuda.Device(1):
	# 	x = cp.array(np.random.randn(2,4,10,4)).astype(cp.float32)

	# x = sigmoid(x)

	# linear = Linear(4, 10, device=1)
	# _x = linear(x)

