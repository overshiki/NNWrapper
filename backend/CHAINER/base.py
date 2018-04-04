import chainer
import chainer.functions as F
import chainer.links as L

from . import np, cp, Variable, Parameter, operation, device_guard

import re, copy

class node(chainer.Chain):
	def __init__(self,):
		super().__init__()

class Graph(node):
	def __init__(self, device=0, link=False):
		super().__init__()
		# self._children = set()

		self.device = device
		self.link = link
		self.op = operation(device=self.device)

		# self.weights_key = set()
		# self.subgraphs_key = set()

		# self.type = self.__class__.__name__
		self.name = self.__class__.__name__

	def __str__(self):
		return "model for device: {}, model name: {}".format(self.device, self.name)

	def guard(self):
		return self.init_scope()


	def forward(self, *args, **kwargs):
		pass

	def __call__(self, *args, **kwargs):
		if self.device!=-1:
			with self.gpu():
				if(self.link==False):
					args = tuple(map(lambda i:Variable(i, device=self.device), args))
				else:
					args = tuple(map(lambda i:Variable(i, device=self.device).var, args))
				return self.forward(*args, **kwargs)
		# else:

		# 	if(self.link==False):
		# 		args = tuple(map(lambda i:Variable(i, device=self.device), args))
		# 	else:
		# 		args = tuple(map(lambda i:Variable(i, device=self.device).var, args))
		# 	return self.forward(*args, **kwargs)


	def to_gpu(self, device):
		'''
		TODO: with problem
		'''
		self.device = device 
		self.op = operation(device=self.device)
		super().to_gpu(device)

	def gpu(self):
		return cp.cuda.Device(self.device)

	def register_weights(self, name, ndarray):
		with self.init_scope():
			if(type(ndarray)==self.op.arrayType):
				var = Parameter(ndarray, device=self.device).var
			elif(type(ndarray)==chainer.Parameter):
				var = ndarray
			elif(type(ndarray)==self.op.counterType):
				var = Parameter(ndarray, device=self.device).var
			else:
				raise ValueError("type of input parameters is neither {} nor {}, but {}".format(self.op.arrayType, "chainer.Parameter", type(ndarray)))
				
			#in chainer's __setattr__, name is registrated into self._childern
			setattr(self, name, var)

	def get_grad_dict(self, cpu=True):
		grad_dict = {}
		for key, params in self.namedparams():
			if cpu==True:
				grad_dict[key] = params.grad.get()
			else:
				grad_dict[key] = params.grad
		return grad_dict


	def params(self, include_uninit=True):
		# for param in super().params(include_uninit):
		# 	yield param

		d = self.__dict__
		for name in self._params:
			if include_uninit or d[name].data is not None:
				yield d[name]

		d = self.__dict__
		for name in self._children:
			for param in d[name].params(include_uninit):
				yield param


	def namedparams(self, include_uninit=True):
		# for ret in super().namedparams(include_uninit):
		# 	yield ret
		d = self.__dict__
		for name in self._params:
			if include_uninit or d[name].data is not None:
				yield '/' + name, d[name]

		for name in self._children:
			prefix = '/' + name
			for path, param in d[name].namedparams(include_uninit):
				yield prefix + path, param

	def namednodes(self, skipself=False):
		'''
		if the child type is node, then will be ok
		but if the child is with type link, will cause problem

		TODO: here we need a better way to wrap link into node
		'''
		if not skipself:
			yield '/', self
		d = self.__dict__
		for name in self._children:
			child = d[name]
			prefix = '/' + name
			yield prefix, child
			if issubclass(type(d[name]), node):
				for path, _node in d[name].namednodes(True):
					yield prefix + path, _node

			elif isinstance(d[name], chainer.Link):
				#type guard here, if it is leaf node, then it may be chainer link type
				for path, link in d[name].namedlinks(True):
					yield prefix + path, link

	def add_node(self, name, _node):
		if name in self.__dict__:
			raise AttributeError(
				'cannot register a new link %s: attribute exists' % name)
		if (not isinstance(_node, node)) and (not isinstance(_node, chainer.Link)):
			raise TypeError('cannot register a non-Node object as a child')
		with self.init_scope():
			#in chainer's __setattr__, name is registrated into self._childern
			setattr(self, name, _node)


class GraphList(Graph):
	def __init__(self, *nodes):
		super().__init__()
		self._children = []

		for node in nodes:
			self.add_node(node)

	def __setattr__(self, name, value):
		if self.within_init_scope and (isinstance(value, chainer.Link) or (isinstance(value, node))):
			raise TypeError(
				'cannot register a new node'
				' within a "with chainlist.init_scope():" block.')
		super().__setattr__(name, value)

	def __getitem__(self, index):
		"""Returns the child at given index.
		Args:
			index (int): Index of the child in the list.
		Returns:
			Link: The ``index``-th child link.
		"""
		return self._children[index]

	def __iter__(self):
		return iter(self._children)

	def __len__(self):
		"""Returns the number of children."""
		return len(self._children)

	def append(self, node):
		"""Registers a child link and adds it to the tail of the list.
		This is equivalent to :meth:`add_link`. This method has been added to
		emulate the ``list`` interface.
		Args:
			link (Link): The link object to be regsitered.
		"""
		self.add_node(node)

	def add_node(self, node):
		"""Registers a child link and adds it to the tail of the list.
		Args:
			link (Link): The link object to be registered.
		"""
		node.name = str(len(self._children))
		self._children.append(node)

	# def copy(self):
	#     ret = super(ChainList, self).copy()
	#     ret._children = list(ret._children)  # copy
	#     children = ret._children
	#     for i, child in enumerate(children):
	#         child = child.copy()
	#         child.name = str(i)
	#         children[i] = child
	#     return ret

	# def to_cpu(self):
	#     super(ChainList, self).to_cpu()
	#     for link in self._children:
	#         link.to_cpu()
	#     return self

	# def to_gpu(self, device=None):
	#     with cuda._get_device(device):
	#         super(ChainList, self).to_gpu()
	#         for link in self._children:
	#             link.to_gpu()
	#     return self

	# def to_intel64(self):
	#     super(ChainList, self).to_intel64()
	#     for link in self._children:
	#         link.to_intel64()
	#     return self

	def params(self, include_uninit=True):
		# for param in super(ChainList, self).params(include_uninit):
		#     yield param
		for link in self._children:
			for param in link.params(include_uninit):
				yield param

	def namedparams(self, include_uninit=True):
		# for ret in super().namedparams(include_uninit):
		# 	yield ret
		d = self.__dict__
		for name in self._params:
			if include_uninit or d[name].data is not None:
				yield '/' + name, d[name]

		for idx, node in enumerate(self._children):
			prefix = '/%d' % idx
			for path, param in node.namedparams(include_uninit):
				yield prefix + path, param

	def namednodes(self, skipself=False):
		if not skipself:
			yield '/', self
		for idx, child in enumerate(self._children):
			prefix = '/%d' % idx
			yield prefix, child
			for path, link in child.namednodes(True):
				yield prefix + path, link


# class Graph(node):
# 	def __init__(self, device=0, link=False):
# 		super().__init__(device=device)


# 	def register_graphs(self, nodes_info):
# 		'''
# 		after this operation, all corresponding subgraphs, parameters are connected by self.subgraphs_dict and self.weights_dict, hierarchily
# 		'''
# 		with self.guard():
# 			for key, name, data in nodes_info:
# 				if(key=='graph'):
# 					data.name = name
# 					setattr(self, name, data)
# 					# self.subgraphs_dict[name] = getattr(self, name) 
# 					self.subgraphs_key.add(name)
# 					# print(type(data))
# 					# print(self._children)
# 					# self.add_link(data)
# 					# self._children.add(data)
# 				elif(key=='weights'):
# 					self.register_weights(name, data)





if __name__ == '__main__':
	sigmoid = Sigmoid(device=1)

	with cp.cuda.Device(1):
		x = cp.array(np.random.randn(2,4,10,4)).astype(cp.float32)

	# print(type(x))
	x = sigmoid(x)
	# print(type(x))

	linear = Linear(4, 10, device=1)
	_x = linear(x)
	# print(_x)

