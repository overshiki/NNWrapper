import torch

from . import np, Variable, operation

import re, copy

class node(torch.nn.Module):
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

	# def guard(self):
	# 	return self.init_scope()


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
		else:

			if(self.link==False):
				args = tuple(map(lambda i:Variable(i, device=self.device), args))
			else:
				args = tuple(map(lambda i:Variable(i, device=self.device).var, args))
			return self.forward(*args, **kwargs)


	def to_gpu(self, device):
		'''
		TODO: with problem
		'''
		self.device = device 
		self.op = operation(device=self.device)
		super().to_gpu(device)

	def gpu(self):
		return cp.cuda.Device(self.device)


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

		for name, param in self.namedparams():
			yield param


	def namedparams(self, memo=None, prefix='/'):

		#compared that in chainer, the pytorch version use memo set to manage the visited names, this may not be neccessary since parameters of the child module is always different from that from root modules
		#in chainer, self._params is a list contains the name of the parameters, while the paramters are in self.__dict__ through setattr, getattr method
		#while in pytorch, self._parameters is dict contains both name and paramters

		#similarly, in chainer, self._children is a list contains the name of the child modules, while the child modules are in self.__dict__ through setattr, getattr method
		#while in pytorch, self._modules is dict contains both name and child module

		#in our wrapper implementation here, we follow the pytorch version, while in our chainer wrapper we follow chainer version instead. This will make our wrapper fit the two framework better
		if memo is None:
			memo = set()
		for name, p in self._parameters.items():
			if p is not None and p not in memo:
				memo.add(p)
				yield prefix + '/' + name, p
		for mname, _node in self.named_children():
			submodule_prefix = prefix + '/' + mname

			if isinstance(_node, node):
				#if node is NNWrapper node
				for name, p in _node.namedparams(memo, submodule_prefix):
					yield name.replace(".", "/"), p #replace "." with "/" so that match our parameter name format
			elif isinstance(_node, torch.nn.Module):
				#if node is pytorch module
				for name, p in _node.named_parameters(memo, submodule_prefix):
					yield name.replace(".", "/"), p #replace "." with "/" so that match our parameter name format
			else:
				raise TypeError("node is neighter node type nor torch.nn.Module type")

		#this is a recursive approach, since for every named_childern loop, there is a namedparams call, which will resulting a recursive all of named_children
		#this means named_children should just return immediate childern nodes


	def named_children(self):
		#unique in pytorch version, return immediate childern nodes
		memo = set()
		for name, module in self._modules.items():
			if module is not None and module not in memo:
				memo.add(module)
				yield name, module 


	def namednodes(self, skipself=False):
		'''An iterator over all children modules of the network
		if the child type is node, then will be ok
		but if the child is with type link, will cause problem

		TODO: here we need a better way to wrap link into node
		'''
		#for pytorch, it seems Returns an iterator over immediate children modules
		if not skipself:
			yield '/', self
		memo = set()
		for name, module in self._modules.items():
			if module is not None and module not in memo:
				memo.add(module)
				yield name, module
				if isinstance(module, node):
					for path, submodule in module.namednodes():
						yield path, submodule

				elif isinstance(module, torch.nn.Module):
					for path, submodule in module.named_modules(memo=memo, prefix='/'):
						yield path, submodule				



	def register_weights(self, name, ndarray):

		if isinstance(ndarray, tensor):
			if ndarray.device!=self.device:
				ndarray = ndarray.to_device(self.device)
			param = Parameter(ndarray, device=self.device).var
		elif isinstance(ndarray, torch.nn.Parameter):
			param = ndarray
		else:
			raise ValueError("type of input parameters is neither tensor nor torch.nn.Parameter, but {}".format(type(ndarray)))

		self.register_parameter(name, param) #inherited from torch.nn.Module
		# if '_parameters' not in self.__dict__:
		# 	raise AttributeError(
		# 		"cannot assign parameter before Module.__init__() call")

		# if hasattr(self, name) and name not in self._parameters:
		# 	raise KeyError("attribute '{}' already exists".format(name))

		# if param is None:
		# 	self._parameters[name] = None
		# elif not isinstance(param, torch.nn.Parameter):
		# 	raise TypeError("cannot assign '{}' object to parameter '{}' "
		# 					"(torch.nn.Parameter or None required)"
		# 					.format(torch.typename(param), name))
		# elif param.grad_fn:
		# 	raise ValueError(
		# 		"Cannot assign non-leaf Variable to parameter '{0}'. Model "
		# 		"parameters must be created explicitly. To express '{0}' "
		# 		"as a function of another variable, compute the value in "
		# 		"the forward() method.".format(name))
		# else:
		# 	self._parameters[name] = param




	def add_node(self, name, _node):
		'''
		for pytorch, module are registrated in self._modules and can be accessed through __getattr__
		'''
		if not isinstance(_node, torch.nn.Module) and not isinstance(_node, node) and module is not None:
			raise TypeError('cannot register a non-Node object as a child')
		if hasattr(self, name) and name not in self._modules:
			raise AttributeError(
				'cannot register a new node %s: attribute exists' % name)
		self._modules[name] = _node


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

