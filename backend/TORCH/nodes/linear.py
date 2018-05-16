from .. import np, Variable, device_guard, Graph
import torch


class Linear(Graph):
	'''
	TODO: consider a better to wrap a link into node

	we considering not directly registrate any weights, but just treat self.layer as child. we then handle this link type child in base
	'''
	def __init__(self, n_in, n_hidden, bias=True, device=0):
		'''
		linear = Ax+b
		'''
		super().__init__(device=device)
		self.link = True #input for forward is Variable type

		self.n_in, self.n_hidden = n_in, n_hidden

		# with self.guard():
			# self.layer = L.Linear(self.n_in, self.n_hidden)
			# if(self.device!=-1):
			# 	self.layer.to_gpu(device=self.device)


		layer = torch.nn.Linear(self.n_in, self.n_hidden, bias=bias)
		if(self.device!=-1):
			layer.cuda(device=self.device)
		self.add_node('layer', layer)


	def forward(self, x):
		x = x.float()
		shape = list(x.shape)
		shape.pop(-1)
		x = x.view(-1, self.n_in)
		x = self.layer(x)
		if(self.n_hidden==1):
			x = x.view(*shape)
		else:
			shape.append(self.n_hidden)
			x = x.view(*shape)
		return Variable(x, device=self.device)
