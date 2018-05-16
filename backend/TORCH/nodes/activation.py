from .. import Graph, Variable

class Sigmoid(Graph):
	def __init__(self, device=0):
		'''
		sigmoid = 1./(1+exp(-x))
		'''
		super().__init__(device=device)
		self.link = True

	def forward(self, x):
		x = F.sigmoid(x)
		return Variable(x)

class Tanh(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class LogSigmoid(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class Softplus(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class Hardtanh(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class Threshold(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class LeakyReLU(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class PReLU(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class SELU(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class ELU(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class ReLU6(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class ReLU(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class Softshrink(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class Softsign(Graph):
	# raise NotImplementedError("not implemented currently")
	pass

class Tanhshrink(Graph):
	# raise NotImplementedError("not implemented currently")
	pass
