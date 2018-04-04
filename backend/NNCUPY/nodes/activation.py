from .. import node, np, Variable

class Sigmoid(node):
	def __init__(self, device=0):
		'''
		sigmoid = 1./(1+exp(-x))
		'''
		super().__init__(device=device)

	def forward(self, x):
		x = (self.op.exp(x*(-1.)) + 1.) ** -1
		return Variable(x)

class Tanh(node):
	# raise NotImplementedError("not implemented currently")
	pass

class LogSigmoid(node):
	# raise NotImplementedError("not implemented currently")
	pass

class Softplus(node):
	# raise NotImplementedError("not implemented currently")
	pass

class Hardtanh(node):
	# raise NotImplementedError("not implemented currently")
	pass

class Threshold(node):
	# raise NotImplementedError("not implemented currently")
	pass

class LeakyReLU(node):
	# raise NotImplementedError("not implemented currently")
	pass

class PReLU(node):
	# raise NotImplementedError("not implemented currently")
	pass

class SELU(node):
	# raise NotImplementedError("not implemented currently")
	pass

class ELU(node):
	# raise NotImplementedError("not implemented currently")
	pass

class ReLU6(node):
	# raise NotImplementedError("not implemented currently")
	pass

class ReLU(node):
	# raise NotImplementedError("not implemented currently")
	pass

class Softshrink(node):
	# raise NotImplementedError("not implemented currently")
	pass

class Softsign(node):
	# raise NotImplementedError("not implemented currently")
	pass

class Tanhshrink(node):
	# raise NotImplementedError("not implemented currently")
	pass
