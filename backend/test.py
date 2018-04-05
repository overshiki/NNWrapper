

class tensor_trial:
	def __init__(self, x):
		self.data = x

	def __add__(self, other):
		return self.data+other

	def __radd__(self, other):
		return self.__add__(other)

	def __neg__(self):
		return -1*self.data

	def __sub__(self, other):
		return self.data-other

	def __rsub__(self, other):
		return self.__neg__().__add__(other)

A = tensor_trial(3)

print(A-4)

print(4-A)