
from CHAINER.tensor import device_guard, cp, np

def device_guard_decorator(fun):
	def guard_fun(self, x):
		with self.guard:
			return fun(self, x)
	return guard_fun


class tensor_trial:
	def __init__(self, x, device=0):
		self.data = x
		self.device = device
		self.guard = device_guard(device=self.device)

	# def device_guard_decorator(self, fun):
	# 	with self.guard:
	# 		return fun

	@device_guard_decorator
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

tensor_trial.device_guard_decorator = device_guard_decorator


with device_guard(1):
	data = cp.asarray(np.random.randn(10, 10))
	other = cp.asarray(np.random.randn(10, 10))

# A = tensor_trial(data, device=1)

# print(A+other)
with device_guard(0):
	A = data.get()
	# B = data.astype('float32')
	C = data.reshape((100,1))
	# D = data.sum()
	# E = data.max()
	F = data.transpose()
	E = data[:5,:5]
	G = data.copy()
	# H = data+other

from CHAINER.tensor import tensor
from CHAINER.variable import Variable


A = Variable(tensor(np.random.randn(10,10), device=1))
B = Variable(tensor(np.random.randn(10,10), device=1))
A+B

with device_guard(0):
	A+B
	A-B
	A*B
# def p_decorate(func):
#    def func_wrapper(name):
#        return "<p>{0}</p>".format(func(name))
#    return func_wrapper

# @p_decorate
# def get_text(name):
#    return "lorem ipsum, {0} dolor sit amet".format(name)

# print(get_text("John"))

