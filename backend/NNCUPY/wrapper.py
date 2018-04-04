from . import Variable, np, cp, operation
import chainer



def wrapper(fun, device, *x, **kwargs):
	op = operation(device=device)
	_x = []
	for i in x:
		if(type(i)==Variable):
			_x.append(i.chainer)
		elif(type(i)==op.arrayType):
			_x.append(chainer.Variable(i))
		elif(type(i)==chainer.Variable):
			_x.append(i)
		else:
			raise ValueError("type is not Variable nor cp.core.ndarray nor chainer.Variable, but to be {}".format(type(i)))
	return fun(*_x, **kwargs)


