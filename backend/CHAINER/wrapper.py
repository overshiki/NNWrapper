from . import Variable, np, cp, operation, tensor
import chainer



def wrapper(fun, device, *x, **kwargs):
	op = operation(device=device)
	_x = []
	for i in x:
		if isinstance(i, tensor):
			if device != i.device:
				i.to_device(device)
			_x.append(chainer.Variable(i.ndarray))

		elif isinstance(i, (chainer.Variable, chainer.Parameter)):
			#TODO:
			# if device != i.device:
			# 	i = i.to_gpu(device)
			_x.append(i)
		elif isinstance(i, Variable):
			if device != i.device:
				i = i.to_device(device)
			_x.append(i.var)
		else:
			raise ValueError("input type is neither tensor, chainer.Variable, nor chainer.Parameter, but {}".format(type(i)))
	return fun(*_x, **kwargs)


