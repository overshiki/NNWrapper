from . import Variable, tensor
import torch



def wrapper(fun, device, *x, **kwargs):
	_x = []
	for i in x:
		if isinstance(i, tensor):
			if device != i.device:
				i.to_device(device)
			_x.append(chainer.Variable(i.ndarray))

		elif isinstance(i, torch.autograd.Variable):
			if device != i.device:
				i = i.cuda(device)
			_x.append(i)
		elif isinstance(i, Variable):
			if device != i.device:
				i = i.to_device(device)
			_x.append(i.var)
		else:
			raise ValueError("input type is neither tensor, nor torch.autograd.Variable, but {}".format(type(i)))
	return fun(*_x, **kwargs)


