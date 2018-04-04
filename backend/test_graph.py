from NNWrapper.backend.CHAINER.nodes.linear import Linear
from NNWrapper.backend.CHAINER import Graph, GraphList, np, cp, Variable, operation, device_guard, wrapper

import chainer, re
from timeit import default_timer as timer
from IO.basic import load_obj, save_obj, mkdir
from BASIC_LIST.basic import groupby

# class model(node):
# 	def __init__(self, device=0):
# 		super().__init__(device=device)
# 		l = Linear(4, 10, device=0)
# 		self.add_node('l', l)
# 		# with self.gpu():
# 			# weigths = cp.asarray(np.random.randn(10,10,10))
# 		self.register_weights('weights', np.random.randn(10,10,10))

# _model = model(device=0)


# nodes = nodeList(*[_model, _model])
# _str = str(nodes)
# print(_str)


from NNWrapper.zoo.vinn.model.vinn import model


design5 = {}
design5['size_min'], design5['size_max'], design5['size_step'] = 1, 100, 10
design5['stride_min'], design5['stride_max'], design5['stride_step'] = 5, 20, 5
design5['dilation_min'], design5['dilation_max'], design5['dilation_step'] = 1, 3, 1


# _model = model(design=design5, device=0)

from chainer.backends import cuda
class Lasso(object):
    """Optimizer/UpdateRule hook function for Lasso regularization.
    This hook function adds a scaled parameter to the sign of each weight.
    It can be used as a regularization.
    Args:
        rate (float): Coefficient for the weight decay.
    Attributes:
        rate (float): Coefficient for the weight decay.
    """
    name = 'Lasso'
    call_for_each_param = True

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        p, g = param.array, param.grad_var
        if p is None or g is None:
            return
        xp = cuda.get_array_module(p)
        with cuda.get_device_from_array(p) as dev:
            sign = xp.sign(p)
            # if int(dev) == -1:
            g -= self.rate * sign


def train(X, Y, minibatch=1000, num_epoch=1000, patch_size=31, checkPath=None, savePath="./save/mix/feed/", device=1):
	op = operation(device=device)

	loss = chainer.functions.softmax_cross_entropy
	optimizer = chainer.optimizers.Adam(alpha=0.0002)

	embedding_dim = patch_size//2 + 1 
	# _model = model_fun(design=design, device=device, embedding_dim=embedding_dim)
	_model = model(design=design5, device=0, embedding_dim=embedding_dim)

	if(checkPath is not None):
		param_dict = load_obj(checkPath)
		with device_guard(device):
			for key, param in _model.namedparams():
				param.copydata(chainer.Parameter(cp.asarray(param_dict[key])))	

	optimizer.setup(_model)



	with device_guard(device):

		for epoch in range(num_epoch):

			if(epoch%10==0):
				check = epoch/10
				if(check%2==0):
					print("L1, and only for pooling")
					for key, l in _model.namednodes():
						if bool(re.search("linear", key)):
							l.disable_update()
						else:
							l.enable_update()

					optimizer.add_hook(Lasso(rate=0.001), name='lasso')
				else:
					print("No regularization, and only for linear")
					for key, l in _model.namednodes():
						if bool(re.search("pool", key)):
							l.disable_update()
						else:
							l.enable_update()

					optimizer.remove_hook('lasso')

			start_time = timer()
			num = 0
			count = 0 
			correct = 0
			l = 0

			indices = np.arange(0, len(X), minibatch)
			for start in indices:
				if start+minibatch<len(X):
					end = start+minibatch
				else:
					end = len(X)

				x, y = X[start:end], Y[start:end]
				x, y = op.array(x), op.array(y)

				x.cast(op.run.float32)

				x = _model(x)

				L = wrapper(loss, device, x, y)

				_model.cleargrads()

				L.backward()

				optimizer.update()

				# print(type(x), type(y))
				correct = (op.argmax(x, axis=1)==y).sum() + correct
				l = Variable(L, device=device) + l
				num = num + y.shape[0]
				count = count + 1

			print("epoch: ", epoch, "num ", num, "correct: ", correct/num, "loss: ", l/count, "time: ", timer()-start_time)

			param_dict = {}
			for key, params in _model.namedparams():
				param_dict[key] = params.array.get()

			if(savePath is not None):
				mkdir(savePath)
				save_obj(param_dict, savePath+"/{}.pkl".format(epoch))



def debug(X, Y, minibatch=1000, patch_size=31, checkPath=None, device=1):
	op = operation(device=device)

	loss = chainer.functions.softmax_cross_entropy
	optimizer = chainer.optimizers.Adam(alpha=0.0002)

	embedding_dim = patch_size//2 + 1 
	# _model = model_fun(design=design, device=device, embedding_dim=embedding_dim)
	_model = model(design=design5, device=0, embedding_dim=embedding_dim)

	if(checkPath is not None):
		param_dict = load_obj(checkPath)
		with device_guard(device):
			for key, param in _model.namedparams():
				param.copydata(chainer.Parameter(cp.asarray(param_dict[key])))	

	optimizer.setup(_model)



	with device_guard(device):

		indices = np.arange(0, len(X), minibatch)
		for start in indices:
			if start+minibatch<len(X):
				end = start+minibatch
			else:
				end = len(X)

			x, y = X[start:end], Y[start:end]
			x, y = op.array(x), op.array(y)

			x.cast(op.run.float32)

			x = _model(x)

			print(x)





X, Y = np.load("/home/polaris/CODE/Polaris_v5/NN/NNWrapper/zoo/VINN/save/data/X_test.npy"), np.load("/home/polaris/CODE/Polaris_v5/NN/NNWrapper/zoo/VINN/save/data/Y_test.npy")
X, Y = X.astype(np.float32), Y.astype(np.int32)
X, Y = X[:20], Y[:20]
print(X.shape, Y.shape)

train(X, Y, minibatch=5, num_epoch=100, patch_size=31, checkPath=None, savePath="./save/", device=0)

# debug(X, Y, minibatch=5, patch_size=31, checkPath=None, device=0)