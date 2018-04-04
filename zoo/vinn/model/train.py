
import os, re
from IO.basic import load_obj, save_obj, mkdir
from BASIC_LIST.basic import groupby


# from NN.NNWrapper.backend.NNCUPY import Variable, operation, node, Graph, np, cp, device_guard, wrapper

from . import Variable, operation, node, Graph, np, cp, device_guard, wrapper

# from .model import model
from .old_vinn import model
from .prepare import prepare, pertubation

import chainer
from timeit import default_timer as timer


class loader:
	def __init__(self, posi, nege, imgs, minigroup, channels_pooling_stride=5, patch_size=61, step=15, shuffle=True, device=0, dataLimit=15000, channels=4095):
		self.posi, self.nege, self.imgs, self.minigroup, self.shuffle = posi, nege, imgs, minigroup, shuffle
		self.step, self.patch_size = step, patch_size
		self.dataLimit = dataLimit
		self.channels = channels
		self.channels_pooling_stride = channels_pooling_stride

		INDEX = 0 #INDEX is meaningless here
		self.X_posi, self.Y_posi, self.POS_posi, self.IMG_posi = prepare(INDEX, self.imgs, self.posi, 1, channels_pooling_stride=self.channels_pooling_stride, device=device, patch_size=self.patch_size, key='return', step=step, dataLimit=self.dataLimit, channels=channels)
		self.X_nege, self.Y_nege, self.POS_nege, self.IMG_nege = prepare(INDEX, self.imgs, self.nege, 0, channels_pooling_stride=self.channels_pooling_stride, device=device, patch_size=self.patch_size, key='return', step=step, dataLimit=self.dataLimit, channels=channels)


		self.X, self.Y = [], []

		self.device = device
		self.op = operation(device=device)

		print("finished loading normal")

	def _shuffle(self):
		p_posi = np.random.permutation(len(self.X_posi)).tolist()
		p_nege = np.random.permutation(len(self.X_nege)).tolist()
		return p_posi, p_nege

	def get(self, epoch, pertube=True):
		#in additional to X, Y we get in __init__
		#next we need add pertubation training data using pertubation method 
		if(self.shuffle==True):
			p_posi, p_nege = self._shuffle()
		else:
			p_posi, p_nege = np.arange(len(self.X_posi)).tolist(), np.arange(len(self.X_nege)).tolist()

		groups_posi = groupby(p_posi, self.minigroup, key='mini')
		groups_nege = groupby(p_nege, self.minigroup, key='mini')

		# if(epoch==0):
		# 	if(pertube==True):
		# 		print("minibatch is :{}(step) * {}(minigroup) * {}(num) = {} with 1 pertubation, where step should be smaller than 15, total should be smaller than 80".format(self.step, self.minigroup, 3, self.step*self.minigroup*3))
		# 	else:
		# 		print("minibatch is :{}(step) * {}(minigroup) * {}(num) = {} with no pertubation, where step should be smaller than 15, total should be smaller than 80".format(self.step, self.minigroup, 2, self.step*self.minigroup*2))

		for index, (g_posi, g_nege) in enumerate(zip(groups_posi, groups_nege)):

			x_posi = [self.X_posi[x] for x in g_posi]
			y_posi = [self.Y_posi[x] for x in g_posi]
			POS_posi = [self.POS_posi[x] for x in g_posi]
			img_posi = [self.IMG_posi[x] for x in g_posi]

			x_nege = [self.X_nege[x] for x in g_nege]
			y_nege = [self.Y_nege[x] for x in g_nege]
			POS_nege = [self.POS_nege[x] for x in g_nege]
			img_nege = [self.IMG_nege[x] for x in g_nege]



			x_posi, y_posi = self.op.concatenate([self.op.array(x) for x in x_posi]), self.op.concatenate([self.op.array(x) for x in y_posi])
			x_nege, y_nege = self.op.concatenate([self.op.array(x) for x in x_nege]), self.op.concatenate([self.op.array(x) for x in y_nege])


			if(pertube==True):
				pertub = [pertubation(x.shape[0])+x for x in POS_posi]

				# print("start pertubation")
				x_pertub, y_pertub, _, _ = prepare(0, img_posi, pertub, 0, channels_pooling_stride=self.channels_pooling_stride, device=self.device, patch_size=self.patch_size, key='return', step=self.step, cpu=False, dataLimit=self.dataLimit, channels=self.channels)

				# print("finished pertubation")

				if(len(x_pertub)>0):
					x_pertub, y_pertub = self.op.concatenate(x_pertub), self.op.concatenate(y_pertub)

					x, y = self.op.concatenate([x_posi, x_nege, x_pertub]), self.op.concatenate([y_posi, y_nege, y_pertub])
				else:
					x, y = self.op.concatenate([x_posi, x_nege]), self.op.concatenate([y_posi, y_nege])

			else:
				x, y = self.op.concatenate([x_posi, x_nege]), self.op.concatenate([y_posi, y_nege])

			yield x, y, index

	def refresh_data(self, epoch, pertube=True):
		self.X, self.Y = [], []
		for x, y, index in self.get(epoch):
			self.X.append(x)
			self.Y.append(y)
			# print("finished {}".format(index))

	def periodic_get(self, epoch):
		if epoch%10==0:
			self.refresh_data(epoch)
			# print("refresh_data")

		for index, (x, y) in enumerate(zip(self.X, self.Y)):
			yield x, y, index 

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



def train(model_fun, loader, device=0, num_epoch=100, patch_size=61, checkPath=None, pertubation=False, savePath=None):
	op = operation(device=device)

	loss = chainer.functions.softmax_cross_entropy
	optimizer = chainer.optimizers.Adam(alpha=0.0002)

	embedding_dim = patch_size//2 +1 
	_model = model_fun(device=device, embedding_dim=embedding_dim)

	if(checkPath is not None):
		param_dict = load_obj(checkPath)
		# _model.param_from_dict(param_dict)
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
					for key, l in _model.namedlinks():
						if bool(re.search("linear", key)):
							# print(key, "disable_update")
							l.disable_update()
						else:
							l.enable_update()

					# optimizer.add_hook(chainer.optimizer.Lasso(rate=0.005), name='lasso')
					# optimizer.add_hook(Lasso(rate=0.005), name='lasso')
					optimizer.add_hook(Lasso(rate=0.001), name='lasso')
				else:
					print("No regularization, and only for linear")
					for key, l in _model.namedlinks():
						if bool(re.search("pool", key)):
							# print(key, "disable_update")
							l.disable_update()
						else:
							l.enable_update()

					optimizer.remove_hook('lasso')
					# optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.005), name='ridge')				

			start_time = timer()
			num = 0
			count = 0 
			correct = 0
			l = 0
			if(pertubation==False):
				ld_fun = loader.get(epoch, pertube=False)
			else:
				ld_fun = loader.periodic_get(epoch) 

			for x, y, index in ld_fun:
				x, y = op.array(x), op.array(y)

				x.cast(op.run.float32)

				x = _model(x)

				L = wrapper(loss, device, x, y)

				_model.cleargrads()

				L.backward()

				optimizer.update()

				correct = (op.argmax(x, axis=1)==y).sum() + correct
				l = Variable(L, device=device) + l
				num = num + y.shape[0]
				count = count + 1

				# param_dict = _model.param_to_dict()
				# for key in param_dict.keys():
				# 	print(key)

				# for name, params in _model.namedparams():
				# 	if name=="/pooling_chain/0/weights":
				# 		print(name, params)
				# 		print(type(params._data[0].get()))

			print("epoch: ", epoch, "num ", num, "correct: ", correct/num, "loss: ", l/count, "time: ", timer()-start_time)

			# param_dict = _model.param_to_dict()
			param_dict = {}
			for key, params in _model.namedparams():
				# if name=="/pooling_chain/0/weights":
					# print(name, params)
					# print(type(params._data[0].get()))
				param_dict[key] = params.array.get()

			if(savePath is not None):
				mkdir(savePath)
				save_obj(param_dict, savePath+"/{}.pkl".format(epoch))



def validate(loader, name, device=0, num_epoch=100, patch_size=61, checkPath=None, pertubation=False):
	op = operation(device=device)
	# loss = chainer.functions.softmax_cross_entropy

	embedding_dim = patch_size//2 +1 
	_model = model(device=device, embedding_dim=embedding_dim)

	if(checkPath is not None):
		param_dict = load_obj(checkPath)
		_model.param_from_dict(param_dict)


	with device_guard(device):

		start_time = timer()
		num = 0
		count = 0 
		correct = 0
		l = 0
		epoch = 0

		if(pertubation==False):
			ld_fun = loader.get(epoch, pertube=False)
		else:
			ld_fun = loader.periodic_get(epoch) 

		for x, y, index in ld_fun:
			x, y = op.array(x), op.array(y)

			x.cast(op.run.float32)

			x = _model(x)

			# L = wrapper(loss, device, x, y)

			correct = (op.argmax(x, axis=1)==y).sum() + correct
			# l = Variable(L) + l
			num = num + y.shape[0]
			# count = count + 1

		print("correct: ", correct/num, "time: ", timer()-start_time)



















































# loss = chainer.functions.softmax_cross_entropy
def single(x, y, param_dict, loss, q, device):
	with device_guard(device):
		op = operation(device=device)
		_model = model(device=device)
		_model.param_from_dict(param_dict)

		x = Variable(x, device=device)
		y = Variable(y, device=device)

		x.cast(op.run.float32)
		x = _model(x)
		L = wrapper(loss, device, x, y)
		_model.cleargrads()
		L.backward()
		grad_dict = _model.grad_to_dict()

		print(type(x), type(x.ndarray()), x.shape, device)
		q.put(grad_dict)

	# print(type(param_dict))

def initial(x, y, _model, loss, device):
	with device_guard(device):
		op = operation(device=device)
		x = Variable(x, device=device)
		y = Variable(y, device=device)

		x.cast(op.run.float32)
		x = _model(x)
		L = wrapper(loss, device, x, y)
		_model.cleargrads()
		L.backward()
		# grad_dict = _model.grad_to_dict()

		# print(type(x), type(x.ndarray()), x.shape, device)
		# q.put(grad_dict)


import multiprocessing as mp
def train_multi(loader, device_list=[1,2,3], num_epoch=100):

	loss = chainer.functions.softmax_cross_entropy
	optimizer = chainer.optimizers.Adam()

	_model = model(device=3)

	optimizer.setup(_model)


	ctx = mp.get_context('forkserver')

	for epoch in range(num_epoch):
		start_time = timer()
		num = 0
		count = 0 
		correct = 0
		l = 0
		for x, y, index in loader.get():
			if(index==0):
				initial(x, y, _model, loss, 3)

			q = ctx.Queue()
			param_dict = _model.param_to_dict()

			groups = groupby([x for x in range(y.shape[0])], len(device_list), key='num')
			# half_size = y.shape[0]//2
			# x_0, x_1 = x[:half_size], x[half_size:]
			# y_0, y_1 = y[:half_size], y[half_size:]


			jobs = []

			for device, group in zip(device_list, groups):
				_x, _y = x[group], y[group]

				p = ctx.Process(name='device {}'.format(0), target=single, args=(_x, _y, param_dict, loss, q, device))
				p.daemon = True
				jobs.append(p)
				p.start()


			for job in jobs:
				grad_dict = q.get()
				print(grad_dict.keys())
				print(_model.grad_to_dict().keys())
				_model.grad_add_from_dict(grad_dict)
				job.join()

			optimizer.update()


	# 	# 	correct = (op.argmax(x, axis=1)==y).sum() + correct
	# 	# 	l = Variable(L) + l
	# 	# 	num = num + y.shape[0]
	# 	# 	count = count + 1

	# 	# print("epoch: ", epoch, "correct: ", correct/num, "loss: ", l/count, "time: ", timer()-start_time)
	# 	print("time: ", timer()-start_time)



if __name__ == '__main__':
	end = timer()

	# X_files = list(filter(lambda x: re.search("X_", x), os.listdir("./log/size_61/")))
	# X, Y = [], []
	# num = 0
	# for x_file in X_files:
	# 	y_file = x_file.replace("X_", "Y_")	
	# 	X.append(np.load("./log/size_61/"+x_file))
	# 	Y.append(np.load("./log/size_61/"+y_file))
	# 	num = num+X[-1].shape[0]
	# 	print(num)
	# 	if(num>1500):
	# 		break
	# X, Y = np.concatenate(X), np.concatenate(Y)
	# print(timer()-end)
	# print("X.shape: ", X.shape, "Y.shape: ", Y.shape)



	# np.save("./log/X.npy", X)
	# np.save("./log/Y.npy", Y)


	# end = timer()
	# X, Y = np.load("./log/X.npy"), np.load("./log/Y.npy")
	# print(timer()-end)

	# minibatch = 44
	# ld = loader(X, Y, minibatch, shuffle=True)

	# train(ld, device=1, num_epoch=100, size=61)
	# train_multi(ld, num_epoch=10)


	POSI = load_obj("./log/POSI.pkl")
	NEGE = load_obj("./log/NEGE.pkl")
	IMG = load_obj("./log/IMG.pkl")

	POSI = POSI[:10]
	NEGE = NEGE[:10]
	IMG = IMG[:10]

	minibatch = 25*3
	size = 31
	ld = pertube_loader(POSI, NEGE, IMG, minibatch, size=size, step=25, shuffle=True, device=2, dataLimit=8000, cpu=True)
	# # ld.get()
	train(ld, "pertube", device=3, num_epoch=100, size=size)


	# POSI = load_obj("./log/POSI_test.pkl")
	# NEGE = load_obj("./log/NEGE_test.pkl")
	# IMG = load_obj("./log/IMG_test.pkl")
	# minibatch = 150

	# ld = pertube_loader(POSI, NEGE, IMG, minibatch, size=61, step=15, shuffle=True, device=0, dataLimit=300, cpu=True)
	# train_multi(ld, num_epoch=10)