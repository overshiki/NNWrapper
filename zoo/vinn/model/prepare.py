from .preprocessing import preprocessing
from .embedding import Embedding
from IO.basic import load_obj, save_obj, mkdir
import os, re
from BASIC_LIST.basic import groupby

from . import Variable, operation, node, Graph, np, cp

# from NN.NNWrapper.backend.NNCUPY import Variable, operation, node, Graph, np, cp

import chainer

from timeit import default_timer as timer



def pts_generater(files):
	# work_dir = "/home/DATA/ADDITIONAL_STORAGE/ADD2/DATA/TRAININGSETS/"
	# work_dir = "/media/huge/DATA/TRAININGSETS/"
	# work_dir = "/home/polaris-nn/CODE/Polaris_v5/NN/NNWrapper/zoo/VINN/DataSet/"
	# mainDir = work_dir+"/"+name+"/"
	# targetDir = work_dir+"/"+name+"/COLLECT"
	# files = list(filter(lambda x: re.search("pkl", x), os.listdir(targetDir)))

	POSI, NEGE, IMG = [], [], []

	for index, file in enumerate(files):
		# path = targetDir+"/"+file
		path = file
		# print(path)
		_dict = load_obj(path)
		positive, negetive, img = _dict['pos'], _dict['choice'], _dict['img_out']
		POSI.append(positive)
		NEGE.append(negetive)
		IMG.append(img)
		# if(index==99):
			# break
	return POSI, NEGE, IMG






def img_pos_label(positive, negetive):
	'''
	positive & positive is 3d list contains: imgs->img->pts
	'''
	index = []
	for index_img, img in enumerate(positive):
		for pt in img:
			index.append((index_img, pt, 0))
	for index_img, img in enumerate(negetive):
		for index_pt in enumerate(img):
			index.append((index_img, index_pt, 1))
	return index


class Channels_pooling(Graph):
	'''
	inputs: C*N*S*k
	operation:
		skip the even number channels
	'''
	def __init__(self, stride=5, device=0):
		super().__init__(device=device)
		self.device = device
		self.stride = stride
		self.link = False 

	def forward(self, x):
		C = x.shape[0]
		select_indices = [i for i in range(0, C, self.stride)]
		x = x[select_indices]
		return x


def prepare(INDEX, IMG, POS, label, channels_pooling_stride=5, device=0, patch_size=61, key='save', step=15, cpu=True, dataLimit=15000, channels=4095):
	op = operation(device=device)
	embedding = Embedding(patch_size, device=device)
	channels_pooling = Channels_pooling(stride=channels_pooling_stride, device=device)

	X, Y, _POS, _IMG = [], [], [], []
	num = 0

	for _index, (img, pos) in enumerate(zip(IMG, POS)):

		preprocess = preprocessing(img, channels=channels, half_size=patch_size//2, device=device, step=step)

		pos = op.array(pos).astype(op.run.int32)

		_x, pos, groups_start, groups_end = preprocess(pos)

		for index in range(groups_start.shape[0]):
			# print(groups_start.shape, type(groups_start))
			start = groups_start[index]
			end = groups_end[index]

			start, end = int(start.ndarray()), int(end.ndarray())

			x = _x[:,:,start:end]

			y = Variable((op.run.ones((x.shape[-1]))*label).astype(op.run.int32))

			x = preprocess.beforeEmbedding(x)

			x = embedding(x)

			# x = x.transpose([1,0,2,3])
			# x = channels_pooling(x)
			# x = x.transpose([1,0,2,3])
			# print(x.shape)


			if(cpu==True):
				X.append(x.cpu(delete=True))
				Y.append(y.cpu(delete=True))
				_POS.append(pos[start:end].cpu(delete=True))
			else:
				X.append(x)
				Y.append(y)
				_POS.append(pos[start:end])

			_IMG.append(img)
			num = num+end-start
			if(num>dataLimit):
				break

		# print("device: ", device, "index: ", _index, "progress: ", len(X), "num: ", num)
		if(num>dataLimit):
			break


	if(key=='save'):
		X = np.concatenate(X)
		Y = np.concatenate(Y)
		print("X.shape: ", X.shape, "Y.shape: ", Y.shape)

		np.save("../log/X_{}.npy".format(INDEX), X)
		np.save("../log/Y_{}.npy".format(INDEX), Y)
	else:
		return X, Y, _POS, _IMG






def prepare_single(INDEX, IMG, POSI, NEGE, size, device=0):
	op = operation(device=device)
	embedding = Embedding(size, device=device)

	X, Y = [], []


	# for _index, (img, positive, negetive) in enumerate(zip(IMG, POSI, NEGE)):

	img, positive, negetive = IMG, POSI, NEGE

	preprocess = preprocessing(img, half_size=size//2, device=device, step=15)

	pos = op.array(positive)

	_x, pos, groups_start, groups_end = preprocess(pos)

	for index in range(groups_start.shape[0]):
		# print(groups_start.shape, type(groups_start))
		start = groups_start[index]
		end = groups_end[index]

		start, end = int(start.ndarray), int(end.ndarray)

		x = _x[:,:,start:end]

		y = Variable(op.run.ones((x.shape[-1])).astype(op.run.int32))

		x = preprocess.beforeEmbedding(x)

		x = embedding(x)

		X.append(x.cpu(delete=True))
		Y.append(y.cpu(delete=True))

	pos = op.array(negetive)

	_x, pos, groups_start, groups_end = preprocess(pos)


	for index in range(groups_start.shape[0]):
		# print(groups_start.shape, type(groups_start))
		start = groups_start[index]
		end = groups_end[index]

		start, end = int(start.ndarray), int(end.ndarray)

		x = _x[:,:,start:end]

		y = Variable(op.run.zeros((x.shape[-1])).astype(op.run.int32))

		x = preprocess.beforeEmbedding(x)

		x = embedding(x)

		X.append(x.cpu(delete=True))
		Y.append(y.cpu(delete=True))

	print("device: ", device, "index: ", INDEX, "progress: ", len(X))

	X = np.concatenate(X)
	Y = np.concatenate(Y)
	print("X.shape: ", X.shape, "Y.shape: ", Y.shape)
	mkdir("../log/size_{}".format(size))
	np.save("../log/size_{}/X_{}.npy".format(size, INDEX), X)
	np.save("../log/size_{}/Y_{}.npy".format(size, INDEX), Y)

def pertubation(size):
	pertub_x = np.random.randint(5, 15, size)*(np.random.randint(0,2,size)*2-1)
	pertub_y = np.random.randint(5, 15, size)*(np.random.randint(0,2,size)*2-1)
	pertub = np.stack([pertub_x, pertub_y], axis=1)
	return pertub


def single_pertubation(INDEX, IMG, POSI, NEGE, size, device=0):
	posi_shape = POSI.shape[0]
	pertub = np.random.randint(5, 10, 100)*(np.random.randint(0,2,100)*2-1)




if __name__ == '__main__':
	pass