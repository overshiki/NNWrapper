from .preprocessing import preprocessing
from .embedding import Embedding
# from .model import model
from .old_vinn import model

from . import Variable, operation, Linear, Graph, node, Sigmoid, cp, np, F, device_guard
# from NN.NNWrapper import operation, Variable, wrapper, operation, device_guard

from timeit import default_timer
from BASIC_LIST.basic import groupby
from IO.basic import load_obj

import chainer
from .prepare import Channels_pooling


def inference(img, pos, checkPath, device, design=None, patch_size=61, step=30, channels=4095):
	op = operation(device=device)

	preprocess = preprocessing(img, channels=channels, half_size=patch_size//2, device=device, step=step)
	embedding = Embedding(patch_size, device=device)
	# channels_pooling = Channels_pooling(stride=channels_pooling_stride, device=device)


	embedding_dim = patch_size//2 +1 
	_model = model(design=design, device=device, embedding_dim=embedding_dim)
	param_dict = load_obj(checkPath)
	# _model.param_from_dict(param_dict)
	with device_guard(device):
		for key, param in _model.namedparams():
			param.copydata(chainer.Parameter(cp.asarray(param_dict[key])))	

	pos = op.array(pos)

	_x, pos, groups_start, groups_end = preprocess(pos)

	PRED = []

	with device_guard(device):

		for index in range(groups_start.shape[0]):
			_end = default_timer()

			start = groups_start[index]
			end = groups_end[index]

			start, end = int(start.ndarray()), int(end.ndarray())

			x = _x[:,:,start:end]

			x = preprocess.beforeEmbedding(x)

			x = embedding(x)

			# x = x.transpose([1,0,2,3])
			# x = channels_pooling(x)
			# x = x.transpose([1,0,2,3])

			x.cast(cp.float32)

			x = _model(x)

			# x = _model(x)

			pred = op.argmax(x, axis=1)
			PRED.append(pred)

		PRED = op.concatenate(PRED, axis=0)

	return PRED, pos

def inference_kernel(img, pos, checkPath, device):
	op = operation(device=device)
	PRED, pos = inference(img, pos, checkPath, device, size=31)
	PRED = op.asnumpy(PRED)
	pos = op.asnumpy(pos)
	print(type(PRED), type(pos))

	np.save("./log/{}_PRED.npy".format(device), PRED)
	np.save("./log/{}_pos.npy".format(device), pos)

import multiprocessing as mp
def multi_inference(img, pos, checkPath, device_list=[1,2,3]):
	'''
	img should be numpy array
	pos should be list
	'''
	num_gpu = len(device_list)
	pos_groups = groupby(pos, num_gpu, key='num')

	ctx = mp.get_context('forkserver')

	# ctx = mp.get_context('spawn')
	# que = mp.Queue()

	# que = ctx.Array('i', range(10))

	processes = []
	img = img.astype(np.int32)
	compare = np.arange(4095, dtype=np.int32)
	compare = np.expand_dims(compare, axis=0)

	for device, pos in zip(device_list, pos_groups):
		# with device_guard(device):
		p = ctx.Process(target=inference_kernel, args=(img, pos, checkPath, device))
		p.start()
		processes.append(p)
		print(device, "start")


	for index, p in enumerate(processes):
		p.join()

	PRED = []
	pos = []
	for device in device_list:
		PRED.append(np.load("./log/{}_PRED.npy".format(device)))
		pos.append(np.load("./log/{}_pos.npy".format(device)))

	PRED = np.concatenate(PRED, axis=0)
	pos = np.concatenate(pos, axis=0)
	
	return PRED, pos

def analyze(img, checkPath, targetPath, device):
	stride = 5
	pos_x = np.arange(0, 2048, stride)
	pos_y = np.arange(0, 2048, stride)
	vx, vy = np.meshgrid(pos_x, pos_y)
	pos = np.stack([vx, vy]).reshape((2, -1)).transpose([1,0])

	PRED, pos = inference(img, pos, checkPath, device, size=31)

	op = operation(device=device)

	INDEX = op.where(PRED>0)
	print("finished where")
	pos = pos[INDEX]
	pos = pos.ndarray().get()

	img = rescale_intensity(img.astype('uint8'))

	img = drawRedPoints(img, pos, half_size=2)

	imwrite(img, targetPath)

def doubleLayer_analyze(img, device=0):
	op = operation(device=device)
	stride = 31
	pos_x = np.arange(0, 2048, stride)
	pos_y = np.arange(0, 2048, stride)
	vx, vy = np.meshgrid(pos_x, pos_y)
	pos = np.stack([vx, vy]).reshape((2, -1)).transpose([1,0])

	checkPath = "./log/params/size_61/20.pkl"

	PRED, pos = inference(img, pos, checkPath, device, size=61, step=10)

	pos = pos[op.where(PRED>0)]

	return pos



if __name__ == '__main__':
	from CV.ImageIO import center_cell, imreadTif, imshow_plt, plt, gray2rgb, drawMarker, drawRedPoints, imwrite, rescale_intensity
	import os, re
	checkPath = "./log/params/pertube/third_stage.pkl"

	# files = list(filter(lambda x:re.search("tif", x), os.listdir("./log/imgs")))
	# for file in files:
	# 	path = "./log/imgs/"+file 
	# 	targetPath = path.replace("tif", "png")
	# 	img = imreadTif(path)
	# 	analyze(img, checkPath, targetPath, 0)


	path = "../../../../DATA/CELL/10.tif"
	img = imreadTif(path)
	pos = doubleLayer_analyze(img)

	pos = pos.cpu()
	_pos = []

	half_size = 15

	for x in range(-1*half_size, half_size, 5):
		for y in range(-1*half_size, half_size, 5):
			_pos.append(pos+np.array([x, y]))

	pos = np.concatenate(_pos, axis=0)
	print(pos.shape)

	img = rescale_intensity(img.astype('uint8'))

	img = drawRedPoints(img, pos, half_size=2)

	imwrite(img, "./log/result.png")

# if __name__ == '__main__':
# 	from CV.ImageIO import center_cell, imreadTif, imshow_plt, plt, gray2rgb, drawMarker, drawRedPoints, imwrite, rescale_intensity
# 	import os, re
# 	from timeit import default_timer as timer
# 	path = "../../../../DATA/CELL/10.tif"
# 	# path = "/media/processed/Dec_2017/CST-20171211-01/changhai12.11/DATA/"
# 	# path = path+list(filter(lambda x: re.search("tif", x), os.listdir(path)))[0]
# 	print(path)

# 	# checkPath = "./log/params/size_61/25.pkl"
# 	# checkPath = "./log/params/size_31/60.pkl"
# 	# checkPath = "./log/params/size_61/17.pkl"
# 	checkPath = "./log/params/pertube/third_stage.pkl"

# 	end = timer()

# 	img = imreadTif(path)

# 	stride = 60

# 	pos_x = np.arange(0, 2048, stride)
# 	pos_y = np.arange(0, 2048, stride)
# 	vx, vy = np.meshgrid(pos_x, pos_y)
# 	pos = np.stack([vx, vy]).reshape((2, -1)).transpose([1,0]).astype(np.int32)
# 	print(pos.shape)
# 	# pos = pos[:1000]

# 	# multi_inference(img, pos.tolist(), checkPath)
# 	# PRED, pos = multi_inference(img, pos.tolist(), checkPath, device_list=[1,2,3])

# 	device = 0

# 	PRED, pos = inference(img, pos, checkPath, device, size=31)

# 	print(type(PRED), type(pos))
# 	op = operation(device=device)

# 	INDEX = op.where(PRED>0)
# 	print("finished where")
# 	pos = pos[INDEX]
# 	pos = pos.ndarray().get()


# 	img = rescale_intensity(img.astype('uint8'))

# 	# for pt in pos:
# 	# 	img = drawMarker(img, pt)
# 	img = drawRedPoints(img, pos,half_size=2)

# 	imwrite(img, "./log/result.png")

# 	print(timer()-end)