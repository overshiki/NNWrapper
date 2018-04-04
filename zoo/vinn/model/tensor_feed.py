'''
generate training tensors from imgs
the shape of the tensor should be:
	
'''
from .prepare import pts_generater
from .train import loader
import random, os, re
import numpy as np
from IO.serialize import data2pa

def feed(channels=4095, step=10, minigroup=2, data_step=40):
	# op = operation(device=device)

	# embedding_dim = patch_size//2 +1 

	work_dir = "/home/polaris-nn/CODE/Polaris_v5/NN/NNWrapper/zoo/VINN/DataSet/"

	names = ["CST-20170705-01", "CST-20171108-02", "CTC-20171106-01", "CST-20171128-01", "CST-20171123-01", "CST-20171208-01", "CST-20170804-01", "CST-20170810-01", "CST-20170822-01", "CST-20170823-01", "CST-20170907-01"]


	files = []
	for _dir in names:
		targetDir = work_dir+"/"+_dir+"/COLLECT/"

		_files = list(map(lambda x: targetDir+x, filter(lambda x: re.search("pkl", x), os.listdir(targetDir))))
		if(len(_files)>100):
			random.shuffle(_files)
			_files = _files[:100]
		print(_dir, len(_files))
		files.extend(_files)

	random.shuffle(files)
	
	POSI, NEGE, IMG = pts_generater(files)


	# POSI, NEGE, IMG = POSI[:100], NEGE[:100], IMG[:100]
	print("total num of imgs: {}".format(len(IMG)))

	if(channels!=4095):
		IMG = list(map(lambda x:(x/4095.*channels).astype(np.uint16), IMG))
	else:
		IMG = list(map(lambda x:x.astype(np.uint16), IMG))

	print("convert finished")

	dataLimit = 150000
	patch_size = 31



	for data_index in range(0, len(POSI), data_step):
		X, Y = [], []
		print(data_index)

		posi = POSI[data_index:data_index+data_step]
		nege = NEGE[data_index:data_index+data_step]
		imgs = IMG[data_index:data_index+data_step]


		ld = loader(posi, nege, imgs, minigroup, patch_size=patch_size, shuffle=True, device=1, dataLimit=dataLimit, channels=channels, step=step)

		ld_fun = ld.periodic_get(0) 

		for x, y, index in ld_fun:
			x, y = x.ndarray().get().astype(np.float32), y.ndarray().get().astype(np.bool_)
			X.append(x)
			Y.append(y)

		X = np.concatenate(X, axis=0)
		Y = np.concatenate(Y, axis=0)
		# print(X.shape, Y.shape, X.dtype, Y.dtype)
		data2pa("/media/nvme0n1/DATA/TRAININGSETS/vinn/X_c{}_i{}.pa".format(channels, data_index), X)
		data2pa("/media/nvme0n1/DATA/TRAININGSETS/vinn/X_c{}_i{}.pa".format(channels, data_index), X)
