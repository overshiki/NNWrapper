

'''
USING LOCAL_THRESHOLD MECHANISM, GENERATE KEYPOINTS 
'''

import os
import re
from GPU.local_threshold import threshold_loader
from GPU import ccl_map
from CV.ImageIO import imreadTif, rescale_intensity, gray2rgb, imwrite, drawMarker, symmetry_mapping, center_cell
import numpy as np
from yaml import load
from CELL.basic import boundary
from CV.drawMarker import drawMarkers
from skimage.exposure import equalize_adapthist
from IO.basic import mkdir, save_obj


class keypoints():
	def __init__(self, files, yaml_file, targetDir, brightShift=0, paramsDict={}, verbose=False):
		self.files = files
		f = open(yaml_file,'r')
		self.params = load(f)
		self.verbose = verbose

		self.highPrior = paramsDict

		self.default = {'local_threshold':0.5, 
						'device': 0,
						'use_sobel': False,
						'ccl': 'tree',
						'cellsize': 200, 
						'symmetry': 0.5, 
						'targetDir': "STATIC"}


		for key in ['local_threshold', 'device', 'use_sobel', 'ccl', 'cellsize', 'symmetry', 'targetDir']:
			if key not in self.params.keys():
				self.params[key] = self.default[key]
			if key in self.highPrior.keys():
				self.params[key] = self.highPrior[key]

		self.targetDir = targetDir
		self.brightShift = brightShift


	def get_points(self, half_size=80, extra=0):
		local_threshold = float(self.params['local_threshold'])
		_device = int(self.params['device'])
		symmetry = float(self.params['symmetry'])
		# targetDir = str(self.params['targetDir'])
		cellsize = int(self.params['cellsize'])
		ccl = str(self.params['ccl'])

		print("[get_insider] device: {}, verbose: {}".format(_device, self.verbose))
		for file in self.files:
			img = imreadTif(file)
			binary = threshold_loader(img, stride=32, grid_size=32, color_channel=4095, thres=local_threshold, device=_device, FLAG="matrix")

			cells = ccl_map(binary, ccl, cellsize, device=_device)

			binary, cells = symmetry_mapping(cells, thres=symmetry)

			pos = center_cell(cells)

			img_bound = [0, 2048]

			mapping = np.zeros((2048, 2048))
			num = len(cells)+extra
			img_out = imreadTif(file)

			for cell in cells:
				_bound, _ = boundary(cell=cell, _range=None)
				if(_bound[0]-half_size >= img_bound[0]):
					_bound[0] = _bound[0]-half_size
				else:
					_bound[0] = img_bound[0]
				if(_bound[2]-half_size >= img_bound[0]):
					_bound[2] = _bound[2]-half_size
				else:
					_bound[2] = img_bound[0]

				if(_bound[1]+half_size <= img_bound[1]):
					_bound[1] = _bound[1]+half_size
				else:
					_bound[1] = img_bound[1]
				if(_bound[3]-half_size <= img_bound[1]):
					_bound[3] = _bound[3]+half_size
				else:
					_bound[3] = img_bound[1]
					
				mapping[_bound[0]:_bound[1], _bound[2]:_bound[3]] = 1

			mapping[0:half_size, :] = 1
			mapping[img_bound[1]-half_size:img_bound[1], :] = 1

			mapping[:, 0:half_size] = 1
			mapping[:, img_bound[1]-half_size:img_bound[1]] = 1
			


			pts = np.array(np.where(mapping==0)).transpose()
			if(pts.shape[0]>=num):
				index = np.random.choice(pts.shape[0], num)
				choice = pts[index]
			else:
				choice = np.zeros((0,2))


			if(len(pos)>0):
				if(self.verbose==True):
					img = img_out.copy()
					img = rescale_intensity(img.astype('uint8'))

					mask_true = drawMarkers(np.array(pos), 10, 2)
					mask_false = drawMarkers(choice, 10 ,2)

					img = np.stack([img, img, img], axis=2)

					index = list(np.where(mask_true==1))
					index.append((np.ones(index[0].shape[0])*2).astype('int32'))
					# img[index] = 55535
					img[index] = 255

					index = list(np.where(mask_false==1))
					index.append((np.ones(index[0].shape[0])*1).astype('int32'))
					# img[index] = 55535
					img[index] = 255

					_path = self.targetDir+"/"+os.path.basename(file).replace("tif","png")
					print("[verbose]: ", _path)
					imwrite(img,_path)

					yield file, cells, pos, choice, img_out
				else:
					yield file, cells, pos, choice, img_out






	# def get_outsider_points(self, half_size_list, extra=30):
	# 	local_threshold = float(self.params['local_threshold'])
	# 	_device = int(self.params['device'])
	# 	symmetry = float(self.params['symmetry'])
	# 	targetDir = str(self.params['targetDir'])
	# 	cellsize = int(self.params['cellsize'])
	# 	ccl = str(self.params['ccl'])

	# 	print("[get_outsider] device: {}, verbose: {}".format(_device, self.verbose))

	# 	for file in self.files:
	# 		img = imreadTif(file)
	# 		binary = threshold_loader(img, stride=32, grid_size=32, color_channel=4095, thres=local_threshold, device=_device, FLAG="matrix")
	# 		cells = ccl_map(binary, ccl, cellsize, device=_device)

	# 		img_bound = [0, 2048]

	# 		mapping = np.zeros((2048, 2048))
	# 		num = len(cells)+extra
	# 		img_out = imreadTif(file)

	# 		pos_dict = {}
	# 		for half_size in half_size_list:
	# 			for cell in cells:
	# 				_bound, _ = boundary(cell=cell, _range=None)
	# 				if(_bound[0]-half_size >= img_bound[0]):
	# 					_bound[0] = _bound[0]-half_size
	# 				else:
	# 					_bound[0] = img_bound[0]
	# 				if(_bound[2]-half_size >= img_bound[0]):
	# 					_bound[2] = _bound[2]-half_size
	# 				else:
	# 					_bound[2] = img_bound[0]
						

	# 				if(_bound[1]+half_size <= img_bound[1]):
	# 					_bound[1] = _bound[1]+half_size
	# 				else:
	# 					_bound[1] = img_bound[1]
	# 				if(_bound[3]-half_size <= img_bound[1]):
	# 					_bound[3] = _bound[3]+half_size
	# 				else:
	# 					_bound[3] = img_bound[1]
						
	# 				mapping[_bound[0]:_bound[1], _bound[2]:_bound[3]] = 1
				
	# 			pts = np.array(np.where(mapping==0)).transpose()
	# 			if(pts.shape[0]>0):
	# 				index = np.random.choice(pts.shape[0], num)
	# 				choice = pts[index]
	# 				pos_dict[half_size] = choice
	# 			else:
	# 				pos_dict[half_size] = np.zeros((0,2))

	# 		yield file, pos_dict, img_out





	# def get_keyPoints(self):
	# 	local_threshold = float(self.params['local_threshold'])
	# 	_device = int(self.params['device'])
	# 	symmetry = float(self.params['symmetry'])
	# 	targetDir = str(self.params['targetDir'])
	# 	cellsize = int(self.params['cellsize'])
	# 	ccl = str(self.params['ccl'])

	# 	print("[get_keyPoints] device: {}, verbose: {}".format(_device, self.verbose))
	# 	for file in self.files:
	# 		img = imreadTif(file)
	# 		binary = threshold_loader(img, stride=32, grid_size=32, color_channel=4095, thres=local_threshold, device=_device, FLAG="matrix")

	# 		cells = ccl_map(binary, ccl, cellsize, device=_device)

	# 		binary, cells = symmetry_mapping(cells, thres=symmetry)

	# 		pos = center_cell(cells)

	# 		img_out = imreadTif(file)

	# 		if(self.verbose==True):
				
	# 			img = img_out.astype('uint16')
	# 			img = gray2rgb(img)
	# 			img = rescale_intensity(img)
	# 			for point in pos:
	# 				img = drawMarker(img, point)
	# 			_path = targetDir+"/"+file.replace("tif","png")

	# 			imwrite(img,_path)
	# 			imwrite(binary, _path.replace(".png","_mapping.jpeg"))
	# 			yield file, cells, pos, img_out
	# 		else:
	# 			yield file, cells, pos, img_out







def kp_generator(name, TRAIN_root, _type='override', extra=30):
	work_dir = "/home/polaris-nn/CODE/Polaris_v5/NN/NNWrapper/zoo/VINN/DataSet/"
	
	targetDir = work_dir+"/"+name+"/COLLECT"
	mainDir = work_dir+"/"+name+"/"
	
	mkdir(targetDir)

	finished_files = list(filter(lambda x: re.search("pkl", x), os.listdir(targetDir)))


	TRAINDATA = TRAIN_root+"DATA"
	yaml_file = TRAIN_root+"local.yml"

	files = list(filter(lambda x: re.search("tif", x), os.listdir(TRAINDATA)))
	# files = list(filter(lambda x: re.search("DAPI", x), files))

	if(_type!="override"):
		for file in finished_files:
		    print("finished: {}".format(file.replace("pkl", "tif")))
		    files.remove(file.replace("pkl", "tif"))
	else:
		files = list(map(lambda x:x.replace("pkl", "tif"), finished_files))
	    
	files = list(map(lambda x: TRAINDATA+"/"+x, files))
	    
	kp = keypoints(files, yaml_file, targetDir, brightShift=67000, verbose=True)

	for file, cells, pos, choice, img_out in kp.get_points(extra=extra):
	    _dict = {'file':file, 'cells':cells, 'pos':pos, 'choice':choice, 'img_out':img_out}
	    save_obj(_dict, targetDir+"/"+os.path.basename(file).replace("tif","pkl"))


if __name__ == '__main__':
	# name = "CST-20171108-02"
	# TRAIN_root = "/media/processed/Nov_2017/CST-20171108-02/"
	# kp_generator(name, TRAIN_root, _type='override', extra=100)

	# name = "CST-20171028-01"
	# TRAIN_root = "/media/processed/Nov_2017/"+name+"/"
	# kp_generator(name, TRAIN_root, _type='initial')

	# name = "CST-20171106-01"
	# TRAIN_root = "/media/processed/Nov_2017/CST-20171106-01/"
	# kp_generator(name, TRAIN_root, _type='initial', extra=0)

	# name = "CST-20171128-01"
	# TRAIN_root = "/media/processed/Dec_2017/"+name+"/"
	# kp_generator(name, TRAIN_root, _type='initial')

	# name = "CST-20171123-01"
	# TRAIN_root = "/media/processed/Dec_2017/"+name+"/"
	# kp_generator(name, TRAIN_root, _type='initial')

	name = "CST-20171208-01"
	TRAIN_root = "/media/processed/Dec_2017/"+name+"/"
	kp_generator(name, TRAIN_root, _type='initial')