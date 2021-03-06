
'''
IMPLEMENT VINN WITH CHAINER, CUPY
'''

'''
IN THIS MODULE, I JUST IMPLEMENT THE INFERENCE PART OF VINN WITH ONLY CUPY, I THOUGH THIS WOULD BE SUPER FAST
'''


# from . import Variable, operation, Sigmoid, cp, np, F

from . import Sigmoid, Graph, GraphList, np, Variable, operation, device_guard, wrapper, Linear, tensor, VariableOperation

import math, re

import chainer

#TODO: consider directly wrap chainer.Variable operation in local Variable

class model(Graph):
	'''
	inputs: N*C*S*k

	secondary inputs: N*C*k
	'''
	def __init__(self, design=None, checkPath=None, device=0, embedding_dim=31, class_num=2):
		super().__init__(device=device)
		self.device = device
		self.checkPath = checkPath
		self.link = False

		size_min, size_max, size_step = design['size_min'], design['size_max'], design['size_step']
		stride_min, stride_max, stride_step = design['stride_min'], design['stride_max'], design['stride_step']
		dilation_min, dilation_max, dilation_step = design['dilation_min'], design['dilation_max'], design['dilation_step']


		# with self.init_scope():

		self.inter = []
		self.pooling = []
		index = 0
		for size in range(size_min, size_max, size_step):
			for stride in range(stride_min, stride_max, stride_step):
				for dilation in range(dilation_min, dilation_max, dilation_step):
					self.inter.append(pnn_layer(size, stride, dilation, device=self.device))
					self.pooling.append(pooling(3, key='max', device=self.device))
					index = index+1

		inter_chain = GraphList(*self.inter)
		pooling_chain = GraphList(*self.pooling)

		linear1 = Linear(3, 1, device=self.device)

		linear2 = Linear(embedding_dim, 1, device=self.device)

		linear3 = Linear(index, class_num, device=self.device)
		self.sigmoid = Sigmoid(device=self.device)
		# self.channels_pooling = Channels_pooling(device=self.device)

		self.add_node('linear1', linear1)
		self.add_node('linear2', linear2)
		self.add_node('linear3', linear3)
		self.add_node('inter_chain', inter_chain)
		self.add_node('pooling_chain', pooling_chain)

		# for index, pool in enumerate(self.pooling):
		# 	self.add_node('pool'+str(index), pool)
		self.op = VariableOperation(device=self.device)


	def forward(self, x):
		# print("vinn start")
		#embedding and change to C*N*S*k
		print(type(x))
		x = x.transpose([1,0,2,3])
		# x = chainer.functions.transpose(x, axes=[1,0,2,3])

		# x = self.channels_pooling(x)
		# print(x.shape)

		out = []

		for treat, pool in zip(self.inter_chain, self.pooling_chain):
			out.append(pool(treat(x)))
		# for index in range(len(self.inter)):
		# 	out.append(getattr(self, 'pool'+str(index))(self.inter[index](x)))

		# print(type(out[0]))
		#now N*P*S*k
		# out = self.op.stack(out).transpose([1,0,2,3])
		out = self.op.stack(out)

		out = out.transpose([1,0,2,3])
		# out = self.op.transpose(out, axes=[1,0,2,3])

		#now N*P*S
		out = self.sigmoid(self.linear1(out))
		# out = self.sigmoid(self.linear1(out).squeeze())

		#now N*P
		out = self.sigmoid(self.linear2(out))
		# out = self.sigmoid(self.linear2(out).squeeze())

		#now N*2
		out = self.sigmoid(self.linear3(out))

		return out





class pooling(Graph):
	'''
	inputs: F*N*S*k
	outputs: N*S*k

	by defaut, param(if tensor), should be cupy tensor allocated in specific device set by device parameters
	'''
	def __init__(self, param, key='max', device=0, own=False):
		super().__init__(device=device)
		self.device = device
		self.own = own
		self.link = False

		# if(type(param)==int):
		# 	self.register_weights('weights', np.random.randn(param))
		# else:
		# 	self.register_weights('weights', param)
		self.register_weights('weights', tensor(np.random.randn(param).astype(np.float32), device=self.device))

		'''
		this two with could be combined together
		'''

		self.key = key
		self.op = VariableOperation(device=self.device)

	def forward(self, x):
		# print(x.shape, self.weights.shape)

		# x, weights = self.op.run.broadcast(x, self.weights)
		# print(x.shape, weights.shape)

		# x = x*self.weights.chainer
		x = x*self.weights
		# print(type(x))

		if(self.key=='sum'):
			x = x.sum(axis=0)
			# x = torch.sum(x, dim=0)
		elif(self.key=='max'):
			x = x.max(axis=0)
			# x = self.op.max(x, axis=0)
			# x = torch.max(x, dim=0)[0]
		else:
			raise ValueError("key is neither sum nor max")
		return x



class pnn_layer(Graph):
	'''
	inputs: C*N*S*k  k:embedding dim, S:slides dim
	outputs: F*N*S*k  k: embedding dim
	'''
	def __init__(self, size, stride, dilation, channel=4095, device=0):
		super().__init__(device=device)
		self.device = device

		self.size = size
		self.stride = stride
		self.channel = channel
		self.dilation = dilation
		self.link = False

	def forward(self, X):
		INDEX = []
		for i in range(0, self.channel-self.dilation*self.size, self.stride):
			INDEX.append([i+x*self.dilation for x in range(self.size)])

		INDEX = tensor(INDEX, device=self.device)
		outputs = X.ndarray[INDEX.astype('int64').ndarray.squeeze()]
		outputs = outputs.sum(axis=1)
		#now is of F*N*S*k

		# outputs = tensor(outputs, device=self.device)
		return outputs






if __name__ == '__main__':
	checkPath = "./log/000065.pth.tar"
	params_dict = load_params(checkPath)
	_model = model(params_dict)

