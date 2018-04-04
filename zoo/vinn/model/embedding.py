'''
EMBEDDING SPEED UP WITH CUPY
'''

# from NN.NNWrapper.backend.NNCUPY import Variable, operation, node, Graph, np, cp

from . import Variable, operation, node, Graph, np, cp

import math

class Embedding(node):
	def __init__(self, figsize, device=0):
		super().__init__(device=device)
		self.figsize = figsize
		self.device = device
		if(self.device!=-1):
			with self.gpu():
				self.LUT = cp.asarray(self.ball())
		else:
			self.LUT = self.ball()

	def ball(self):
		LUT = []
		coef = self.figsize*1./(2*math.pi)
		for i in range(self.figsize):
			for j in range(self.figsize):
				feature = [coef*math.cos(i*1./self.figsize * 2*math.pi)*(-1)+1.,
						   coef*math.sin(i*1./self.figsize * 2*math.pi), 
						   coef*math.sin(j*1./self.figsize * 2*math.pi)]
				LUT.append(feature)

		LUT = np.array(LUT)
		return LUT

	def slides_index(self):
		INDEX = []
		for i in range(self.figsize, 0, -2):
			padding = int((self.figsize-i)/2)
			index = [padding+x+self.figsize*padding for x in range(i)]
			index.extend([padding+x+self.figsize*(self.figsize-padding-1) for x in range(i)])
			index.extend([padding+self.figsize*(x+padding) for x in range(i)])
			index.extend([self.figsize-padding-1+self.figsize*(x+padding) for x in range(i)])
			INDEX.append(index)
		return INDEX


	def forward(self, binary):
		'''
		binary is N*C*x*y, torch tensor
		output is N*C*S*k or N*C*k
		'''
		shape = binary.shape

		N, C, x, y = shape

		# binary = binary.view(N, C, x*y, 1)
		binary = binary.reshape(N, C, x*y, 1)
		binary = binary*self.LUT

		# binary = torch.unsqueeze(binary, 3) #axis 3 is the feature vector dim
		# binary = binary*binary.new(method().tolist()).float()

		INDEX = self.slides_index()

		LIST = []
		for index in INDEX:
			LIST.append(binary[:,:,index,:].sum(axis=2, keepdims=True))
		
		binary = self.op.concatenate(LIST, axis=2)

		# 	LIST.append(torch.index_select(binary, 2, binary.new(index).long()).sum(dim=2, keepdim=True))
		# binary = torch.cat(LIST, dim=2)
		# # N*C*S*k

		return binary

