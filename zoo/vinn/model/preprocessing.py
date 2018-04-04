# from NN.NNWrapper.backend.NNCUPY import Variable, operation, node, Graph, np, cp

from . import Variable, operation, node, Graph, np, cp


class preprocessing(node):
	def __init__(self, img, channels=4095, half_size=30, device=0, step=15):
		super().__init__(device=device)
		# self.device = device

		# self.op = operation(device=self.device)
		self.link = False
		self.img = img
		self.channels = channels
		if(self.device!=-1):
			with self.gpu():
				self.img = self.op.asarray(self.img)
				self.compare = self.op.asarray(np.expand_dims(np.arange(self.channels, dtype=np.int32), axis=0))
		else:
			self.img = self.op.asarray(self.img)
			self.compare = self.op.asarray(np.expand_dims(np.arange(self.channels, dtype=np.int32), axis=0))

		self.step = step
		self.half_size = half_size

	def forward(self, pos):
		# print(self.op, self.device)


		pos_x_left, pos_x_right = pos[:,0]-self.half_size, pos[:,0]+self.half_size
		pos_y_left, pos_y_right = pos[:,1]-self.half_size, pos[:,1]+self.half_size

		#TODO: current using filtering option, in the near future, change to padding option
		SELECT_MAP = (pos_x_left>=0)*(pos_y_left>=0)*(pos_x_right<2048)*(pos_y_right<2048)
		SELECT_INDEX = self.op.where(SELECT_MAP>0)[0]

		pos_x_left, pos_x_right, pos_y_left, pos_y_right = pos_x_left[SELECT_INDEX], pos_x_right[SELECT_INDEX], pos_y_left[SELECT_INDEX], pos_y_right[SELECT_INDEX]

		pos = pos[SELECT_INDEX]


		_len = pos.shape[0]
		groups_start = self.op.arange(0, _len-1, self.step)
		groups_end = self.op.minimum(self.op.array([_len]), groups_start+self.step)

		# shape should be dx * dy * dz
		pos_x = pos_x_left
		pos_x = self.op.expand_dims(pos_x, axis=0)
		adding = self.op.expand_dims(self.op.arange(2*self.half_size+1), 1)
		pos_x = pos_x+adding

		pos_y = pos_y_left
		pos_y = self.op.expand_dims(pos_y, axis=0)
		adding = self.op.expand_dims(self.op.arange(2*self.half_size+1), 1)
		pos_y = pos_y+adding

		# x * y * N
		_x = self.img[pos_x[:, self.op.newaxis, :].ndarray(), pos_y.ndarray()]

		return _x, pos, groups_start, groups_end

	def beforeEmbedding(self, x):
		# with self.gpu():
		x = self.op.expand_dims(x, axis=3)

		x = self.op.equal(x, self.compare)
		# N * C * x * y
		x = x.transpose([2,3,0,1])

		x = self.op.cumsum(x, axis=1)

		return x