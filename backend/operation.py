r'''base interface for operation
'''

class _OperationBase:
	def __init__(self, OpType=None, VarType=None, device=0):
		self.device = device
		self.type = OpType
		self.VarType = VarType
		
		# self.run = np #we may not want to see run object anymore

		# self.newaxis = self.run.newaxis #this currently only support numpy and cupy, we will consider enable this syntax for pytorch in the near future

	def __str__(self):
		return "operation on device: {}".format(self.device)


	def unskin_element(self, x):
		if isinstance(x, self.type):
			if x.device!=self.device:
				x.to_device(self.device)
			_x = x.var
		elif isinstance(x, self.VarType):
			_x = x
		else:
			raise TypeError("input type is not type {}, but {}".format(self.type, type(x)))
		return _x


	def unskin(self, x):
		_x = []
		for i in x:
			_x.append(self.unskin_element(i))
		return _x