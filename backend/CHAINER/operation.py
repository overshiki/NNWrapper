class operation:
	def __init__(self, device=0):
		self.device = device
		if self.device==-1:
			self.run = np 
			self.arrayType = np.ndarray
			self.counterType = cp.core.ndarray
		else:
			self.run = cp
			self.arrayType = cp.core.ndarray
			self.counterType = np.ndarray

		self.newaxis = self.run.newaxis
		self.guard = device_guard(device=self.device)

	def __str__(self):
		return "operation for device: {}, of arrayType should be: {}".format(self.device, self.arrayType)


	def array_to_device(self, array):
		if self.device==-1:
			if(type(array)!=self.arrayType):
				return array.get()
		else:
			if(type(array)!=self.arrayType):
				with self.guard:
					return self.run.asarray(array)
			#TODO, considering array in different gpus
		return array


	def unskin_element(self, x):
		if(type(x)==Variable):
			_x = x.ndarray()
		elif(type(x)==self.arrayType):
			#TODO, considering array in different gpus
			_x = x
		elif(type(x)==self.counterType):
			_x = self.array_to_device(x)
		else:
			_x = x
		return _x


	def unskin(self, x):
		_x = []
		for i in x:
			_x.append(self.unskin_element(i))
		return _x

	def concatenate(self, x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.concatenate(_x, **kwargs), device=self.device)

	def argmax(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.argmax(*_x, **kwargs), device=self.device)

	def expand_dims(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.expand_dims(*_x, **kwargs), device=self.device)

	def where(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return self.run.where(*_x, **kwargs)

	def stack(self, x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.stack(_x, **kwargs), device=self.device)

	def array(self, x):
		with self.guard:
			return Variable(self.run.array(self.unskin_element(x)), device=self.device)

	def exp(self, *x):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.exp(*_x), device=self.device)

	def arange(self, *x):
		with self.guard:
			return Variable(self.run.arange(*x), device=self.device)

	def minimum(self, *x):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.minimum(*_x), device=self.device)

	def equal(self, *x):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.equal(*_x), device=self.device)		

	def cumsum(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return Variable(self.run.cumsum(*_x, **kwargs), device=self.device)

	def asarray(self, x):
		with self.guard:
			if(type(x)==Variable):
				return x
			elif(type(x)==self.run.core.ndarray):
				return Variable(x, device=self.device)
			else:
				return Variable(self.run.asarray(x), device=self.device)

	def asnumpy(self, x):
		with self.guard:
			if(type(x)==Variable):
				return self.run.asnumpy(x.ndarray)
			elif(type(x)==self.run.core.ndarray):
				return self.run.asnumpy(x)