from . import np, cp, tensor, device_guard, Variable

class operation:
	def __init__(self, device=0):
		self.device = device
		if self.device==-1:
			self.run = np 
		else:
			self.run = cp

		self.newaxis = self.run.newaxis
		self.guard = device_guard(device=self.device)

	def __str__(self):
		return "operation on device: {}".format(self.device)


	def unskin_element(self, x):
		if isinstance(x, tensor):
			if x.device!=self.device:
				x.to_device(self.device)
			_x = x.ndarray
		elif isinstance(x, (cp.ndarray, np.ndarray)):
			#TODO, considering array in different gpus
			_x = tensor(x, device=self.device).ndarray
		elif isinstance(x, Variable):
			_x = tensor(x.ndarray, device=self.device).ndarray
		else:
			raise TypeError("input type is neither tensor, nor cp.ndarray, np.ndarray, but {}".format(type(x)))
		return _x

	def unskin(self, x):
		_x = []
		for i in x:
			_x.append(self.unskin_element(i))
		return _x

	def concatenate(self, x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return tensor(self.run.concatenate(_x, **kwargs), device=self.device)

	def argmax(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return tensor(self.run.argmax(*_x, **kwargs), device=self.device)

	def expand_dims(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return tensor(self.run.expand_dims(*_x, **kwargs), device=self.device)

	def where(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return self.run.where(*_x, **kwargs)

	def stack(self, x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return tensor(self.run.stack(_x, **kwargs), device=self.device)

	def array(self, x):
		with self.guard:
			return tensor(self.run.array(self.unskin_element(x)), device=self.device)

	def exp(self, *x):
		with self.guard:
			_x = self.unskin(x)
			return tensor(self.run.exp(*_x), device=self.device)

	def arange(self, *x):
		with self.guard:
			return tensor(self.run.arange(*x), device=self.device)

	def minimum(self, *x):
		with self.guard:
			_x = self.unskin(x)
			return tensor(self.run.minimum(*_x), device=self.device)

	def equal(self, *x):
		with self.guard:
			_x = self.unskin(x)
			return tensor(self.run.equal(*_x), device=self.device)		

	def cumsum(self, *x, **kwargs):
		with self.guard:
			_x = self.unskin(x)
			return tensor(self.run.cumsum(*_x, **kwargs), device=self.device)

	def asarray(self, x):
		with self.guard:
			if(type(x)==Variable):
				return x
			elif(type(x)==self.run.core.ndarray):
				return tensor(x, device=self.device)
			else:
				return tensor(self.run.asarray(x), device=self.device)

	def asnumpy(self, x):
		with self.guard:
			if(type(x)==Variable):
				return self.run.asnumpy(x.ndarray)
			elif(type(x)==self.run.core.ndarray):
				return self.run.asnumpy(x)