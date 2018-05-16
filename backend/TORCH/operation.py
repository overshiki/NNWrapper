from . import np, tensor, device_guard, Variable, _OperationBase
import torch

r"""we consider using two type of operation, operation for tensor and operation for variable, since for many framwork, this two operations are different. 
For some framwork such as pytorch, Variable is actually the same type as tensor, thus this two operations should be exactly the same
"""
class OperationBase(_OperationBase):
	def __init__(self, OpType=tensor, VarType=np.ndarray, device=0):
		super().__init__(OpType=OpType, VarType=VarType, device=device)

		self.run = np #we may not want to see run object anymore

		self.newaxis = self.run.newaxis #this currently only support numpy and cupy, we will consider enable this syntax for pytorch in the near future

	def __str__(self):
		return "operation on device: {}".format(self.device)



	def concatenate(self, x, **kwargs):		
		_x = self.unskin(x)
		return self.type(self.run.cat(_x, **kwargs), device=self.device)

	def argmax(self, *x, **kwargs):		
		_x = self.unskin(x)
		return self.type(self.run.argmax(*_x, **kwargs), device=self.device)

	def expand_dims(self, *x, **kwargs):		
		_x = self.unskin(x)
		return self.type(self.run.expand_dims(*_x, **kwargs), device=self.device)

	def where(self, *x, **kwargs):		
		_x = self.unskin(x)
		return self.run.where(*_x, **kwargs)

	def stack(self, x, **kwargs):		
		_x = self.unskin(x)
		return self.type(self.run.stack(_x, **kwargs), device=self.device)

	def array(self, x):		
		return self.type(self.run.array(self.unskin_element(x)), device=self.device)

	def exp(self, *x):		
		_x = self.unskin(x)
		return self.type(self.run.exp(*_x), device=self.device)

	def arange(self, *x):		
		return self.type(self.run.arange(*x), device=self.device)

	def minimum(self, *x):		
		_x = self.unskin(x)
		return self.type(self.run.minimum(*_x), device=self.device)

	def equal(self, *x):		
		_x = self.unskin(x)
		return self.type(self.run.equal(*_x), device=self.device)		

	def cumsum(self, *x, **kwargs):		
		_x = self.unskin(x)
		return self.type(self.run.cumsum(*_x, **kwargs), device=self.device)

	def asarray(self, x):		
		if(type(x)==Variable):
			return x
		elif(type(x)==self.run.core.ndarray):
			return self.type(x, device=self.device)
		else:
			return self.type(self.run.asarray(x), device=self.device)

	def asnumpy(self, x):		
		if(type(x)==Variable):
			return self.run.asnumpy(x.ndarray)
		elif(type(x)==self.run.core.ndarray):
			return self.run.asnumpy(x)


class TensorOperation(OperationBase):
	def __init__(self, device=0):
		super().__init__(OpType=tensor, device=device)
		self.run = torch

class VariableOperation(OperationBase):
	def __init__(self, device=0):
		super().__init__(OpType=Variable, device=device)
		self.run = torch

