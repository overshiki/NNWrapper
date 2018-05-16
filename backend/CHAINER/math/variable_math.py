from .. import tensor, VarBase, np, cp, device_guard
import chainer

def type_check(self, other):
	number_check = False
	variable_check = False
	if isinstance(other, VarBase):
		var = other
	elif isinstance(other, tensor):
		#TODO: may be we should add support for chainer.Variable and chainer.Parameter
		var = self.new(other, device=self.device)
	elif np.isscalar(other):
		var = other
		number_check = True
	elif isinstance(other, (chainer.Variable, chainer.Parameter)):
		if self.var.ndim!=other.ndim:
			self.var, var = chainer.functions.broadcast(self.var, other)
		variable_check = True
	else:
		raise TypeError("input other is not VarBase nor tensor nor scalar variable, but to be: {}".format(type(other)))

	if number_check==False:
		if variable_check==False:
			if var.device!=self.device:
				var.to_device(self.device)
			return var.var
		else:
			return var 
			#TODO: device transfer
	else:
		return var

def add(self, rhs):  # lhs + rhs
	"""Element-wise addition.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.var + var, device=self.device)

def neg(self): #-x
	"""Element-wise negation.
	Returns:
		tensor: Output tensor.
	"""	
	return self.new((-1)*self.var, device=self.device)

def sub(self, rhs):  # lhs - rhs
	"""Element-wise subtraction.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.var - var, device=self.device)

def rsub(self, rhs): # rhs - lhs 
	"""Element-wise subtraction.
	Returns:
		tensor: Output tensor.
	"""
	return self.__neg__().__add__(rhs)

def mul(self, rhs):  # lhs * rhs
	"""Element-wise multiplication.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.var * var, device=self.device)

def div(self, rhs):  # lhs / rhs
	"""Element-wise division.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.var / var, device=self.device)

def pow(self, rhs):  # lhs ** rhs
	"""Element-wise power function.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.var ** var, device=self.device)

def rpow(self, rhs):  # rhs ** lhs
	"""Element-wise power function.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	if self.device==-1:
		var = np.array([var])
	else:
		with device_guard(self.device):
			var = cp.asarray(np.array([var]))
	return self.new(var ** self.var, device=self.device)

def rdiv(self, rhs):  # rhs / lhs
	"""Element-wise division.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(var*(self.var**-1), device=self.device)

def rdiv(self, rhs):  # rhs / lhs
	"""Element-wise division.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(var*(self.var**-1), device=self.device)

def absolute(self):  # abs(x)
	"""Element-wise absolute.
	Returns:
		tensor: Output tensor.
	"""
	return self.new(abs(self.var), device=self.device)


def eq(self, rhs):  # rhs == lhs
	"""Element-wise equal.
	Returns:
		tensor: Output tensor.
	"""
	# var = self.type_check(rhs)
	# return self.new(self.var == var, device=self.device)
	raise NotImplementedError("not implemented in chainer.Variable")

def lt(self, rhs):  # rhs < lhs
	"""Element-wise less than.
	Returns:
		tensor: Output tensor.
	"""
	# var = self.type_check(rhs)
	# return self.new(self.var < var, device=self.device)
	raise NotImplementedError("not implemented in chainer.Variable")

def le(self, rhs):  # rhs <= lhs
	"""Element-wise less equal.
	Returns:
		tensor: Output tensor.
	"""
	# var = self.type_check(rhs)
	# return self.new(self.var <= var, device=self.device)
	raise NotImplementedError("not implemented in chainer.Variable")

def gt(self, rhs):  # rhs > lhs
	"""Element-wise greater than.
	Returns:
		tensor: Output tensor.
	"""
	# var = self.type_check(rhs)
	# return self.new(self.var > var, device=self.device)
	raise NotImplementedError("not implemented in chainer.Variable")

def ge(self, rhs):  # rhs >= lhs
	"""Element-wise greater than.
	Returns:
		tensor: Output tensor.
	"""
	# var = self.type_check(rhs)
	# return self.new(self.var >= var, device=self.device)
	raise NotImplementedError("not implemented in chainer.Variable")

def ne(self, rhs):  # rhs != lhs
	"""Element-wise not equal.
	Returns:
		tensor: Output tensor.
	"""
	# var = self.type_check(rhs)
	# return self.new(self.var != var, device=self.device)
	raise NotImplementedError("not implemented in chainer.Variable")


def install_variable_arithmetics():
	VarBase.type_check = type_check
	VarBase.__neg__ = neg
	VarBase.__abs__ = absolute
	VarBase.__add__ = add
	VarBase.__radd__ = add
	VarBase.__sub__ = sub
	VarBase.__rsub__ = rsub
	VarBase.__mul__ = mul
	VarBase.__rmul__ = mul
	VarBase.__div__ = div
	VarBase.__truediv__ = div
	VarBase.__rdiv__ = rdiv
	VarBase.__rtruediv__ = rdiv
	VarBase.__pow__ = pow
	VarBase.__rpow__ = rpow
	VarBase.__gt__ = gt
	VarBase.__lt__ = lt
	VarBase.__ge__ = ge
	VarBase.__le__ = le 
	VarBase.__eq__ = eq 
	VarBase.__ne__ = ne