from .. import TensorBase, np

def type_check(self, other):
	number_check = False
	if isinstance(other, TensorBase):
		var = other
	elif np.isscalar(other):
		var = other
		number_check = True
	else:
		raise TypeError("input other is not numpy cupy ndarray nor tensor nor python variable, but to be: {}".format(type(other)))

	if number_check==False:
		if var.device!=self.device:
			var.to_device(self.device)
		return var.ndarray
	else:
		return var

# @device_guard_decorator
def add(self, rhs):  # lhs + rhs
	"""Element-wise addition.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray + var, device=self.device)

# @device_guard_decorator
def neg(self): #-x
	"""Element-wise negation.
	Returns:
		tensor: Output tensor.
	"""	
	return self.new((-1)*self.ndarray, device=self.device)

# @device_guard_decorator
def sub(self, rhs):  # lhs - rhs
	"""Element-wise subtraction.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray - var, device=self.device)

# @device_guard_decorator
def rsub(self, rhs): # rhs - lhs 
	"""Element-wise subtraction.
	Returns:
		tensor: Output tensor.
	"""
	return self.__neg__().__add__(rhs)

# @device_guard_decorator
def mul(self, rhs):  # lhs * rhs
	"""Element-wise multiplication.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray * var, device=self.device)

# @device_guard_decorator
def div(self, rhs):  # lhs / rhs
	"""Element-wise division.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray / var, device=self.device)

# @device_guard_decorator
def pow(self, rhs):  # lhs ** rhs
	"""Element-wise power function.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray ** var, device=self.device)

# @device_guard_decorator
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
	return self.new(var ** self.ndarray, device=self.device)

# @device_guard_decorator
def rdiv(self, rhs):  # rhs / lhs
	"""Element-wise division.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(var*(self.ndarray**-1), device=self.device)

# @device_guard_decorator
def rdiv(self, rhs):  # rhs / lhs
	"""Element-wise division.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(var*(self.ndarray**-1), device=self.device)

# @device_guard_decorator
def absolute(self):  # abs(x)
	"""Element-wise absolute.
	Returns:
		tensor: Output tensor.
	"""
	return self.new(self.ndarray.abs(), device=self.device)

# @device_guard_decorator
def eq(self, rhs):  # rhs == lhs
	"""Element-wise equal.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray == var, device=self.device)

# @device_guard_decorator
def lt(self, rhs):  # rhs < lhs
	"""Element-wise less than.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray < var, device=self.device)

# @device_guard_decorator
def le(self, rhs):  # rhs <= lhs
	"""Element-wise less equal.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray <= var, device=self.device)

# @device_guard_decorator
def gt(self, rhs):  # rhs > lhs
	"""Element-wise greater than.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray > var, device=self.device)

# @device_guard_decorator
def ge(self, rhs):  # rhs >= lhs
	"""Element-wise greater than.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray >= var, device=self.device)

# @device_guard_decorator
def ne(self, rhs):  # rhs != lhs
	"""Element-wise not equal.
	Returns:
		tensor: Output tensor.
	"""
	var = self.type_check(rhs)
	return self.new(self.ndarray != var, device=self.device)


def install_variable_arithmetics():
	TensorBase.type_check = type_check
	TensorBase.__neg__ = neg
	TensorBase.__abs__ = absolute
	TensorBase.__add__ = add
	TensorBase.__radd__ = add
	TensorBase.__sub__ = sub
	TensorBase.__rsub__ = rsub
	TensorBase.__mul__ = mul
	TensorBase.__rmul__ = mul
	TensorBase.__div__ = div
	TensorBase.__truediv__ = div
	TensorBase.__rdiv__ = rdiv
	TensorBase.__rtruediv__ = rdiv
	TensorBase.__pow__ = pow
	TensorBase.__rpow__ = rpow
	TensorBase.__gt__ = gt
	TensorBase.__lt__ = lt
	TensorBase.__ge__ = ge
	TensorBase.__le__ = le 
	TensorBase.__eq__ = eq 
	TensorBase.__ne__ = ne