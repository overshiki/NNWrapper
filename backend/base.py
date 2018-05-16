r'''
base interface with some common method for modules
'''
from ..utils.serialization import save_obj, load_obj

class base:
	def __init__(self, *wrags, **kwrags):
		pass

	def save_params(self, name):
		param_dict = self.params_to_dict()
		save_obj(name, param_dict)

	def load_params(self, name):
		return load_obj(name)

	def namedparamsOut(self):
		for name, param in self.namedparams():
			yield self.paramNameOut(name), param

	def namedparamsIn(self, param_dict):
		for name, param in param_dict.items():
			yield self.paramNameIn(name), param

