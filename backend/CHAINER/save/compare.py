import os, re
from IO.basic import load_obj
import numpy as np

def compare(d1, d2):
	_bool = []
	_key = []
	for key in d1.keys():
		_bool.append((d1[key]==d2[key]).all())
		_key.append(key)
	_bool = np.array(_bool).astype(np.bool_)
	# print(_bool)
	return [_key[x] for x in np.where(_bool==False)[0]]

if __name__ == '__main__':

	# compare(d1, d2)
	for i in range(9):
		for j in range(i+1, 9):
			d1 = load_obj("{}.pkl".format(i))
			d2 = load_obj("{}.pkl".format(j))
			# print(compare(d1, d2))
			result = compare(d1, d2)
			liner_check = False
			for res in result:
				if bool(re.search("linear", res)):
					liner_check = True 
			pool_check = False
			for res in result:
				if bool(re.search("pool", res)):
					pool_check = True 
			print("linear: ", liner_check, "pool: ", pool_check)

	print("#"*100)

	for i in range(10, 20):
		for j in range(i+1, 20):
			d1 = load_obj("{}.pkl".format(i))
			d2 = load_obj("{}.pkl".format(j))
			# print(compare(d1, d2))
			result = compare(d1, d2)
			liner_check = False
			for res in result:
				if bool(re.search("linear", res)):
					liner_check = True 
			pool_check = False
			for res in result:
				if bool(re.search("pool", res)):
					pool_check = True 
			print("linear: ", liner_check, "pool: ", pool_check)