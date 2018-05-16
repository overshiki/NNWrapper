import pickle, re

def save_obj(name, obj):
	if(not re.search('.pkl', name)):
		name = name+'.pkl'
	with open(name, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	if(not re.search('.pkl', name)):
		name = name+'.pkl'
	with open(name, 'rb') as f:
		return pickle.load(f)