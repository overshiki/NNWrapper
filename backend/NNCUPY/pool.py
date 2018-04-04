import multiprocessing as mp
from BASIC_LIST.basic import groupby
from IO.basic import save_obj
from CSV.basic import list2csv

def task_parallel(f, jobs, device_list=[0,1,2,3]):
	'''
	jobs is list where each element is params for f
	f still need device parameters as keyword parameter
	'''
	jobs_group = groupby(jobs, len(device_list), key='num')
	jobs, results = jobs_claim(f, jobs_group, device_list=device_list)


def jobs_claim(f, args, device_list=[0,1,2,3]):
	'''
	args is 2 level list, where the deepest level is the parameter list
	'''

	def pool_wrapper(*args):
		args = list(args)
		device = args.pop(-1)	
		for arg in args:
			print("arg", [type(x) for x in arg], "device: ", device)
			f(*arg, device=device)
			# save_obj(arg, "/home/polaris-nn/CODE/Polaris_v5/NN/NNWrapper/log/{}".format(device))
			# list2csv(arg, "/home/polaris-nn/CODE/Polaris_v5/NN/NNWrapper/log/{}".format(device))
			# print(arg)
		# print(args)

	ctx = mp.get_context('fork')
	jobs = []
	results = []
	for index, (device, param) in enumerate(zip(device_list, args)):
		param.append(device)
		p = ctx.Process(name='device {}'.format(device), target=pool_wrapper, args=param)
		p.daemon = True
		jobs.append(p)
		p.start()
		# print(index, device)

	for job in jobs:
		job.join()

	return jobs, results

