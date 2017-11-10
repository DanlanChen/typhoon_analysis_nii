from multiprocessing import Pool
import numpy as np
import os, time,json,math
# from read_h5 import read_h5
import h5py,config
def maping_check_2(file, subdir, mean_sumVal, var_sumVal, minVal,maxVal,suspected_file_list):
	# print'mappingcheck'
	# k1,k2,k3,k4 = file.split("-")
	# key_1 = "".join((k1, k2))
	# key_1 = int(key_1)
	file_path = os.path.join(subdir,file)
	#read h5 
	f = h5py.File(file_path, 'r')
	data = np.array(f.get(f.keys()[0]))
	f.close()
	# if maxVal < data.max():
	# 	maxVal = data.max()
	# if minVal > data.min():
	# 	minVal = data.min()
	# mean_sumVal += data.mean()
	# var_sumVal += np.var(data)
	# data = read_h5(file_path)
	# print data,'data'

	if data.max() > 400.0 :
		if os.path.exists(file_path):
			print data.max(),'maxVal suspicious'
			
			# print file_path, 'suspicious'
			suspected_file_list.append(file_path)
	if data.min() < 0.0:
		if os.path.exists(file_path):
			# print data.max(),'maxVal suspicious'
			print data.min(),'minVal suspicious'
			# print file_path, 'suspicious'
			suspected_file_list.append(file_path)
	if maxVal < data.max() and data.max() < 400.0:
		maxVal = data.max()
		# max_key = key_1
	if minVal > data.min() and data.min() > 0.0:
		minVal = data.min()
		# min_key = key_1
	if data.max() < 400.0 and data.min() >0.0:
		mean_sumVal += data.mean()
		var_sumVal += np.var(data)
	return minVal,maxVal,mean_sumVal, var_sumVal, suspected_file_list

def reducing_1(results):
	mean_sumVal, var_sumVal, minVal,maxVal,count,suspected_file_list = 0.0, 0.0,+np.inf, -np.inf, 0.0,[]
	# results = np.array(results)
	for result in results:
		# print result,'result'
		if minVal > result[0]:
			minVal = result[0]
			# min_key = result[-3]
		if maxVal < result[1]:
			maxVal = result[1]
			# max_key = result[-2]
		mean_sumVal += result[2]
		var_sumVal += result[3]
		# print result[2]
		count += result[-2]
		suspected_file_list += result[-1]

	meanVal = mean_sumVal/count
	var = var_sumVal/count
	std = math.sqrt(var)
	return minVal, maxVal, mean_sumVal,var_sumVal,count,meanVal,var,std,suspected_file_list
def reducing_2(results):
	mean_sumVal,var_sumVal,minVal,maxVal,count,suspected_file_list = 0.0,0.0, +np.inf, -np.inf, 0.0,[]
	# results = np.array(results)
	for result in results:
		# print result,'result'
		if len(result) > 0:
			if minVal > result[0]:
				minVal = result[0]
				# min_key = result[-2]
			if maxVal < result[1]:
				maxVal = result[1]
				# max_key = result[-1]
			mean_sumVal += result[2]
			var_sumVal += result[3]
			# print result[2]
			if len(result[4]) == 0:
				count += 1.0
			if len(result[4]) != 0:
				suspected_file_list += result[4]
				# suspected_file_list.append(result[4])
	# meanVal = sumVal/count
	
	return minVal, maxVal, mean_sumVal, var_sumVal, count, suspected_file_list
def multi_process_image_2(rootdir):
	pool = Pool()
	outputs = []
	mean_sumVal, var_sumVal, minVal,maxVal,suspected_file_list, = 0.0, 0.0, +np.inf, -np.inf, []
	# min_key,max_key = 0,0
	for subdir, dirs, files in os.walk(rootdir):
		# print files,'files'
		results = [pool.apply_async(maping_check_2,(x,subdir,mean_sumVal, var_sumVal, minVal,maxVal,suspected_file_list)) for x in files]
		# print results,'results'
		roots = [r.get() for r in results]
		outputs.append(roots)
	results2 = [pool.apply_async(reducing_2,(output,)) for output in outputs[1:] ]
	roots2 = [r.get() for r in results2]
		# outputs += roots
	# return outputs
	return roots2

def main():
	t1 = time.time()
	file_path = config.image_path
	suspected_file_path = config.suspicious_file_list_path
	outputs = multi_process_image_2(file_path)
	roots2 = reducing_1(outputs)
	t2 = time.time()

	# print len(outputs)
	# print outputs[1:]
	# print outputs
	#print file_path
	print roots2[:-1],'minVal, maxVal, mean_sumVal,var_sumVal,count,meanVal,var,std'
	suspected_file_list = roots2[-1]
	print suspected_file_list,'suspected_file_list'
	with open(suspected_file_path,'w') as f:
		json.dump(suspected_file_list,f)
		print ('load into ' + suspected_file_path)
	# print 'minVal,maxVal,sumVal,min_key,max_key,count,meanVal ',minVal,maxVal,sumVal,str(min_key),str(max_key),count,meanVal 
	print ('using' + str(t2 - t1) + 'time')
	#using1.82275295258time method 1; using0.5691010952time method 2
	# print len(outputs[0])
if __name__ == "__main__":
	main()


