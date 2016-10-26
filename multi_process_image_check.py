from multiprocessing import Pool
import numpy as np
import os, time,json
from read_h5 import read_h5
def maping_check(subdir, dirs, files,sumVal,minVal,maxVal,max_key, min_key):
	count = 0
	for file in files:
		count += 1
		k1,k2,k3,k4 = file.split("-")
		key_1 = "".join((k1, k2))
		key_1 = int(key_1)
		file_path = os.path.join(subdir,file)
		data = read_h5(file_path)
		if maxVal < data.max():
			maxVal = data.max()
			max_key = key_1
		if minVal > data.min():
			minVal = data.min()
			min_key = key_1
		sumVal += data.mean()
	return minVal,maxVal,sumVal,min_key,max_key,count
def maping_check_2(file, subdir, sumVal,minVal,maxVal):
	k1,k2,k3,k4 = file.split("-")
	key_1 = "".join((k1, k2))
	key_1 = int(key_1)
	file_path = os.path.join(subdir,file)
	data = read_h5(file_path)
	if maxVal < data.max():
		maxVal = data.max()
		# max_key = key_1
	if minVal > data.min():
		minVal = data.min()
		# min_key = key_1
	sumVal += data.mean()
	if minVal < 0.0:
		if os.path.exists(file_path):
			os.remove(file_path)
			print file_path, 'removed'
	# if maxVal > 400.0:
		# print file_path
	return minVal,maxVal,sumVal
def reducing_1(results):
	sumVal,minVal,maxVal,count = 0.0, +np.inf, -np.inf, 0
	# results = np.array(results)
	for result in results:
		# print result,'result'
		if minVal > result[0]:
			minVal = result[0]
			# min_key = result[-3]
		if maxVal < result[1]:
			maxVal = result[1]
			# max_key = result[-2]
		sumVal += result[2]
		# print result[2]
		count += result[-1]
	meanVal = sumVal/count
	return minVal, maxVal, sumVal,count,meanVal
def reducing_2(results):
	sumVal,minVal,maxVal,count = 0.0, +np.inf, -np.inf, 0
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
			sumVal += result[2]
			# print result[2]
			count += 1
	# meanVal = sumVal/count
	
	return minVal,maxVal,sumVal,count


def multi_process_image(rootdir):
	pool = Pool()
	outputs = []
	sumVal,minVal,maxVal,count = 0.0, +np.inf, -np.inf, 0
	min_key,max_key = 0,0
	# for subdir, dirs, files in os.walk(rootdir):
		
		# results = [pool.apply_async(maping_check,(x,subdir,sumVal,minVal,maxVal,max_key,min_key)) for x in files]

	results = [pool.apply_async(maping_check,(subdir, dirs, files,sumVal,minVal,maxVal,max_key,min_key)) for subdir, dirs, files in os.walk(rootdir)]
	roots = [r.get() for r in results]
	# outputs.append(roots)
	return roots

def multi_process_image_2(rootdir):
	pool = Pool()
	outputs = []
	sumVal,minVal,maxVal,count = 0.0, +np.inf, -np.inf, 0
	# min_key,max_key = 0,0
	for subdir, dirs, files in os.walk(rootdir):
		results = [pool.apply_async(maping_check_2,(x,subdir,sumVal,minVal,maxVal)) for x in files]
		roots = [r.get() for r in results]

		outputs.append(roots)
	results2 = [pool.apply_async(reducing_2,(output,)) for output in outputs[1:] ]
	roots2 = [r.get() for r in results2]
		# outputs += roots
	# return outputs
	return roots2
t1 = time.time()
file_path = '/NOBACKUP/nii/typhoon_data/orig_image/'
file_path = '/NOBACKUP/nii/typhoon_data/resize-image-cubic-interpolation-224-224/'
# file_path = '/NOBACKUP/nii/cnn_data/data_test/image'	
# outputs = multi_process_image(file_path)
# minVal,maxVal,sumVal,min_key,max_key,count,meanVal = reducing_1(outputs)

outputs = multi_process_image_2(file_path)
roots2 = reducing_1(outputs)
t2 = time.time()

# print len(outputs)
# print outputs[1:]
# print outputs

print roots2
# print 'minVal,maxVal,sumVal,min_key,max_key,count,meanVal ',minVal,maxVal,sumVal,str(min_key),str(max_key),count,meanVal 
# print ('using' + str(t2 - t1) + 'time')
#using1.82275295258time method 1; using0.5691010952time method 2
# print len(outputs[0])



