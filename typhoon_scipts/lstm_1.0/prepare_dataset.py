import os,csv,load
import numpy as np
from read_h5 import read_h5
import config
img_rows,img_cols,mean_v,std_v= config.img_rows,config.img_cols,config.mean_v,config.std_v
""""
the following code is supposed the image path has exact the number of track data
"""
def dataset_1(file_path):
	intensity = []
	# for subdir,dirs,files in os.walk(folder):
	# 	for file in files:
	# 		file_path = os.join(subdir,file)
	with open(file_path,'rb') as tsv_file:
		tsv_reader = csv.reader(tsv_file, delimiter='\t')
		for row in tsv_reader:
			intensity.append(float(row[7]))
	intensity = np.array(intensity)
	return intensity
def dataset_1_type(file_path):
	types = []
	# for subdir,dirs,files in os.walk(folder):
	# 	for file in files:
	# 		file_path = os.join(subdir,file)
	with open(file_path,'rb') as tsv_file:
		tsv_reader = csv.reader(tsv_file, delimiter='\t')
		for row in tsv_reader:
			types.append(float(row[4]))
	types = np.array(types)
	return types
def dataset_2(image_path):
	file_path_list = []
	for  subdirs, dirs, files in os.walk(image_path):
		for file in files:
			file_path = os.path.join(subdirs,file)
			file_path_list.append(file_path)
	# sorted_file_list = sorted(file_path_list,key = lambda x : int(x.split('/')[-1].split('-')[0]))
	sorted_file_list = sorted(file_path_list,key = lambda x : int(x.split('/')[-1].split('-')[-4]))
	return np.array(load.get_x(sorted_file_list,img_rows,img_cols,mean_v,std_v))
def create_dataset_y_zero(dataset_y,look_back=1):
	dataY=[]
	for i in range(len(dataset_y)-look_back+1):
		dataY.append(dataset_y[i+look_back-1])
	return dataY
# def create_dataset(dataset, look_back=1):
# 	# print dataset,'dataset'
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-look_back+1):
# 		a = dataset[i:(i+look_back)]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back:i+2*look_back])
# 	return dataX, dataY
	# return numpy.array(dataX), numpy.array(dataY)

def create_dataset(dataset, look_back=1):
	# print dataset,'dataset'
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return dataX, dataY
def normalize_intensity(intensity, mean,std):
	return (intensity-mean)/std
def reverse_normalize_intensity(intensity,mean,std):
	return intensity*std + mean
def create_dataset_2(dataset_x,dataset_y,look_back=1):
	#dataset_x =load.get_x()
	#dataset_y ~ load.get_y()
	dataX, dataY = [], []
	for i in range(len(dataset_y)-look_back):
		a = dataset_x[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset_y[i+look_back])
	return dataX,dataY
def create_dataset_2_zero(dataset_x,dataset_y,look_back=1):
	#dataset_x =load.get_x()
	#dataset_y ~ load.get_y()
	dataX, dataY = [], []
	for i in range(len(dataset_y)-look_back+1):
		a = dataset_x[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset_y[i+look_back-1])
	return dataX,dataY
def train_generator(dataset_x,dataset_y,look_back=1):
	while True:
		for i in range(len(dataset_y)-look_back):
			a = dataset_x[i:(i+look_back)]
			b = dataset_y[i+look_back]
			yield (a,b)
def test_generator(dataset_x,look_back =1):
	while True:
		for i in range(len(dataset_y)-look_back-1):
			a = dataset_x[i:(i+look_back)]
			yield a
def extend_dataset_2_zero(dataset_x,dataset_y,look_back = 1):
	dataX, dataY = [], []
	for i in range(20):
		a = dataset_x[0:look_back]
		b = dataset_y[look_back-1]
		dataX.append(a)
		dataY.append(b)
	for i in range(len(dataset_y)-look_back+1):
		a = dataset_x[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset_y[i+look_back-1])
	return dataX,dataY
def extend_dataset_2(dataset_x,dataset_y,look_back = 1):
	dataX, dataY = [], []
	for i in range(20):
		a = dataset_x[0:look_back]
		b = dataset_y[look_back]
		dataX.append(a)
		dataY.append(b)
	for i in range(len(dataset_y)-look_back):
		a = dataset_x[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset_y[i+look_back])
	return dataX,dataY
def extend_dataset_2_zero_method_2(dataset_x,dataset_y,look_back =1):
	dataset_x =dataset_x[0]*36+dataset_x
	dataset_y = dataset_y[0]*36 + dataset_y
	dataX, dataY = [], []
	for i in range(len(dataset_y)-look_back+1):
		a = dataset_x[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset_y[i+look_back-1])
	return dataX,dataY
# def extend_dataset_3_zero()
