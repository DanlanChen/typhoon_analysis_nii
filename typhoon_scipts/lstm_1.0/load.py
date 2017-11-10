
import os,json
import config	
import numpy as np
from scipy.misc import bytescale
np.random.seed(1337)
from one_hot import one_hot
from keras.utils import np_utils
from read_h5 import read_h5
from multiprocessing import Pool
import cv2
import random
def get_data_2(track_dict, file):

	[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
	key = "".join((k1,k2))
	yy = int(track_dict[key][0])
	mm = int(track_dict[key][1])
	dd = int(track_dict[key][2])
	hh = int(track_dict[key][3])
	lat = float(track_dict[key][5])
	lont = float(track_dict[key][6])
	landorsea = int(track_dict[key][-1])
	return [yy,mm,dd,hh,lat,lont,landorsea]
def rotate_image(file_path):
	# file_path = '/NOBACKUP/nii/typhoon_data/orig_image/197908/1979072512-197908-GMS1-1.h5'
	image = read_h5(file_path)
	# print image.shape

# 	image = cv2.imread('1979072512-197908-GMS1-1.jpg')
# 	print image
# 	print image.shape
	image = bytescale(image)
	(h, w) = image.shape[:2]
	center = (w / 2, h / 2)
 	a = random.uniform(0,90)
# rotate the image by 180 degrees
	M = cv2.getRotationMatrix2D(center, a, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h))
	# print rotated
	# print rotated.shape
	img_i = bytescale(rotated)
	# pic_name = '1979072512-197908-GMS1-1_rotate_90.jpg'
	# cv2.imwrite(pic_name, rotated)
	x0,x1 = 143, 367
	crop_np_data = img_i[x0:x1,x0:x1]
	# print crop_np_data
	# print crop_np_data.shape
	# pic_name = '1979072512-197908-GMS1-1_rotate_90_crop_224.jpg'
	# cv2.imwrite(pic_name, crop_np_data)
	return crop_np_data
def get_crop_224_x(file_path):
	image = read_h5(file_path)
	image = bytescale(image)
	x0,x1 = 143, 367
	crop_np_data = image[x0:x1,x0:x1]
	return crop_np_data
mean_v,std_v = config.mean_v,config.std_v
def load_json(file):
	with open(file,'r') as f:
		return json.load(f)
def get_subdirs_list(image_path):
	file_list = []
	for subdirs, dirs, files in os.walk(image_path):
		# for subdir in subdirs:
		# print subdirs
		file_list.append(subdirs)
		# for file in files:
		# 	file_path = os.path.join(subdir,file)
		# 	file_list.append(file_path)
	# print file_list
	# len_file = [len(file) for file in file_list[1:]]
	# mark = True
	# for lens in len_file:
	# 	if lens != 44:
	# 		mark =False
	# print mark
	# print len_file
	return file_list[1:]
def order_subdirs_list(subdirs_list):
	return sorted(subdirs_list, key = lambda file:int(file.split('/')[-1]))
def subtract_suspicious_list(orig_file_list, suspicious_file_list):
	return list(set(orig_file_list) - set(suspicious_file_list))

# def subtrack_from_track_file(file_list, track_dict):
# 	not_in_track = []
# 	for file in file_list:
# 		[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
# 		key = "".join((k1,k2))
# 		try:
# 			mark = track_dict[key]
# 		except KeyError:
# 			# print file,'not in track'
# 			not_in_track.append(file)
# 	# print not_in_track,'not_in_track'
# 	return list(set(file_list) - set(not_in_track))
def subtrack_from_track_file(file_list, track_dict):
	not_in_track = []
	not_include_7=[]
	not_include_6 = []
	for file in file_list:
		[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
		key = "".join((k1,k2))
		try:
			mark = track_dict[key]
			# print (mark[-2])
			if mark[-3] == "7":
				not_include_7.append(file)
			if mark[-3] == "6":
				not_include_6.append(file)
		except KeyError:
			# print file,'not in track'
			not_in_track.append(file)
	# print not_in_track,'not_in_track'
	print len(not_include_7),'not_include_7'
	print len(not_include_6),'not_include_6'
	file_list = list(set(file_list) - set(not_in_track))
	file_list = list(set(file_list) - set(not_include_7))
	file_list = list(set(file_list) - set(not_include_6))
	return file_list
def get_file_list(subdirs_list):
	file_list = []
	for image_path in subdirs_list:
		for subdir, dirs, files in os.walk(image_path):
			for file in files:
				file_path = os.path.join(subdir,file)
				# print file
				file_list.append(file_path)
	# print file_list
	return file_list
def get_train_validation_test_file_split(train_subdirs_list, validation_subdirs_list, test_subdirs_list,track_dict,suspicious_file_list,json_file):
	train_file_list = get_file_list(train_subdirs_list)
	validation_file_list = get_file_list(validation_subdirs_list)
	test_file_list = get_file_list(test_subdirs_list)
	train_file_list = subtract_suspicious_list(train_file_list,suspicious_file_list)
	validation_file_list = subtract_suspicious_list(validation_file_list,suspicious_file_list)
	test_file_list = subtract_suspicious_list(test_file_list,suspicious_file_list)
	train_file_list = subtrack_from_track_file(train_file_list,track_dict)
	validation_file_list = subtrack_from_track_file(validation_file_list,track_dict)
	test_file_list = subtrack_from_track_file(test_file_list,track_dict)
	train_file_list = np.array(train_file_list)
	validation_file_list = np.array(validation_file_list)
	test_file_list = np.array(test_file_list)
	np.random.shuffle(train_file_list)
	np.random.shuffle(validation_file_list)
	np.random.shuffle(test_file_list)
	train_file_list = list(train_file_list)
	validation_file_list = list(validation_file_list)
	test_file_list = list(test_file_list)
	# with open(json_file,'w') as f:
	# 	json.dump({'train_file_list' : train_file_list, 'test_file_list' : test_file_list,'validation_file_list' :validation_file_list},f)
	
	return train_file_list, validation_file_list,test_file_list
def get_train_test_file_split(train_subdirs_list, validation_subdirs_list, test_subdirs_list,track_dict,suspicious_file_list):
	train_subdirs_list = train_subdirs_list + validation_subdirs_list
	train_file_list = get_file_list(train_subdirs_list)
	# validation_file_list = get_file_list(validation_subdirs_list)
	test_file_list = get_file_list(test_subdirs_list)
	train_file_list = subtract_suspicious_list(train_file_list,suspicious_file_list)
	# validation_file_list = subtract_suspicious_list(validation_file_list,suspicious_file_list)
	test_file_list = subtract_suspicious_list(test_file_list,suspicious_file_list)
	train_file_list = subtrack_from_track_file(train_file_list,track_dict)
	# validation_file_list = subtrack_from_track_file(validation_file_list,track_dict)
	test_file_list = subtrack_from_track_file(test_file_list,track_dict)
	train_file_list = np.array(train_file_list)
	# validation_file_list = np.array(validation_file_list)
	test_file_list = np.array(test_file_list)
	np.random.shuffle(train_file_list)
	# np.random.shuffle(validation_file_list)
	np.random.shuffle(test_file_list)
	train_file_list = list(train_file_list)
	# validation_file_list = list(validation_file_list)
	test_file_list = list(test_file_list)
	# with open(json_file,'w') as f:
	# 	json.dump({'train_file_list' : train_file_list, 'test_file_list' : test_file_list,'validation_file_list' :validation_file_list},f)
	
	return train_file_list, test_file_list

def load_train_validation_test_file_list(json_file):
	with open(json_file,'r')  as f:
		train_test = json.load(f)
		train_file_list = train_test['train_file_list']
		test_file_list = train_test['test_file_list']
		validation_file_list = train_test['validation_file_list']
	return train_file_list, validation_file_list, test_file_list

def one_hot_y(y_train, y_valid,y_test):
	nb_classes = None
	y_train = np.array(y_train, dtype = 'float32')
	y_test = np.array(y_test, dtype = 'float32')
	y_valid = np.array(y_valid,dtype = 'float32')
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_valid = np_utils.to_categorical(y_valid, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	return Y_train, Y_valid,Y_test
	# print(y_train,'y_train')
	# print (y_test,'y_test')
	# y_train_len = len(y_train)
	# y = y_train + y_test
	# y = np.array(y)
	# one_hot_y = one_hot(y, num_labels = 7)
	# y_train_one_hot = one_hot_y[:y_train_len]
	# y_test_one_hot = one_hot_y[y_train_len:]
	# y_train_one_hot = list(y_train_one_hot)
	# y_test_one_hot = list(y_test_one_hot)
	# return y_train_one_hot, y_test_one_hot
def float_y(y_train, y_valid, y_test):
	y_train = np.array(y_train,dtype = 'float32')
	y_valid = np.array(y_valid, dtype = 'float32')
	y_test = np.array(y_test, dtype = 'float32')
	y_train = list(y_train)
	y_valid = list(y_valid)
	y_test = list(y_test)
	return y_train, y_valid, y_test
# def get_input_2(file_list, trackDictPath):
# 	data =[]
# 	with open(trackDictPath,'r') as f:
# 		track_dict = json.load(f)
# 	for file in file_list:
# 		[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
# 		key = "".join((k1,k2))
# 		latitude = float(track_dict[key][2])
# 		longtitude = float(track_dict[key][3])
# 		time = track_dict[key][0]
# 		year = time[:4]
# 		month = time[4:6]
# 		day = time[6:8]
# 		hour = time[8:]
# 		print (year,'year',month,'month',day,'day',hour,'hour')
# 		print (latitude,'latitude',longtitude,'longtitude')
# 		break
# 		x = [year,month,day,hour,longtitude,latitude]
# 		data.append(x)
# 	return data

def get_y_file(file,track_Dict,yType):
	
	[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
	key = "".join((k1,k2))
	if yType =='type':
		y = track_Dict[key][7]
	if yType =='tropicalornot':
		if track_Dict[key][7] == "6":
				y = 0
		else:
				y = 1
	if yType == 'intensity':
		y = track_Dict[key][-2]
	return y

def get_y(train_file_list, trackDictPath, yType):
	y_train =[]
	with open(trackDictPath,'r') as f:
		track_Dict = json.load(f)
	if yType =='type':
		for file in train_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
			key = "".join((k1,k2))
			y_train.append(track_Dict[key][7])
		get_statistic_y(y_train)
		y_train = np.array(y_train, dtype = 'float32')
		y_train = np_utils.to_categorical(y_train, None)
	if yType =='tropicalornot':
		for file in train_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
			key = "".join((k1,k2))
			if track_Dict[key][7] == "6":
				y_train.append(0)
			else:
				y_train.append(1)
		get_statistic_y(y_train)
		y_train = np.array(y_train, dtype = 'float32')
		y_train = np_utils.to_categorical(y_train, None)
	if yType == 'intensity':
		for file in train_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4: -2]
			key = "".join((k1,k2))
			y_train.append(track_Dict[key][-2])
		y_train = np.array(y_train,dtype = 'float32')
	return y_train
def get_train_validation_test_y(train_file_list, validation_file_list,test_file_list,trackDictPath, yType):
	y_train =[]
	y_valid = []
	y_test = []
	with open(trackDictPath,'r') as f:
		track_Dict = json.load(f)
	if yType =='type':
		for file in train_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
			key = "".join((k1,k2))
			try:
				y_train.append(track_Dict[key][7])
			except:
				print file
		for file in validation_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
			key = "".join((k1,k2))
			y_valid.append(track_Dict[key][7])

		for file in test_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4: -2]
			key = "".join((k1,k2))
			y_test.append(track_Dict[key][4])
		get_statistic_y(y_train)
		get_statistic_y(y_valid)
		get_statistic_y(y_test)
		y_train, y_valid, y_test = one_hot_y(y_train, y_valid, y_test)

	if yType =='tropicalornot':
		for file in train_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
			key = "".join((k1,k2))
			try:
				if track_Dict[key][7] == "6":
					y_train.append(0)
				else:
					y_train.append(1)
			except:
				print file
		for file in validation_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4:-2]
			key = "".join((k1,k2))
			if track_Dict[key][7] == "6":
					y_valid.append(0)
			else:
					y_valid.append(1)

		for file in test_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4: -2]
			key = "".join((k1,k2))
			if track_Dict[key][7] == "6":
					y_test.append(0)
			else:
					y_test.append(1)
		get_statistic_y(y_train)
		get_statistic_y(y_valid)
		get_statistic_y(y_test)
		y_train, y_valid, y_test = one_hot_y(y_train, y_valid, y_test)


	if yType == 'intensity':
		for file in train_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4: -2]
			key = "".join((k1,k2))
			y_train.append(track_Dict[key][-2])

		for file in validation_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4: -2]
			key = "".join((k1,k2))
			y_valid.append(track_Dict[key][-2])

		for file in test_file_list:
			[k1, k2] = file.split("/")[-1].split("-")[-4: -2]
			key = "".join((k1,k2))
			y_test.append(track_Dict[key][-2])
		y_train, y_valid, y_test = float_y(y_train, y_valid, y_test)
	return y_train, y_valid, y_test
def normalize_image(data,mean_v,std_v):
	return (data - mean_v)/std_v
# def get_x(file_list):
# 	x = []
# 	for file in file_list:
# 		data = get_crop_224_x(file)
# 		x.append(data)
# 	x = np.array(x)
# 	print x.shape
# 	x = np.reshape(x,(-1,1,224,224))
# 	return x
# def get_x(file_list,img_rows,img_cols,mean_v,std_v):
# 	x = []
# 	for file in file_list:
# 		data = read_h5(file)
#  		data = normalize_image(data,mean_v,std_v)
# 		data = np.array(data)
# 		data = np.reshape(data,(1,img_rows,img_cols))
# 		x.append(data)
# 	x = np.array(x)
# 	x = np.reshape(x,(-1,1,img_rows,img_cols))
# 	return x
def get_x(file_list,img_rows,img_cols,mean_v,std_v):
	x = []
	for file in file_list:
		data = read_h5(file)
 		data = normalize_image(data,mean_v,std_v)
		data = np.array(data)
		data = np.array([data]*3)
		# data = np.reshape(data,(1,img_rows,img_cols))
		# print (data.shape)
		x.append(data)
	x = np.array(x,dtype ='float32')
	# print (x.shape)
	x = np.reshape(x,(-1,3,img_rows,img_cols))
	return x
def get_single_x(file_path,img_rows,img_cols,mean_v,std_v):
	data = read_h5(file_path)
	data = normalize_image(data,mean_v,std_v)
	data = np.array(data)
	data = np.array([data]*3)
	return data
def get_chunk(images,tracks,img_rows,img_cols,num_labels):
	while True:
		for stepStart in range(len(images)):
	 		file = images[stepStart]
	 		data = read_h5(file)
	 		data = normalize_image(data,mean_v,std_v)
			data = np.array(data)
			data = np.reshape(data,(1,1,img_rows,img_cols))
			# print dats.shape
			y = np.reshape(np.array(tracks[stepStart]),(1,num_labels))
			# y = tracks[stepStart]
			# print y.shape
			yield (data,y)
def get_test_chunk(images,img_rows,img_cols):
	while True:
		for stepStart in range(len(images)):
	 		file = images[stepStart]
	 		data = read_h5(file)
	 		data = normalize_image(data,mean_v,std_v)
			data = np.array(data)
			data = np.reshape(data,(-1,1,img_rows,img_cols))
			yield data	
#new
def split_subdirs(subdirs_list,subdirs_list_json):
	len_subdir_list = len(subdirs_list)
	
	split_indice_1 = int(len(subdirs_list) *0.1)
	split_indice_2 = int(0.05*(len(subdirs_list) - split_indice_1))
	subdirs_list = np.array(subdirs_list)
	np.random.shuffle(subdirs_list)
	subdirs_list = list(subdirs_list)
	test_subdirs_list = subdirs_list[:split_indice_1]
	validation_subdirs_list = subdirs_list[split_indice_1:split_indice_1 + split_indice_2]
	train_subdirs_list = subdirs_list[split_indice_1 + split_indice_2:]
	# print len(train_subdirs_list),len(validation_subdirs_list),len(test_subdirs_list)
	print train_subdirs_list,'train'
	print validation_subdirs_list,'validation'
	print test_subdirs_list,'test'
	# with open(subdirs_list_json,'w') as f:
	# 	json.dump({'train_subdirs_list':train_subdirs_list,'validation_subdirs_list':validation_subdirs_list,'test_subdirs_list':test_subdirs_list},f)

	return train_subdirs_list, validation_subdirs_list, test_subdirs_list
# def split_subdirs(subdirs_list,subdirs_list_json):
# 	len_subdir_list = len(subdirs_list)
# 	print len_subdir_list,'len_subdir_list'
# 	number_each_chunk = len_subdir_list/10
# 	chunks = [subdirs_list[x: x + number_each_chunk] for x in xrange(0,len(subdirs_list),number_each_chunk)]
# 	# print chunks,'chunks'
# 	train_subdirs_list = []
# 	validation_subdirs_list = []
# 	test_subdirs_list = []
# 	for chunk in chunks:
# 		chunk = np.array(chunk)
# 		# print chunk,'chunk'
# 		np.random.shuffle(chunk)
# 		test_divide = int(chunk.shape[0] * 0.1)
# 		# print test_divide,'test_divide'
# 		validation_divide = int((len(chunk) - test_divide) * 0.05)
# 		# print validation_divide,'validation_divide'
# 		test_subdirs_list += list(chunk[:test_divide])
# 		validation_subdirs_list += list(chunk[test_divide: test_divide + validation_divide])
# 		train_subdirs_list += list(chunk[test_divide + validation_divide:])
# 	print len(train_subdirs_list),'len_train_subdirs_list'
# 	print len(test_subdirs_list),'len_test_subdirs_list'
# 	print len(validation_subdirs_list),'validation_subdirs_list'
# 	train_subdirs_list = np.array(train_subdirs_list)
# 	validation_subdirs_list = np.array(validation_subdirs_list)
# 	test_subdirs_list = np.array(test_subdirs_list)
# 	np.random.shuffle(train_subdirs_list)
# 	np.random.shuffle(validation_subdirs_list)
# 	np.random.shuffle(test_subdirs_list)
# 	train_subdirs_list = list(train_subdirs_list)
# 	test_subdirs_list = list(test_subdirs_list)
# 	validation_subdirs_list = list(validation_subdirs_list)
# 	# split_indice_1 = int(len(subdirs_list) *0.1)
# 	# split_indice_2 = int(0.05*(len(subdirs_list) - split_indice_1))
# 	# subdirs_list = np.array(subdirs_list)
# 	# np.random.shuffle(subdirs_list)
# 	# subdirs_list = list(subdirs_list)
# 	# test_subdirs_list = subdirs_list[:split_indice_1]
# 	# validation_subdirs_list = subdirs_list[split_indice_1:split_indice_1 + split_indice_2]
# 	# train_subdirs_list = subdirs_list[split_indice_1 + split_indice_2:]
# 	# print len(train_subdirs_list),len(validation_subdirs_list),len(test_subdirs_list)
# 	print train_subdirs_list,'train'
# 	print validation_subdirs_list,'validation'
# 	print test_subdirs_list,'test'
# 	with open(subdirs_list_json,'w') as f:
# 		json.dump({'train_subdirs_list':train_subdirs_list,'validation_subdirs_list':validation_subdirs_list,'test_subdirs_list':test_subdirs_list},f)

# 	return train_subdirs_list, validation_subdirs_list, test_subdirs_list
def get_split_subdirs(json_file):
	with open(json_file,'r')  as f:
		train_test = json.load(f)
		train_subdirs_list = train_test['train_subdirs_list']
		test_subdirs_list = train_test['test_subdirs_list']
		validation_subdirs_list = train_test['validation_subdirs_list']
	print len(train_subdirs_list),len(validation_subdirs_list),len(test_subdirs_list)
	return train_subdirs_list, validation_subdirs_list, test_subdirs_list
def get_statistic_y(yy):
	# dicts={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
	dicts={}
	dicts_ratio={}
	# y0 = 0
	# y1 = 0
	for y in yy:
		if y not in dicts.keys():
			# print type(y)
			# print y
			dicts[y] = 1.0
		else:
			dicts[y]+=1.0
	for k,v in dicts.iteritems():
		dicts_ratio[k] = v /len(yy)
		# if y ==0.0:
		# 	y0 +=1.0
		# else:
		# 	y1 = 1.0
	# print y0/len(yy),y1/len(yy),'0 1 ratio'
	print dicts
	print dicts_ratio


def main():
	train_test_file_list_path = config.train_test_file_path_divid
	image_path = config.image_path
	suspicious_file_list_path = config.suspicious_file_list_path
	suspicious_file_list = load_json(suspicious_file_list_path)
	train_validation_test_subdirs_split = config.train_validation_test_subdirs_split
	trackDictPath = config.track_dic_path
	track_dict = load_json(trackDictPath)
	
	if not os.path.exists(train_validation_test_subdirs_split):
		print 'subdirs not split'
		subdirs_list = get_subdirs_list(image_path)
		subdirs_list = order_subdirs_list(subdirs_list)
		print subdirs_list,'order_subdirs_list'
		train_subdirs_list, validation_subdirs_list, test_subdirs_list = split_subdirs(subdirs_list,train_validation_test_subdirs_split)
	else:
		print 'subdirs splitted'

		train_subdirs_list, validation_subdirs_list, test_subdirs_list= get_split_subdirs(train_validation_test_subdirs_split)
	


	# file_list = subtract_suspicious_list(file_list,suspicious_file_list)
	# trackDictPath = config.track_dic_path
	yType = config.yType
	if not os.path.exists(train_test_file_list_path):
		print 'file_list not splited'
		train_file_list ,validation_file_list,test_file_list =  get_train_validation_test_file_split(train_subdirs_list, validation_subdirs_list, test_subdirs_list,track_dict,suspicious_file_list,train_test_file_list_path)
	else:
		print 'file list splitted'
		train_file_list, validation_file_list,test_file_list = load_train_validation_test_file_list(train_test_file_list_path)
	# # print len(file_list)
	print len(train_file_list)
	print len(validation_file_list)
	print len(test_file_list)
	y_train, y_valid, y_test = get_train_validation_test_y(train_file_list,validation_file_list, test_file_list,trackDictPath, yType)
	print ('y_train',len(y_train))
	print ('y_valid', len(y_valid))
	print ('y_test',len(y_test))
	# get_statistic_y(y_train)
	# get_statistic_y(y_valid)
	# get_statistic_y(y_test)
	# print (type(y_train))

if __name__ == "__main__":
	# main()
	pass

