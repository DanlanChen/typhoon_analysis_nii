import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import MaxPooling2D,Dropout
from keras.layers.convolutional import Convolution2D,Convolution3D,ZeroPadding2D
# from keras.layers.recurrent_convolutional import LSTMConv2D
from keras.layers.core import Dense,Flatten
from keras.layers.recurrent import LSTM
# from recurrent_convolutional import LSTMConv2D
from keras import backend as K
import os,config,load,csv,datetime,json
# K.set_image_dim_ordering('tf')
import time,h5py,random
import prepare_dataset,math
import numpy as np
from keras.layers.wrappers import TimeDistributed
from sklearn.metrics import mean_squared_error
from vgg_pretrain_model import VGG_16, get_fc2
from keras.optimizers import SGD,Adam,Adadelta
from scipy.misc import bytescale
import cv2
def conv_lstm(look_back,img_row,img_col,batch_size):
	seq = Sequential()
	seq.add(LSTMConv2D(nb_filter=15,nb_row=2,nb_col=2,batch_input_shape=(batch_size,look_back,img_row,img_col,1),
	border_mode="same",return_sequences=True,stateful = True))
	seq.add(LSTMConv2D(nb_filter=15,nb_row=2,nb_col=2,
	border_mode="same",return_sequences=True, stateful =True))
	seq.add(Convolution3D(nb_filter=1,kernel_dim1=1,kernel_dim2=2,kernel_dim3=2
	,activation='sigmoid',
	border_mode="same",dim_ordering="tf"))
	seq.add(Flatten())
	seq.add(Dense(1024))
	seq.add(Dense(1))
	seq.compile(loss='mean_squared_error', optimizer='adam')
	return seq
def conv_lstm_2(look_back,img_row,img_col,batch_size):
	model = Sequential()
	# model.add(TimeDistributed(ZeroPadding2D(1,1), input_shape=(look_back,1,img_row,img_col)))
	model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu'), input_shape = (look_back,1,img_row,img_col),batch_input_shape=(batch_size,look_back, 1, img_row, img_col)))
	model.add(TimeDistributed(Convolution2D(64, 3, 3,activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
	model.add(TimeDistributed(Convolution2D(128, 3, 3,activation='relu')))
	model.add(TimeDistributed(Convolution2D(128, 3, 3, activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
	model.add(TimeDistributed(Convolution2D(256, 3, 3, activation='relu')))
	model.add(TimeDistributed(Convolution2D(256, 3, 3, activation='relu')))
	model.add(TimeDistributed(Convolution2D(256, 3, 3, activation='relu')))
	model.add(TimeDistributed(Convolution2D(256, 3, 3, activation='relu')))
	model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu')))
	model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu')))
	model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
	model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu')))
	model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu')))
	model.add(TimeDistributed(Convolution2D(512, 3, 3, activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
	# model.add(TimeDistributed(Dropout(0.2)))
	model.add(Dropout(0.2))
	# model.add(TimeFlatten())
	model.add(TimeDistributed(Flatten()))
	model.add(TimeDistributed(Dense(1024, activation='relu')))
	model.add(TimeDistributed(Dense(1024, activation='relu')))
	# model.add(LSTM(300))
	model.add(LSTM(300,stateful= True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	# seq.compile(loss="binary_crossentropy",optimizer="adadelta")
def pretrain_model(look_back,batch_size):
	model =Sequential()
	model.add(LSTM(300,stateful= True,batch_input_shape=(batch_size,look_back,4096)))
	model.add(Dense(1))
	# adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	adadelta = Adadelta(lr=0.0001, rho=0.95, epsilon=1e-08, decay=0.0000001)
	model.compile(loss='mean_squared_error', optimizer=adadelta)
	return model
def show_image(np_data,pic_name):
	img_i = bytescale(np_data)
	cv2.imwrite(pic_name, img_i)
def main():
	np.random.seed(7)
	t1 = time.time()
	image_path = config.image_path
	track_path = config.track_path
	track_dic_path = config.track_dic_path
	track_dict = load.load_json(track_dic_path)
	intensity_mean,intensity_std = config.intensity_mean, config.intensity_std
	batch_size = config.batch_size
	ModelCheckpoint_file = config.ModelCheckpoint_file
	look_back = config.look_back
	img_rows,img_cols = config.img_rows,config.img_cols
	subdir_list = []
	hist_path = config.hist_path
	mean_v,std_v = config.mean_v,config.std_v
	intensity_mean,intensity_std = config.intensity_mean , config.intensity_std
	model = pretrain_model(look_back,batch_size)
	if os.path.exists(ModelCheckpoint_file):
		print ('load  load_weights',ModelCheckpoint_file)
		model.load_weights(ModelCheckpoint_file)
	print(model.summary())
	# train_x = np.random.uniform(0,1,(17, 3, 1, 512, 512))
	# train_y = np.random.uniform(0,1,(17,1))
	# print (train_x)
	# train_x = np.array(train_x,dtype = 'float32')
	# train_y = np.array(train_y,dtype= 'float32')
	# hist = model.fit(train_x, train_y, nb_epoch=1, batch_size=batch_size, verbose=2, validation_split=0.1,shuffle=False)

	"""
	count the number of image in each typhoon sequence
	"""
	image_number_dictionary={}
	for  subdirs, dirs, files in os.walk(image_path):
		# print (subdirs)
		subdir_list.append(subdirs)
	for subdir in subdir_list:
		count = 0
		for subdirs, dirs, files in os.walk(subdir):
			for file in files:
				count += 1
		key = subdir.split('/')[-1]
		image_number_dictionary[key] = count
		if count < 24:
			print (key,count)
	# print (image_number_dictionary)

	"""
	check the number of images equals the number of track data?
	"""
	# for subdir in subdir_list:
	# 	for subdirs, dirs, files in os.walk(subdir):
	# 		for file in files:
	# 			# print (file)
	# 			[k1, k2] = file.split("-")[:2]
	# 			key = "".join((k1,k2))
	# 			try:
	# 				mark = track_dict[key]
	# 			except KeyError:
	# 				print (file +'do not have track value')
	

# for k in track_dict.keys():
# 	k2 = k[-6:] # typhoon number
# 	k1 = k[:-6]
# 	file = k1 +'-' + k2 +'*'
# 	file_path = image_path + k2 +'/' + file
# 	if not os.path.isfile(file_path):
# 		print (file_path not exists)
	track_dict_number ={}
	equal_track_image_list = []
	not_equal_track_image_list = []
	for subdir in subdir_list:
		key =subdir.split('/')[-1] 

		if len(key) > 0 and key not in ['201620','201621','201622']:
			track_file_path = track_path + key+'.itk'
			with open(track_file_path,'rb') as tsv_file:
				tsv_reader = csv.reader(tsv_file, delimiter='\t')
				count = 0
				for row in tsv_reader:
					count += 1
				track_dict_number[key] = count
				if count != image_number_dictionary[key]:
					not_equal_track_image_list.append(key)
					# print (key,count,image_number_dictionary[key],'not equal')
				if count == image_number_dictionary[key]:
					# print  (key,count,image_number_dictionary[key],' equal')
					equal_track_image_list.append(key)
	# print (not_equal_track_image_list,'not_equal_track_image_list')
	# print (equal_track_image_list,'equal_track_image_list')
	
	print (len(equal_track_image_list),'lenth of eqaual track image list')
	# "check if track file difference is one hour, result is yes for both equal and not_eqaul_image_list "

	for key in not_equal_track_image_list:
			ts =[]
			track_file_path = track_path + key+'.itk'
			with open(track_file_path,'rb') as tsv_file:
				tsv_reader = csv.reader(tsv_file, delimiter='\t')
				for row in tsv_reader:
					yy = row[0]
					mm = row[1]
					dd = row[2]
					hh = row[3]
					t = datetime.datetime.strptime(yy +":" + mm +":" + dd +':' +hh, '%Y:%m:%d:%H')
					ts.append(t)
			tmp = ts[0]
			for i in range(1,len(ts)):
				dif = (ts[i] - tmp).total_seconds()
				# print (dif,'dif')
				if dif != 3600:
					print (dif,i,key)
				tmp = ts[i]
			# break
	dataset_imageset_path = 'test_file/dataset_image_unequal.hdf5'
	dataset_intensity_path = 'test_file/dataset_intensity_unequal.hdf5'
	# hf_image = h5py.File(dataset_imageset_path)
	# hf_intensity = h5py.File(dataset_intensity_path)
	vgg_model = VGG_16('vgg16_weights.h5')
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
   	vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy')
	for key in not_equal_track_image_list:
	# # for key in equal_track_image_list:
		image_folder = image_path + key +'/'
	# 	dataset_x,dataset_y = prepare_dataset.dataset_1_2(image_folder,track_dict)
	# 	print dataset_x.shape
	# 	print dataset_y.shape
	# 	break
		file_path_list = []
		# print key
		dataset_image =[]
		dataset_intensity =[]
		for  subdirs, dirs, files in os.walk(image_folder):
			for file in files:
				file_path = os.path.join(subdirs,file)
				file_path_list.append(file_path)
		sorted_file_list = sorted(file_path_list,key = lambda x : int(x.split('/')[-1].split('-')[-4]))
		# print (len(sorted_file_list),'len of sorted_file_list')
		ts =[]
		intensities =[]
		for file_path in sorted_file_list:
			yymmddhh = file_path.split('/')[-1].split('-')[-4]
			track_key = yymmddhh + key
			intensities.append(float(track_dict[track_key][-2]))
			t = datetime.datetime.strptime(yymmddhh,'%Y%m%d%H')
			ts.append(t)
		# print len(ts),'len ts'
		tmp = ts[0]
		orig_image = load.get_x(sorted_file_list,img_rows,img_cols,mean_v,std_v)
		tmp_image = orig_image[0]
	# 		dataset_input = get_fc2(vgg_model,dataset_image)
	# 		dataset_input = np.array(dataset_input)

		dataset_image.append(orig_image[0])
		dataset_intensity.append(intensities[0])
		for i in range(1,len(ts)):
			dif = (ts[i] - tmp).total_seconds()
			# print (dif,'dif')
			if dif != 3600:
				print (dif/3600.0,i,key,ts[i])
				for j in range(1, int(dif/3600.0)):
					t2 = tmp +datetime.timedelta(seconds = 3600)
					yy = t2.year
					mm = str(t2.month).zfill(2)
					dd = str(t2.day).zfill(2)
					hh = str(t2.hour).zfill(2)
					yymmddhh = str(yy) +mm + dd +hh
					track_key = yymmddhh + key
					intensity = float(track_dict[track_key][-2])
					image = (1-(float(j)/(dif/3600.0))) * tmp_image + (float(j)/(dif/3600.0)) * orig_image[i]
					dataset_image.append(image)
					dataset_intensity.append(intensity)
			dataset_image.append(orig_image[i])
			dataset_intensity.append(intensities[i])

			tmp = ts[i]
			tmp_image = orig_image[i]
		# dataset_image = np.array(dataset_image)
		for i in range(len(dataset_image)):
			show_image(dataset_image[i][0],'test_file/unequal_image_generate_test/' + str(key)+'_'+str(i) +'.jpg')

		# dataset_input = get_fc2(vgg_model,dataset_image)
		# dataset_intensity = np.array(dataset_intensity)
		# dataset_intensity = prepare_dataset.normalize_intensity(dataset_intensity, intensity_mean,intensity_std)
		# hf_image.create_dataset(key, data = dataset_input)
		# hf_intensity.create_dataset(key, data = dataset_intensity)
		# break
	# hf_image.close()
	# hf_intensity.close()
	

	t2 = time.time()
	print ("using  %s seconds" % (t2-t1))


if __name__ == '__main__':
	main()

	
