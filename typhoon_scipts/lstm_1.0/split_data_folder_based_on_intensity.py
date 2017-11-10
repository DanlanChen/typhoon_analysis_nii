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
import os,load,csv,datetime,json
import config_sorted as config
# K.set_image_dim_ordering('tf')
import time,h5py
import prepare_dataset,math
import numpy as np
from keras.layers.wrappers import TimeDistributed
from sklearn.metrics import mean_squared_error
from vgg_pretrain_model import VGG_16, get_fc2
from keras.optimizers import SGD,Adam,Adadelta
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
	data_folder = 'test_file/sorted_intensity_data_folder.json'
	# intensity_min = {}
	# for key in equal_track_image_list:
	# 	track_file_path = track_path + key + '.itk'
	# 	dataset_intensity = prepare_dataset.dataset_1(track_file_path)
	# 	intensity_min[key] = min(dataset_intensity)
	# sorted_intensity_min = sorted(intensity_min.iteritems(), key=lambda (k,v): (v,k))
	# sorted_intensity_min =list(sorted_intensity_min)
	# sorted_intensity_min = np.array(sorted_intensity_min)
	# sorted_data_folder = sorted_intensity_min[:,0]
	# train_folder=[]
	# test_folder=[]
	# for i in range(0,len(sorted_data_folder),10):
	# 	if (i+10) <= len(sorted_data_folder):
	# 		j = i+10
	# 	else:
	# 		j = len(sorted_data_folder) +1
	# 	small_list = sorted_data_folder[i:j]
	# 	small_list = np.array(small_list)
	# 	np.random.shuffle(small_list)
	# 	small_list = list(small_list)
	# 	train_folder += small_list[:int(0.9*len(small_list))]
	# 	test_folder += small_list[int(0.9*len(small_list)):]
	# print train_folder
	# print test_folder
	# print len(train_folder)
	# print len(test_folder)
	# with open(data_folder,'w') as f:
	# 	json.dump({'train_folder':train_folder,'test_folder':test_folder},f)

	"""

	# get train test data from pre_built dataset
	
	"""
	# dataset_imageset
	# 0.423964 mean data
	# 0.569374 std data
	# 0.0 min
	# 4.71836 max
	dataset_image_path = 'test_file/dataset_imageset.hdf5'
	dataset_intensity_path = 'test_file/dataset_intensity.hdf5'


	hf_image = h5py.File(dataset_image_path)
	hf_intensity = h5py.File(dataset_intensity_path)

	train_x = []
	train_y = []
	test_x = []
	test_y = []

	vgg_fc2_mean = config.vgg_fc2_mean
	vgg_fc2_std = config.vgg_fc2_std
	with open(data_folder,'r') as f:
		data_folder = json.load(f)
		train_folder = data_folder['train_folder']
		test_folder = data_folder['test_folder']
	train_folder = np.array(train_folder)
	test_folder = np.array(test_folder)
	np.random.shuffle(train_folder)
	np.random.shuffle(test_folder)
	train_folder = list(train_folder)
	test_folder = list(test_folder)
	for key in train_folder:
		print(key)
		dataset_image = np.array(hf_image.get(key))
		# dataset_image = prepare_dataset.normalize_intensity(dataset_image,vgg_fc2_mean,vgg_fc2_std) #normalize image (the same function of normalize intensity)
		dataset_intensity = np.array(hf_intensity.get(key))
		if len(dataset_intensity) > look_back:
			data_x,data_y = prepare_dataset.extend_dataset_2_zero(dataset_image, dataset_intensity,look_back = look_back)
			train_x += data_x
			train_y += data_y

	for key in test_folder:
		print (key)
		dataset_image = np.array(hf_image.get(key))
		# dataset_image = prepare_dataset.normalize_intensity(dataset_image,vgg_fc2_mean,vgg_fc2_std)
		dataset_intensity = np.array(hf_intensity.get(key))
		if len(dataset_intensity) > look_back:
			data_x,data_y = prepare_dataset.extend_dataset_2_zero(dataset_image, dataset_intensity,look_back = look_back)
			test_x += data_x
			test_y += data_y
	# train = train_x + test_x
	train_x = np.array(train_x,dtype = 'float32')
	train_y = np.array(train_y,dtype = 'float32')
	test_x = np.array(test_x,dtype = 'float32')
	test_y = np.array(test_y,dtype = 'float32')
	print (train_x.shape,train_y.shape)
	print (test_x.shape,test_y.shape)
	
	




	train_hists=[]
	validation_hists=[]
	val_loss =  sys.float_info.max

	for i in range(1000):
		print (i,'epoch')
		# ModelCheckpoint_file = 'test_file/orig_weights_lstm_1.0_image_lookback_'+str(look_back)+str(i)+'_whole_equal.hdf5'
		# print('start train')
		hist = model.fit(train_x, train_y, nb_epoch=1, batch_size=batch_size, verbose=2, validation_split=0.1,shuffle=False)
		model.reset_states()
		train_hists.append(hist.history['loss'][0])
		validation_hists.append(hist.history['val_loss'][0])
		if val_loss > hist.history['val_loss'][0]:
			model.save_weights(ModelCheckpoint_file)
			print(i,val_loss,'->',hist.history['val_loss'][0],'save_weights',ModelCheckpoint_file)
			val_loss = hist.history['val_loss'][0]
	print (train_hists,'train_hists')
	print (validation_hists, 'validation_hists')
	with open(hist_path,'w') as f:
		json.dump({'train_loss':train_hists,'val_loss':validation_hists},f)
	# hist = model.fit(train_x, train_y, nb_epoch=2, batch_size=batch_size, verbose=2, validation_split = 0.1,shuffle=False)
		# break
	# with open(hist_path,'w') as j:
	# 	json.dump(hist.history,j)
	# validation_hists_least_index = validation_hists.index(min(validation_hists))
	# print ('ModelCheckpoint_file','test_file/orig_weights_lstm_1.0_image_lookback_'+str(look_back)+str(validation_hists_least_index)+'_whole_equal.hdf5')
	# model.load_weights('test_file/orig_weights_lstm_1.0_image_lookback_'+str(look_back)+str(validation_hists_least_index)+'_whole_equal.hdf5')
	
	print('load_weights',ModelCheckpoint_file)
	model.load_weights(ModelCheckpoint_file)
	trainPredict = model.predict(train_x, batch_size=batch_size)
	trainPredict = prepare_dataset.reverse_normalize_intensity(trainPredict,intensity_mean,intensity_std)
	
	trainY = prepare_dataset.reverse_normalize_intensity(train_y,intensity_mean,intensity_std)
	trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
	model.reset_states()

	print('Train Score: %.2f RMSE' % (trainScore))
	testPredict = model.predict(test_x, batch_size=batch_size)
	# # invert predictions
	testPredict = prepare_dataset.reverse_normalize_intensity(testPredict,intensity_mean,intensity_std)
	testY = prepare_dataset.reverse_normalize_intensity(test_y,intensity_mean,intensity_std)
	# # calculate root mean squared error


	testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	print(look_back,'look_back')
	
	
	t2 = time.time()
	print ("using  %s seconds" % (t2-t1))
	
	
if __name__ == '__main__':
	main()

	
