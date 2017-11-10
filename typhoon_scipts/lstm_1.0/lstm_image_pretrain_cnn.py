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
def pretrain_model_2(look_back,batch_size):
	model =Sequential()
	model.add(LSTM(1024,stateful= True,batch_input_shape=(batch_size,look_back,4096)))
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
	print ('look_back',look_back,'ModelCheckpoint_file',ModelCheckpoint_file)
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
	data_folder_path = config.data_folder_path
	if not os.path.exists(data_folder_path): 
		equal_track_image_list = np.array(equal_track_image_list)
		np.random.shuffle(equal_track_image_list)
		equal_track_image_list = list(equal_track_image_list)
		# equal_track_image_list = equal_track_image_list[:2]
		train_folder = equal_track_image_list[:int(0.9 * len(equal_track_image_list))]
		test_folder = equal_track_image_list[int(0.9* len(equal_track_image_list)):]
		with open(data_folder_path,'w') as f:
			json.dump({'train_folder':train_folder,'test_folder': test_folder},f)
			print ('data_folder_path dumped to: ',data_folder_path)
	else:
		with open(data_folder_path,'r') as f:
			data_folder = json.load(f)
			train_folder = data_folder['train_folder']
			test_folder = data_folder['test_folder']
			print ('load data folder from: ' , data_folder_path)





	# data_path = config.data_path
	# if not os.path.exists(data_path):
	# 	train_x =[]
	# 	train_y=[]
	# 	test_x = []
	# 	test_y = []
	# 	vgg_model = VGG_16('vgg16_weights.h5')
	# 	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#    	vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy')
	# 	for key in test_folder:
	# 		print(key)
	# 		image_folder = image_path + key +'/'
	# 		track_file_path = track_path + key + '.itk'
	# 		dataset_image = prepare_dataset.dataset_2(image_folder)
	# 		print (dataset_image.shape)
	# 		dataset_input = get_fc2(vgg_model,dataset_image)
	# 		dataset_intensity = prepare_dataset.dataset_1(track_file_path)
	# 		dataset_intensity = prepare_dataset.normalize_intensity(dataset_intensity,intensity_mean,intensity_std)
	# 		print (dataset_image.shape,'dataset_image.shape')
	# 		print (dataset_intensity.shape,'dataset_intensity')
	# 		data_x,data_y = prepare_dataset.create_dataset_2(dataset_input, dataset_intensity,look_back = look_back)
	# 		test_x += data_x
	# 		test_y += data_y
	# 	# print test_y.shape,test_y
	# 	# train_histss =[]
	# 	# validation_histss=[]
	# 	for key in train_folder:
	# 		print(key)
	# 		image_folder = image_path + key +'/'
	# 		track_file_path = track_path + key + '.itk'
	# 		dataset_image = prepare_dataset.dataset_2(image_folder)
	# 		dataset_input = get_fc2(vgg_model,dataset_image)
	# 		dataset_intensity = prepare_dataset.dataset_1(track_file_path)
	# 		dataset_intensity = prepare_dataset.normalize_intensity(dataset_intensity,intensity_mean,intensity_std)
	# 		print (dataset_image.shape,'dataset_image.shape')
	# 		print (dataset_intensity.shape,'dataset_intensity')
	# 		data_x,data_y = prepare_dataset.create_dataset_2(dataset_input, dataset_intensity,look_back = look_back)
	# 		# print (len(data_x))
	# 		train_x += data_x
	# 		train_y += data_y
	# 		data_x = np.array(data_x)
	# 		data_y = np.array(data_y)
	# 		# print (data_x.shape,data_y.shape,'data_x,data_y')
	# 		# train_hists=[]
	# 		# validation_hists=[]
	# 		# for i in range(20):
	# 		# 	print('start train')
	# 		# 	hist = model.fit(data_x, data_y, nb_epoch=1, batch_size=batch_size, verbose=2, validation_split=0.1,shuffle=False)
	# 		# 	model.reset_states()
	# 		# 	train_hists.append(hist.history['loss'][0])
	# 		# 	validation_hists.append(hist.history['val_loss'][0])
	# 		# # print (hists,'hists')
	# 		# train_histss.append(train_hists)
	# 		# validation_histss.append(validation_hists)

	# 	# print (train_histss,'train_histss')
	# 	# print (validation_histss, 'validation_histss')
		
	# 		# print ((data_x.shape),data_y.shape)
	# 	train_x = np.array(train_x,dtype = 'float32')
	# 	train_y = np.array(train_y,dtype = 'float32')
	# 	test_x = np.array(test_x,dtype = 'float32')
	# 	test_y = np.array(test_y,dtype = 'float32')
		
	# 	hf = h5py.File(data_path)
	# 	hf.create_dataset('train_x',data = train_x)
	# 	hf.create_dataset('train_y',data = train_y)
	# 	hf.create_dataset('test_x', data= test_x)
	# 	hf.create_dataset('test_y', data= test_y)
	# 	hf.close()
	# 	print ('dump train test data to' ,data_path)

	# else:
	# 	with h5py.File(data_path,'r') as hf:
	# 		train_x = np.array(hf.get('train_x'))
	# 		train_y = np.array(hf.get('train_y'))
	# 		test_x = np.array(hf.get('test_x'))
	# 		test_y = np.array(hf.get('test_y'))
	# 	print ('loaded train test data from ', data_path)
	# print (train_x.shape,train_y.shape)
	# print (test_x.shape,test_y.shape)
	

	dataset_image_path = 'test_file/dataset_imageset.hdf5'
	dataset_intensity_path = 'test_file/dataset_intensity.hdf5'


	hf_image = h5py.File(dataset_image_path)
	hf_intensity = h5py.File(dataset_intensity_path)

	train_x = []
	train_y = []
	test_x = []
	test_y = []
	# train_folder = train_folder[:2]
	# test_folder = test_folder[:1]
	for key in train_folder:
		print(key)
		dataset_image = np.array(hf_image.get(key))
		dataset_intensity = np.array(hf_intensity.get(key))
		# data_x,data_y = prepare_dataset.extend_dataset_2(dataset_image, dataset_intensity,look_back = look_back)
		data_x,data_y = prepare_dataset.create_dataset_2(dataset_image, dataset_intensity,look_back = look_back)
		train_x += data_x
		train_y += data_y
	for key in test_folder:
		print (key)
		dataset_image = np.array(hf_image.get(key))
		dataset_intensity = np.array(hf_intensity.get(key))
		# data_x,data_y = prepare_dataset.extend_dataset_2(dataset_image, dataset_intensity,look_back = look_back)
		data_x,data_y = prepare_dataset.create_dataset_2(dataset_image, dataset_intensity,look_back = look_back)
		test_x += data_x
		test_y += data_y
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
		# ModelCheckpoint_file = 'test_file/orig_weights_lstm_1.0_image_lookback_'+str(look_back)+str(i)+'_whole_equal.hdf5'
		print('start train')
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
	
	if os.path.exists(ModelCheckpoint_file):
		print ('load  load_weights',ModelCheckpoint_file)
		model.load_weights(ModelCheckpoint_file)
	
	trainPredict = model.predict(train_x, batch_size=batch_size)
	trainPredict = prepare_dataset.reverse_normalize_intensity(trainPredict,intensity_mean,intensity_std)
	
	trainY = prepare_dataset.reverse_normalize_intensity(train_y,intensity_mean,intensity_std)
	trainScore = math.sqrt(mean_squared_error(trainY[20:], trainPredict[20:,0]))
	model.reset_states()

	print('Train Score: %.2f RMSE' % (trainScore))
	testPredict = model.predict(test_x, batch_size=batch_size)
	# # invert predictions
	testPredict = prepare_dataset.reverse_normalize_intensity(testPredict,intensity_mean,intensity_std)
	testY = prepare_dataset.reverse_normalize_intensity(test_y,intensity_mean,intensity_std)
	# # calculate root mean squared error


	testScore = math.sqrt(mean_squared_error(testY[20:], testPredict[20:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	print(look_back,'look_back')

	"""
	train_predict_image = config.train_predict_image
	test_predict_image = config.test_predict_image
	fig = plt.figure()
	plt.title('train_predicts_look_back')
	plt.plot(list(trainPredict[:20000,0]),'r--',label= 'train_predict')
	plt.plot(list(trainY[:20000]), 'g--',label = 'train')
	plt.xlabel('typhoon_image')
	plt.ylabel('typhoon intensity')
	plt.legend(loc = 'upper left', shadow =True)
	plt.savefig(train_predict_image)
	plt.close(fig)
	fig = plt.figure()
	plt.title('test_predicts_look_back')
	plt.plot(list(testPredict[:10000,0]),'r--',label= 'test_predict')
	plt.plot(list(testY[:10000]), 'g--',label = 'test')
	plt.xlabel('typhoon_image')
	plt.ylabel('typhoon intensity')
	plt.legend(loc = 'upper left', shadow =True)
	plt.savefig(test_predict_image)
	plt.close(fig)
	"""
	t2 = time.time()
	print ("using  %s seconds" % (t2-t1))

"""
	2016101415-201621-HMW8-1.h5do not have track value
2016101414-201621-HMW8-1.h5do not have track value
2016101223-201621-HMW8-1.h5do not have track value
2016101610-201621-HMW8-1.h5do not have track value
2016101612-201621-HMW8-1.h5do not have track value
2016101607-201621-HMW8-1.h5do not have track value
2016101508-201621-HMW8-1.h5do not have track value
2016101318-201621-HMW8-1.h5do not have track value
2016101520-201621-HMW8-1.h5do not have track value
2016101204-201621-HMW8-1.h5do not have track value
2016101521-201621-HMW8-1.h5do not have track value
2016101218-201621-HMW8-1.h5do not have track value
2016101304-201621-HMW8-1.h5do not have track value
2016101500-201621-HMW8-1.h5do not have track value
2016101402-201621-HMW8-1.h5do not have track value
2016101212-201621-HMW8-1.h5do not have track value
2016101418-201621-HMW8-1.h5do not have track value
2016101509-201621-HMW8-1.h5do not have track value
2016101407-201621-HMW8-1.h5do not have track value
2016101121-201621-HMW8-1.h5do not have track value
2016101603-201621-HMW8-1.h5do not have track value
2016101619-201621-HMW8-1.h5do not have track value
2016101501-201621-HMW8-1.h5do not have track value
2016101408-201621-HMW8-1.h5do not have track value
2016101510-201621-HMW8-1.h5do not have track value
2016101301-201621-HMW8-1.h5do not have track value
2016101120-201621-HMW8-1.h5do not have track value
2016101623-201621-HMW8-1.h5do not have track value
2016101311-201621-HMW8-1.h5do not have track value
2016101405-201621-HMW8-1.h5do not have track value
2016101704-201621-HMW8-1.h5do not have track value
2016101515-201621-HMW8-1.h5do not have track value
2016101419-201621-HMW8-1.h5do not have track value
2016101420-201621-HMW8-1.h5do not have track value
2016101606-201621-HMW8-1.h5do not have track value
2016101605-201621-HMW8-1.h5do not have track value
2016101702-201621-HMW8-1.h5do not have track value
2016101505-201621-HMW8-1.h5do not have track value
2016101413-201621-HMW8-1.h5do not have track value
2016101622-201621-HMW8-1.h5do not have track value
2016101601-201621-HMW8-1.h5do not have track value
2016101206-201621-HMW8-1.h5do not have track value
2016101315-201621-HMW8-1.h5do not have track value
2016101617-201621-HMW8-1.h5do not have track value
2016101700-201621-HMW8-1.h5do not have track value
2016101618-201621-HMW8-1.h5do not have track value
2016101216-201621-HMW8-1.h5do not have track value
2016101703-201621-HMW8-1.h5do not have track value
2016101514-201621-HMW8-1.h5do not have track value
2016101412-201621-HMW8-1.h5do not have track value
2016101422-201621-HMW8-1.h5do not have track value
2016101410-201621-HMW8-1.h5do not have track value
2016101320-201621-HMW8-1.h5do not have track value
2016101421-201621-HMW8-1.h5do not have track value
2016101323-201621-HMW8-1.h5do not have track value
2016101504-201621-HMW8-1.h5do not have track value
2016101611-201621-HMW8-1.h5do not have track value
2016101404-201621-HMW8-1.h5do not have track value
2016101122-201621-HMW8-1.h5do not have track value
2016101310-201621-HMW8-1.h5do not have track value
2016101409-201621-HMW8-1.h5do not have track value
2016101503-201621-HMW8-1.h5do not have track value
2016101201-201621-HMW8-1.h5do not have track value
2016101322-201621-HMW8-1.h5do not have track value
2016101706-201621-HMW8-1.h5do not have track value
2016101519-201621-HMW8-1.h5do not have track value
2016101511-201621-HMW8-1.h5do not have track value
2016101203-201621-HMW8-1.h5do not have track value
2016101314-201621-HMW8-1.h5do not have track value
2016101417-201621-HMW8-1.h5do not have track value
2016101302-201621-HMW8-1.h5do not have track value
2016101502-201621-HMW8-1.h5do not have track value
2016101512-201621-HMW8-1.h5do not have track value
2016101215-201621-HMW8-1.h5do not have track value
2016101416-201621-HMW8-1.h5do not have track value
2016101517-201621-HMW8-1.h5do not have track value
2016101210-201621-HMW8-1.h5do not have track value
2016101219-201621-HMW8-1.h5do not have track value
2016101209-201621-HMW8-1.h5do not have track value
2016101319-201621-HMW8-1.h5do not have track value
2016101222-201621-HMW8-1.h5do not have track value
2016101300-201621-HMW8-1.h5do not have track value
2016101406-201621-HMW8-1.h5do not have track value
2016101211-201621-HMW8-1.h5do not have track value
2016101615-201621-HMW8-1.h5do not have track value
2016101214-201621-HMW8-1.h5do not have track value
2016101221-201621-HMW8-1.h5do not have track value
2016101309-201621-HMW8-1.h5do not have track value
2016101313-201621-HMW8-1.h5do not have track value
2016101613-201621-HMW8-1.h5do not have track value
2016101220-201621-HMW8-1.h5do not have track value
2016101513-201621-HMW8-1.h5do not have track value
2016101321-201621-HMW8-1.h5do not have track value
2016101208-201621-HMW8-1.h5do not have track value
2016101604-201621-HMW8-1.h5do not have track value
2016101317-201621-HMW8-1.h5do not have track value
2016101621-201621-HMW8-1.h5do not have track value
2016101123-201621-HMW8-1.h5do not have track value
2016101620-201621-HMW8-1.h5do not have track value
2016101507-201621-HMW8-1.h5do not have track value
2016101308-201621-HMW8-1.h5do not have track value
2016101423-201621-HMW8-1.h5do not have track value
2016101202-201621-HMW8-1.h5do not have track value
2016101205-201621-HMW8-1.h5do not have track value
2016101522-201621-HMW8-1.h5do not have track value
2016101411-201621-HMW8-1.h5do not have track value
2016101614-201621-HMW8-1.h5do not have track value
2016101705-201621-HMW8-1.h5do not have track value
2016101207-201621-HMW8-1.h5do not have track value
2016101602-201621-HMW8-1.h5do not have track value
2016101701-201621-HMW8-1.h5do not have track value
2016101401-201621-HMW8-1.h5do not have track value
2016101609-201621-HMW8-1.h5do not have track value
2016101303-201621-HMW8-1.h5do not have track value
2016101400-201621-HMW8-1.h5do not have track value
2016101608-201621-HMW8-1.h5do not have track value
2016101600-201621-HMW8-1.h5do not have track value
2016101119-201621-HMW8-1.h5do not have track value
2016101307-201621-HMW8-1.h5do not have track value
2016101523-201621-HMW8-1.h5do not have track value
2016101200-201621-HMW8-1.h5do not have track value
2016101316-201621-HMW8-1.h5do not have track value
2016101516-201621-HMW8-1.h5do not have track value
2016101306-201621-HMW8-1.h5do not have track value
2016101403-201621-HMW8-1.h5do not have track value
2016101118-201621-HMW8-1.h5do not have track value
2016101305-201621-HMW8-1.h5do not have track value
2016101506-201621-HMW8-1.h5do not have track value
2016101213-201621-HMW8-1.h5do not have track value
2016101217-201621-HMW8-1.h5do not have track value
2016101616-201621-HMW8-1.h5do not have track value
2016101312-201621-HMW8-1.h5do not have track value
2016101518-201621-HMW8-1.h5do not have track value
2016101614-201622-HMW8-1.h5do not have track value
2016101622-201622-HMW8-1.h5do not have track value
2016101618-201622-HMW8-1.h5do not have track value
2016101402-201622-HMW8-1.h5do not have track value
2016101522-201622-HMW8-1.h5do not have track value
2016101416-201622-HMW8-1.h5do not have track value
2016101613-201622-HMW8-1.h5do not have track value
2016101501-201622-HMW8-1.h5do not have track value
2016101513-201622-HMW8-1.h5do not have track value
2016101519-201622-HMW8-1.h5do not have track value
2016101400-201622-HMW8-1.h5do not have track value
2016101615-201622-HMW8-1.h5do not have track value
2016101512-201622-HMW8-1.h5do not have track value
2016101410-201622-HMW8-1.h5do not have track value
2016101608-201622-HMW8-1.h5do not have track value
2016101411-201622-HMW8-1.h5do not have track value
2016101515-201622-HMW8-1.h5do not have track value
2016101420-201622-HMW8-1.h5do not have track value
2016101616-201622-HMW8-1.h5do not have track value
2016101607-201622-HMW8-1.h5do not have track value
2016101611-201622-HMW8-1.h5do not have track value
2016101604-201622-HMW8-1.h5do not have track value
2016101601-201622-HMW8-1.h5do not have track value
2016101603-201622-HMW8-1.h5do not have track value
2016101511-201622-HMW8-1.h5do not have track value
2016101421-201622-HMW8-1.h5do not have track value
2016101415-201622-HMW8-1.h5do not have track value
2016101706-201622-HMW8-1.h5do not have track value
2016101518-201622-HMW8-1.h5do not have track value
2016101704-201622-HMW8-1.h5do not have track value
2016101407-201622-HMW8-1.h5do not have track value
2016101514-201622-HMW8-1.h5do not have track value
2016101705-201622-HMW8-1.h5do not have track value
2016101517-201622-HMW8-1.h5do not have track value
2016101612-201622-HMW8-1.h5do not have track value
2016101406-201622-HMW8-1.h5do not have track value
2016101423-201622-HMW8-1.h5do not have track value
2016101702-201622-HMW8-1.h5do not have track value
2016101403-201622-HMW8-1.h5do not have track value
2016101418-201622-HMW8-1.h5do not have track value
2016101502-201622-HMW8-1.h5do not have track value
2016101504-201622-HMW8-1.h5do not have track value
2016101701-201622-HMW8-1.h5do not have track value
2016101619-201622-HMW8-1.h5do not have track value
2016101623-201622-HMW8-1.h5do not have track value
2016101617-201622-HMW8-1.h5do not have track value
2016101500-201622-HMW8-1.h5do not have track value
2016101508-201622-HMW8-1.h5do not have track value
2016101509-201622-HMW8-1.h5do not have track value
2016101404-201622-HMW8-1.h5do not have track value
2016101507-201622-HMW8-1.h5do not have track value
2016101510-201622-HMW8-1.h5do not have track value
2016101602-201622-HMW8-1.h5do not have track value
2016101503-201622-HMW8-1.h5do not have track value
2016101516-201622-HMW8-1.h5do not have track value
2016101417-201622-HMW8-1.h5do not have track value
2016101620-201622-HMW8-1.h5do not have track value
2016101405-201622-HMW8-1.h5do not have track value
2016101703-201622-HMW8-1.h5do not have track value
2016101505-201622-HMW8-1.h5do not have track value
2016101609-201622-HMW8-1.h5do not have track value
2016101606-201622-HMW8-1.h5do not have track value
2016101401-201622-HMW8-1.h5do not have track value
2016101621-201622-HMW8-1.h5do not have track value
2016101700-201622-HMW8-1.h5do not have track value
2016101521-201622-HMW8-1.h5do not have track value
2016101605-201622-HMW8-1.h5do not have track value
2016101520-201622-HMW8-1.h5do not have track value
2016101523-201622-HMW8-1.h5do not have track value
2016101506-201622-HMW8-1.h5do not have track value
2016101419-201622-HMW8-1.h5do not have track value
2016101600-201622-HMW8-1.h5do not have track value
2016101422-201622-HMW8-1.h5do not have track value
2016101413-201622-HMW8-1.h5do not have track value
2016101409-201622-HMW8-1.h5do not have track value
2016101414-201622-HMW8-1.h5do not have track value
2016101412-201622-HMW8-1.h5do not have track value
2016101610-201622-HMW8-1.h5do not have track value
2016101408-201622-HMW8-1.h5do not have track value
2016101300-201620-HMW8-1.h5do not have track value
2016101222-201620-HMW8-1.h5do not have track value
2016101305-201620-HMW8-1.h5do not have track value
2016101302-201620-HMW8-1.h5do not have track value
2016101301-201620-HMW8-1.h5do not have track value
2016101306-201620-HMW8-1.h5do not have track value
2016101304-201620-HMW8-1.h5do not have track value
2016101303-201620-HMW8-1.h5do not have track value
2016101223-201620-HMW8-1.h5do not have track value
2016101415-201621-HMW8-1.h5do not have track value
2016101414-201621-HMW8-1.h5do not have track value
2016101223-201621-HMW8-1.h5do not have track value
2016101610-201621-HMW8-1.h5do not have track value
2016101612-201621-HMW8-1.h5do not have track value
2016101607-201621-HMW8-1.h5do not have track value
2016101508-201621-HMW8-1.h5do not have track value
2016101318-201621-HMW8-1.h5do not have track value
2016101520-201621-HMW8-1.h5do not have track value
2016101204-201621-HMW8-1.h5do not have track value
2016101521-201621-HMW8-1.h5do not have track value
2016101218-201621-HMW8-1.h5do not have track value
2016101304-201621-HMW8-1.h5do not have track value
2016101500-201621-HMW8-1.h5do not have track value
2016101402-201621-HMW8-1.h5do not have track value
2016101212-201621-HMW8-1.h5do not have track value
2016101418-201621-HMW8-1.h5do not have track value
2016101509-201621-HMW8-1.h5do not have track value
2016101407-201621-HMW8-1.h5do not have track value
2016101121-201621-HMW8-1.h5do not have track value
2016101603-201621-HMW8-1.h5do not have track value
2016101619-201621-HMW8-1.h5do not have track value
2016101501-201621-HMW8-1.h5do not have track value
2016101408-201621-HMW8-1.h5do not have track value
2016101510-201621-HMW8-1.h5do not have track value
2016101301-201621-HMW8-1.h5do not have track value
2016101120-201621-HMW8-1.h5do not have track value
2016101623-201621-HMW8-1.h5do not have track value
2016101311-201621-HMW8-1.h5do not have track value
2016101405-201621-HMW8-1.h5do not have track value
2016101704-201621-HMW8-1.h5do not have track value
2016101515-201621-HMW8-1.h5do not have track value
2016101419-201621-HMW8-1.h5do not have track value
2016101420-201621-HMW8-1.h5do not have track value
2016101606-201621-HMW8-1.h5do not have track value
2016101605-201621-HMW8-1.h5do not have track value
2016101702-201621-HMW8-1.h5do not have track value
2016101505-201621-HMW8-1.h5do not have track value
2016101413-201621-HMW8-1.h5do not have track value
2016101622-201621-HMW8-1.h5do not have track value
2016101601-201621-HMW8-1.h5do not have track value
2016101206-201621-HMW8-1.h5do not have track value
2016101315-201621-HMW8-1.h5do not have track value
2016101617-201621-HMW8-1.h5do not have track value
2016101700-201621-HMW8-1.h5do not have track value
2016101618-201621-HMW8-1.h5do not have track value
2016101216-201621-HMW8-1.h5do not have track value
2016101703-201621-HMW8-1.h5do not have track value
2016101514-201621-HMW8-1.h5do not have track value
2016101412-201621-HMW8-1.h5do not have track value
2016101422-201621-HMW8-1.h5do not have track value
2016101410-201621-HMW8-1.h5do not have track value
2016101320-201621-HMW8-1.h5do not have track value
2016101421-201621-HMW8-1.h5do not have track value
2016101323-201621-HMW8-1.h5do not have track value
2016101504-201621-HMW8-1.h5do not have track value
2016101611-201621-HMW8-1.h5do not have track value
2016101404-201621-HMW8-1.h5do not have track value
2016101122-201621-HMW8-1.h5do not have track value
2016101310-201621-HMW8-1.h5do not have track value
2016101409-201621-HMW8-1.h5do not have track value
2016101503-201621-HMW8-1.h5do not have track value
2016101201-201621-HMW8-1.h5do not have track value
2016101322-201621-HMW8-1.h5do not have track value
2016101706-201621-HMW8-1.h5do not have track value
2016101519-201621-HMW8-1.h5do not have track value
2016101511-201621-HMW8-1.h5do not have track value
2016101203-201621-HMW8-1.h5do not have track value
2016101314-201621-HMW8-1.h5do not have track value
2016101417-201621-HMW8-1.h5do not have track value
2016101302-201621-HMW8-1.h5do not have track value
2016101502-201621-HMW8-1.h5do not have track value
2016101512-201621-HMW8-1.h5do not have track value
2016101215-201621-HMW8-1.h5do not have track value
2016101416-201621-HMW8-1.h5do not have track value
2016101517-201621-HMW8-1.h5do not have track value
2016101210-201621-HMW8-1.h5do not have track value
2016101219-201621-HMW8-1.h5do not have track value
2016101209-201621-HMW8-1.h5do not have track value
2016101319-201621-HMW8-1.h5do not have track value
2016101222-201621-HMW8-1.h5do not have track value
2016101300-201621-HMW8-1.h5do not have track value
2016101406-201621-HMW8-1.h5do not have track value
2016101211-201621-HMW8-1.h5do not have track value
2016101615-201621-HMW8-1.h5do not have track value
2016101214-201621-HMW8-1.h5do not have track value
2016101221-201621-HMW8-1.h5do not have track value
2016101309-201621-HMW8-1.h5do not have track value
2016101313-201621-HMW8-1.h5do not have track value
2016101613-201621-HMW8-1.h5do not have track value
2016101220-201621-HMW8-1.h5do not have track value
2016101513-201621-HMW8-1.h5do not have track value
2016101321-201621-HMW8-1.h5do not have track value
2016101208-201621-HMW8-1.h5do not have track value
2016101604-201621-HMW8-1.h5do not have track value
2016101317-201621-HMW8-1.h5do not have track value
2016101621-201621-HMW8-1.h5do not have track value
2016101123-201621-HMW8-1.h5do not have track value
2016101620-201621-HMW8-1.h5do not have track value
2016101507-201621-HMW8-1.h5do not have track value
2016101308-201621-HMW8-1.h5do not have track value
2016101423-201621-HMW8-1.h5do not have track value
2016101202-201621-HMW8-1.h5do not have track value
2016101205-201621-HMW8-1.h5do not have track value
2016101522-201621-HMW8-1.h5do not have track value
2016101411-201621-HMW8-1.h5do not have track value
2016101614-201621-HMW8-1.h5do not have track value
2016101705-201621-HMW8-1.h5do not have track value
2016101207-201621-HMW8-1.h5do not have track value
2016101602-201621-HMW8-1.h5do not have track value
2016101701-201621-HMW8-1.h5do not have track value
2016101401-201621-HMW8-1.h5do not have track value
2016101609-201621-HMW8-1.h5do not have track value
2016101303-201621-HMW8-1.h5do not have track value
2016101400-201621-HMW8-1.h5do not have track value
2016101608-201621-HMW8-1.h5do not have track value
2016101600-201621-HMW8-1.h5do not have track value
2016101119-201621-HMW8-1.h5do not have track value
2016101307-201621-HMW8-1.h5do not have track value
2016101523-201621-HMW8-1.h5do not have track value
2016101200-201621-HMW8-1.h5do not have track value
2016101316-201621-HMW8-1.h5do not have track value
2016101516-201621-HMW8-1.h5do not have track value
2016101306-201621-HMW8-1.h5do not have track value
2016101403-201621-HMW8-1.h5do not have track value
2016101118-201621-HMW8-1.h5do not have track value
2016101305-201621-HMW8-1.h5do not have track value
2016101506-201621-HMW8-1.h5do not have track value
2016101213-201621-HMW8-1.h5do not have track value
2016101217-201621-HMW8-1.h5do not have track value
2016101616-201621-HMW8-1.h5do not have track value
2016101312-201621-HMW8-1.h5do not have track value
2016101518-201621-HMW8-1.h5do not have track value
2016101614-201622-HMW8-1.h5do not have track value
2016101622-201622-HMW8-1.h5do not have track value
2016101618-201622-HMW8-1.h5do not have track value
2016101402-201622-HMW8-1.h5do not have track value
2016101522-201622-HMW8-1.h5do not have track value
2016101416-201622-HMW8-1.h5do not have track value
2016101613-201622-HMW8-1.h5do not have track value
2016101501-201622-HMW8-1.h5do not have track value
2016101513-201622-HMW8-1.h5do not have track value
2016101519-201622-HMW8-1.h5do not have track value
2016101400-201622-HMW8-1.h5do not have track value
2016101615-201622-HMW8-1.h5do not have track value
2016101512-201622-HMW8-1.h5do not have track value
2016101410-201622-HMW8-1.h5do not have track value
2016101608-201622-HMW8-1.h5do not have track value
2016101411-201622-HMW8-1.h5do not have track value
2016101515-201622-HMW8-1.h5do not have track value
2016101420-201622-HMW8-1.h5do not have track value
2016101616-201622-HMW8-1.h5do not have track value
2016101607-201622-HMW8-1.h5do not have track value
2016101611-201622-HMW8-1.h5do not have track value
2016101604-201622-HMW8-1.h5do not have track value
2016101601-201622-HMW8-1.h5do not have track value
2016101603-201622-HMW8-1.h5do not have track value
2016101511-201622-HMW8-1.h5do not have track value
2016101421-201622-HMW8-1.h5do not have track value
2016101415-201622-HMW8-1.h5do not have track value
2016101706-201622-HMW8-1.h5do not have track value
2016101518-201622-HMW8-1.h5do not have track value
2016101704-201622-HMW8-1.h5do not have track value
2016101407-201622-HMW8-1.h5do not have track value
2016101514-201622-HMW8-1.h5do not have track value
2016101705-201622-HMW8-1.h5do not have track value
2016101517-201622-HMW8-1.h5do not have track value
2016101612-201622-HMW8-1.h5do not have track value
2016101406-201622-HMW8-1.h5do not have track value
2016101423-201622-HMW8-1.h5do not have track value
2016101702-201622-HMW8-1.h5do not have track value
2016101403-201622-HMW8-1.h5do not have track value
2016101418-201622-HMW8-1.h5do not have track value
2016101502-201622-HMW8-1.h5do not have track value
2016101504-201622-HMW8-1.h5do not have track value
2016101701-201622-HMW8-1.h5do not have track value
2016101619-201622-HMW8-1.h5do not have track value
2016101623-201622-HMW8-1.h5do not have track value
2016101617-201622-HMW8-1.h5do not have track value
2016101500-201622-HMW8-1.h5do not have track value
2016101508-201622-HMW8-1.h5do not have track value
2016101509-201622-HMW8-1.h5do not have track value
2016101404-201622-HMW8-1.h5do not have track value
2016101507-201622-HMW8-1.h5do not have track value
2016101510-201622-HMW8-1.h5do not have track value
2016101602-201622-HMW8-1.h5do not have track value
2016101503-201622-HMW8-1.h5do not have track value
2016101516-201622-HMW8-1.h5do not have track value
2016101417-201622-HMW8-1.h5do not have track value
2016101620-201622-HMW8-1.h5do not have track value
2016101405-201622-HMW8-1.h5do not have track value
2016101703-201622-HMW8-1.h5do not have track value
2016101505-201622-HMW8-1.h5do not have track value
2016101609-201622-HMW8-1.h5do not have track value
2016101606-201622-HMW8-1.h5do not have track value
2016101401-201622-HMW8-1.h5do not have track value
2016101621-201622-HMW8-1.h5do not have track value
2016101700-201622-HMW8-1.h5do not have track value
2016101521-201622-HMW8-1.h5do not have track value
2016101605-201622-HMW8-1.h5do not have track value
2016101520-201622-HMW8-1.h5do not have track value
2016101523-201622-HMW8-1.h5do not have track value
2016101506-201622-HMW8-1.h5do not have track value
2016101419-201622-HMW8-1.h5do not have track value
2016101600-201622-HMW8-1.h5do not have track value
2016101422-201622-HMW8-1.h5do not have track value
2016101413-201622-HMW8-1.h5do not have track value
2016101409-201622-HMW8-1.h5do not have track value
2016101414-201622-HMW8-1.h5do not have track value
2016101412-201622-HMW8-1.h5do not have track value
2016101610-201622-HMW8-1.h5do not have track value
2016101408-201622-HMW8-1.h5do not have track value
2016101300-201620-HMW8-1.h5do not have track value
2016101222-201620-HMW8-1.h5do not have track value
2016101305-201620-HMW8-1.h5do not have track value
2016101302-201620-HMW8-1.h5do not have track value
2016101301-201620-HMW8-1.h5do not have track value
2016101306-201620-HMW8-1.h5do not have track value
2016101304-201620-HMW8-1.h5do not have track value
2016101303-201620-HMW8-1.h5do not have track value
2016101223-201620-HMW8-1.h5do not have track value
	"""




"""
numbers of image less than 24, do not use it
	'197830', 3)
('201314', 13)
('198401', 0)
('198402', 0)
('197923', 2)
('198414', 21)
('198403', 16)
('198111', 21)
('197917', 12)
"""

""""
201620,201621,201622 do not have track ,so discard
"""

"""
TRACK IMAGE NOT EQUAL
('/NOBACKUP/nii/typhoon_data/orig_image/198722', 259, 164)
('/NOBACKUP/nii/typhoon_data/orig_image/199915', 193, 166)
('/NOBACKUP/nii/typhoon_data/orig_image/200901', 283, 282)
('/NOBACKUP/nii/typhoon_data/orig_image/199215', 391, 367)
('/NOBACKUP/nii/typhoon_data/orig_image/198711', 277, 263)
('/NOBACKUP/nii/typhoon_data/orig_image/199229', 73, 69)
('/NOBACKUP/nii/typhoon_data/orig_image/200320', 187, 186)
('/NOBACKUP/nii/typhoon_data/orig_image/198824', 289, 240)
('/NOBACKUP/nii/typhoon_data/orig_image/200712', 133, 132)
('/NOBACKUP/nii/typhoon_data/orig_image/200804', 235, 234)
('/NOBACKUP/nii/typhoon_data/orig_image/200806', 223, 221)
('/NOBACKUP/nii/typhoon_data/orig_image/200007', 61, 58)
('/NOBACKUP/nii/typhoon_data/orig_image/197908', 139, 45)
('/NOBACKUP/nii/typhoon_data/orig_image/198413', 145, 25)
('/NOBACKUP/nii/typhoon_data/orig_image/199701', 301, 264)
('/NOBACKUP/nii/typhoon_data/orig_image/199128', 277, 268)
('/NOBACKUP/nii/typhoon_data/orig_image/200406', 403, 402)
('/NOBACKUP/nii/typhoon_data/orig_image/199814', 91, 88)
('/NOBACKUP/nii/typhoon_data/orig_image/198423', 253, 85)
('/NOBACKUP/nii/typhoon_data/orig_image/200509', 235, 230)
('/NOBACKUP/nii/typhoon_data/orig_image/199512', 175, 153)
('/NOBACKUP/nii/typhoon_data/orig_image/198920', 211, 192)
('/NOBACKUP/nii/typhoon_data/orig_image/201007', 217, 208)
('/NOBACKUP/nii/typhoon_data/orig_image/198309', 319, 106)
('/NOBACKUP/nii/typhoon_data/orig_image/200418', 373, 325)
('/NOBACKUP/nii/typhoon_data/orig_image/198215', 319, 104)
('/NOBACKUP/nii/typhoon_data/orig_image/200421', 331, 286)
('/NOBACKUP/nii/typhoon_data/orig_image/199716', 247, 231)
('/NOBACKUP/nii/typhoon_data/orig_image/200621', 259, 258)
('/NOBACKUP/nii/typhoon_data/orig_image/199211', 343, 329)
('/NOBACKUP/nii/typhoon_data/orig_image/200314', 289, 253)
('/NOBACKUP/nii/typhoon_data/orig_image/197907', 205, 67)
('/NOBACKUP/nii/typhoon_data/orig_image/199026', 67, 64)
('/NOBACKUP/nii/typhoon_data/orig_image/200021', 193, 179)
('/NOBACKUP/nii/typhoon_data/orig_image/198116', 181, 60)
('/NOBACKUP/nii/typhoon_data/orig_image/198710', 157, 153)
('/NOBACKUP/nii/typhoon_data/orig_image/201613', 55, 43)
('/NOBACKUP/nii/typhoon_data/orig_image/201503', 271, 270)
('/NOBACKUP/nii/typhoon_data/orig_image/197913', 223, 66)
('/NOBACKUP/nii/typhoon_data/orig_image/200315', 193, 169)
('/NOBACKUP/nii/typhoon_data/orig_image/199807', 175, 156)
('/NOBACKUP/nii/typhoon_data/orig_image/198617', 343, 107)
('/NOBACKUP/nii/typhoon_data/orig_image/199220', 145, 129)
('/NOBACKUP/nii/typhoon_data/orig_image/199006', 235, 222)
('/NOBACKUP/nii/typhoon_data/orig_image/200603', 325, 323)
('/NOBACKUP/nii/typhoon_data/orig_image/199017', 175, 160)
('/NOBACKUP/nii/typhoon_data/orig_image/199207', 181, 180)
('/NOBACKUP/nii/typhoon_data/orig_image/200120', 175, 139)
('/NOBACKUP/nii/typhoon_data/orig_image/200607', 259, 257)
('/NOBACKUP/nii/typhoon_data/orig_image/198510', 175, 59)
('/NOBACKUP/nii/typhoon_data/orig_image/198820', 139, 126)
('/NOBACKUP/nii/typhoon_data/orig_image/200014', 439, 383)
('/NOBACKUP/nii/typhoon_data/orig_image/198404', 163, 27)
('/NOBACKUP/nii/typhoon_data/orig_image/198626', 391, 126)
('/NOBACKUP/nii/typhoon_data/orig_image/198816', 151, 139)
('/NOBACKUP/nii/typhoon_data/orig_image/200523', 151, 149)
('/NOBACKUP/nii/typhoon_data/orig_image/199415', 121, 120)
('/NOBACKUP/nii/typhoon_data/orig_image/198502', 109, 37)
('/NOBACKUP/nii/typhoon_data/orig_image/198602', 217, 73)
('/NOBACKUP/nii/typhoon_data/orig_image/199315', 151, 137)
('/NOBACKUP/nii/typhoon_data/orig_image/201003', 151, 150)
('/NOBACKUP/nii/typhoon_data/orig_image/200708', 295, 294)
('/NOBACKUP/nii/typhoon_data/orig_image/199523', 241, 236)
('/NOBACKUP/nii/typhoon_data/orig_image/198902', 211, 197)
('/NOBACKUP/nii/typhoon_data/orig_image/199625', 217, 209)
('/NOBACKUP/nii/typhoon_data/orig_image/199125', 181, 174)
('/NOBACKUP/nii/typhoon_data/orig_image/198619', 157, 53)
('/NOBACKUP/nii/typhoon_data/orig_image/200204', 175, 171)
('/NOBACKUP/nii/typhoon_data/orig_image/198307', 157, 53)
('/NOBACKUP/nii/typhoon_data/orig_image/199004', 133, 129)
('/NOBACKUP/nii/typhoon_data/orig_image/200105', 85, 82)
('/NOBACKUP/nii/typhoon_data/orig_image/199326', 205, 197)
('/NOBACKUP/nii/typhoon_data/orig_image/201224', 325, 306)
('/NOBACKUP/nii/typhoon_data/orig_image/201001', 97, 96)
('/NOBACKUP/nii/typhoon_data/orig_image/199510', 103, 85)
('/NOBACKUP/nii/typhoon_data/orig_image/199226', 283, 274)
('/NOBACKUP/nii/typhoon_data/orig_image/201005', 73, 70)
('/NOBACKUP/nii/typhoon_data/orig_image/198601', 193, 56)
('/NOBACKUP/nii/typhoon_data/orig_image/198524', 253, 85)
('/NOBACKUP/nii/typhoon_data/orig_image/198811', 121, 118)
('/NOBACKUP/nii/typhoon_data/orig_image/199808', 61, 53)
('/NOBACKUP/nii/typhoon_data/orig_image/199007', 229, 215)
('/NOBACKUP/nii/typhoon_data/orig_image/198620', 259, 86)
('/NOBACKUP/nii/typhoon_data/orig_image/198108', 157, 53)
('/NOBACKUP/nii/typhoon_data/orig_image/200515', 205, 180)
('/NOBACKUP/nii/typhoon_data/orig_image/198427', 205, 69)
('/NOBACKUP/nii/typhoon_data/orig_image/198213', 409, 135)
('/NOBACKUP/nii/typhoon_data/orig_image/199802', 97, 94)
('/NOBACKUP/nii/typhoon_data/orig_image/198921', 205, 187)
('/NOBACKUP/nii/typhoon_data/orig_image/201509', 331, 330)
('/NOBACKUP/nii/typhoon_data/orig_image/199728', 373, 353)
('/NOBACKUP/nii/typhoon_data/orig_image/198830', 139, 63)
('/NOBACKUP/nii/typhoon_data/orig_image/200906', 103, 102)
('/NOBACKUP/nii/typhoon_data/orig_image/198322', 121, 38)
('/NOBACKUP/nii/typhoon_data/orig_image/198612', 427, 143)
('/NOBACKUP/nii/typhoon_data/orig_image/199517', 157, 128)
('/NOBACKUP/nii/typhoon_data/orig_image/199604', 187, 179)
('/NOBACKUP/nii/typhoon_data/orig_image/199906', 127, 115)
('/NOBACKUP/nii/typhoon_data/orig_image/198203', 145, 49)
('/NOBACKUP/nii/typhoon_data/orig_image/200301', 139, 137)
('/NOBACKUP/nii/typhoon_data/orig_image/199003', 157, 156)
('/NOBACKUP/nii/typhoon_data/orig_image/198419', 115, 39)
('/NOBACKUP/nii/typhoon_data/orig_image/198609', 109, 36)
('/NOBACKUP/nii/typhoon_data/orig_image/199118', 187, 169)
('/NOBACKUP/nii/typhoon_data/orig_image/200618', 211, 210)
('/NOBACKUP/nii/typhoon_data/orig_image/199024', 85, 81)
('/NOBACKUP/nii/typhoon_data/orig_image/198516', 193, 65)
('/NOBACKUP/nii/typhoon_data/orig_image/199219', 235, 216)
('/NOBACKUP/nii/typhoon_data/orig_image/198119', 157, 50)
('/NOBACKUP/nii/typhoon_data/orig_image/200009', 295, 277)
('/NOBACKUP/nii/typhoon_data/orig_image/198120', 457, 147)
('/NOBACKUP/nii/typhoon_data/orig_image/199804', 379, 341)
('/NOBACKUP/nii/typhoon_data/orig_image/199801', 61, 59)
('/NOBACKUP/nii/typhoon_data/orig_image/200303', 223, 210)
('/NOBACKUP/nii/typhoon_data/orig_image/198114', 205, 61)
('/NOBACKUP/nii/typhoon_data/orig_image/200201', 91, 89)
('/NOBACKUP/nii/typhoon_data/orig_image/200808', 187, 186)
('/NOBACKUP/nii/typhoon_data/orig_image/199705', 139, 126)
('/NOBACKUP/nii/typhoon_data/orig_image/200305', 205, 203)
('/NOBACKUP/nii/typhoon_data/orig_image/199417', 349, 338)
('/NOBACKUP/nii/typhoon_data/orig_image/200512', 199, 187)
('/NOBACKUP/nii/typhoon_data/orig_image/200516', 91, 80)
('/NOBACKUP/nii/typhoon_data/orig_image/199718', 259, 234)
('/NOBACKUP/nii/typhoon_data/orig_image/199618', 241, 209)
('/NOBACKUP/nii/typhoon_data/orig_image/200106', 319, 307)
('/NOBACKUP/nii/typhoon_data/orig_image/199713', 457, 433)
('/NOBACKUP/nii/typhoon_data/orig_image/199806', 103, 90)
('/NOBACKUP/nii/typhoon_data/orig_image/199316', 133, 119)
('/NOBACKUP/nii/typhoon_data/orig_image/199726', 85, 59)
('/NOBACKUP/nii/typhoon_data/orig_image/200102', 145, 140)
('/NOBACKUP/nii/typhoon_data/orig_image/199605', 307, 288)
('/NOBACKUP/nii/typhoon_data/orig_image/200604', 205, 204)
('/NOBACKUP/nii/typhoon_data/orig_image/201110', 223, 221)
('/NOBACKUP/nii/typhoon_data/orig_image/198211', 271, 91)
('/NOBACKUP/nii/typhoon_data/orig_image/199719', 379, 333)
('/NOBACKUP/nii/typhoon_data/orig_image/198926', 151, 138)
('/NOBACKUP/nii/typhoon_data/orig_image/199432', 313, 283)
('/NOBACKUP/nii/typhoon_data/orig_image/198214', 343, 115)
('/NOBACKUP/nii/typhoon_data/orig_image/197904', 205, 66)
('/NOBACKUP/nii/typhoon_data/orig_image/198806', 187, 142)
('/NOBACKUP/nii/typhoon_data/orig_image/199902', 133, 126)
('/NOBACKUP/nii/typhoon_data/orig_image/200312', 307, 271)
('/NOBACKUP/nii/typhoon_data/orig_image/198304', 97, 32)
('/NOBACKUP/nii/typhoon_data/orig_image/198814', 169, 160)
('/NOBACKUP/nii/typhoon_data/orig_image/198828', 211, 164)
('/NOBACKUP/nii/typhoon_data/orig_image/199028', 301, 273)
('/NOBACKUP/nii/typhoon_data/orig_image/198715', 319, 263)
('/NOBACKUP/nii/typhoon_data/orig_image/197830', 139, 3)
('/NOBACKUP/nii/typhoon_data/orig_image/200206', 361, 346)
('/NOBACKUP/nii/typhoon_data/orig_image/199012', 295, 294)
('/NOBACKUP/nii/typhoon_data/orig_image/199108', 145, 144)
('/NOBACKUP/nii/typhoon_data/orig_image/198616', 289, 97)
('/NOBACKUP/nii/typhoon_data/orig_image/198916', 97, 96)
('/NOBACKUP/nii/typhoon_data/orig_image/200013', 157, 137)
('/NOBACKUP/nii/typhoon_data/orig_image/201215', 325, 312)
('/NOBACKUP/nii/typhoon_data/orig_image/200709', 271, 264)
('/NOBACKUP/nii/typhoon_data/orig_image/200302', 391, 358)
('/NOBACKUP/nii/typhoon_data/orig_image/198818', 247, 225)
('/NOBACKUP/nii/typhoon_data/orig_image/200402', 235, 233)
('/NOBACKUP/nii/typhoon_data/orig_image/200321', 355, 354)
('/NOBACKUP/nii/typhoon_data/orig_image/200413', 223, 216)
('/NOBACKUP/nii/typhoon_data/orig_image/200208', 139, 132)
('/NOBACKUP/nii/typhoon_data/orig_image/198222', 241, 81)
('/NOBACKUP/nii/typhoon_data/orig_image/198618', 235, 79)
('/NOBACKUP/nii/typhoon_data/orig_image/198508', 427, 142)
('/NOBACKUP/nii/typhoon_data/orig_image/199301', 229, 207)
('/NOBACKUP/nii/typhoon_data/orig_image/200211', 205, 196)
('/NOBACKUP/nii/typhoon_data/orig_image/198219', 289, 96)
('/NOBACKUP/nii/typhoon_data/orig_image/199626', 229, 215)
('/NOBACKUP/nii/typhoon_data/orig_image/199911', 151, 143)
('/NOBACKUP/nii/typhoon_data/orig_image/198821', 79, 72)
('/NOBACKUP/nii/typhoon_data/orig_image/200419', 127, 112)
('/NOBACKUP/nii/typhoon_data/orig_image/199218', 355, 319)
('/NOBACKUP/nii/typhoon_data/orig_image/199612', 361, 348)
('/NOBACKUP/nii/typhoon_data/orig_image/198911', 199, 196)
('/NOBACKUP/nii/typhoon_data/orig_image/198210', 307, 103)
('/NOBACKUP/nii/typhoon_data/orig_image/200504', 295, 294)
('/NOBACKUP/nii/typhoon_data/orig_image/199117', 163, 149)
('/NOBACKUP/nii/typhoon_data/orig_image/198913', 283, 281)
('/NOBACKUP/nii/typhoon_data/orig_image/199912', 151, 143)
('/NOBACKUP/nii/typhoon_data/orig_image/198810', 265, 261)
('/NOBACKUP/nii/typhoon_data/orig_image/199029', 253, 244)
('/NOBACKUP/nii/typhoon_data/orig_image/198611', 175, 59)
('/NOBACKUP/nii/typhoon_data/orig_image/198905', 205, 199)
('/NOBACKUP/nii/typhoon_data/orig_image/200005', 151, 146)
('/NOBACKUP/nii/typhoon_data/orig_image/199113', 121, 118)
('/NOBACKUP/nii/typhoon_data/orig_image/199419', 211, 198)
('/NOBACKUP/nii/typhoon_data/orig_image/199425', 169, 152)
('/NOBACKUP/nii/typhoon_data/orig_image/198721', 151, 69)
('/NOBACKUP/nii/typhoon_data/orig_image/200417', 313, 271)
('/NOBACKUP/nii/typhoon_data/orig_image/199727', 169, 162)
('/NOBACKUP/nii/typhoon_data/orig_image/198418', 139, 47)
('/NOBACKUP/nii/typhoon_data/orig_image/200420', 205, 181)
('/NOBACKUP/nii/typhoon_data/orig_image/201316', 175, 168)
('/NOBACKUP/nii/typhoon_data/orig_image/198405', 355, 60)
('/NOBACKUP/nii/typhoon_data/orig_image/198504', 175, 59)
('/NOBACKUP/nii/typhoon_data/orig_image/198305', 481, 161)
('/NOBACKUP/nii/typhoon_data/orig_image/200221', 193, 113)
('/NOBACKUP/nii/typhoon_data/orig_image/201111', 241, 230)
('/NOBACKUP/nii/typhoon_data/orig_image/199613', 157, 152)
('/NOBACKUP/nii/typhoon_data/orig_image/198501', 205, 69)
('/NOBACKUP/nii/typhoon_data/orig_image/198603', 223, 71)
('/NOBACKUP/nii/typhoon_data/orig_image/201315', 127, 122)
('/NOBACKUP/nii/typhoon_data/orig_image/200308', 187, 185)
('/NOBACKUP/nii/typhoon_data/orig_image/198101', 151, 47)
('/NOBACKUP/nii/typhoon_data/orig_image/201605', 235, 234)
('/NOBACKUP/nii/typhoon_data/orig_image/198704', 217, 188)
('/NOBACKUP/nii/typhoon_data/orig_image/199518', 193, 165)
('/NOBACKUP/nii/typhoon_data/orig_image/199514', 259, 222)
('/NOBACKUP/nii/typhoon_data/orig_image/198807', 187, 182)
('/NOBACKUP/nii/typhoon_data/orig_image/201114', 115, 114)
('/NOBACKUP/nii/typhoon_data/orig_image/198319', 247, 82)
('/NOBACKUP/nii/typhoon_data/orig_image/200018', 205, 180)
('/NOBACKUP/nii/typhoon_data/orig_image/199809', 115, 99)
('/NOBACKUP/nii/typhoon_data/orig_image/199019', 247, 225)
('/NOBACKUP/nii/typhoon_data/orig_image/199401', 211, 186)
('/NOBACKUP/nii/typhoon_data/orig_image/200103', 103, 99)
('/NOBACKUP/nii/typhoon_data/orig_image/198129', 169, 57)
('/NOBACKUP/nii/typhoon_data/orig_image/199436', 271, 265)
('/NOBACKUP/nii/typhoon_data/orig_image/199123', 349, 347)
('/NOBACKUP/nii/typhoon_data/orig_image/200213', 337, 321)
('/NOBACKUP/nii/typhoon_data/orig_image/199025', 289, 282)
('/NOBACKUP/nii/typhoon_data/orig_image/199717', 85, 79)
('/NOBACKUP/nii/typhoon_data/orig_image/199522', 133, 126)
('/NOBACKUP/nii/typhoon_data/orig_image/199920', 229, 191)
('/NOBACKUP/nii/typhoon_data/orig_image/200222', 187, 160)
('/NOBACKUP/nii/typhoon_data/orig_image/198127', 145, 46)
('/NOBACKUP/nii/typhoon_data/orig_image/199115', 241, 219)
('/NOBACKUP/nii/typhoon_data/orig_image/200006', 217, 211)
('/NOBACKUP/nii/typhoon_data/orig_image/199615', 85, 79)
('/NOBACKUP/nii/typhoon_data/orig_image/199314', 223, 202)
('/NOBACKUP/nii/typhoon_data/orig_image/198515', 205, 69)
('/NOBACKUP/nii/typhoon_data/orig_image/198312', 127, 42)
('/NOBACKUP/nii/typhoon_data/orig_image/200224', 109, 106)
('/NOBACKUP/nii/typhoon_data/orig_image/198318', 73, 24)
('/NOBACKUP/nii/typhoon_data/orig_image/198604', 319, 103)
('/NOBACKUP/nii/typhoon_data/orig_image/199317', 157, 145)
('/NOBACKUP/nii/typhoon_data/orig_image/199431', 469, 427)
('/NOBACKUP/nii/typhoon_data/orig_image/198924', 127, 115)
('/NOBACKUP/nii/typhoon_data/orig_image/197911', 259, 86)
('/NOBACKUP/nii/typhoon_data/orig_image/200215', 301, 230)
('/NOBACKUP/nii/typhoon_data/orig_image/199509', 151, 132)
('/NOBACKUP/nii/typhoon_data/orig_image/200801', 163, 157)
('/NOBACKUP/nii/typhoon_data/orig_image/198827', 139, 131)
('/NOBACKUP/nii/typhoon_data/orig_image/198825', 67, 57)
('/NOBACKUP/nii/typhoon_data/orig_image/198822', 139, 126)
('/NOBACKUP/nii/typhoon_data/orig_image/200623', 121, 118)
('/NOBACKUP/nii/typhoon_data/orig_image/198407', 223, 38)
('/NOBACKUP/nii/typhoon_data/orig_image/199116', 121, 111)
('/NOBACKUP/nii/typhoon_data/orig_image/198521', 151, 50)
('/NOBACKUP/nii/typhoon_data/orig_image/198311', 109, 37)
('/NOBACKUP/nii/typhoon_data/orig_image/200110', 61, 60)
('/NOBACKUP/nii/typhoon_data/orig_image/198109', 109, 37)
('/NOBACKUP/nii/typhoon_data/orig_image/197906', 217, 72)
('/NOBACKUP/nii/typhoon_data/orig_image/198815', 211, 199)
('/NOBACKUP/nii/typhoon_data/orig_image/198624', 193, 64)
('/NOBACKUP/nii/typhoon_data/orig_image/199111', 175, 163)
('/NOBACKUP/nii/typhoon_data/orig_image/198907', 127, 125)
('/NOBACKUP/nii/typhoon_data/orig_image/199015', 253, 245)
('/NOBACKUP/nii/typhoon_data/orig_image/198702', 163, 91)
('/NOBACKUP/nii/typhoon_data/orig_image/200016', 157, 136)
('/NOBACKUP/nii/typhoon_data/orig_image/199206', 85, 83)
('/NOBACKUP/nii/typhoon_data/orig_image/200209', 355, 341)
('/NOBACKUP/nii/typhoon_data/orig_image/199714', 301, 282)
('/NOBACKUP/nii/typhoon_data/orig_image/201008', 55, 53)
('/NOBACKUP/nii/typhoon_data/orig_image/199216', 205, 193)
('/NOBACKUP/nii/typhoon_data/orig_image/198306', 97, 33)
('/NOBACKUP/nii/typhoon_data/orig_image/198812', 301, 297)
('/NOBACKUP/nii/typhoon_data/orig_image/199901', 205, 194)
('/NOBACKUP/nii/typhoon_data/orig_image/199429', 343, 309)
('/NOBACKUP/nii/typhoon_data/orig_image/201414', 133, 131)
('/NOBACKUP/nii/typhoon_data/orig_image/198207', 109, 37)
('/NOBACKUP/nii/typhoon_data/orig_image/199703', 97, 94)
('/NOBACKUP/nii/typhoon_data/orig_image/198518', 127, 43)
('/NOBACKUP/nii/typhoon_data/orig_image/200307', 253, 250)
('/NOBACKUP/nii/typhoon_data/orig_image/199515', 175, 147)
('/NOBACKUP/nii/typhoon_data/orig_image/199327', 313, 302)
('/NOBACKUP/nii/typhoon_data/orig_image/199516', 181, 147)
('/NOBACKUP/nii/typhoon_data/orig_image/199914', 139, 115)
('/NOBACKUP/nii/typhoon_data/orig_image/197901', 349, 111)
('/NOBACKUP/nii/typhoon_data/orig_image/199101', 271, 237)
('/NOBACKUP/nii/typhoon_data/orig_image/201313', 205, 199)
('/NOBACKUP/nii/typhoon_data/orig_image/198220', 109, 36)
('/NOBACKUP/nii/typhoon_data/orig_image/200517', 217, 190)
('/NOBACKUP/nii/typhoon_data/orig_image/200125', 307, 294)
('/NOBACKUP/nii/typhoon_data/orig_image/200122', 277, 256)
('/NOBACKUP/nii/typhoon_data/orig_image/197910', 325, 108)
('/NOBACKUP/nii/typhoon_data/orig_image/198706', 145, 112)
('/NOBACKUP/nii/typhoon_data/orig_image/198716', 325, 288)
('/NOBACKUP/nii/typhoon_data/orig_image/198409', 229, 38)
('/NOBACKUP/nii/typhoon_data/orig_image/200513', 193, 171)
('/NOBACKUP/nii/typhoon_data/orig_image/201119', 217, 212)
('/NOBACKUP/nii/typhoon_data/orig_image/200124', 133, 129)
('/NOBACKUP/nii/typhoon_data/orig_image/199027', 331, 300)
('/NOBACKUP/nii/typhoon_data/orig_image/200119', 199, 166)
('/NOBACKUP/nii/typhoon_data/orig_image/198507', 313, 104)
('/NOBACKUP/nii/typhoon_data/orig_image/198610', 253, 80)
('/NOBACKUP/nii/typhoon_data/orig_image/197902', 151, 45)
('/NOBACKUP/nii/typhoon_data/orig_image/200212', 121, 116)
('/NOBACKUP/nii/typhoon_data/orig_image/200015', 175, 153)
('/NOBACKUP/nii/typhoon_data/orig_image/199909', 127, 124)
('/NOBACKUP/nii/typhoon_data/orig_image/199609', 301, 284)
('/NOBACKUP/nii/typhoon_data/orig_image/199104', 295, 289)
('/NOBACKUP/nii/typhoon_data/orig_image/199210', 253, 241)
('/NOBACKUP/nii/typhoon_data/orig_image/199119', 433, 391)
('/NOBACKUP/nii/typhoon_data/orig_image/199105', 127, 120)
('/NOBACKUP/nii/typhoon_data/orig_image/198308', 319, 107)
('/NOBACKUP/nii/typhoon_data/orig_image/199103', 121, 119)
('/NOBACKUP/nii/typhoon_data/orig_image/199722', 187, 136)
('/NOBACKUP/nii/typhoon_data/orig_image/200117', 127, 105)
('/NOBACKUP/nii/typhoon_data/orig_image/199313', 229, 215)
('/NOBACKUP/nii/typhoon_data/orig_image/198906', 139, 138)
('/NOBACKUP/nii/typhoon_data/orig_image/198829', 235, 159)
('/NOBACKUP/nii/typhoon_data/orig_image/199720', 241, 213)
('/NOBACKUP/nii/typhoon_data/orig_image/200216', 301, 249)
('/NOBACKUP/nii/typhoon_data/orig_image/198123', 133, 41)
('/NOBACKUP/nii/typhoon_data/orig_image/200104', 139, 133)
('/NOBACKUP/nii/typhoon_data/orig_image/197921', 163, 55)
('/NOBACKUP/nii/typhoon_data/orig_image/198202', 325, 107)
('/NOBACKUP/nii/typhoon_data/orig_image/199919', 103, 89)
('/NOBACKUP/nii/typhoon_data/orig_image/199422', 133, 120)
('/NOBACKUP/nii/typhoon_data/orig_image/199614', 361, 314)
('/NOBACKUP/nii/typhoon_data/orig_image/198314', 121, 39)
('/NOBACKUP/nii/typhoon_data/orig_image/201109', 469, 467)
('/NOBACKUP/nii/typhoon_data/orig_image/198121', 133, 43)
('/NOBACKUP/nii/typhoon_data/orig_image/198113', 211, 63)
('/NOBACKUP/nii/typhoon_data/orig_image/198216', 85, 28)
('/NOBACKUP/nii/typhoon_data/orig_image/198522', 265, 88)
('/NOBACKUP/nii/typhoon_data/orig_image/199121', 307, 272)
('/NOBACKUP/nii/typhoon_data/orig_image/198701', 157, 50)
('/NOBACKUP/nii/typhoon_data/orig_image/198621', 241, 80)
('/NOBACKUP/nii/typhoon_data/orig_image/200914', 211, 210)
('/NOBACKUP/nii/typhoon_data/orig_image/197922', 157, 50)
('/NOBACKUP/nii/typhoon_data/orig_image/201309', 139, 138)
('/NOBACKUP/nii/typhoon_data/orig_image/199511', 127, 101)
('/NOBACKUP/nii/typhoon_data/orig_image/200121', 169, 155)
('/NOBACKUP/nii/typhoon_data/orig_image/200311', 43, 37)
('/NOBACKUP/nii/typhoon_data/orig_image/199319', 217, 195)
('/NOBACKUP/nii/typhoon_data/orig_image/200502', 175, 152)
('/NOBACKUP/nii/typhoon_data/orig_image/198313', 151, 50)
('/NOBACKUP/nii/typhoon_data/orig_image/198320', 169, 57)
('/NOBACKUP/nii/typhoon_data/orig_image/199228', 265, 259)
('/NOBACKUP/nii/typhoon_data/orig_image/198225', 103, 35)
('/NOBACKUP/nii/typhoon_data/orig_image/200123', 157, 151)
('/NOBACKUP/nii/typhoon_data/orig_image/200910', 247, 244)
('/NOBACKUP/nii/typhoon_data/orig_image/199430', 259, 229)
('/NOBACKUP/nii/typhoon_data/orig_image/198918', 97, 94)
('/NOBACKUP/nii/typhoon_data/orig_image/201112', 349, 335)
('/NOBACKUP/nii/typhoon_data/orig_image/199608', 181, 172)
('/NOBACKUP/nii/typhoon_data/orig_image/198719', 247, 213)
('/NOBACKUP/nii/typhoon_data/orig_image/198831', 127, 110)
('/NOBACKUP/nii/typhoon_data/orig_image/200218', 145, 126)
('/NOBACKUP/nii/typhoon_data/orig_image/200107', 85, 79)
('/NOBACKUP/nii/typhoon_data/orig_image/198401', 85, 0)
('/NOBACKUP/nii/typhoon_data/orig_image/197912', 253, 75)
('/NOBACKUP/nii/typhoon_data/orig_image/200820', 103, 102)
('/NOBACKUP/nii/typhoon_data/orig_image/198819', 145, 132)
('/NOBACKUP/nii/typhoon_data/orig_image/198930', 199, 196)
('/NOBACKUP/nii/typhoon_data/orig_image/199325', 163, 157)
('/NOBACKUP/nii/typhoon_data/orig_image/198519', 193, 65)
('/NOBACKUP/nii/typhoon_data/orig_image/200023', 169, 160)
('/NOBACKUP/nii/typhoon_data/orig_image/198801', 289, 153)
('/NOBACKUP/nii/typhoon_data/orig_image/199620', 295, 257)
('/NOBACKUP/nii/typhoon_data/orig_image/198115', 229, 77)
('/NOBACKUP/nii/typhoon_data/orig_image/199208', 91, 89)
('/NOBACKUP/nii/typhoon_data/orig_image/199513', 241, 206)
('/NOBACKUP/nii/typhoon_data/orig_image/198208', 187, 61)
('/NOBACKUP/nii/typhoon_data/orig_image/198223', 349, 117)
('/NOBACKUP/nii/typhoon_data/orig_image/199711', 307, 293)
('/NOBACKUP/nii/typhoon_data/orig_image/200505', 259, 258)
('/NOBACKUP/nii/typhoon_data/orig_image/200109', 223, 215)
('/NOBACKUP/nii/typhoon_data/orig_image/199428', 64, 60)
('/NOBACKUP/nii/typhoon_data/orig_image/198503', 205, 69)
('/NOBACKUP/nii/typhoon_data/orig_image/199506', 79, 74)
('/NOBACKUP/nii/typhoon_data/orig_image/199917', 145, 127)
('/NOBACKUP/nii/typhoon_data/orig_image/199434', 343, 321)
('/NOBACKUP/nii/typhoon_data/orig_image/200004', 205, 184)
('/NOBACKUP/nii/typhoon_data/orig_image/199426', 361, 327)
('/NOBACKUP/nii/typhoon_data/orig_image/199410', 175, 172)
('/NOBACKUP/nii/typhoon_data/orig_image/201006', 181, 174)
('/NOBACKUP/nii/typhoon_data/orig_image/200616', 187, 185)
('/NOBACKUP/nii/typhoon_data/orig_image/199202', 115, 111)
('/NOBACKUP/nii/typhoon_data/orig_image/200510', 121, 119)
('/NOBACKUP/nii/typhoon_data/orig_image/200115', 235, 194)
('/NOBACKUP/nii/typhoon_data/orig_image/198614', 523, 175)
('/NOBACKUP/nii/typhoon_data/orig_image/198931', 337, 310)
('/NOBACKUP/nii/typhoon_data/orig_image/198212', 241, 79)
('/NOBACKUP/nii/typhoon_data/orig_image/198803', 61, 60)
('/NOBACKUP/nii/typhoon_data/orig_image/199312', 181, 179)
('/NOBACKUP/nii/typhoon_data/orig_image/199222', 283, 259)
('/NOBACKUP/nii/typhoon_data/orig_image/199908', 151, 144)
('/NOBACKUP/nii/typhoon_data/orig_image/199812', 79, 78)
('/NOBACKUP/nii/typhoon_data/orig_image/200306', 289, 284)
('/NOBACKUP/nii/typhoon_data/orig_image/199619', 187, 162)
('/NOBACKUP/nii/typhoon_data/orig_image/198110', 157, 52)
('/NOBACKUP/nii/typhoon_data/orig_image/198629', 307, 95)
('/NOBACKUP/nii/typhoon_data/orig_image/199231', 181, 178)
('/NOBACKUP/nii/typhoon_data/orig_image/200219', 127, 111)
('/NOBACKUP/nii/typhoon_data/orig_image/198514', 217, 73)
('/NOBACKUP/nii/typhoon_data/orig_image/199710', 343, 294)
('/NOBACKUP/nii/typhoon_data/orig_image/198823', 127, 116)
('/NOBACKUP/nii/typhoon_data/orig_image/198904', 109, 108)
('/NOBACKUP/nii/typhoon_data/orig_image/198805', 79, 55)
('/NOBACKUP/nii/typhoon_data/orig_image/198506', 325, 109)
('/NOBACKUP/nii/typhoon_data/orig_image/199404', 109, 99)
('/NOBACKUP/nii/typhoon_data/orig_image/198623', 151, 51)
('/NOBACKUP/nii/typhoon_data/orig_image/198606', 253, 83)
('/NOBACKUP/nii/typhoon_data/orig_image/199402', 193, 186)
('/NOBACKUP/nii/typhoon_data/orig_image/200012', 253, 227)
('/NOBACKUP/nii/typhoon_data/orig_image/200911', 127, 122)
('/NOBACKUP/nii/typhoon_data/orig_image/198323', 79, 26)
('/NOBACKUP/nii/typhoon_data/orig_image/199702', 103, 100)
('/NOBACKUP/nii/typhoon_data/orig_image/198628', 325, 104)
('/NOBACKUP/nii/typhoon_data/orig_image/200520', 253, 232)
('/NOBACKUP/nii/typhoon_data/orig_image/198224', 355, 119)
('/NOBACKUP/nii/typhoon_data/orig_image/201317', 127, 122)
('/NOBACKUP/nii/typhoon_data/orig_image/199413', 283, 281)
('/NOBACKUP/nii/typhoon_data/orig_image/199424', 253, 224)
('/NOBACKUP/nii/typhoon_data/orig_image/198917', 217, 204)
('/NOBACKUP/nii/typhoon_data/orig_image/199611', 103, 99)
('/NOBACKUP/nii/typhoon_data/orig_image/200205', 235, 226)
('/NOBACKUP/nii/typhoon_data/orig_image/199433', 109, 97)
('/NOBACKUP/nii/typhoon_data/orig_image/199725', 337, 289)
('/NOBACKUP/nii/typhoon_data/orig_image/200202', 253, 202)
('/NOBACKUP/nii/typhoon_data/orig_image/199505', 139, 131)
('/NOBACKUP/nii/typhoon_data/orig_image/199021', 211, 191)
('/NOBACKUP/nii/typhoon_data/orig_image/198426', 217, 72)
('/NOBACKUP/nii/typhoon_data/orig_image/200620', 157, 156)
('/NOBACKUP/nii/typhoon_data/orig_image/200506', 235, 226)
('/NOBACKUP/nii/typhoon_data/orig_image/199913', 163, 149)
('/NOBACKUP/nii/typhoon_data/orig_image/198221', 271, 90)
('/NOBACKUP/nii/typhoon_data/orig_image/199610', 181, 173)
('/NOBACKUP/nii/typhoon_data/orig_image/200108', 205, 197)
('/NOBACKUP/nii/typhoon_data/orig_image/200902', 247, 246)
('/NOBACKUP/nii/typhoon_data/orig_image/199803', 133, 129)
('/NOBACKUP/nii/typhoon_data/orig_image/200408', 433, 430)
('/NOBACKUP/nii/typhoon_data/orig_image/198103', 217, 72)
('/NOBACKUP/nii/typhoon_data/orig_image/200403', 133, 132)
('/NOBACKUP/nii/typhoon_data/orig_image/199110', 223, 211)
('/NOBACKUP/nii/typhoon_data/orig_image/201113', 175, 170)
('/NOBACKUP/nii/typhoon_data/orig_image/199707', 187, 181)
('/NOBACKUP/nii/typhoon_data/orig_image/199018', 253, 232)
('/NOBACKUP/nii/typhoon_data/orig_image/199521', 193, 181)
('/NOBACKUP/nii/typhoon_data/orig_image/199001', 139, 138)
('/NOBACKUP/nii/typhoon_data/orig_image/201117', 187, 182)
('/NOBACKUP/nii/typhoon_data/orig_image/198402', 217, 0)
('/NOBACKUP/nii/typhoon_data/orig_image/198302', 103, 35)
('/NOBACKUP/nii/typhoon_data/orig_image/198513', 223, 75)
('/NOBACKUP/nii/typhoon_data/orig_image/199213', 121, 117)
('/NOBACKUP/nii/typhoon_data/orig_image/201013', 271, 270)
('/NOBACKUP/nii/typhoon_data/orig_image/198303', 211, 71)
('/NOBACKUP/nii/typhoon_data/orig_image/199308', 199, 198)
('/NOBACKUP/nii/typhoon_data/orig_image/200619', 301, 300)
('/NOBACKUP/nii/typhoon_data/orig_image/198509', 169, 56)
('/NOBACKUP/nii/typhoon_data/orig_image/198107', 133, 45)
('/NOBACKUP/nii/typhoon_data/orig_image/200220', 145, 118)
('/NOBACKUP/nii/typhoon_data/orig_image/198117', 133, 45)
('/NOBACKUP/nii/typhoon_data/orig_image/198415', 133, 43)
('/NOBACKUP/nii/typhoon_data/orig_image/200316', 229, 200)
('/NOBACKUP/nii/typhoon_data/orig_image/200310', 217, 216)
('/NOBACKUP/nii/typhoon_data/orig_image/200113', 115, 97)
('/NOBACKUP/nii/typhoon_data/orig_image/199102', 181, 166)
('/NOBACKUP/nii/typhoon_data/orig_image/198104', 211, 68)
('/NOBACKUP/nii/typhoon_data/orig_image/199519', 103, 93)
('/NOBACKUP/nii/typhoon_data/orig_image/199421', 187, 170)
('/NOBACKUP/nii/typhoon_data/orig_image/200416', 439, 383)
('/NOBACKUP/nii/typhoon_data/orig_image/198218', 247, 81)
('/NOBACKUP/nii/typhoon_data/orig_image/199407', 349, 348)
('/NOBACKUP/nii/typhoon_data/orig_image/197905', 187, 62)
('/NOBACKUP/nii/typhoon_data/orig_image/199225', 199, 189)
('/NOBACKUP/nii/typhoon_data/orig_image/200518', 217, 190)
('/NOBACKUP/nii/typhoon_data/orig_image/201202', 235, 233)
('/NOBACKUP/nii/typhoon_data/orig_image/198125', 241, 73)
('/NOBACKUP/nii/typhoon_data/orig_image/199212', 115, 112)
('/NOBACKUP/nii/typhoon_data/orig_image/199107', 151, 139)
('/NOBACKUP/nii/typhoon_data/orig_image/200514', 283, 242)
('/NOBACKUP/nii/typhoon_data/orig_image/198912', 247, 245)
('/NOBACKUP/nii/typhoon_data/orig_image/199508', 181, 157)
('/NOBACKUP/nii/typhoon_data/orig_image/198915', 295, 292)
('/NOBACKUP/nii/typhoon_data/orig_image/199311', 223, 222)
('/NOBACKUP/nii/typhoon_data/orig_image/198909', 157, 155)
('/NOBACKUP/nii/typhoon_data/orig_image/198425', 367, 123)
('/NOBACKUP/nii/typhoon_data/orig_image/198607', 307, 102)
('/NOBACKUP/nii/typhoon_data/orig_image/200226', 229, 220)
('/NOBACKUP/nii/typhoon_data/orig_image/200203', 199, 194)
('/NOBACKUP/nii/typhoon_data/orig_image/198112', 145, 49)
('/NOBACKUP/nii/typhoon_data/orig_image/198928', 265, 248)
('/NOBACKUP/nii/typhoon_data/orig_image/199602', 211, 187)
('/NOBACKUP/nii/typhoon_data/orig_image/198106', 211, 67)
('/NOBACKUP/nii/typhoon_data/orig_image/199120', 463, 420)
('/NOBACKUP/nii/typhoon_data/orig_image/200210', 25, 24)
('/NOBACKUP/nii/typhoon_data/orig_image/199109', 211, 209)
('/NOBACKUP/nii/typhoon_data/orig_image/198826', 253, 241)
('/NOBACKUP/nii/typhoon_data/orig_image/198923', 307, 275)
('/NOBACKUP/nii/typhoon_data/orig_image/198316', 181, 61)
('/NOBACKUP/nii/typhoon_data/orig_image/198201', 217, 72)
('/NOBACKUP/nii/typhoon_data/orig_image/200401', 367, 317)
('/NOBACKUP/nii/typhoon_data/orig_image/199805', 151, 134)
('/NOBACKUP/nii/typhoon_data/orig_image/200423', 265, 259)
('/NOBACKUP/nii/typhoon_data/orig_image/198613', 481, 161)
('/NOBACKUP/nii/typhoon_data/orig_image/199322', 169, 168)
('/NOBACKUP/nii/typhoon_data/orig_image/200022', 103, 99)
('/NOBACKUP/nii/typhoon_data/orig_image/199715', 91, 87)
('/NOBACKUP/nii/typhoon_data/orig_image/199201', 271, 249)
('/NOBACKUP/nii/typhoon_data/orig_image/199706', 283, 275)
('/NOBACKUP/nii/typhoon_data/orig_image/199204', 193, 191)
('/NOBACKUP/nii/typhoon_data/orig_image/198520', 211, 71)
('/NOBACKUP/nii/typhoon_data/orig_image/198411', 277, 46)
('/NOBACKUP/nii/typhoon_data/orig_image/199302', 325, 311)
('/NOBACKUP/nii/typhoon_data/orig_image/199230', 463, 450)
('/NOBACKUP/nii/typhoon_data/orig_image/198718', 115, 105)
('/NOBACKUP/nii/typhoon_data/orig_image/198416', 139, 42)
('/NOBACKUP/nii/typhoon_data/orig_image/198625', 187, 56)
('/NOBACKUP/nii/typhoon_data/orig_image/199904', 103, 97)
('/NOBACKUP/nii/typhoon_data/orig_image/199020', 235, 215)
('/NOBACKUP/nii/typhoon_data/orig_image/199223', 181, 165)
('/NOBACKUP/nii/typhoon_data/orig_image/200001', 289, 277)
('/NOBACKUP/nii/typhoon_data/orig_image/198315', 73, 25)
('/NOBACKUP/nii/typhoon_data/orig_image/200614', 259, 256)
('/NOBACKUP/nii/typhoon_data/orig_image/199221', 265, 237)
('/NOBACKUP/nii/typhoon_data/orig_image/200207', 325, 311)
('/NOBACKUP/nii/typhoon_data/orig_image/199321', 169, 147)
('/NOBACKUP/nii/typhoon_data/orig_image/201606', 217, 216)
('/NOBACKUP/nii/typhoon_data/orig_image/199504', 139, 128)
('/NOBACKUP/nii/typhoon_data/orig_image/199816', 73, 71)
('/NOBACKUP/nii/typhoon_data/orig_image/199704', 199, 186)
('/NOBACKUP/nii/typhoon_data/orig_image/199112', 235, 230)
('/NOBACKUP/nii/typhoon_data/orig_image/197903', 253, 83)
('/NOBACKUP/nii/typhoon_data/orig_image/199810', 241, 204)
('/NOBACKUP/nii/typhoon_data/orig_image/197916', 283, 83)
('/NOBACKUP/nii/typhoon_data/orig_image/198708', 277, 238)
('/NOBACKUP/nii/typhoon_data/orig_image/199723', 331, 311)
('/NOBACKUP/nii/typhoon_data/orig_image/199907', 175, 163)
('/NOBACKUP/nii/typhoon_data/orig_image/197915', 205, 61)
('/NOBACKUP/nii/typhoon_data/orig_image/199203', 241, 238)
('/NOBACKUP/nii/typhoon_data/orig_image/198417', 91, 31)
('/NOBACKUP/nii/typhoon_data/orig_image/200519', 193, 177)
('/NOBACKUP/nii/typhoon_data/orig_image/199503', 211, 192)
('/NOBACKUP/nii/typhoon_data/orig_image/198714', 187, 167)
('/NOBACKUP/nii/typhoon_data/orig_image/198301', 85, 29)
('/NOBACKUP/nii/typhoon_data/orig_image/201511', 373, 372)
('/NOBACKUP/nii/typhoon_data/orig_image/198927', 223, 203)
('/NOBACKUP/nii/typhoon_data/orig_image/198424', 211, 68)
('/NOBACKUP/nii/typhoon_data/orig_image/200422', 163, 137)
('/NOBACKUP/nii/typhoon_data/orig_image/199435', 277, 266)
('/NOBACKUP/nii/typhoon_data/orig_image/199122', 241, 213)
('/NOBACKUP/nii/typhoon_data/orig_image/199124', 343, 342)
('/NOBACKUP/nii/typhoon_data/orig_image/201312', 187, 184)
('/NOBACKUP/nii/typhoon_data/orig_image/198523', 181, 61)
('/NOBACKUP/nii/typhoon_data/orig_image/198925', 211, 177)
('/NOBACKUP/nii/typhoon_data/orig_image/200116', 385, 319)
('/NOBACKUP/nii/typhoon_data/orig_image/199310', 349, 342)
('/NOBACKUP/nii/typhoon_data/orig_image/198932', 157, 152)
('/NOBACKUP/nii/typhoon_data/orig_image/199318', 181, 167)
('/NOBACKUP/nii/typhoon_data/orig_image/198712', 337, 271)
('/NOBACKUP/nii/typhoon_data/orig_image/200812', 139, 138)
('/NOBACKUP/nii/typhoon_data/orig_image/199126', 205, 204)
('/NOBACKUP/nii/typhoon_data/orig_image/198321', 217, 72)
('/NOBACKUP/nii/typhoon_data/orig_image/199624', 271, 254)
('/NOBACKUP/nii/typhoon_data/orig_image/198102', 211, 71)
('/NOBACKUP/nii/typhoon_data/orig_image/200425', 289, 288)
('/NOBACKUP/nii/typhoon_data/orig_image/198126', 217, 67)
('/NOBACKUP/nii/typhoon_data/orig_image/199601', 91, 77)
('/NOBACKUP/nii/typhoon_data/orig_image/201002', 157, 156)
('/NOBACKUP/nii/typhoon_data/orig_image/199721', 139, 113)
('/NOBACKUP/nii/typhoon_data/orig_image/198505', 205, 69)
('/NOBACKUP/nii/typhoon_data/orig_image/200507', 259, 255)
('/NOBACKUP/nii/typhoon_data/orig_image/199724', 277, 258)
('/NOBACKUP/nii/typhoon_data/orig_image/200225', 157, 151)
('/NOBACKUP/nii/typhoon_data/orig_image/198813', 61, 59)
('/NOBACKUP/nii/typhoon_data/orig_image/200019', 175, 169)
('/NOBACKUP/nii/typhoon_data/orig_image/200424', 313, 310)
('/NOBACKUP/nii/typhoon_data/orig_image/200214', 115, 111)
('/NOBACKUP/nii/typhoon_data/orig_image/198723', 229, 102)
('/NOBACKUP/nii/typhoon_data/orig_image/199910', 223, 208)
('/NOBACKUP/nii/typhoon_data/orig_image/197923', 343, 2)
('/NOBACKUP/nii/typhoon_data/orig_image/198713', 385, 331)
('/NOBACKUP/nii/typhoon_data/orig_image/199114', 115, 107)
('/NOBACKUP/nii/typhoon_data/orig_image/200002', 169, 167)
('/NOBACKUP/nii/typhoon_data/orig_image/199008', 307, 303)
('/NOBACKUP/nii/typhoon_data/orig_image/200617', 91, 90)
('/NOBACKUP/nii/typhoon_data/orig_image/199224', 373, 352)
('/NOBACKUP/nii/typhoon_data/orig_image/200008', 283, 272)
('/NOBACKUP/nii/typhoon_data/orig_image/200511', 307, 281)
('/NOBACKUP/nii/typhoon_data/orig_image/198512', 259, 87)
('/NOBACKUP/nii/typhoon_data/orig_image/199403', 211, 209)
('/NOBACKUP/nii/typhoon_data/orig_image/198128', 313, 97)
('/NOBACKUP/nii/typhoon_data/orig_image/198206', 271, 91)
('/NOBACKUP/nii/typhoon_data/orig_image/198317', 199, 66)
('/NOBACKUP/nii/typhoon_data/orig_image/198209', 265, 89)
('/NOBACKUP/nii/typhoon_data/orig_image/199921', 121, 114)
('/NOBACKUP/nii/typhoon_data/orig_image/198929', 97, 90)
('/NOBACKUP/nii/typhoon_data/orig_image/199617', 391, 339)
('/NOBACKUP/nii/typhoon_data/orig_image/199905', 151, 122)
('/NOBACKUP/nii/typhoon_data/orig_image/199811', 379, 353)
('/NOBACKUP/nii/typhoon_data/orig_image/198817', 211, 192)
('/NOBACKUP/nii/typhoon_data/orig_image/198414', 109, 21)
('/NOBACKUP/nii/typhoon_data/orig_image/198105', 265, 83)
('/NOBACKUP/nii/typhoon_data/orig_image/199606', 337, 323)
('/NOBACKUP/nii/typhoon_data/orig_image/198403', 121, 16)
('/NOBACKUP/nii/typhoon_data/orig_image/199501', 151, 146)
('/NOBACKUP/nii/typhoon_data/orig_image/200118', 217, 180)
('/NOBACKUP/nii/typhoon_data/orig_image/198204', 199, 66)
('/NOBACKUP/nii/typhoon_data/orig_image/199323', 205, 204)
('/NOBACKUP/nii/typhoon_data/orig_image/201009', 211, 206)
('/NOBACKUP/nii/typhoon_data/orig_image/200020', 205, 195)
('/NOBACKUP/nii/typhoon_data/orig_image/201510', 199, 198)
('/NOBACKUP/nii/typhoon_data/orig_image/197909', 241, 78)
('/NOBACKUP/nii/typhoon_data/orig_image/199320', 247, 221)
('/NOBACKUP/nii/typhoon_data/orig_image/200003', 193, 172)
('/NOBACKUP/nii/typhoon_data/orig_image/199412', 115, 113)
('/NOBACKUP/nii/typhoon_data/orig_image/199227', 265, 262)
('/NOBACKUP/nii/typhoon_data/orig_image/200521', 121, 118)
('/NOBACKUP/nii/typhoon_data/orig_image/198205', 205, 67)
('/NOBACKUP/nii/typhoon_data/orig_image/198111', 61, 21)
('/NOBACKUP/nii/typhoon_data/orig_image/200017', 151, 133)
('/NOBACKUP/nii/typhoon_data/orig_image/198627', 145, 49)
('/NOBACKUP/nii/typhoon_data/orig_image/200805', 259, 258)
('/NOBACKUP/nii/typhoon_data/orig_image/199127', 139, 137)
('/NOBACKUP/nii/typhoon_data/orig_image/198707', 337, 288)
('/NOBACKUP/nii/typhoon_data/orig_image/198406', 217, 37)
('/NOBACKUP/nii/typhoon_data/orig_image/197920', 433, 131)
('/NOBACKUP/nii/typhoon_data/orig_image/200503', 193, 172)
('/NOBACKUP/nii/typhoon_data/orig_image/198608', 175, 58)
('/NOBACKUP/nii/typhoon_data/orig_image/199813', 79, 77)
('/NOBACKUP/nii/typhoon_data/orig_image/199423', 169, 150)
('/NOBACKUP/nii/typhoon_data/orig_image/199607', 133, 127)
('/NOBACKUP/nii/typhoon_data/orig_image/198408', 379, 62)
('/NOBACKUP/nii/typhoon_data/orig_image/199307', 319, 316)
('/NOBACKUP/nii/typhoon_data/orig_image/199011', 229, 226)
('/NOBACKUP/nii/typhoon_data/orig_image/198605', 115, 39)
('/NOBACKUP/nii/typhoon_data/orig_image/198118', 295, 99)
('/NOBACKUP/nii/typhoon_data/orig_image/198919', 217, 197)
('/NOBACKUP/nii/typhoon_data/orig_image/199922', 85, 82)
('/NOBACKUP/nii/typhoon_data/orig_image/199328', 277, 273)
('/NOBACKUP/nii/typhoon_data/orig_image/200605', 223, 220)
('/NOBACKUP/nii/typhoon_data/orig_image/199411', 217, 213)
('/NOBACKUP/nii/typhoon_data/orig_image/198124', 235, 77)
('/NOBACKUP/nii/typhoon_data/orig_image/199502', 277, 248)
('/NOBACKUP/nii/typhoon_data/orig_image/200101', 145, 140)
('/NOBACKUP/nii/typhoon_data/orig_image/198910', 121, 116)
('/NOBACKUP/nii/typhoon_data/orig_image/197918', 175, 49)
('/NOBACKUP/nii/typhoon_data/orig_image/198420', 121, 41)
('/NOBACKUP/nii/typhoon_data/orig_image/200217', 313, 261)
('/NOBACKUP/nii/typhoon_data/orig_image/200114', 103, 81)
('/NOBACKUP/nii/typhoon_data/orig_image/200313', 151, 132)
('/NOBACKUP/nii/typhoon_data/orig_image/199708', 283, 279)
('/NOBACKUP/nii/typhoon_data/orig_image/198901', 139, 137)
('/NOBACKUP/nii/typhoon_data/orig_image/197914', 301, 88)
('/NOBACKUP/nii/typhoon_data/orig_image/199916', 151, 130)
('/NOBACKUP/nii/typhoon_data/orig_image/199616', 139, 121)
('/NOBACKUP/nii/typhoon_data/orig_image/198717', 223, 200)
('/NOBACKUP/nii/typhoon_data/orig_image/200415', 199, 175)
('/NOBACKUP/nii/typhoon_data/orig_image/197919', 391, 113)
('/NOBACKUP/nii/typhoon_data/orig_image/198709', 229, 160)
('/NOBACKUP/nii/typhoon_data/orig_image/199603', 283, 272)
('/NOBACKUP/nii/typhoon_data/orig_image/199420', 277, 255)
('/NOBACKUP/nii/typhoon_data/orig_image/199621', 283, 248)
('/NOBACKUP/nii/typhoon_data/orig_image/200309', 91, 88)
('/NOBACKUP/nii/typhoon_data/orig_image/199918', 295, 260)
('/NOBACKUP/nii/typhoon_data/orig_image/199815', 145, 140)
('/NOBACKUP/nii/typhoon_data/orig_image/198804', 187, 138)
('/NOBACKUP/nii/typhoon_data/orig_image/200010', 211, 204)
('/NOBACKUP/nii/typhoon_data/orig_image/199622', 253, 229)
('/NOBACKUP/nii/typhoon_data/orig_image/200304', 241, 238)
('/NOBACKUP/nii/typhoon_data/orig_image/198922', 97, 89)
('/NOBACKUP/nii/typhoon_data/orig_image/199903', 187, 181)
('/NOBACKUP/nii/typhoon_data/orig_image/198517', 127, 43)
('/NOBACKUP/nii/typhoon_data/orig_image/200111', 223, 213)
('/NOBACKUP/nii/typhoon_data/orig_image/198410', 193, 32)
('/NOBACKUP/nii/typhoon_data/orig_image/198705', 265, 240)
('/NOBACKUP/nii/typhoon_data/orig_image/199022', 49, 43)
('/NOBACKUP/nii/typhoon_data/orig_image/201010', 175, 174)
('/NOBACKUP/nii/typhoon_data/orig_image/198720', 301, 263)
('/NOBACKUP/nii/typhoon_data/orig_image/198511', 235, 79)
('/NOBACKUP/nii/typhoon_data/orig_image/199507', 193, 174)
('/NOBACKUP/nii/typhoon_data/orig_image/199005', 235, 224)
('/NOBACKUP/nii/typhoon_data/orig_image/199129', 235, 229)
('/NOBACKUP/nii/typhoon_data/orig_image/198622', 145, 49)
('/NOBACKUP/nii/typhoon_data/orig_image/200821', 67, 66)
('/NOBACKUP/nii/typhoon_data/orig_image/199324', 211, 209)
('/NOBACKUP/nii/typhoon_data/orig_image/199520', 307, 292)
('/NOBACKUP/nii/typhoon_data/orig_image/199217', 307, 280)
('/NOBACKUP/nii/typhoon_data/orig_image/197917', 37, 12)
('/NOBACKUP/nii/typhoon_data/orig_image/200414', 121, 118)
('/NOBACKUP/nii/typhoon_data/orig_image/198908', 199, 197)
('/NOBACKUP/nii/typhoon_data/orig_image/199214', 277, 273)
('/NOBACKUP/nii/typhoon_data/orig_image/198526', 211, 67)
('/NOBACKUP/nii/typhoon_data/orig_image/198421', 181, 61)
('/NOBACKUP/nii/typhoon_data/orig_image/198310', 283, 94)
('/NOBACKUP/nii/typhoon_data/orig_image/198122', 313, 101)
('/NOBACKUP/nii/typhoon_data/orig_image/200011', 103, 99)
('/NOBACKUP/nii/typhoon_data/orig_image/199023', 247, 241)
('/NOBACKUP/nii/typhoon_data/orig_image/199709', 313, 261)
('/NOBACKUP/nii/typhoon_data/orig_image/200610', 217, 210)
('/NOBACKUP/nii/typhoon_data/orig_image/198412', 205, 34)
('/NOBACKUP/nii/typhoon_data/orig_image/198615', 253, 85)
('/NOBACKUP/nii/typhoon_data/orig_image/198903', 157, 155)
('/NOBACKUP/nii/typhoon_data/orig_image/198527', 133, 41)
('/NOBACKUP/nii/typhoon_data/orig_image/198217', 271, 87)
('/NOBACKUP/nii/typhoon_data/orig_image/199623', 157, 135)
('/NOBACKUP/nii/typhoon_data/orig_image/200405', 187, 186)
('/NOBACKUP/nii/typhoon_data/orig_image/198525', 139, 47)
('/NOBACKUP/nii/typhoon_data/orig_image/199427', 229, 210)
('/NOBACKUP/nii/typhoon_data/orig_image/199414', 331, 329)
('/NOBACKUP/nii/typhoon_data/orig_image/199712', 115, 110)
('/NOBACKUP/nii/typhoon_data/orig_image/200223', 91, 88)
('/NOBACKUP/nii/typhoon_data/orig_image/201214', 355, 343)
('/NOBACKUP/nii/typhoon_data/orig_image/200612', 271, 258)
('/NOBACKUP/nii/typhoon_data/orig_image/198422', 223, 75)
('/NOBACKUP/nii/typhoon_data/orig_image/201403', 211, 205)
('/NOBACKUP/nii/typhoon_data/orig_image/200112', 217, 184)
('/NOBACKUP/nii/typhoon_data/orig_image/199010', 331, 329)
"""
if __name__ == '__main__':
	main()

	
