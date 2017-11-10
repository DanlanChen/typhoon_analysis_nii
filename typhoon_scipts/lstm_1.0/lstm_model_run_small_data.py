import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import config,os,load
import numpy as np
import prepare_dataset
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
def lstm_model_1(batch_size,look_back):
	model = Sequential()
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))

	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	# # checkpointer = ModelCheckpoint(filepath=ModelCheckpoint_file, verbose=2, save_best_only=True)
	
def main():
	np.random.seed(7)
	# trackDictPath = config.track_dic_path
	# track_dict = load.load_json(trackDictPath)
	track_path = config.track_path
	suspicious_file_list_path = config.suspicious_file_list_path
	suspicious_file_list = load.load_json(suspicious_file_list_path)
	train_validation_test_subdirs_split = config.train_validation_test_subdirs_split
	intensity_mean,intensity_std = config.intensity_mean, config.intensity_std
	batch_size = config.batch_size
	ModelCheckpoint_file = 'test_file/orig_weights_lstm_1.0_lookback_1.hdf5'#config.ModelCheckpoint_file
	look_back = 6
	batch_size = 1
	file_list = []
	model = lstm_model_1(batch_size,look_back)
	model.load_weights(ModelCheckpoint_file)
	for subdir,dirs, files in os.walk(track_path):
		for file in files:
			file_path = os.path.join(subdir,file)
			file_list.append(file_path)
	file_list = np.array(file_list)
	np.random.shuffle(file_list)
	file_list = list(file_list)
	file_list = file_list[:10]
	# print (file_list)
	# for file in file_list:
	# 	if len(file) <=2:
	# 		print (file)
	# 		print (file_list.index(file))
	# file_list = file_list[:10]
	train_file_list = file_list[:int(0.9*len(file_list))]
	test_file_list = file_list[int(0.9*len(file_list)):]
	print(len(train_file_list))
	print (len(test_file_list))
	
	testX = []
	testY = []
	# dataset_count = 0
	histss = []
	train_file_list_copy = train_file_list
	# trainXS=np.array([]).reshape(0,look_back)
	# print (trainXS.shape,'trainxs shape')
	# trainYS = np.array([]).reshape(0,1)
	trainXS =[]
	trainYS =[]
	for i in np.arange(0,len(train_file_list_copy),12):#len(train_file_list_copy)
		trainX = []
		trainY = []
		hists = []
		print (i,'i')
		train_file_list = train_file_list_copy[i:i+12]
		# print len(train_file_list)
		for file in train_file_list:
			print file
			# try:
			data = prepare_dataset.dataset_1(file)
			data = prepare_dataset.normalize_intensity(data, intensity_mean,intensity_std)
			# data = list(data)
			trainXx , trainYy = prepare_dataset.create_dataset(data,look_back)
			trainX += trainXx
			trainY += trainYy
			# print (trainX,'trainX')
			# print (trainY,'trainY')
			# break
			# dataset_count += data.shape[0]
			# except:
			# 	print(file,'error')
		trainX = np.array(trainX,dtype = 'float32')
		trainY = np.array(trainY, dtype = 'float32')
		# print (trainX.shape)
		# print(trainY.shape,'trainY SHAPE')
		trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
		# trainXS = np.vstack((trainXS, trainX))
		# trainYS = np.vstack((trainYS, trainY))
		# print (trainXS.shape,'trainxs shape')
		# break
		# return
		
		trainXS.append(trainX)
		trainYS.append(trainY)

		
		"""
		training
		"""



		"""
	
		for i in range(100):
			hist = model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
			model.reset_states()
			hists.append(hist.history['loss'][0])
		# print (hists,'hists')
		histss.append(hists)
	print (histss,'histss')
	"""
	"""
	for file in test_file_list:
		try:
			data = prepare_dataset.dataset_1(file)
			data = prepare_dataset.normalize_intensity(data, intensity_mean,intensity_std)
			# data = list(data)
			testXx , testYy = prepare_dataset.create_dataset(data,look_back)
			testX += testXx
			testY += testYy
			
		except:
			print (file)

	
	
	testY = np.array(testY, dtype = 'float32')


	testX = np.array(testX,dtype = 'float32')

	
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	
	# 
	# 
	# model.save_weights(ModelCheckpoint_file)
	# make predictions
	
	

	trainX =trainXS[0]
	trainY = list(trainYS[0])
	# print(trainY)
	for i in xrange (1,len(trainXS)):
		trainX = np.vstack((trainX,trainXS[i]))
		trainY += list(trainYS[i])
		# trainY = np.vstack((trainY, trainYS[i]))
	trainY = np.array(trainY)
	trainPredict = model.predict(trainX, batch_size=batch_size)
	model.reset_states()
	trainPredict = prepare_dataset.reverse_normalize_intensity(trainPredict,intensity_mean,intensity_std)
	trainY = prepare_dataset.reverse_normalize_intensity(trainY,intensity_mean,intensity_std)
	trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
	

	print('Train Score: %.2f RMSE' % (trainScore))
	testPredict = model.predict(testX, batch_size=batch_size)
	# # invert predictions
	testPredict = prepare_dataset.reverse_normalize_intensity(testPredict,intensity_mean,intensity_std)
	testY = prepare_dataset.reverse_normalize_intensity(testY,intensity_mean,intensity_std)
	# # calculate root mean squared error


	testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	fig = plt.figure()
	plt.title('train_predicts_look_back_1')
	plt.plot(list(trainPredict[:20000,0]),'r--',label= 'train_predict')
	plt.plot(list(trainY[:20000]), 'g--',label = 'train')
	plt.legend(loc = 'upper left', shadow =True)
	plt.savefig('test_file/train_predict_look_back_1.png')
	plt.close(fig)
	fig = plt.figure()
	plt.title('test_predicts_look_back_1')
	plt.plot(list(testPredict[:10000,0]),'r--',label= 'test_predict')
	plt.plot(list(testY[:10000]), 'g--',label = 'test')
	plt.legend(loc = 'upper left', shadow =True)
	plt.savefig('test_file/test_predict_look_back_1.png')
	plt.close(fig)
	"""

if __name__ == '__main__':
	main()