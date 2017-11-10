import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
import config,os,load
import numpy as np
import prepare_dataset
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
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
	ModelCheckpoint_file = config.ModelCheckpoint_file
	train_predict_image = config.train_predict_image
	test_predict_image = config.test_predict_image
	look_back = 3
	file_list = []
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
	# print(train_file_list)
	trainX = []
	trainY = []
	testX = []
	testY = []
	dataset_count = 0
	for file in train_file_list:
		try:
			data = prepare_dataset.dataset_1(file)
			data = prepare_dataset.normalize_intensity(data, intensity_mean,intensity_std)
			# data = list(data)
			trainXx , trainYy = prepare_dataset.create_dataset(data,look_back)
			trainX += trainXx
			trainY += trainYy
			dataset_count += data.shape[0]
		except:
			print(file)


	for file in test_file_list:
		try:
			data = prepare_dataset.dataset_1(file)
			data = prepare_dataset.normalize_intensity(data, intensity_mean,intensity_std)
			# data = list(data)
			testXx , testYy = prepare_dataset.create_dataset(data,look_back)
			testX += testXx
			testY += testYy
			dataset_count += data.shape[0]
		except:
			print (file)

	trainX = np.array(trainX,dtype = 'float32')
	trainY = np.array(trainY, dtype = 'float32')
	testX = np.array(testX, dtype = 'float32')
	testY = np.array(testY, dtype = 'float32')


	print (trainX.shape)
	print (testX.shape)

	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	batch_size = 1
	model = Sequential()
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
	model.add(Dense(3))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# checkpointer = ModelCheckpoint(filepath=ModelCheckpoint_file, verbose=2, save_best_only=True)
	hists = []
	for i in range(10):
		hist = model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
		model.reset_states()
		hists.append(hist.history['loss'][0])
	print (hists,'hists')
	# model.save_weights(ModelCheckpoint_file)
	# make predictions

	trainPredict = model.predict(trainX, batch_size=batch_size)
	model.reset_states()
	testPredict = model.predict(testX, batch_size=batch_size)
	# invert predictions
	trainPredict = prepare_dataset.reverse_normalize_intensity(trainPredict,intensity_mean,intensity_std)
	trainY = prepare_dataset.reverse_normalize_intensity(trainY,intensity_mean,intensity_std)
	testPredict = prepare_dataset.reverse_normalize_intensity(testPredict,intensity_mean,intensity_std)
	testY = prepare_dataset.reverse_normalize_intensity(testY,intensity_mean,intensity_std)
	# calculate root mean squared error
	# print (trainPredict[:,0], 'trainPredict')
	# print (trainPredict.shape,'len_train_predict')
	# print(trainY[0],'trainY')
	trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	dataset = np.zeros((dataset_count,1),dtype = 'float32')

	# trainPredictPlot = np.empty_like(dataset)
	# trainPredictPlot[:, :] = np.nan
	# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# # shift test predictions for plotting
	# testPredictPlot = np.empty_like(dataset)
	# testPredictPlot[:, :] = np.nan
	# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	# # plt.plot(dataset))
	fig = plt.figure()
	plt.title('train_predicts_look_back')
	plt.plot(list(trainPredict[:,0]),'r--',label= 'train_predict')
	plt.plot(list(trainY), 'g--',label = 'train')
	plt.legend(loc = 'upper left', shadow =True)
	plt.xlabel('typhoon_image')
	plt.ylael('typhoon intensity')
	plt.savefig(train_predict_image)
	plt.close(fig)
	fig = plt.figure()
	plt.title('test_predicts_look_back')
	plt.plot(list(testPredict[:,0]),'r--',label= 'test_predict')
	plt.plot(list(testY), 'g--',label = 'test')
	plt.xlabel('typhoon_image')
	plt.ylael('typhoon intensity')
	plt.legend(loc = 'upper left', shadow =True)
	plt.savefig(test_predict_image)
	plt.close(fig)

if __name__ == '__main__':
	main()