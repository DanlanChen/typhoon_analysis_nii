# try cnn on small dataset 201601 with 181 images
# first just use pure image array data and internsity information
from __future__ import absolute_import
from __future__ import print_function
from merge_dic import get_image_dict, get_dict2,merge_dict
import numpy as np
import pandas as pd 
np.random.seed(1337)
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
#from keras.datasets import cifar10
# from sklearn.cross_validation import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
import pickle,os
import h5py,csv
import time,json
# import hickle as hkl
from keras.optimizers import Adam,SGD,RMSprop
from keras.regularizers import l1l2,l2,activity_l2
from itertools import izip
from keras import backend as K
from sklearn.preprocessing import scale
from data_file_save_load import write_to_h5,load_h5
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')
def create_simple_cnn_model(img_rows,img_cols,file):
	model = Sequential()
	#(a)
	# model.add(Reshape((1,img_rows,img_cols), input_shape = (1,)))
	model.add(Convolution2D(16, 3, 3, border_mode='valid',input_shape=(1,img_rows, img_cols)))
	model.add(Activation('relu'))
	#512 -3 +1 =510

	#(b)
	model.add(Convolution2D(16,5,5))
	model.add(Activation('relu'))
	# 510 -5 + 1 = 506

	#(c)
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.2))
	#506/2=253

	#(d)
	model.add(Convolution2D(16, 4, 4))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	# #253-4+1 =250

	#(e)
	model.add(MaxPooling2D(pool_size = (2,2)))
	# #250/2=125

	# #(f)
	model.add(Convolution2D(32,3,3))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	# # #125 - 3 + 1 =123

	#(g)
	model.add(MaxPooling2D(pool_size = (3,3)))
	#123 /3 = 41

	(h)
	model.add(Convolution2D(32, 3,3))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	# 41 -3 + 1 =39

	#(i)
	model.add(MaxPooling2D(pool_size = (3,3)))
 #    #39/3 = 13

	#(j)
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	# model.add(Dropout(0.2))

	#(k)
	model.add(Dense(1))
	model.add(Activation("linear")) 
	
	model_json = model.to_json()
	with open(file,'w') as outfile:
		json.dump(model_json,outfile)
	return model

def train(img_rows,img_cols,batch_size,nb_epoch,X_train,Y_train,filepath,f1,optimizer,f2):
	model = create_simple_cnn_model(img_rows,img_cols,f1)
	if os.path.exists(filepath):
		model.load_weights(filepath)
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
	early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min')
	print (X_train.shape)
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_split=0.1,callbacks=[checkpointer])
	with open(f2,'w') as outputfile:
		json.dump(hist.history,outputfile)
	print(hist.history)


def predict(x,y,idd,f1,f2,optimizer,f3,f5):
    with open(f1) as in_file:
        in_file =json.load(in_file)
    model = model_from_json(in_file)
    model.load_weights(f2)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    y_predict = model.predict(x, batch_size = 32, verbose = 0)
    print(y_predict[0].shape)
    
    # reshape array from (#numsample,1) to (#numsample,)
    y_predict = np.array(np.reshape(y_predict, -1))

    print (y_predict, 'y_predict')
    print (y,'y')
    rmse = np.sqrt(((y_predict - y) ** 2).mean(axis=0))
    # rmse = np.sqrt((y_predict - y) ** 2)
    print (rmse, 'test loss')

    # tolist, to str is for json serilization convenience
    with open(f5,'w') as outfile:
        json.dump({'rmse':str(rmse),'y_pred':y_predict.tolist(),"y_act":y.tolist()},outfile)

    with open(f3,'wb') as f:
        writer = csv.writer(f)
        writer.writerow( [ 'id', 'predict','actual' ] )
        writer.writerows(izip(idd,y_predict, y))
    print ('test out file saved ' + str(f3))
    
def main():
	rootdir1 = '/fs9/danlan/typhoon/data/image'
	rootdir2 ='/fs9/danlan/typhoon/data/track'
	# rootdir1 = '/home/danlan/typhoon_scripts/data_test/image'
	# rootdir2 ='/home/danlan/typhoon_scripts/data_test/track'
	f1,f2,f3,f4,f5 = "params/model_small_cnn.json", "params/weights_small_cnn.hdf5", "result/predic_actual_small_cnn.csv", "result/small_cnn_output.json", "result/predict_y_small_cnn.json"
	# data_path = "typhoon_data.hkl"
	data_path = "data_1/typhoon_data.bin"
	train_test_path = "data_1/train_test_data.bin"
	lr_sgd = 0.1
	lr_rmsprop = 0.5
	momentum = 0.9
	img_rows,img_cols, batch_size, nb_epoch = 512, 512, 16, 20
	decay_rate = lr_rmsprop / (nb_epoch + 1)
	# sgd = SGD(lr=lr_sgd, momentum=momentum, nesterov=False)
	rmsprop = RMSprop(lr = lr_rmsprop, rho = 0.9, decay = decay_rate, epsilon = 1e-06)
	# load data
	# keys = ['image','meta','merge']
	keys = 'merge'
	if not os.path.exists(train_test_path):
		print ("train_test_path file not created")
		if not os.path.exists(data_path):
			print ("data path not prepared")
			dict1,arr1 = get_image_dict(rootdir1)
			dict2,arr2 = get_dict2(rootdir2)
			dd,arr_dd = merge_dict(dict1, dict2)
			# datas = [arr1, arr2, arr_dd]
			# datas = [arr_dd]
			print (arr_dd.shape)
			print (type(arr_dd))
			f = file(data_path,"wb")
			# hkl.dump(arr_dd,data_path, mode = 'w', compression = 'gzip')
			np.save(f,arr_dd)
			f.close()
			# write_to_h5(arr_dd,data_path,keys)
		else:
			print ("data path already prepared")
			# [arr1,arr2,arr_dd] = load_h5(data_path,keys)
			# [arr_dd] = load_h5(data_path,keys)
			f = file(data_path,"rb")
			arr_dd = np.load(f)
			f.close()
			# arr_dd = hkl.load(data_path)

		# v = np.array(dd.values())
		#shuffle
		np.random.shuffle(arr_dd)

		# print (v[:,0])
		# print (v[:,-1])
		# print (v.shape)
		# print (v)
		# print (type(v))
		idd = arr_dd[:,0]
		print (idd)
		x = arr_dd[:, 1]
		print (x.shape)
		# reshape x (181,) to (181,512,512)
		x = np.array(map(lambda item: np.array(item), x))

		x_max = x.max()
		x_min = x.min()
		x_mean = np.mean(x)
		x_std = np.std(x)
		# x_scaled = 1.0* (x - x_max)/(x_max - x_min)
		x_scaled = (x - x_mean)/x_std
		print (x[0],'x0')
		print (x_scaled[0], 'x_scaled 0')
		# x = scale(x)
		# print (x.shape)
		y = arr_dd[:, -1]
		# print (y.shape)

		# print (x[0].shape)
		# print (type(x))
		# # print (x.shape)
		# # x = np.array(x)
		# # print (x.shape)
		x_reshape = x_scaled.reshape((x.shape[0],1,img_rows,img_cols))
		x_reshape = x_reshape.astype("float32")
		y = np.array(y.astype("float32"))
		print (x_reshape.shape)
		# # print (x_reshape.shape)
		train_size = int(0.9 * len(x))
		x_train = x_reshape[:train_size]
		y_train = y[: train_size]
		idd_train = idd[:train_size]
		x_test = x_reshape[train_size:]
		y_test = y[train_size:]
		idd_test = idd[:train_size]
		print (x_train.shape, 'x_train.shape')
		print (x_test.shape,'x_test.shape')
		#write to memo
		f = file(train_test_path,"wb")
		np.save(f,x_train)
		np.save(f,x_test)
		np.save(f,y_train)
		np.save(f,y_test)
		np.save(f,idd_test)
		f.close()
	else:
		print ("train_test_path file already created")
		#load from memo
		f = file(train_test_path,"rb")
		x_train = np.load(f)
		x_test = np.load(f)
		y_train = np.load(f)
		y_test = np.load(f)
		idd_test = np.load(f)
		f.close()


	train(img_rows,img_cols,batch_size,nb_epoch,x_train,y_train,f2,f1,rmsprop,f4)
	predict(x_test, y_test,idd_test, f1,f2,rmsprop,f3,f5)


if __name__ == '__main__':
	t1 = time.time()
	main()
	t2 = time.time()
	print ("using  %s seconds" % (t2-t1))




 






