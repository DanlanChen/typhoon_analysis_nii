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
from keras.optimizers import Adam,SGD,RMSprop
from keras.regularizers import l1l2,l2,activity_l2
from itertools import izip
from keras import backend as K
from sklearn.preprocessing import scale
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

	# #(d)
	# model.add(Convolution2D(16, 4, 4))
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))
	#253-4+1 =250

	#(e)
	# model.add(MaxPooling2D(pool_size = (2,2)))
	#250/2=125

	# #(f)
	# model.add(Convolution2D(32,3,3))
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))
	# #125 - 3 + 1 =123

	#(g)
	# model.add(MaxPooling2D(pool_size = (3,3)))
	#123 /3 = 41

	# (h)
	# model.add(Convolution2D(32, 3,3))
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))
	# 41 -3 + 1 =39

	# #(i)
	# model.add(MaxPooling2D(pool_size = (3,3)))
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


def predict(x,y,f1,f2,optimizer,f3):
    with open(f1) as in_file:
        in_file =json.load(in_file)
    model = model_from_json(in_file)
    model.load_weights(f2)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    y_predict = model.predict(x, batch_size = 32, verbose = 0)
    rmse = np.sqrt(((y_predict - y) ** 2).mean(axis=0))
    print (rmse, 'test loss')

    plt.title("test prediction and actual test data")
    plt.plot(y_predict[:100],'g^', label = 'predict')
    plt.plot(y_test[:100], 'r--',label = 'actual')
    plt.legend(loc = 'low center', shadow = True)
    plt.savefig(f3)
    plt.close()



    """
    not working in server
    pd.DataFrame(y_predict[:100]).plot()  
    pd.DataFrame(y_test[:100]).plot()
    """
def main():
	rootdir1 = '/fs9/danlan/typhoon/data/image'
	rootdir2 ='/fs9/danlan/typhoon/data/track'
	f1,f2,f3,f4 = "model_small_cnn.json", "weights_small_cnn.hdf5", "predic_actual_small_cnn.jpg", "small_cnn_output.json"
	lr = 0.1
	momentum = 0.9
	img_rows,img_cols, batch_size, nb_epoch = 512, 512, 32, 2
	sgd = SGD(lr=lr, momentum=momentum, nesterov=False)
	rmsprop = RMSprop(lr = 0.5, rho = 0.9, epsilon = 1e-06)
	dict1 = get_image_dict(rootdir1)
	# print (type(dict1.values()))
	# print (dict1.values().shape)

	# print (dict1)
	dict2 = get_dict2(rootdir2)
	dd = merge_dict(dict1, dict2)
	# print (dd.values())
	# print (type(dd.values()))
	v = np.array(dd.values())
	#shuffle
	np.random.shuffle(v)
	# print (v[:,0])
	# print (v[:,-1])
	# print (v.shape)
	# print (v)
	# print (type(v))
	x = v[:, 0]

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
	y = v[:, -1]
	# print (y.shape)

	# print (x[0].shape)
	# print (type(x))
	# # print (x.shape)
	# # x = np.array(x)
	# # print (x.shape)
	x_reshape = x_scaled.reshape((x.shape[0],1,img_rows,img_cols))
	x_reshape = x_reshape.astype("float32")
	y = y.astype("float32")
	print (x_reshape.shape)
	# # print (x_reshape.shape)
	train_size = int(0.9 * len(x))
	x_train = x_reshape[:train_size]
	y_train = y[: train_size]
	x_test = x_reshape[train_size:]
	y_test = y[train_size:]
	print (x_train.shape, 'x_train.shape')

	train(img_rows,img_cols,batch_size,nb_epoch,x_train,y_train,f2,f1,rmsprop,f4)
	predict(x_test, y_test, f1,f2,rmsprop,f3)


if __name__ == '__main__':
	t1 = time.time()
	main()
	t2 = time.time()
	print ("using  %s seconds" % (t2-t1))




 






