from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import Adam,SGD,RMSprop,SGD
from keras import backend as K
from sklearn.metrics import confusion_matrix
from keras.layers.convolutional import ZeroPadding2D
K.set_image_dim_ordering('th')
import cv2,os
from scipy.misc import bytescale
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import Merge
def optimizer_selection(choice,nb_epoch):
	if choice == 'rmsprop':
		lr_rmsprop = 0.001
		decay_rate = lr_rmsprop * 1.0/nb_epoch
		optimizer = RMSprop(lr = lr_rmsprop, rho = 0.9, decay = decay_rate, epsilon = 1e-06)
	if choice == 'sgd':
		lr_sgd = 0.01
		decay = 0.00001
		# decay = lr_sgd * 1.0 /nb_epoch 0.001
		optimizer = SGD(lr=0.01, momentum=0.9, decay=decay, nesterov=False)

	return optimizer
def vgg_16(img_rows,img_cols,num_labels,optimizer):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(1,img_rows,img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(num_labels, activation='softmax'))
    # model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model
def vgg_19(img_rows,img_cols,num_labels,optimizer):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(1,img_rows,img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
 
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model
def small_model(img_rows,img_cols,num_labels,optimizer):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(1,img_rows,img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu'))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
 
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(num_labels, activation='softmax'))
    # model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model
def vgg_19_with_l2_regularizer(img_rows,img_cols,num_labels,optimizer):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(1,img_rows,img_cols)))
    model.add(Convolution2D(64, 3, 3, W_regularizer =l2(0.0001),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3,W_regularizer =l2(0.0001), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3,W_regularizer =l2(0.0001), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, W_regularizer =l2(0.0001),activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3,W_regularizer =l2(0.0001), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3,W_regularizer =l2(0.0001), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3,W_regularizer =l2(0.0001),activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, W_regularizer =l2(0.0001),activation='relu'))
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, W_regularizer =l2(0.0001),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, W_regularizer =l2(0.0001),activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, W_regularizer =l2(0.0001),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, W_regularizer =l2(0.0001),activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3,W_regularizer =l2(0.0001), activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, W_regularizer =l2(0.0001),activation='relu'))
 
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, W_regularizer =l2(0.0001),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, W_regularizer =l2(0.0001),activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model
def vgg_19_with_drop_out(img_rows,img_cols,num_labels,optimizer):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(1,img_rows,img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model
def model_2():
    model = Sequential()
    model.add(Dense(16, input_dim=7))
    return model
def merge_model(model_1,mode_2,optimizer,num_labels):
    merged = Merge([model_1,mode_2], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model


def create_simple_cnn_model_classification(img_rows,img_cols,num_labels,optimizer):
	model = Sequential()
	#(a)
	# model.add(Reshape((1,img_rows,img_cols), input_shape = (1,)))
	model.add(Convolution2D(64, 3, 3, border_mode='valid',input_shape=(1,img_rows, img_cols)))
	model.add(Activation('relu'))
	#512 -3 +1 =510

	#(b)
	model.add(Convolution2D(32,5,5))
	model.add(Activation('relu'))
	# 510 -5 + 1 = 506

	#(c)
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.2))
	#506/2=253

	#(d)
	model.add(Convolution2D(32, 4, 4))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	# #253-4+1 =250

	#(e)
	model.add(MaxPooling2D(pool_size = (2,2)))
	# #250/2=125

	# #(f)
	model.add(Convolution2D(128,5,5))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	# # #125 - 3 + 1 =123

	#(g)
	model.add(MaxPooling2D(pool_size = (3,3)))
	#123 /3 = 41

	#(h)
	model.add(Convolution2D(128, 4,4))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	# 41 -3 + 1 =39

	#(i)
	model.add(MaxPooling2D(pool_size = (3,3)))
    #39/3 = 13

	#(j)
	model.add(Flatten())
	model.add(Dense(2048))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(2048))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	#(k)
	model.add(Dense(num_labels))
	model.add(Activation("softmax")) 
	model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
	# model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	return model
def create_simple_cnn_model_classification_tropical(img_rows,img_cols,optimizer):
	model = Sequential()
	#(a)
	# model.add(Reshape((1,img_rows,img_cols), input_shape = (1,)))
	model.add(Convolution2D(64, 3, 3, border_mode='valid',input_shape=(1,img_rows, img_cols)))
	model.add(Activation('relu'))
	#512 -3 +1 =510

	#(b)
	model.add(Convolution2D(32,5,5))
	model.add(Activation('relu'))
	# 510 -5 + 1 = 506

	#(c)
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.2))
	#506/2=253

	#(d)
	model.add(Convolution2D(32, 4, 4))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	# #253-4+1 =250

	#(e)
	model.add(MaxPooling2D(pool_size = (2,2)))
	# #250/2=125

	# #(f)
	model.add(Convolution2D(128,5,5))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	# # #125 - 3 + 1 =123

	#(g)
	model.add(MaxPooling2D(pool_size = (3,3)))
	#123 /3 = 41

	#(h)
	model.add(Convolution2D(128, 4,4))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	# 41 -3 + 1 =39

	#(i)
	model.add(MaxPooling2D(pool_size = (3,3)))
    #39/3 = 13

	#(j)
	model.add(Flatten())
	model.add(Dense(2048))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(2048))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	#(k)
	model.add(Dense(2))
	model.add(Activation("softmax")) 
	model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
	# model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	return model

def create_test_cnn_model(img_rows,img_cols,optimizer):
	model = Sequential()
	#(a)
	# model.add(Reshape((1,img_rows,img_cols), input_shape = (1,)))
	model.add(Convolution2D(64, 3, 3, border_mode='valid',input_shape=(1,img_rows, img_cols)))
	model.add(Activation('relu'))
	#512 -3 +1 =510

	#(b)
	model.add(Convolution2D(32,5,5))
	model.add(Activation('relu'))
	# 510 -5 + 1 = 506

	#(c)
	model.add(MaxPooling2D(pool_size = (2,2)))
	# model.add(Dropout(0.2))
	#506/2=253

	#(d)
	model.add(Convolution2D(32, 4, 4))
	model.add(Activation('relu'))
	# model.add(Dropout(0.2))
	# #253-4+1 =250

	#(e)
	model.add(MaxPooling2D(pool_size = (2,2)))
	# #250/2=125

	# #(f)
	model.add(Convolution2D(128,5,5))
	model.add(Activation('relu'))
	# model.add(Dropout(0.2))
	# # #125 - 5 + 1 =121

	#(g)
	model.add(MaxPooling2D(pool_size = (3,3)))
	#121 /3 = 40

	#(h)
	model.add(Convolution2D(128, 4,4))
	model.add(Activation('relu'))
	# model.add(Dropout(0.2))
	# 40 -4 + 1 =37

	#(i)
	model.add(MaxPooling2D(pool_size = (3,3)))
    #39/3 = 13

	#(j)
	model.add(Flatten())
	model.add(Dense(2048))
	model.add(Activation('relu'))
	# model.add(Dropout(0.2))

	model.add(Dense(2048))
	model.add(Activation('relu'))
	# model.add(Dropout(0.2))

	#(k)
	model.add(Dense(8))
	model.add(Activation("softmax")) 
	model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
	# model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	return model
	
def model_training(model,train_generator, validation_generator, img_rows,img_cols,batch_size,nb_epoch,samples_per_epoch,ModelCheckpoint_file):
	
	checkpointer = ModelCheckpoint(filepath=ModelCheckpoint_file, verbose=1, save_best_only=True)
	# early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min')
	early_stop = EarlyStopping(monitor = 'val_loss', patience = 60, mode = 'min')
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
	             
	hist = model.fit_generator(generator = train_generator,samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,callbacks=[checkpointer,early_stop,reduce_lr],validation_data= validation_generator, nb_val_samples=6525, nb_worker=10)
	# hist = model.fit(train_generator, validation_generator, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_split=0.1,callbacks=[checkpointer])
	print (hist.history)
	return hist
def model_training_whole(model,X_train, Y_train, X_test, Y_test,batch_size, nb_epoch,ModelCheckpoint_file):
	
	checkpointer = ModelCheckpoint(filepath=ModelCheckpoint_file, verbose=1, save_best_only=True)
	# early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min')
	early_stop = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min')
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
	hist = model.fit(X_train,Y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test),callbacks=[checkpointer,early_stop,reduce_lr],shuffle = True)
	# hist = model.fit_generator(generator = train_generator,samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,callbacks=[checkpointer,early_stop],validation_data= validation_generator, nb_val_samples=100, nb_worker=10)
	# hist = model.fit(train_generator, validation_generator, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_split=0.1,callbacks=[checkpointer])
	print (hist.history)
	return hist
def model_predicting(model,test_generator,test_samples):
	print ("predicting")
	predictions = model.predict_generator(test_generator , val_samples= test_samples)
	return (predictions)
# def evaluating(model,val_generator, val_samples):
# 	model.evaluate_generator(val_generator, val_samples = val_samples)
def get_accuracy(_predictions, _labels, need_confusion_matrix):
	cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
	# == is overloaded for numpy array

	accuracy = (100.0 * np.sum(_predictions == _labels) / len(_predictions))
	return accuracy,cm
def visualize_intermidiate_output(model,X,layer_number,folder_path,visualize_train_or_test):

	get_layer_output = K.function([model.layers[0].input,K.learning_phase()],[model.layers[layer_number].output])
	if visualize_train_or_test == 'train':
		layer_output = get_layer_output([X,1])[0]
	if visualize_train_or_test == 'test':
		layer_output = get_layer_output([X,0])[0]
	print (layer_number, layer_output.shape)

	if not os.path.exists(folder_path + str(layer_number) + '/'):
		os.makedirs(folder_path + str(layer_number )+'/')
	count = 1
	for data in layer_output[0]:
		img_i = bytescale(data)
		pic_name = folder_path + str(layer_number) + '/' + str(count) + '.jpg'
		cv2.imwrite(pic_name,img_i)
		count += 1


