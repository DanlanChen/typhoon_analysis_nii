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
from mpl_toolkits.mplot3d import Axes3D
from tsne import tsne
from sklearn.manifold import TSNE

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )
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
def get_lstm_intermidiate_layer_output(model,im,layer=0):
	 get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[layer].output])
	 layer_output = get_layer_output([im, 0])[0]
	 return layer_output
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
	"""
	# check_intensities statistics


	data_folder = not_equal_track_image_list + equal_track_image_list
	intensities=[]
	for folder in data_folder:
		file_name = track_path + folder+'.itk'
		with open(file_name,'rb') as tsv_file:
			tsv_reader = csv.reader(tsv_file, delimiter='\t')
			for row in tsv_reader:
				#print row.type
				intensity = float(row[-2])
				intensities.append(intensity)

	intensities = np.array(intensities)
	print intensities
	print intensities.shape
	print np.mean(intensities,axis=0),'mean'
	print np.std(intensities,axis=0),'std'
	print np.min(intensities,axis=0),'min'
	print np.max(intensities,axis =0),'max'
	"""
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
	dataset_image_dic = {}
	dataset_intensity_dic ={}
	dataset_type_dic={}
	dataset_image_path = 'test_file/dataset_imageset.hdf5'
	dataset_intensity_path = 'test_file/dataset_intensity.hdf5'
	dataset_type_path='test_file/dataset_type.hdf5'

	# for key in equal_track_image_list:
	# 	print(key)
	# 	image_folder = image_path + key +'/'
	# 	track_file_path = track_path + key + '.itk'
	# 	dataset_type = prepare_dataset.dataset_1_type(track_file_path)
	# 	print (dataset_type.shape)
	# 	dataset_type_dic[key] = dataset_type
	# 	hf_type.create_dataset(key, data = dataset_type)

	# hf_type.close()






	# equal_track_image_list=equal_track_image_list[:2]
	# if not os.path.exists(dataset_image_path) :
	# 	vgg_model = VGG_16('vgg16_weights.h5')
	# 	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#    	vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy')
	# 	hf_image = h5py.File(dataset_image_path)
	# 	hf_intensity = h5py.File(dataset_intensity_path)
		
	# 	

	# 	print ('dumped data into hf_image,intensity')
	# else:
	# 	print ('hf_image intensity exists')
	# 	for key in equal_track_image_list:
	# 		with h5py.File(dataset_image_path,'r') as hf_image:
			
	# 			dataset_image = np.array(hf_image.get(key))
	# 		with h5py.File(dataset_intensity_path,'r') as hf_intensity:
	# 			dataset_intensity = np.array(hf_intensity.get(key))
	# 		print (key, dataset_image.shape,dataset_intensity.shape)
	# train_selected_folder_index = random.sample(range(0,len(train_folder)),10)
	# test_selected_folder_index = random.sample(range(0,len(test_folder)),10)

	hf_image = h5py.File(dataset_image_path)
	hf_intensity = h5py.File(dataset_intensity_path)
	hf_type = h5py.File(dataset_type_path)
	# for i in train_selected_folder_index:
	# 	key = train_folder[i]
	# train_folder=['201314']
	train_y_types=[]
	train_layer_outputs=[]
	# train_folder=train_folder[:2]
	# test_folder = test_folder[:2]
	train_folder=['198811']
	#scatter points
	for key in train_folder:
		print(key)
		if os.path.exists(ModelCheckpoint_file):
			print ('load  load_weights',ModelCheckpoint_file)
		model.load_weights(ModelCheckpoint_file)
		dataset_image = np.array(hf_image.get(key))
		dataset_intensity = np.array(hf_intensity.get(key))
		dataset_type = np.array(hf_type.get(key))
		# print (dataset_image.shape,'dataset_image')
		train_x,train_y = prepare_dataset.create_dataset_2_zero(dataset_image, dataset_intensity,look_back = look_back)
		train_y_type = prepare_dataset.create_dataset_y_zero(dataset_type,look_back=look_back)
		train_x = np.array(train_x,dtype = 'float32')
		train_y = np.array(train_y,dtype = 'float32')
		train_y_types += train_y_type
		print len(train_y_type),key
		if train_x.shape[0] >0:
			train_predict_image = 'test_file/tsne_visualization_12_zero_arrow/' + str(key)+'_'+str(look_back)+'_train.png' 
			# test_sample = np.array(train_x[0],
			train_outputs=[]
			for sample in train_x:
				sample = np.reshape(sample,(-1,look_back,2048))
				train_output_layer = get_lstm_intermidiate_layer_output(model,sample,layer=-2)
				model.reset_states()

				train_outputs.append(train_output_layer[0])
			# print train_output_layer.shape
			# print train_output_layer
			train_layer_outputs += train_outputs
			train_outputs=np.array(train_outputs)
			print train_outputs.shape
			Y = tsne(train_outputs, 2, 50, 20.0);
			colors = plt.cm.rainbow(np.linspace(0, 1, 8))
			labels_sets = set(train_y_type)
			scatter_dic ={}
			fig = plt.figure()
			for i in labels_sets:
				ii = np.where(train_y_type == i)[0]
				x = [Y[index,0] for index in ii]
				y = [Y[index,1] for index in ii]
				scatter_dic[i] = plt.scatter(x,y,color = colors[int(i)])
			line=plt.plot(Y[:,0],Y[:,1],'k')[0]
			add_arrow(line)
			plt.legend(scatter_dic.values(),scatter_dic.keys(),scatterpoints=1,loc='lower left',ncol = 6,fontsize=8)
			plt.xlabel(' x')
			plt.ylabel('y')
			plt.title('tsne of lstm feature,' +'train_predicts_look_back ' + str(look_back) + ', typhoon number ' + str(key))
			# plt.savefig('test_tsne_2.png')
			plt.savefig(train_predict_image)
			plt.close(fig)
			# fig = plt.figure()
			# ax = fig.add_subplot(111, projection='3d')
			# for i in labels_sets:
			# 	ii = np.where(train_y_type == i)[0]
			# 	x = [Y[index,0] for index in ii]
			# 	y = [Y[index,1] for index in ii]
			# 	z = [train_y[index] for index in ii]
			# 	scatter_dic[i] = ax.scatter(x,y,z,color = colors[int(i)])
			# plt.legend(scatter_dic.values(),scatter_dic.keys(),scatterpoints=1,loc='lower left',ncol = 6,fontsize=8)
			# plt.xlabel(' x')
			# plt.ylabel('y')
			# plt.title('3d tsne of lstm feature,' +'train_predicts_look_back ' + str(look_back) + ', typhoon number ' + str(key))
			# plt.savefig('test_tsne_3d_2.png')
			# # plt.savefig(train_predict_image)
			# plt.close(fig)
			# fig = plt.figure()
			#
			# ax.scatter(Y[:,0], Y[:,1], train_y)
			# ax.set_xlabel('X Label')
			# ax.set_ylabel('Y Label')
			# ax.set_zlabel('intensity Label')

			# plt.savefig('tsne_test.png')
			# break
	"""
	test_y_types=[]
	test_layer_outputs=[]
	for key in test_folder:
		# key = test_folder[i]
		print (key)
		if os.path.exists(ModelCheckpoint_file):
			print ('load  load_weights',ModelCheckpoint_file)
		model.load_weights(ModelCheckpoint_file)
		dataset_image = np.array(hf_image.get(key))
		dataset_intensity = np.array(hf_intensity.get(key))
		dataset_type = np.array(hf_type.get(key))
		test_x,test_y = prepare_dataset.create_dataset_2_zero(dataset_image, dataset_intensity,look_back = look_back)
		test_x = np.array(test_x,dtype = 'float32')
		test_y = np.array(test_y,dtype = 'float32')
		test_y_type = prepare_dataset.create_dataset_y_zero(dataset_type,look_back=look_back)
		test_y_types += test_y_type
		print len(test_y_type),key
		if test_x.shape[0] > 0:
			test_predict_image = 'test_file/tsne_visualization_12_zero_arrow/' + str(key)+'_'+str(look_back)+'_test.png' 
			# test_sample = np.array(train_x[0],
			test_outputs=[]
			for sample in test_x:
				sample = np.reshape(sample,(-1,look_back,2048))
				test_output_layer = get_lstm_intermidiate_layer_output(model,sample,layer=-2)
				model.reset_states()


				test_outputs.append(test_output_layer[0])
			# print train_output_layer.shape
			# print train_output_layer
			test_layer_outputs += test_outputs
			test_outputs=np.array(test_outputs)
			print test_outputs.shape
			Y = tsne(test_outputs, 2, 50, 20.0);
			colors = plt.cm.rainbow(np.linspace(0, 1, 8))
			labels_sets = set(test_y_type)
			scatter_dic ={}
			fig = plt.figure()
			for i in labels_sets:
				ii = np.where(test_y_type == i)[0]
				x = [Y[index,0] for index in ii]
				y = [Y[index,1] for index in ii]
				scatter_dic[i] = plt.scatter(x,y,color = colors[int(i)])
			line=plt.plot(Y[:,0],Y[:,1],'k')[0]
			add_arrow(line)
			plt.legend(scatter_dic.values(),scatter_dic.keys(),scatterpoints=1,loc='lower left',ncol = 6,fontsize=8)
			plt.xlabel(' x')
			plt.ylabel('y')
			plt.title('tsne of lstm feature,' +'test_predicts_look_back ' + str(look_back) + ', typhoon number ' + str(key))
			# plt.savefig('test_tsne_2.png')
			plt.savefig(test_predict_image)
			plt.close(fig)
			# fig = plt.figure()
			# ax = fig.add_subplot(111, projection='3d')
			# for i in labels_sets:
			# 	ii = np.where(test_y_type == i)[0]
			# 	x = [Y[index,0] for index in ii]
			# 	y = [Y[index,1] for index in ii]
			# 	z = [test_y[index] for index in ii]
			# 	scatter_dic[i] = ax.scatter(x,y,z,color = colors[int(i)])
			# plt.legend(scatter_dic.values(),scatter_dic.keys(),scatterpoints=1,loc='lower left',ncol = 6,fontsize=8)
			# plt.xlabel(' x')
			# plt.ylabel('y')
			# plt.title('3d tsne of lstm feature,' +'test_predicts_look_back ' + str(look_back) + ', typhoon number ' + str(key))
			# plt.savefig('test_tsne_3d_2_test.png')
			# # plt.savefig(train_predict_image)
			# plt.close(fig)
			# fig = plt.figure()
			#
			# ax.scatter(Y[:,0], Y[:,1], train_y)
			# ax.set_xlabel('X Label')
			# ax.set_ylabel('Y Label')
			# ax.set_zlabel('intensity Label')

			# plt.savefig('tsne_test.png')
		# break
	"""
	hf_image.close()
	hf_intensity.close()
	hf_type.close()


	# train_y_types = np.array(train_y_types)
	# train_layer_outputs = np.array(train_layer_outputs)
	# test_y_types = np.array(test_y_types)
	# test_layer_outputs = np.array(test_layer_outputs)
	# print train_y_types.shape
	# print test_y_types.shape
	# Y = tsne(train_layer_outputs, 2, 50, 20.0);
	# colors = plt.cm.rainbow(np.linspace(0, 1, 8))
	# labels_sets = set(train_y_types)
	# scatter_dic ={}
	# fig = plt.figure()
	# train_predict_image = 'test_file/tsne_visualization_12_zero/' +str(look_back)+'_whole_train.png' 
	# for i in labels_sets:
	# 	ii = np.where(train_y_types == i)[0]
	# 	x = [Y[index,0] for index in ii]
	# 	y = [Y[index,1] for index in ii]
	# 	scatter_dic[i] = plt.scatter(x,y,color = colors[int(i)])
	# plt.legend(scatter_dic.values(),scatter_dic.keys(),scatterpoints=1,loc='lower left',ncol = 6,fontsize=8)
	# plt.xlabel(' x')
	# plt.ylabel('y')
	# plt.title('tsne of lstm feature,' +'whole train_predicts_look_back ' + str(look_back))
	# # plt.savefig('test_tsne_2.png')
	# plt.savefig(train_predict_image)
	# plt.close(fig)

	# Y = tsne(test_layer_outputs, 2, 50, 20.0);
	# colors = plt.cm.rainbow(np.linspace(0, 1, 8))
	# labels_sets = set(test_y_types)
	# scatter_dic ={}
	# fig = plt.figure()
	# test_predict_image = 'test_file/tsne_visualization_12_zero/' +str(look_back)+'_whole_test.png' 
	# for i in labels_sets:
	# 	ii = np.where(test_y_types == i)[0]
	# 	x = [Y[index,0] for index in ii]
	# 	y = [Y[index,1] for index in ii]
	# 	scatter_dic[i] = plt.scatter(x,y,color = colors[int(i)])
	# plt.legend(scatter_dic.values(),scatter_dic.keys(),scatterpoints=1,loc='lower left',ncol = 6,fontsize=8)
	# plt.xlabel(' x')
	# plt.ylabel('y')
	# plt.title('tsne of lstm feature,' +'whole_test_predicts_look_back ' + str(look_back) )
	# # plt.savefig('test_tsne_2.png')
	# plt.savefig(test_predict_image)
	# plt.close(fig)
	"""
	train_folder = ['199307']
	for key in train_folder:
		print(key)
		if os.path.exists(ModelCheckpoint_file):
			print ('load  load_weights',ModelCheckpoint_file)
		model.load_weights(ModelCheckpoint_file)
		dataset_image = np.array(hf_image.get(key))
		dataset_intensity = np.array(hf_intensity.get(key))
		dataset_type = np.array(hf_type.get(key))
		# print (dataset_image.shape,'dataset_image')
		train_x,train_y = prepare_dataset.create_dataset_2_zero(dataset_image, dataset_intensity,look_back = look_back)
		train_y_type = prepare_dataset.create_dataset_y_zero(dataset_type,look_back=look_back)
		train_x = np.array(train_x,dtype = 'float32')
		train_y = np.array(train_y,dtype = 'float32')
		train_y_types += train_y_type
		print len(train_y_type),key
		if train_x.shape[0] >0:
			train_predict_image = 'test_file/tsne_visualization_24_zero/' + str(key)+'_'+str(look_back)+'arrow_train.png' 
			# test_sample = np.array(train_x[0],
			train_outputs=[]
			for sample in train_x:
				sample = np.reshape(sample,(-1,look_back,2048))
				train_output_layer = get_lstm_intermidiate_layer_output(model,sample,layer=-2)
				model.reset_states()

				train_outputs.append(train_output_layer[0])
			# print train_output_layer.shape
			# print train_output_layer
			train_layer_outputs += train_outputs
			train_outputs=np.array(train_outputs)
			print train_outputs.shape

			Y = tsne(train_outputs, 2, 50, 20.0);
			# tsne = TSNE(n_components=2, init='pca',random_state = 0)
			# Y = tsne.fit_transform(train_outputs)
			colors = plt.cm.rainbow(np.linspace(0, 1, 8))
			labels_sets = set(train_y_type)
			scatter_dic ={}
			fig = plt.figure()
			for i in labels_sets:
				ii = np.where(train_y_type == i)[0]
				x = [Y[index,0] for index in ii]
				y = [Y[index,1] for index in ii]
				scatter_dic[i] = plt.scatter(x,y,color = colors[int(i)])
			# x = Y[:0]
			# y = Y[:1]
			# print Y.shape
			line=plt.plot(Y[:,0],Y[:,1],'k')[0]
			add_arrow(line)
			# plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
			plt.legend(scatter_dic.values(),scatter_dic.keys(),scatterpoints=1,loc='lower left',ncol = 6,fontsize=8)
			plt.xlabel(' x')
			plt.ylabel('y')
			plt.title('tsne of lstm feature,' +'train_predicts_look_back ' + str(look_back) + ', typhoon number ' + str(key))
			# plt.savefig('test_tsne_2.png')
			plt.savefig(train_predict_image)
			plt.close(fig)
			"""




	t2 = time.time()
	print ("using  %s seconds" % (t2-t1))


if __name__ == '__main__':
	main()

	
