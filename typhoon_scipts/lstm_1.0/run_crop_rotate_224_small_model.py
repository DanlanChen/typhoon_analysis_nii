import load,config,time,os,csv,json
import numpy as np
import classification_model
from classification_model import model_training,model_predicting,create_simple_cnn_model_classification,get_accuracy
from resnet import resnet_main
import random
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,ReduceLROnPlateau

def write_to_csv(test_file_list,_predictions,_labels,csv_path):
	assert len(test_file_list) == len(_predictions)
	assert len(test_file_list) == len(_labels)
	test_file_list = np.reshape(np.array(test_file_list),(len(test_file_list),1))
	_predictions = np.reshape(np.array(_predictions),(len(test_file_list),1))
	_labels = np.reshape(np.array(_labels),(len(test_file_list),1))
	zz = np.concatenate((test_file_list, _predictions, _labels),1)
	# np.savetxt(csv_path,zz,delimiter = ',',header = 'file, predictions,labels')
	with open(csv_path,'wb') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(['file_list', 'predictions', 'labels'])
		writer.writerows(zz)
def get_category_reverse_back(labels):
	ys=[]
	ys_dic={}
	for label in labels:
		y = np.argmax(label)
		ys.append(y)
	for y in ys:
		if y not in ys_dic.keys():
			ys_dic[y] = 1
		else:
			ys_dic[y] += 1
	return ys_dic

def main():
	t1 = time.time()
	train_test_file_list_path = config.train_test_file_path_divid
	image_path = config.image_path
	trackDictPath = config.track_dic_path
	track_dict = load.load_json(trackDictPath)
	suspicious_file_list_path = config.suspicious_file_list_path
	suspicious_file_list = load.load_json(suspicious_file_list_path)
	train_validation_test_subdirs_split = config.train_validation_test_subdirs_split
	yType = config.yType
	csv_path = config.csv_path
	confusion_matrix_path = config.confusion_matrix_path
	hist_path = config.hist_path
	nb_epoch = config.nb_epoch
	optimizer_choice = config.optimizer
	img_rows,img_cols = config.img_rows, config.img_cols
	model_check_pointer_file = config.ModelCheckpoint_file
	nb_worker = config.nb_worker
	num_labels = config.num_labels
	batch_size = config.batch_size
	mean_v,std_v = config.mean_v,config.std_v
	if not os.path.exists(train_validation_test_subdirs_split):
		print 'subdirs not split'
		subdirs_list = load.get_subdirs_list(image_path)
		train_subdirs_list, validation_subdirs_list, test_subdirs_list = load.split_subdirs(subdirs_list,train_validation_test_subdirs_split)
	else:
		print 'subdirs splitted'

		train_subdirs_list, validation_subdirs_list, test_subdirs_list= load.get_split_subdirs(train_validation_test_subdirs_split)
	optimizer = classification_model.optimizer_selection(optimizer_choice,nb_epoch)
	# model = classification_model.vgg_19_with_l2_regularizer(img_rows,img_cols,num_labels,optimizer)
	model_1 = classification_model.vgg_16(img_rows,img_cols,num_labels,optimizer)
	model_2 = classification_model.model_2()
	model = classification_model.merge_model(model_1, model_2, optimizer,num_labels)
	model.summary()


	# file_list = subtract_suspicious_list(file_list,suspicious_file_list)
	# trackDictPath = config.track_dic_path
	# yType = config.yType
	# train_file_list, test_file_list =  load.get_train_test_file_split(train_subdirs_list,validation_subdirs_list,test_subdirs_list,track_dict,suspicious_file_list)
	# validation_file_list = train_file_list[:int(len(train_file_list) * 0.05)]
	# train_file_list = train_file_list[int(len(train_file_list) *0.05):]
	if not os.path.exists(train_test_file_list_path):
		print 'file_list not splited'
		train_file_list ,validation_file_list,test_file_list =  load.get_train_validation_test_file_split(train_subdirs_list, validation_subdirs_list, test_subdirs_list,track_dict,suspicious_file_list,train_test_file_list_path)
	else:
		print 'file list splitted'
		train_file_list, validation_file_list,test_file_list = load.load_train_validation_test_file_list(train_test_file_list_path)
	y_train, y_valid, y_test = load.get_train_validation_test_y(train_file_list,validation_file_list, test_file_list,trackDictPath, yType)

	# print len(file_list)
	# print len(train_file_list)
	# print len(validation_file_list)
	# print len(test_file_list)
	# print ('y_train',len(y_train))
	# print ('y_valid', len(y_valid))
	# print ('y_test',len(y_test))
	# print (type(y_train))
	print (y_train[0].shape,'train shape')
	# train_file_list = train_file_list[:200]
	# validation_file_list = validation_file_list[-100:]
	# test_file_list = test_file_list[:100]
	# y_train = y_train[:200]
	# y_valid = y_valid[-100:]
	# y_test = y_test[:100]
	x_train = load.get_x(train_file_list)
	x_valid = load.get_x(validation_file_list)
	x_test = load.get_x(test_file_list)

	input_2_train = []
	input_2_valid = []
	input_2_test =[]
	for file in train_file_list:
		input_2_train.append(load.get_data_2(track_dict,file))
	for file in validation_file_list:
		input_2_valid.append(load.get_data_2(track_dict,file))	
	for file in test_file_list:
		input_2_test.append(load.get_data_2(track_dict,file))	
	input_2_train = np.array(input_2_train)
	input_2_valid = np.array(input_2_valid)
	input_2_test = np.array(input_2_test)
	

	




	# print (x_train.shape)in
	# print(y_train.shape)
	
	# print (get_category_reverse_back(y_train),'set_y_train')
	# print (get_category_reverse_back(y_valid),'set_y_valid')
	# print (get_category_reverse_back(y_test),'set_y_test')
	# print (y_train.shape)
	# print (train_file_list, 'train_file_list')
	# print (validation_file_list,'validation_file_list')
	# print (test_file_list,'test_file_list')
	random_sample_index = random.sample(xrange(len(train_file_list)), int ( len(train_file_list)))
	x_train_2 = []
	y_train_2 = []
	input_2_train_2 = []
	for index in random_sample_index:
		file_path = train_file_list[index]
		x = load.rotate_image(file_path)
		y = y_train[index]
		# y = load.get_y_file(file_path,track_dict,yType)
		# print x.shape,x
		# print y
		x_train_2.append(x)
		y_train_2.append(y)
		input_2_train_2.append(load.get_data_2(track_dict,file_path))
	x_train_2 = np.array(x_train_2)
	x_train_2 = np.reshape(x_train_2,(-1,1,img_rows,img_cols))
	input_2_train_2 = np.array(input_2_train_2)


	x_train = np.concatenate((x_train,x_train_2), axis = 0)
	y_train = np.concatenate((y_train,y_train_2),axis =0)
	input_2_train = np.concatenate((input_2_train,input_2_train_2),axis = 0)
	print x_train.shape
	print y_train.shape
	print x_train[0]
	print y_train[0]
	print ('input_2_train',input_2_train.shape)
	print ('input_2_valid', input_2_valid.shape)
	print ('input_2_test',  input_2_test.shape)
	r = random.random() 
	random.shuffle(x_train, lambda : r) 
	random.shuffle(y_train, lambda: r)
	random.shuffle(input_2_train_2,lambda:r)
	print x_train.shape
	print y_train.shape
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_valid = x_valid.astype('float32')

	x_train /= 255
	x_valid /= 255
	x_test /= 255
	
	# 	# break












	if os.path.exists(model_check_pointer_file):
		model.load_weights(model_check_pointer_file)
	# hist = training(model,train_generator,validation_generator,img_rows,img_cols,128,nb_epoch,len(train_file_list),100, nb_worker,model_check_pointer_file)
	# hist = model_training(model,train_generator,validation_generator,img_rows,img_cols,32,nb_epoch,len(train_file_list),model_check_pointer_file)
	# hist = classification_model.model_training_whole(model,x_train,y_train,x_valid,y_valid, batch_size, nb_epoch,model_check_pointer_file)
	# # with open(hist_path, 'w') as f:
	# # 	json.dump(hist.history,f)
	checkpointer = ModelCheckpoint(filepath=model_check_pointer_file, verbose=1, save_best_only=True)
	# early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min')
	early_stop = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min')
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
	hist = model.fit([x_train,input_2_train],y_train,batch_size = batch_size,nb_epoch = nb_epoch, validation_data=([x_valid, input_2_valid], y_valid),callbacks=[checkpointer,early_stop,reduce_lr],shuffle = True)
	
	print hist.history,'hist'
	if os.path.exists(model_check_pointer_file):
		model.load_weights(model_check_pointer_file)
	# model.load_weights(model_check_pointer_file)
	# predictions = model_predicting(model,test_generator,len(y_test))
	predictions = model.predict([x_test,input_2_test])
	_predictions = np.argmax(predictions, 1)
	_labels = np.argmax(y_test, 1)
	write_to_csv(test_file_list,_predictions,_labels,csv_path)
	accuracy, cm = get_accuracy(_predictions, _labels, True)
	print (accuracy,'test accuracy')
	print(optimizer_choice,'optimizer_choice')
	print(cm,'cm')
	cm = cm.tolist()
	with open(confusion_matrix_path, 'w') as f:
		json.dump(cm,f)
	t2 = time.time()
	print ('using' + str(t2-t1))

if __name__ == '__main__':
	main()
