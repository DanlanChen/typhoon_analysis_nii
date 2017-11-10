import load,config,time,os,csv,json
import numpy as np
import classification_model
from classification_model import model_training,model_predicting,create_simple_cnn_model_classification,get_accuracy
from resnet import resnet_main
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
	model = classification_model.vgg_19(img_rows,img_cols,num_labels,optimizer)
	model.summary()


	# file_list = subtract_suspicious_list(file_list,suspicious_file_list)
	# trackDictPath = config.track_dic_path
	# yType = config.yType
	train_file_list, test_file_list =  load.get_train_test_file_split(train_subdirs_list,validation_subdirs_list,test_subdirs_list,track_dict,suspicious_file_list)
	validation_file_list = train_file_list[:int(len(train_file_list) * 0.05)]
	train_file_list = train_file_list[int(len(train_file_list) *0.05):]
	# if not os.path.exists(train_test_file_list_path):
	# 	print 'file_list not splited'
	# 	train_file_list ,validation_file_list,test_file_list =  load.get_train_validation_test_file_split(train_subdirs_list, validation_subdirs_list, test_subdirs_list,track_dict,suspicious_file_list,train_test_file_list_path)
	# else:
	# 	print 'file list splitted'
	# 	train_file_list, validation_file_list,test_file_list = load.load_train_validation_test_file_list(train_test_file_list_path)
	# print len(file_list)
	print len(train_file_list)
	print len(validation_file_list)
	print len(test_file_list)
	load.get_input_2(train_file_list,trackDictPath)
	y_train, y_valid, y_test = load.get_train_validation_test_y(train_file_list,validation_file_list, test_file_list,trackDictPath, yType)
	print ('y_train',len(y_train))
	print ('y_valid', len(y_valid))
	print ('y_test',len(y_test))
	print (type(y_train))
	# print (y_train[0].shape,'train shape')
	# train_file_list = train_file_list[:2000]
	# validation_file_list = validation_file_list[-1000:]
	# test_file_list = test_file_list[:1000]
	# y_train = y_train[:2000]
	# y_valid = y_valid[-1000:]
	# y_test = y_test[:1000]
	print (get_category_reverse_back(y_train),'set_y_train')
	print (get_category_reverse_back(y_valid),'set_y_valid')
	print (get_category_reverse_back(y_test),'set_y_test')
	print (y_train.shape)
	print (train_file_list, 'train_file_list')
	print (validation_file_list,'validation_file_list')
	print (test_file_list,'test_file_list')
	x_train = load.get_x(train_file_list,img_rows,img_cols,mean_v,std_v)
	x_valid = load.get_x(validation_file_list,img_rows, img_cols, mean_v, std_v)
	x_test = load.get_x(test_file_list, img_rows, img_cols, mean_v, std_v)
	print (x_train.shape)
	print(y_train.shape)
	train_generator = load.get_chunk(train_file_list, y_train,img_rows,img_cols,num_labels)
	validation_generator = load.get_chunk(validation_file_list,y_valid,img_rows,img_cols,num_labels)
	test_generator = load.get_test_chunk(test_file_list,img_rows,img_cols)
	print (model.layers[0].get_config())
	print (model.layers[-1].get_config())
	if os.path.exists(model_check_pointer_file):
		model.load_weights(model_check_pointer_file)
	# hist = training(model,train_generator,validation_generator,img_rows,img_cols,128,nb_epoch,len(train_file_list),100, nb_worker,model_check_pointer_file)
	# hist = model_training(model,train_generator,validation_generator,img_rows,img_cols,32,nb_epoch,len(train_file_list),model_check_pointer_file)
	hist = classification_model.model_training_whole(model,x_train,y_train,x_valid,y_valid, batch_size, nb_epoch,model_check_pointer_file)
	with open(hist_path, 'w') as f:
		json.dump(hist.history,f)
	model.load_weights(model_check_pointer_file)
	predictions = model_predicting(model,test_generator,len(y_test))
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
