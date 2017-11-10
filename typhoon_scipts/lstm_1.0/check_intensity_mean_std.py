import os
track_path = '/NOBACKUP/nii/typhoon_data/new_track/'
data_folder_path = 'test_file/train_test_folder_equal_pretrain.json'
import json,csv,numpy as np
with open(data_folder_path,'r') as f:
	dic = json.load(f)
train_folder=dic['train_folder']
test_folder =dic['test_folder']
data_folder = train_folder+test_folder
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