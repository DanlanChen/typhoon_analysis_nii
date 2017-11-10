from scipy.misc import bytescale
from read_h5 import read_h5
import os,cv2,json,config	
import numpy as np
import config,json	
def show_image(np_data,pic_name):

	img_i = bytescale(np_data)
	cv2.imwrite(pic_name, img_i)
def main():
	# rootdir = '/NOBACKUP/nii/typhoon_data/resize_image_48_48/'
	#rootdir = config.image_path
	rootdir = '/NOBACKUP/nii/typhoon_data/224_224_crop_typhoon_image/' 
	image_root = 'show_image_crop/'
	with open('test_file/equal_track_image_list.json','r') as f:
		equal_image_list = json.load(f)
#	equal_image_list=['199406','199001']
	for key in equal_image_list:
		image_path = rootdir +key+'/'
		for subdir, dirs, files in os.walk(image_path):
			for file in files:
				# k1,k2,k3,k4=file.split("-")
				file_path = os.path.join(subdir,file)
				np_data = read_h5(file_path)
				if not os.path.isdir(image_root + key):
					os.makedirs(image_root + key)
				pic_name = image_root + key +"/" + file+'.jpg'

				print pic_name
				show_image(np_data,pic_name)
		# break

	# image_root = 'show_image/'
	# for subdir, dirs, files in os.walk(rootdir):
	# 	for file in files:
	# 		k1,k2,k3,k4=file.split("-")
	# 		file_path = os.path.join(subdir,file)
	# 		np_data = read_h5(file_path)
	# 		if not os.path.isdir(image_root + k2):
	# 			os.makedirs(image_root + k2)
	# 		pic_name = image_root + k2 +"/" + file+'.jpg'
	# 		print pic_name
	# 		show_image(np_data,pic_name)
	# suspicious_file = config.suspicious_file_list_path
	# with open(suspicious_file,'r') as f:
	# 	suspicious_file_list = json.load(f)
	# suspicious_file_list =np.array(suspicious_file_list)
	# np.random.shuffle(suspicious_file_list)
	# suspicious_file_list = list(suspicious_file_list)
	# for i in range(10):
	# 	file_path = suspicious_file_list[i]
	# 	np_data = read_h5(file_path)
	# 	pic_name = 'show_image/' + file_path.split('/')[-1] +'.jpg'
	# 	show_image(np_data,pic_name)
if __name__ == '__main__':
	main()
