import numpy as np
import h5py
def read_h5(file_name):
	with h5py.File(file_name,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get("infrared")
        np_data = np.array(data)
        print ('Shape of the array dataset_1: \n', np_data.shape)
file_name = "/fs9/danlan/typhoon/data/image/201601/2016070318-201601-HMW8-1.h5"
read_h5(file_name)
#key "infrared"
