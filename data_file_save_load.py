import h5py
from collections import defaultdict
import numpy as np
def write_to_h5(datas,file_path,keys):
	h5f = h5py.File(file_path, 'w')
	# for i in range(len(keys)):
	h5f.create_dataset(keys, data = datas)
	h5f.close()
def load_h5(file_path,keys):
	arrs = []
	with h5py.File(file_path,'r') as hf:
			arrs=(np.array(hf.get(k)))
	return arrs

