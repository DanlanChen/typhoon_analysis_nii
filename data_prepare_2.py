from merge_dic import get_dict2
import numpy as np
import pickle
rootdir1 = '/fs9/danlan/typhoon/data/image'
rootdir2 ='/fs9/danlan/typhoon/data/track'
save_path_2 ='/home/danlan/typhoon_scripts/data_1/track.bin'
dict2, arr2 = get_dict2(rootdir2)
# f = file(save_path_2,"wb")
# np.save(f,arr2)
# f.close()
# f = file(save_path_2,"rb")
# arr2 = np.load(f)
# print (arr2.shape)
# f.close()

pickle_file_path_1 = '/home/danlan/typhoon_scripts/data_1/track_dict.pickle'
picle_file_1 = open(pickle_file_path_1,'wb')
pickle.dump(dict2,picle_file_1)
picle_file_1.close()

pickle_file_path_2 = '/home/danlan/typhoon_scripts/data_1/track_arr.pickle'
picle_file_2 = open(pickle_file_path_2,'wb')
pickle.dump(arr2,picle_file_2)
picle_file_2.close()