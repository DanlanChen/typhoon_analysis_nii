import pickle
from merge_dic import get_image_dict_2000,get_image_dict_2010
rootdir1='/fs9/danlan/typhoon/data/image'
dict1_2000 = get_image_dict_2000(rootdir1)
pickle_file_path_1 = '/home/danlan/typhoon_scripts/data_1/image_dict_2000.pickle'
picle_file_1 = open(pickle_file_path_1,'wb')
pickle.dump(dict1_2000,picle_file_1)
picle_file_1.close()