from multiprocessing import Pool
import numpy as np
import os, time,json
import csv
def mapping(file,subdir,dict1):
    k2, _ =  file.split(".")
    file_path = os.path.join(subdir, file)
    dict2 ={}
    with open(file_path, 'rb') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            time = "".join(row[:4])
            key_2 = "".join((time,k2))
            key_2 = int(key_2)
            yy = row[0]
            mm = row[1]
            dd = row[2]
            hh = row[3]
            typhoon_type = row[4]
            lat = row[5]
            lont = row[6]
            intensity = row[7]
            landorsea = row[8]
            dict2[key_2]= [yy,mm,dd,hh, k2, lat, lont,typhoon_type,intensity,landorsea]
    return dict2
def reducing(outputs):
    merge_dic = {}
    for dic in outputs:
        merge_dic.update(dic)
    return merge_dic
def multi_process_track(rootdir):
    pool = Pool()
    outputs = []
    for subdir, dirs, files in os.walk(rootdir):
        dict1 ={}
        results = [pool.apply_async(mapping,(x,subdir,dict1)) for x in files]
        roots = [r.get() for r in results]
        outputs += roots
    return outputs
# def multi_process_track(rootdir):
# 	pool = Pool()
# 	for subdir, dirs, files in os.walk(rootdir):
# 		dict1 ={}
# 		results = [pool.apply_async(maping,(x,subdir,dict1)) for x in files]
# 		roots = [r.get() for r in results]
# 		print roots
# rootdir = '/NOBACKUP/nii/cnn_data/data_test/track'
rootdir = '/Volumes/Danlan/nii_typhoon_data/new_track'
outputs = multi_process_track(rootdir)
merge_dic = reducing(outputs)
# track_dic_path = '/NOBACKUP/nii/typhoon_data/track_dic.json'
track_dic_path = 'new_track_dict.json'
with open(track_dic_path,'w') as outputfile:
    json.dump(merge_dic,outputfile)
