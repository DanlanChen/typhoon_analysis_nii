import os,csv
import numpy as np
from read_h5 import read_h5
from collections import defaultdict
# merge the h5 file in image directory with the metadata in track directory, 
#step 1 : create two dictionaries with the same key, step2: merge the two dictory together

rootdir='/fs9/danlan/typhoon/data/image'
def get_image_dict(rootdir):
    #loop in the image directories, create dict1 key based on "time-typhoon_name", value is the 512*512 data array
    dict1 = defaultdict(list)
    for subdir, dirs, files in os.walk(rootdir):
    	#print subdir,'subdir'
    	#print dirs,'dirs'
    	#print files,'files'
            #break
            for file in files:
    	   # print file
                #print type(file)
                file_path = os.path.join(subdir,file)
               # print file.split("-")
                k1,k2,k3,k4=file.split("-")
    	    #print k1, k2
                key_1 = "-".join((k1, k2))
                np_data = read_h5(file_path)
                dict1[key_1] = np_data
    """
    print len(dict1)
    lenths1 = [type(v) for v in dict1.values()]
    print lenths1
    print len(lenths1)
    """
    return dict1


rootdir2='/fs9/danlan/typhoon/data/track'
def get_dict2(rootdir2):
# create the second dictionary from track directory
    dict2 = defaultdict()
    for subdir, dirs,files in os.walk(rootdir2):
        for file in files:
            file_path = os.path.join(subdir, file) 
            k2, _ =  file.split(".")
            #print file_path
            with open(file_path, 'rb') as tsv_file:
                tsv_reader = csv.reader(tsv_file, delimiter='\t')
    	    for row in tsv_reader:
                    time = "".join(row[:4])
                    #print time
                    key_2 = "-".join((time,k2))
                    typhoon_type = row[4]
                    lat = row[5]
                    lont = row[6]
                    intensity = row[7]
                    dict2[key_2]=np.array([time, k2, lat, lont,typhoon_type,intensity])
    		#dict2[key_2]= (time,k2)
    #print dict2
    return dict2


def merge_dict(dict1,dict2):
    #merge two dictionaries together
    #modified from http://stackoverflow.com/questions/5946236/how-to-merge-multiple-dicts-with-same-key
    #imagedata,time,typhoon_name,latitude,longtitude, type,intensity
    #181 images
    dd = defaultdict(list)
    for key1, value1 in dict1.iteritems():
        dd[key1].append(dict1[key1])
        for v in dict2[key1]:
            dd[key1].append(v)
    #check if  dd is correctly generated
    """
    lengths = [len(v) for v in dd.values()]
    print lengths
    for k,v in dd.iteritems():
        print v
        print v[0].shape
        break
    """
    return dd

	

