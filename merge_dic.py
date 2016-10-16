import os,csv,random
import numpy as np
from read_h5 import read_h5
from collections import defaultdict
# merge the h5 file in image directory with the metadata in track directory, 
#step 1 : create two dictionaries with the same key, step2: merge the two dictory together

#rootdir='/fs9/danlan/typhoon/data/image'

# NOT WORKING AS memory issue,should seperate the data

# def get_image_dict(rootdir):
#     #loop in the image directories, create dict1 key based on "time-typhoon_name", value is the 512*512 data array
#     dict1 = defaultdict(list)
#     count = 0
#     for subdir, dirs, files in os.walk(rootdir):
#     	#print subdir,'subdir'
#     	#print dirs,'dirs'
#     	#print files,'files'
#             #break
#             for file in files:
#                 count += 1
#                 if count % 1000 == 0:
#                     print ('process ' + str(count) + 'image files')
#     	   # print file
#                 #print type(file)
#                 file_path = os.path.join(subdir,file)
#                # print file.split("-")
#                 k1,k2,k3,k4=file.split("-")
#     	    #print k1, k2
#                 key_1 = "-".join((k1, k2))
#                 np_data = read_h5(file_path)
#                 dict1[key_1] = np_data
    # k = np.array(dict1.keys()).reshape((len(dict1.keys()),1))
    # v = np.array(dict1.values())
    # print (k.shape,v.shape,'k','v')
    # arr1 = np.hstack((k,v))


    # """
    # print len(dict1)
    # lenths1 = [type(v) for v in dict1.values()]
    # print lenths1
    # print len(lenths1)

    # """
    # arr1=2
    # print ("dict1 made")
    # # print type(dict1.values())
    # return dict1,arr1

# get the image from 2010's 
def get_image_dict_2000(rootdir):
    print ("process 2000's iamges")
    #loop in the image directories, create dict1 key based on "time-typhoon_name", value is the 512*512 data array
    dict1 = defaultdict(list)
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                k1,k2,k3,k4=file.split("-")
                if k1[:3] == '200':
                    count += 1
                    file_path = os.path.join(subdir,file)
                    key_1 = "".join((k1, k2))
                    key_1 = int(key_1)
                    np_data = read_h5(file_path)
                    dict1[key_1] = np_data
                    if count % 1000 == 0:
                        print ('process ' + str(count) + 'image files')
    print (count, 'files')
    return dict1
def get_image_dict_2010(rootdir):
    #loop in the image directories, create dict1 key based on "time-typhoon_name", value is the 512*512 data array
    dict1 = defaultdict(list)
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                k1,k2,k3,k4=file.split("-")
                if k1[:3] == '201':
                    count += 1
                    file_path = os.path.join(subdir,file)
                    key_1 = "".join((k1, k2))
                    key_1 = int(key_1)
                    np_data = read_h5(file_path)
                    dict1[key_1] = np_data
                    if count % 1000 == 0:
                        print ('process ' + str(count) + 'image files')
    print (count, 'files')
    return dict1


def count_year_files(rootdir):
    count_70 = 0
    count_80 = 0
    count_90 = 0
    count_100 = 0
    count_110 = 0
    for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                k1, k2, k3, k4 = file.split("-")
                if k1[:3] == '197':
                    count_70 += 1
                elif k1[:3] == '198':
                    count_80 += 1
                elif k1[:3] == '199':
                    count_90 += 1
                elif k1[:3] == '200':
                    count_100 += 1
                elif k1[:3] == '201':
                    count_110 += 1
                else:
                    print (k1)
    print ("70's files ",  count_70)
    print ("80's files ",  count_80)
    print ("90's files ",  count_90)
    print ("2000's files ",  count_100)
    print ("2010's files ",  count_110)
def binarySearch(alist, item):
    first = 0
    last = len(alist)-1
    found = False  
    while first<=last and not found:
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1
    return found
                

#rootdir2='/fs9/danlan/typhoon/data/track'
def get_dict2(rootdir2):
# create the second dictionary from track directory
    dict2 = defaultdict()
    count = 0
    for subdir, dirs,files in os.walk(rootdir2):
        for file in files:
            count += 1
            if count % 1000 == 0:
                print ('process ' + str(count) + 'track files')
            file_path = os.path.join(subdir, file) 
            k2, _ =  file.split(".")
            #print file_path
            with open(file_path, 'rb') as tsv_file:
                tsv_reader = csv.reader(tsv_file, delimiter='\t')
                for row in tsv_reader:
                    time = "".join(row[:4])
                    #print time
                    key_2 = "".join((time,k2))
                    key_2 = int(key_2)
                    typhoon_type = row[4]
                    lat = row[5]
                    lont = row[6]
                    intensity = row[7]
                    dict2[key_2]=np.array([time, k2, lat, lont,typhoon_type,intensity])
    #dict2[key_2]= (time,k2)
    #print dict2
    k = np.array(dict2.keys()).reshape((len(dict2.keys()),1))
    v = np.array(dict2.values())
    arr2 = np.hstack((k,v))
    return dict2, arr2

# def join_keys(dict1,dict2):
#     keys1 = dict1.keys()
#     dict2 = dict2.keys()




def merge_dict(dict1,dict2):
    #merge two dictionaries together
    #modified from http://stackoverflow.com/questions/5946236/how-to-merge-multiple-dicts-with-same-key
    #imagedata,time,typhoon_name,latitude,longtitude, type,intensity
    #181 images
    dict2_key_loss = []
    sort_dict2_keys =dict2.keys()
    sort_dict2_keys.sort()
    dd = defaultdict(list)
    x = []
    idd = []
    y = []
    for key1, value1 in dict1.iteritems():
        if binarySearch(sort_dict2_keys,key1):
            # dd[key1].append(dict1[key1])
            idd.append(key1)
            x.append(value1)
            y.append(dict2[key1][-1])
                # dd[key1].append(v)
        else:
            # print (key1, "not in dict2")
            dict2_key_loss.append(key1)
            
    idd = np.array(idd)
    x = np.array(x)
    y = np.array(y)
    print (x.shape, y.shape,'x,y shape')
    # k = np.array(dd.keys()).reshape((len(dd.keys()),1))
    # v = np.array(dd.values())
    # arr_dd = np.hstack((k,v))
    #check if  dd is correctly generated
    """
    lengths = [len(v) for v in dd.values()]
    print lengths
    for k,v in dd.iteritems():
        print v
        print v[0].shape
        break
    """
    return idd,x,y,dict2_key_loss
def get_x_y(rootdir1,rootdir2):
    dict2, arr2 = get_dict2(rootdir2)
    # print (dict2,'dict2')
    dict1_2000 = get_image_dict_2000(rootdir1)
    # print (dict1_2000,'dict1_2000')
    idd,x,y,dict2_key_loss  = merge_dict(dict1_2000,dict2)
    print (dict2_key_loss,'dict2_key_loss')

  
    #shuffle data 1st method
    r = random.random()
    random.shuffle(idd,lambda:r)
    random.shuffle(x,lambda:r)
    random.shuffle(y,lambda:r)
    print (x.shape,'x_shape not dimension_shuffle')

    return idd,x,y,dict2_key_loss
def get_train_test(x,y,idd,dict2_key_loss,train_test_path_2000,img_rows,img_cols):
    x_max = x.max()
    x_min = x.min()
    # x_mean = np.mean(x)
    # x_std = 0.0
    # x_std = np.std(x)
    # x_scaled = (x - x_mean)/x_std
    x_scaled =(x - x_min)/(x_max - x_min)
    print (x[0],'x0')
    print (x_scaled[0], 'x_scaled 0')
    # y = arr_dd[:, -1]
    x_reshape = x_scaled.reshape((x.shape[0],1,img_rows,img_cols))
    x_reshape = x_reshape.astype("float32")
    y = np.array(y.astype("float32"))
    print (x_reshape.shape)
    train_size = int(0.9 * len(x))
    x_train = x_reshape[:train_size]
    y_train = y[: train_size]
    idd_train = idd[:train_size]
    x_test = x_reshape[train_size:]
    y_test = y[train_size:]
    idd_test = idd[:train_size]
    print (x_train.shape, 'x_train.shape')
    print (x_test.shape,'x_test.shape')
    
    #write to memo
    f = file(train_test_path_2000,"wb")
    np.save(f,x_train)
    np.save(f,x_test)
    np.save(f,y_train)
    np.save(f,y_test)
    np.save(f,idd_test)
    np.save(f,dict2_key_loss)
    f.close()

    return x_train,x_test,y_train,y_test,idd_train,idd_test

	

