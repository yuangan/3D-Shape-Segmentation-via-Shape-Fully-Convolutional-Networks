# -*- coding: utf-8 -*-
import numpy as np
import lmdb
import sys
sys.path.append("../../../python")
from PIL import Image
import os
import caffe
from copy import deepcopy

all_number=7

MAX_HEIGHT = 5
WIDTH = 4
#fature num
CHANNEL = 600

DIR_old = "/media/iot/mydisk2/FCN/wpy/new/Plier/"
def gen_input(lmdbname, file_list):
    X = np.zeros((len(file_list), CHANNEL, MAX_HEIGHT, WIDTH), dtype=np.float32)
    map_size = X.nbytes * 5 + 1000000000

    env = lmdb.open(lmdbname, map_size=map_size)

    count = 0
    for i in file_list:
        print count
        with env.begin(write=True) as txn:
            filename = os.path.join(DIR, "mydata", i + ".txt")
            f = open(filename)
            data_feature = []
            while True:
                a = f.readline()[:-2]	
                if 0 != len(a):
                    a = a.split()
		    #print len(a)
		   # print a[600]
                    data_feature.append(a)
                else:
                    break
            f.close()
            m = [data_feature]
            m = np.asarray(m).transpose(2, 1, 0)
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = CHANNEL
            datum.height = m.shape[1]
            datum.width = m.shape[2]
            datum.float_data.extend(m.astype(float).flat)
            str_id = i
            txn.put(str_id.encode("ascii"), datum.SerializeToString())
            count += 1

def gen_output(lmdbname, file_list):
    X = np.zeros((len(file_list), 1, MAX_HEIGHT, 1), dtype=np.float)
    map_size = X.nbytes * 3 + 10000000

    env = lmdb.open(lmdbname, map_size=map_size)
    
    count = 0
    for i in file_list:
        print count
        with env.begin(write=True) as txn:
            filename = os.path.join(DIR, "mydata_output", i + ".txt")
            f = open(filename)
            data_label = []
            while True:
                a = f.readline()[:-1]
                if 0 != len(a):
                    data_label.append(float(a))
                else:
                    break
            f.close()
            m = np.asarray(data_label)
            print m
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = m.shape[0]
            datum.width = 1
            datum.float_data.extend(m.astype(float).flat)
            str_id = i
            txn.put(str_id.encode("ascii"), datum.SerializeToString())
            count += 1

def gen_generate(lmdbname, file_list):
    generate_index = [1, 2, 3, 4, 5]
    # size of conv
    conv_size = [4, 4, 4, 4, 4]
    for index in generate_index:
        print "generate_"+str(index)
        X = np.zeros((len(file_list), conv_size[index-1], MAX_HEIGHT, 1), dtype = np.float32)
        map_size = X.nbytes * 3 + 100000000

        env = lmdb.open(lmdbname+"_"+str(index), map_size=map_size)

        count = 0
        for i in file_list:
            print count
            with env.begin(write=True) as txn:
                filename = os.path.join(DIR, "mydata_generate_"+str(index), i + ".txt")
                f = open(filename)
                data_gen = []
                while True:
                    a = f.readline()[:-2]
                    if 0 != len(a):
                        data_gen.append(a.split())
                    else:
                        break
                f.close()
                m = np.asarray(data_gen)
  #              print m
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = 1
                datum.height = m.shape[0]
                datum.width = m.shape[1]
                datum.float_data.extend(m.astype(float).flat)
                str_id = i
                txn.put(str_id.encode("ascii"), datum.SerializeToString())
                count += 1
def gen_area(lmdbname, file_list):
    X = np.zeros((len(file_list), 1, MAX_HEIGHT, 1), dtype=np.float)
    map_size = X.nbytes * 3 + 10000000

    env = lmdb.open(lmdbname, map_size=map_size)
    
    count = 0
    for i in file_list:
        print count
        with env.begin(write=True) as txn:
            filename = os.path.join(DIR, "mydata_area", i + ".txt")
            f = open(filename)
            data_area = []
            while True:
                a = f.readline()[:-1]
                if 0 != len(a):
                    data_area.append(float(a))
                else:
                    break
            f.close()
            m = np.asarray(data_area)
            print m
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = m.shape[0]
            datum.width = 1
            datum.float_data.extend(m.astype(float).flat)
            str_id = i
            txn.put(str_id.encode("ascii"), datum.SerializeToString())
            count += 1
def gen_dist(lmdbname,file_list):
    print "dist_"
    X = np.zeros((len(file_list), 1, MAX_HEIGHT, 1), dtype = np.float32)
    map_size = X.nbytes * 3 + 100000000

    env = lmdb.open(lmdbname, map_size=map_size)

    count = 0
    for i in file_list:
        print count
        with env.begin(write=True) as txn:
            filename = os.path.join(DIR, "mydata_dist", i + ".txt")
            f = open(filename)
            data_gen = []
            while True:
                a = f.readline()[:-2]
                if 0 != len(a):
                    data_gen.append(a.split())
                else:
                    break
            f.close()
            m = np.asarray(data_gen)
  #         print m
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = m.shape[0]
            datum.width = m.shape[1]
            datum.float_data.extend(m.astype(float).flat)
            str_id = i
            txn.put(str_id.encode("ascii"), datum.SerializeToString())
            count += 1

def gen_idx(lmdbname, file_list):
    generate_index = [1]
    # size of conv
    conv_size = [4]
    for index in generate_index:
        print "generate_"+str(index)
        X = np.zeros((len(file_list), conv_size[index-1], MAX_HEIGHT, 1), dtype = np.float32)
        map_size = X.nbytes * 3 + 100000000

        env = lmdb.open(lmdbname+"_"+str(index), map_size=map_size)

        count = 0
        for i in file_list:
            print count
            with env.begin(write=True) as txn:
                filename = os.path.join(DIR, "mydata_idx", i + ".txt")
                f = open(filename)
                data_gen = []
                while True:
                    a = f.readline()[:-2]
                    if 0 != len(a):
                        data_gen.append(a.split())
                    else:
                        break
                f.close()
                m = np.asarray(data_gen)
  #              print m
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = 1
                datum.height = m.shape[0]
                datum.width = m.shape[1]
                datum.float_data.extend(m.astype(float).flat)
                str_id = i
                txn.put(str_id.encode("ascii"), datum.SerializeToString())
                count += 1

#output dir
DIR=""
number_channel=open("N_1.txt","r").read().split()
for kkkk in range(1,5):
    TRAIN_FILE_LIST = open("./train/t"+str(kkkk)+".txt", "r").read().strip().split()
    TEST_FILE_LIST = open("./train/v"+str(kkkk)+".txt", "r").read().strip().split()
    IDX=str(kkkk)+"/"
    for i in range(all_number):
            CHANNEL=int(number_channel[i])
            print "Channel: "+str(CHANNEL) 
            DIR=DIR_old+"feaCombine_"+str(i)+"/"
            dir_out=DIR_old+IDX+"feaCombine_"+str(i)+"/"
            if not os.path.exists(dir_out):
                    os.makedirs(dir_out)
            gen_input(dir_out+"train_input_lmdb", TRAIN_FILE_LIST)
            gen_output(dir_out+"train_output_lmdb", TRAIN_FILE_LIST)
            gen_generate(dir_out+"train_generate_lmdb", TRAIN_FILE_LIST)

            gen_input(dir_out+"test_input_lmdb", TEST_FILE_LIST)
            gen_output(dir_out+"test_output_lmdb", TEST_FILE_LIST)
            gen_generate(dir_out+"test_generate_lmdb", TEST_FILE_LIST)
            gen_area(dir_out+"test_area_lmdb",TEST_FILE_LIST)
            gen_dist(dir_out+"test_dist_lmdb",TEST_FILE_LIST)
