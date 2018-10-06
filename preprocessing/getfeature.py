import numpy as np
import math

FEATURE_CUR_START=0#64
FEATURE_PCA_START=64#48
FEATURE_SC_START=112#270
FEATURE_GEO_START=382#15
FEATURE_SDF_START=397#72
FEATURE_VSI_START=469#24
FEATURE_SI_START=493#100


if_CUR=1
if_PCA=1
if_SC=1
if_GEO=1
if_SDF=1
if_VSI=1
if_SI=1
def set_feature_zero():
    if_CUR=0
    if_PCA=0
    if_SC=0
    if_GEO=0
    if_SDF=0
    if_VSI=0
    if_SI=0
def set_feature(one_list,N):
    if not len(one_list)==N:
        print "error! feature number error!"
        return
    for i in one_list:
        if i==1:
            if_CUR=1
            print "USE CUR"
        if i==2:
            if_PCA=1
            print "USE PCA"
        if i==3:
            if_SC=1
            print "USE SC"
        if i==4:
            if_GEO=1
            print "USE GEO"
        if i==5:
            if_SDF=1
            print "USE SDF"
        if i==6:
            if_VSI=1
            print "USE VSI"
        if i==7:
            if_SI=1
            print "USE SI"

def compute(list_one,name,old_feature_dir,new_feature_dir):
    #name=str(i)
    old_feature_name=old_feature_dir+name+".off.txt"
    
    new_feature_name=new_feature_dir+name+".off.txt"
    
    file_old_feature=open(old_feature_name,"r")
    read_old_feature=file_old_feature.read().split('\n')#628
    read_old_feature=read_old_feature[:-1]
    number_faces=len(read_old_feature)
    print "face's num : "+ str(number_faces)

    w_new_file_name=open(new_feature_name,"w")
    NN=0
    if (list_one[0]==1):
        NN=NN+64
    if (list_one[1]==1):
        NN=NN+48
    if (list_one[2]==1):
        NN=NN+270
    if (list_one[3]==1):
        NN=NN+15
    if (list_one[4]==1):
        NN=NN+72
    if (list_one[5]==1):
        NN=NN+24
    if (list_one[6]==1):
        NN=NN+100
    for ii in range(number_faces):
        aaa=read_old_feature[ii].split()
        if len(aaa)<593:
            continue
        if (list_one[0]==1):
            for iii in range(64):
                a=float(aaa[FEATURE_CUR_START+iii])
                w_new_file_name.write(str(a))
                w_new_file_name.write(" ")
        if (list_one[1]==1):
            for iii in range(48):
                a=float(aaa[FEATURE_PCA_START+iii])
                w_new_file_name.write(str(a))
                w_new_file_name.write(" ")
        if (list_one[2]==1):
            for iii in range(270):
                a=float(aaa[FEATURE_SC_START+iii])
                w_new_file_name.write(str(a))
                w_new_file_name.write(" ")
        if (list_one[3]==1):
            for iii in range(15):
                a=float(aaa[FEATURE_GEO_START+iii])
                w_new_file_name.write(str(a))
                w_new_file_name.write(" ")
        if (list_one[4]==1):
            for iii in range(72):
                a=float(aaa[FEATURE_SDF_START+iii])
                w_new_file_name.write(str(a))
                w_new_file_name.write(" ")
        if (list_one[5]==1):
            for iii in range(24):
                a=float(aaa[FEATURE_VSI_START+iii])
                w_new_file_name.write(str(a))
                w_new_file_name.write(" ")
        if (list_one[6]==1):
            for iii in range(100):
                a=float(aaa[FEATURE_SI_START+iii])
                w_new_file_name.write(str(a))
                w_new_file_name.write(" ")
        
        w_new_file_name.write("\n")
        del aaa
    w_new_file_name.close()
    file_old_feature.close()
    print NN
    del read_old_feature
    return NN
