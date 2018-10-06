import p
import os


#dir_output_all="d:/output/1/"
#dir_feature_origin="D:/big_data/vase_normal/"
#dir_seg_origin="D:/big_data/vase_seg/"
#dir_adj_origin="D:/big_data/vase_adj/"
#dir_dh_angle_orign="D:/big_data/vase_origin/shapes/"


dir_output_all="./test/output/"
dir_feature_origin="./test/"
dir_seg_origin="./test/"
dir_adj_origin="./test/"
dir_dh_angle_orign="./test/"

all_feature_number=593#600.......

p.if_sort=1
neighbor_number=3

if not os.path.exists("./test/output/"):
        os.makedirs("./test/output/")
if not os.path.exists(dir_output_all+'mydata/'):
        os.makedirs(dir_output_all+'mydata/')
if not os.path.exists(dir_output_all+'mydata_dist/'):
        os.makedirs(dir_output_all+'mydata_dist/')
if not os.path.exists(dir_output_all+'mydata_generate_1/'):
        os.makedirs(dir_output_all+'mydata_generate_1/')
if not os.path.exists(dir_output_all+'mydata_generate_2/'):
        os.makedirs(dir_output_all+'mydata_generate_2/')
if not os.path.exists(dir_output_all+'mydata_generate_3/'):
        os.makedirs(dir_output_all+'mydata_generate_3/')
if not os.path.exists(dir_output_all+'mydata_generate_4/'):
        os.makedirs(dir_output_all+'mydata_generate_4/')
if not os.path.exists(dir_output_all+'mydata_generate_5/'):
        os.makedirs(dir_output_all+'mydata_generate_5/')
if not os.path.exists(dir_output_all+'mydata_output/'):
        os.makedirs(dir_output_all+'mydata_output/')
if not os.path.exists(dir_output_all+'perm/'):
        os.makedirs(dir_output_all+'perm/')

for i in range(-3,1):
        name=str(i)
        print "begin "+name+".off"
        if not os.path.exists(dir_feature_origin+name+".off.txt"):
            print "not exist " + name +" feature files"
            continue
        if not os.path.exists(dir_seg_origin+name+".seg"):
            print "not exist " + name +" seg files"
            continue
        if not os.path.exists(dir_adj_origin+name+"_adjacentfaces.txt"):
            print "not exist " + name +" adj files"
            continue
        if not os.path.exists(dir_dh_angle_orign+name+"_Dist.txt"):
            print "not exist " + name +" dh files"
            continue
        
        p.compute(name,dir_output_all,dir_feature_origin,dir_seg_origin,dir_adj_origin,dir_dh_angle_orign,all_feature_number,neighbor_number)
        
        print name+".off done"
