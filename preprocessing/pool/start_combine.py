import p
import os


def compute_one(feature_number,name,dir_output,dir_feature,dir_all):
        dir_output_all=dir_output
        dir_feature_origin=dir_feature
        dir_seg_origin=dir_all
        dir_adj_origin=dir_all
        dir_dh_angle_orign=dir_all
        dir_face_orign=dir_all
        all_feature_number=feature_number#600.......

        p.if_sort=1
        neighbor_number=9

        if not os.path.exists(dir_output_all):
                os.makedirs(dir_output_all)
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
        if not os.path.exists(dir_output_all+'mydata_area/'):
                os.makedirs(dir_output_all+'mydata_area/')
        if not os.path.exists(dir_output_all+'mydata_idx/'):
                os.makedirs(dir_output_all+'mydata_idx/')

        print "begin "+name+".off"
        if not os.path.exists(dir_feature_origin+name+".off.txt"):
            print "not exist " + name +" feature files"
            return
        if not os.path.exists(dir_seg_origin+name+".seg"):
            print "not exist " + name +" seg files"
            return
        if not os.path.exists(dir_adj_origin+name+"_adjacentfaces.txt"):
            print "not exist " + name +" adj files"
            return
        #if not os.path.exists(dir_dh_angle_orign+name+"_DihedralAngle.txt"):
           # print "not exist " + name +" dh files"
            #return
        if not os.path.exists(dir_dh_angle_orign+name+"_FaceArea.txt"):
            print "not exist " + name +" face files"
            return
        
        p.compute(name,dir_output_all,dir_feature_origin,dir_seg_origin,dir_adj_origin,dir_dh_angle_orign,all_feature_number,neighbor_number,dir_face_orign)
        
        print name+".off done"
