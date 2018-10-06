import itertools
import getfeature
import os
import numpy as np
import sys
sys.path.append("./pool/")
import start_combine

Combine_Number=1
name_dir_list=["Plier/"]
for _N in range(0,1):
    dir_name=name_dir_list[_N]
    print dir_name
    old_feature_dir="plier_fea/"#feature files
    dir_all="plier_all/"#other files(dist,area,adj,seg)


    new_feature_dir="fea/"+dir_name+"fea/"#temp feature files
    dir_output="output/"+dir_name#output files

    all_list=list(itertools.combinations([1,2,3,4,5,6,7],Combine_Number))
    print "all number: "+str(len(all_list))

    file_feature=open("./Result_"+str(Combine_Number)+".txt","w")
    feaN=open("./N_"+str(Combine_Number)+".txt","w")
    for one_in_list in all_list:
        idx=all_list.index(one_in_list)
        if (idx==0):
            continue
        if (idx==1):
            continue
        if (idx==4):
            continue
        if (idx==5):
            continue
        print "the idx is: "+str(idx)
        print "the Combine is: "+str(one_in_list)
        print one_in_list
        file_feature.write(str(one_in_list))
        file_feature.write("\n")
        getfeature.set_feature_zero()
        getfeature.set_feature(one_in_list,Combine_Number)
        fea_list=np.zeros(7,np.int32)
        print fea_list
        for i in range(Combine_Number):
            fea_list[int(one_in_list[i])-1]=1
        print fea_list
        temp_feature_dir=new_feature_dir+"feaCombine_"+str(idx)+"/"
        for i in range(201,221):#plier id 201-220
            name=str(i)
            if not os.path.exists(old_feature_dir+name+".off.txt"):
                print "not exist!"+name+".off.txt"
                continue
            if not os.path.exists(temp_feature_dir):
                os.makedirs(temp_feature_dir)
            feature_number=getfeature.compute(fea_list,name,old_feature_dir,temp_feature_dir)
            now_output=dir_output+"feaCombine_"+str(idx)+"/"
            start_combine.compute_one(feature_number,name,now_output,temp_feature_dir,dir_all)
        feaN.write(str(feature_number))
        feaN.write("\n")
    file_feature.close()
    feaN.close()
        
