from __future__ import division
import sys
sys.path.append("../../../python")
import caffe
import numpy as np
import os
import build
from copy import deepcopy

perm_dir_root="/media/iot/mydisk2/FCN/wpy/new/Plier/"#data dir

faces_num=open("faces.txt","r").read().split()
idx_num=open("./train/v5.txt","r").read().split()
result_number=""
idx_num.sort()
caffe.set_mode_gpu()
caffe.set_device(1)
all_N=7
all_test_N=len(idx_num)
result=open("eval_result.txt","w")
build.build()
for i in range(7):
	perm_dir=perm_dir_root+"feaCombine_"+str(i)+"/perm/"
	solver = caffe.SGDSolver("./solve/solver_"+str(i)+".prototxt")
	solver.step(3501)
	ac=0
	result.write(str(i)+" AC: ")
	for ii in range(all_test_N):
                N=int(faces_num[int(idx_num[ii])-1])
                perm=open(perm_dir+idx_num[ii]+".txt","r").read().split()
		a=solver.test_nets[0].forward()
		out=solver.test_nets[0].blobs['bigscore'].data[0].transpose((1,2,0))
		file_output = open("./result/"+str(i)+"/"+idx_num[ii]+".txt", 'w')
                im = np.zeros((out.shape[0],out.shape[1],1),np.uint8)
                for iii in range(len(out)):
                        for jjj in range(len(out[i])):
				k=out[iii][jjj][1:]
                        	im[iii][jjj] = k.argmax()+1
                        	#file_output.write(str(im[iii][jjj][0]) + "\n")
                l=np.zeros(N,np.int32)
                for iii in range(len(perm)):
                        m=int(perm[iii])
                        if m<N:
                                l[m]=im[iii][0][0]
                for iii in range(N):
                        file_output.write(str(l[iii]))
                        file_output.write("\n")
                file_output.close()
		print a['accuracy']
		result.write(str(a['accuracy']))
		result.write(" ")
		ac=ac+a['accuracy']
	acc=ac/all_test_N
	result.write("\n")
	result.write("ave : "+str(acc))
	result.write("\n")
	result.write("\n")
result.close()


