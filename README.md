# 3D Shape Segmentation via Shape Fully Convolutional Networks
[paper](https://arxiv.org/abs/1702.08675)

##Enviroment
201610 version of caffe: [caffe201610](https://drive.google.com/open?id=1WfzLWRakN2luR-qM7SgPD_KDw93bY0qu)
------
## Data
[plier_fea](https://drive.google.com/open?id=1BB7YNz56MdKstRcckJoBM6QY_OkrlJ0P)
------

## 1.segmesh
to calculate distance between two neighbour meshes.

## 2.preprocessing (you need to run the python files to build the data for SFCN)

requirement: python 2.7, numpy, scipy, sklearn

a.edit the all_cimbine.py

--old_feature_dir : 3d shapes' feature files

--dir_all: 3d shapes' other files(dist, area, adj, seg)

b.set the output dir

--dir_output

c.run
### Ps: More details are in ./preprocessing/README.md

## 3.build caffe 

see caffe_read.txt

## 4.caffe-plier

a.run gen_lmdb.py to generate the lmdb dataset for caffe

b.run eval-solve.py
### Ps: More details are in ./caffe-plier/README.md

## 5.SFCN's multilabel graph cut
if you want to run graph cut,
you need multilabel graph cut code(you can find it in gco)
