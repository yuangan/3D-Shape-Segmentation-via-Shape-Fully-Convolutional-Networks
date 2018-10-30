# Scripts

### build.py
Generates train_X.protoxt and solver_X.protoxt files where X is the index of the selected feature from the 7 available features (e.g. PCA, Spin Image). The indexes start from 0 to 6.
This script is called from the eval-solve.py script, before starting the training of the models.


### eval-solve.py
Contrary to gen_lmbd.py, this script only evaluates one pipeline at a time. The testing set is specified in the line: 
idx_num=open("./train/v5.txt","r").read().split()


The loop range should be set based on the features we want to train.
ex1: i to range (7) -> train for all features 
ex2: i to range (6,7) -> train only for Spin Image 

### gen_lmdb.py
Generates lmbd data for all the training and testing mesh lists specified in the /train folder. Example: If five t1 ... t5 and five v1 ... v5 lists are specified, and the loop ranges to 5, this script will generate data for 5 separate end-to-end pipelines.

The first loop specifies the number of pipelines based on different training/testing sets.
The nested loop specifies for which feature we want to generate data for.

# Folders

### /result 
The segmentation results for each of the features used in training and then evaluation.
#### /result/6 
This folder will contain the labels generated from the model trained with Spin Image features.

### /solve
Contains the generated (by the build.py script) train.protoxt and solver.protoxt files for each of the features.
Example: train_6.protoxt corresponds to the training network for Spin Image.

### /train
Contains the list of meshes that present the training and testing sets.
The training files correspond to the SB6 "dataset" mentined in the paper, where the 6 stands for 6 randomly chosen meshes to form the training set.
- The five t1 ... t5 text files correspond to the lists of meshes used for training.
- The five v1 ... v5 text files correspond to the lists meshes used for testing.

From the SFCN paper: "we repeated our approach five times to compute the average performance", that is why there are five separate files for both training and testing/validation.

# Files

### faces.txt
Contains the number of faces/triangles for each mesh on the PSB dataset. Has 400 entries, an integer for each mesh.

### N_1.txt
Contains the feature dimension for each of the 7 features (i.e. PCA, Spin Image, Geodesic Distance etc.).
It is used in the gen_lmdb.py script.

### train.prototxt
Defines the Neural network layers and specifies the data sources.