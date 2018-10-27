# Scripts

### /getfeature.py

Parses the all-feature text files and writes only the selected features on another folder. 

The all-feature text files contain all the 7 features extracted from the previous step of the pipeline (C++ code), while usually we want to train separately for a given feature. 
This script is used in the all_combine.py script where one should pass the list of features to indicate which features need to be parsed/extracted away from the all-feature file.

### /all_combine.py

For each mesh in the feature folder runs the compute methods from getfeature.py and /pool/start_combine.py. Basically generates the necessary feature and graph text files for the next step of the pipeline (i.e. Caffe lmbd generation). 

# Folders

### /plier_all

The following files are all generated from the code provided in: https://people.cs.umass.edu/~kalo/papers/LabelMeshes/index.html

file  | description
------------- | -------------
X_adjacentfaces.txt  | The annotations (labels) for each
X_DihedralAngle.txt  | The dihedral angles between two adjacent meshes (only used in graph-cut optimization)
X_Dist.txt  | The lengths of common edges between two adjacent meshes (only used in graph-cut optimization)
X_FaceArea.txt  | Face area of each face/triangle (only used in the last accuracy layer)
X_labels.txt  | The annotations (labels) grouped into classes
X.seg   | Each row represents a triangle label  



### /fea/ModelName/fea/featureCombine_X
Example: /fea/Plier/fea/featureCombine_6 contains the SPIN IMAGE feature. Feature indexes range from 0 to 6. 

The selected features from the /getfeature.py script end up in this folder. Nothing fancy, the files here are only a subset of the all-feature text files extracted from the feature-extraction (C++) step.

## /pool
Contains scripts for coarsening, building graph adjacency matrix, BreadthFirstSearch etc.
In general graph processing utils

## /output

folder  | description
------------- | -------------
mydata_output  | The labels (annotations) of each triangle (node in graph). <br> Fake nodes get label 0. <br> The number of nodes corresponds to the mydata_generate_1 <br> (i.e. #nodes == #mydata_generate1)
mydata  | The selected features (e.g. spin image) vectors for each node/triangle. <br> Fake nodes get a vector of all zeros as a feature vector. <br> The number of nodes corresponds to the mydata_generate_1 <br> (i.e. #nodes == #mydata_generate1)
mydata_area    | The face areas (only used in the last accuracy layer)
mydata_dist    | Used only in graph-cut optimization 
mydata_generate_1    | Layer of the balanced binary tree 
mydata_generate_2    | Layer of the balanced binary tree 
mydata_generate_3    | Layer of the balanced binary tree
mydata_generate_4    | Layer of the balanced binary tree 
mydata_generate_5    | Layer of the balanced binary tree  
mydata_idx  | not used AFAIK
perm    | the pooling order of the first layer - mydata_generate_1 <br> This folder is used in the evaluation phase in the /caffe-plier/eval-solve.py script

