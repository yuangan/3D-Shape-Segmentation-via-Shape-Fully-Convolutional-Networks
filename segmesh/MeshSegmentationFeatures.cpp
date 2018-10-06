// This file is part of "LearningMeshSegmentation".

// Copyright (c) 2010 Evangelos Kalogerakis
// All rights reserved. 
// If you use this code or its parts, please also cite 
// "Learning 3D Mesh Segmentation and Labeling, 
// E. Kalogerakis, A. Hertzmann, K. Singh, ACM Transactions on Graphics, 
// Vol. 29, No. 3, July 2010"

// "LearningMeshSegmentation" is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// "LearningMeshSegmentation" is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with "LearningMeshSegmentation".  If not, see <http://www.gnu.org/licenses/>.

// note: if dataset is upright oriented, uncomment FeatureExporter, line 678

#include "MeshSegmentationFeatures.h"
#include "JointBoost.h"
#include "TriMesh.h"
//#include "kmeans.h"

int* readLabels( char *, int );
void writeNLOGW( const char*, JointBoost&, vector<TriMesh*>& , int  );
void writeEDGESNLOGW( const char*, JointBoost&, vector<TriMesh*>&  );

#ifdef USE_MATLAB
mxArray* createMxArrayBestFeatures(float**, int, std::vector<int>  );
extern FeatureSet* exportHKSandLaplacianFeatures(TriMesh* m,  FeatureSet*, int, int, Engine*, bool writeDebugInfo = false, bool returnNumFeaturesOnly = false);
#endif

extern TriMesh * processMesh( TriMesh *, bool writeDebugInfo = false );
extern FeatureSet* importFeatures(TriMesh*, FeatureSet*, int, int, bool writeDebugInfo = false, bool returnNumFeaturesOnly = false);
extern FeatureSet* exportCurvatureFeatures(TriMesh*, FeatureSet*, int, int, bool writeDebugInfo = false, bool returnNumFeaturesOnly = false );
extern FeatureSet* exportPCAFeatures(TriMesh*, FeatureSet*, int, int, bool writeDebugInfo = false, bool returnNumFeaturesOnly = false );
extern FeatureSet* exportSCFeatures(TriMesh* m, FeatureSet*, int, int, float, float, bool writeDebugInfo = false, bool returnNumFeaturesOnly = false);
extern FeatureSet* exportSDFFeatures(TriMesh* m, FeatureSet*, int, int, bool writeDebugInfo = false, bool returnNumFeaturesOnly = false );
extern FeatureSet* exportSpinImageFeatures(TriMesh* m,  FeatureSet*, int, int, bool writeDebugInfo = false, bool returnNumFeaturesOnly = false);
extern FeatureSet* exportSCClassProbFeatures(TriMesh* m,  FeatureSet*, int, int, float**, int, bool writeDebugInfo = false, bool returnNumFeaturesOnly = false);
extern FeatureSet* exportEdgeFeatures(TriMesh* m,  FeatureSet*, int, int, int*, float, bool writeDebugInfo = false, bool returnNumFeaturesOnly = false);
extern FeatureSet* exportSCEdgeFeatures(TriMesh* m, FeatureSet*, int, int, float** PH, bool writeDebugInfo );



/////////////////////////////////////////////////////
///////////////// MAIN FUNCTION /////////////////////
/////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	// argv[0] program
	// argv[1] number of labels (use 1 to output features in files and then terminate without learning)
	// argv[2] meshes list filename (if there is a line with -, it means training mode)  
	if ( !JointBoost::selfTest() ) {
		return -100;
	}

#ifdef USE_MATLAB
	Engine* matlabEng = initializeMatlabEngine();
	if (matlabEng == NULL) {
		return -200;
	}
#endif

	// INITIALIZATION
	vector<TriMesh*> meshes;
	vector<TriMesh*> CVmeshes;
	vector<int*> labels;
	vector<int*> CVlabels;

	if (argc < 3) {
		std::cerr << "Specify number of labels and mesh filenames to read (use #labels=1 for feature extraction only)"  << std::endl;
		return -1;
	} else if (argc > 3) {
		std::cerr << "Too many input arguments!" << std::endl;
		return -2;
	}

	int numlabels = atoi( argv[1] );
	if ( numlabels <= 0 ) {
		std::cerr << "Having <=0 number of labels does not make sense!" << std::endl;
		return -3;
	}
	char* filename = argv[2];
	char meshFilename[1024];
	char labelsFilename[1024];
	bool isCV = false;
	std::ifstream fin(filename);
	if (!fin.good() ) {
		std::cerr << "Failed to open mesh list filename " << filename << std::endl;
		return -3;
	}
	while( fin.good() ) {
		fin >> meshFilename;
		if (strcmp(meshFilename, "-") == 0) {
			isCV = true;
			continue;
		}
		if (isemptychar(meshFilename) == true || strlen( meshFilename ) <= 1 || !fin.good() ) {
			continue;
		}
    if (numlabels > 1)
      fin >> labelsFilename;
    else
      labelsFilename[0] = '\0';

		if (!isCV) {			
			meshes.push_back( TriMesh::read(meshFilename) );
			if ( meshes[ meshes.size() - 1 ] == NULL ) {
				std::cerr << "Failed to open mesh file " << meshFilename << std::endl;
				return -5;
			}
      labels.push_back(readLabels(labelsFilename, meshes[meshes.size() - 1]->faces.size()));
      if (labels[labels.size() - 1] == NULL) {
        std::cerr << "Failed to open labels file " << labelsFilename << std::endl;
        return -6;
      }
			std::cout << "Read mesh " << meshFilename << " (" << meshes[ meshes.size() - 1 ]->vertices.size() << " vertices) and labels file " << labelsFilename << std::endl; 
		} else {
			CVmeshes.push_back( TriMesh::read(meshFilename) );
			if ( CVmeshes[ CVmeshes.size() - 1 ] == NULL ) {
				std::cerr << "Failed to open CV mesh file " << meshFilename << std::endl;
				return -7;
			}
      CVlabels.push_back(readLabels(labelsFilename, CVmeshes[CVmeshes.size() - 1]->faces.size()));
      if (CVlabels[CVlabels.size() - 1] == NULL) {
        std::cerr << "Failed to open CV labels file " << labelsFilename << std::endl;
        return -8;
      }
			std::cout << "Read CV mesh " << meshFilename << " (" << CVmeshes[ CVmeshes.size() - 1 ]->vertices.size() << " vertices) and labels file " << labelsFilename << std::endl; 
		}
	}
	fin.close();






	// INITIAL PROCESSING
	int numFaces = 0;
	int numCVFaces = 0;
	float boundariesPercentage = 0;
	for (int i = 0; i < meshes.size(); i++) {
		processMesh( meshes[i], DEBUG_FEATURES );
		numFaces += meshes[i]->faces.size();
	}
	for (int i = 0; i < CVmeshes.size(); i++) {
		processMesh( CVmeshes[i], DEBUG_FEATURES );
		numCVFaces += CVmeshes[i]->faces.size();
	}
	float *INITIALWEIGHTS = new float[ numFaces ];
	float *INITIALCVWEIGHTS = new float[ numCVFaces ];
	int   *LABELS = new int[ numFaces ];
	int *CVLABELS = new int[ numCVFaces ];
	int ii = 0; 
	for (int i = 0; i < meshes.size(); i++) {
		for (int j = 0; j < meshes[i]->faces.size(); j++) {
			LABELS[ii] = labels[i][j];
			INITIALWEIGHTS[ii] = meshes[i]->faces[j].faceArea;
			ii++;
			for (int k = 0; k < 3; k++) {
				boundariesPercentage += ( labels[i][j] != labels[i][ meshes[i]->across_edge[j][k] ] );
			}
		}
	}
	boundariesPercentage /= 3.0f * (float)numFaces;
	ii = 0;
	for (int i = 0; i < CVmeshes.size(); i++) {
		for (int j = 0; j < CVmeshes[i]->faces.size(); j++) {
			CVLABELS[ii] = CVlabels[i][j];
			INITIALCVWEIGHTS[ii] = CVmeshes[i]->faces[j].faceArea;
			ii++;
		}
	}
	int meanGDfeatureID = 0;

	std::ofstream fout_pwparms; fout_pwparms.precision(10); fout_pwparms.setf( std::ios::scientific );
	std::ofstream fout_pwparmsedges; fout_pwparmsedges.precision(10); fout_pwparmsedges.setf( std::ios::scientific );
	std::ifstream fin_pwparms;
	std::ifstream fin_pwparmsedges;
	std::ifstream fin_pwdictionary;
	if (isCV) {
		fout_pwparms.open(PWPARMS_FILE);
		fout_pwparmsedges.open(EDGESPWPARMS_FILE);
		if ( !fout_pwparms.good() || !fout_pwparmsedges.good() ) {
			std::cerr << "Failed to open pw parameters file for writing " << PWPARMS_FILE << std::endl;
			return -9;
		}
	} else {
		fin_pwparms.open(PWPARMS_FILE);
		fin_pwparmsedges.open(EDGESPWPARMS_FILE);
    if (numlabels > 1)
    {
      if (!fin_pwparms.good() || !fin_pwparmsedges.good()) {
        std::cerr << "Failed to open pw parameters file for reading " << PWPARMS_FILE << std::endl;
        return -10;
      }
    }
	}
	FeatureSet *features = NULL, *featuresCV = NULL;
	int totalNumFeatures = 0;


  features = importFeatures(meshes[0], features, 0, 0, false, true);
  totalNumFeatures += features->numFeatures;
  delete features;
  if (totalNumFeatures == 0) // use predefined features
  {
    std::cout << "Will use predefined features." << std::endl;
#ifdef USE_MATLAB
    features = exportHKSandLaplacianFeatures( NULL, features, 0, 0, NULL, false, true );
    totalNumFeatures += features->numFeatures;
    delete features;	
#endif
    features = exportCurvatureFeatures(NULL, features, 0, 0, false, true);
    totalNumFeatures += features->numFeatures;
    delete features;
    features = exportPCAFeatures(NULL, features, 0, 0, false, true);
    totalNumFeatures += features->numFeatures;
    delete features;
    features = exportSCFeatures(NULL, features, 0, 0, 0, 0, false, true);
    totalNumFeatures += features->numFeatures + features->numFeatures2;
    delete features;
    features = exportSDFFeatures(NULL, features, 0, 0, false, true);
    totalNumFeatures += features->numFeatures + features->numFeatures2;
    delete features;
    features = exportSpinImageFeatures(NULL, features, 0, 0, false, true);
    totalNumFeatures += features->numFeatures;
    delete features;
  }
  else // use imported features
  {
    std::cout << "Will use imported features (number = " << totalNumFeatures << "). " << std::endl;
  }
  features = exportSCClassProbFeatures(NULL, features, 0, 0, NULL, numlabels, false, true); // always use these
  totalNumFeatures += features->numFeatures;
  delete features;


	FeatureSet* featuresALL = new FeatureSet();
	featuresALL->numFeatures = totalNumFeatures;
	featuresALL->FEATURES = new float*[featuresALL->numFeatures];
	for (int j = 0; j < featuresALL->numFeatures; j++) {
		featuresALL->FEATURES[j] = new float[ numFaces ];
		for (int i = 0; i < numFaces; i++) {
			featuresALL->FEATURES[j][i] = 0.0f;
		}
	}
	FeatureSet* featuresALLCV = NULL;
	if (isCV) {
		featuresALLCV = new FeatureSet();
		featuresALLCV->numFeatures = totalNumFeatures;
		featuresALLCV->FEATURES = new float*[featuresALLCV->numFeatures];
		for (int j = 0; j < featuresALLCV->numFeatures; j++) {
			featuresALLCV->FEATURES[j] = new float[ numCVFaces ];
			for (int i = 0; i < numCVFaces; i++) {
				featuresALLCV->FEATURES[j][i] = 0.0f;
			}
		}
	}
	int jj = 0;





  ii = 0;
  for (int i = 0; i < meshes.size(); i++) {
    features = importFeatures(meshes[i], features, ii, numFaces, DEBUG_FEATURES);
    ii += meshes[i]->faces.size();
  }
  ii = 0;
  for (int i = 0; i < CVmeshes.size(); i++) {
    featuresCV = importFeatures(CVmeshes[i], featuresCV, ii, numCVFaces, DEBUG_FEATURES);
    ii += CVmeshes[i]->faces.size();
  }
  for (int j = 0; j < features->numFeatures; j++) {
    for (int i = 0; i < numFaces; i++) {
      featuresALL->FEATURES[j+jj][i] = features->FEATURES[j][i];
    }
    for (int i = 0; i < numCVFaces; i++) {
      featuresALLCV->FEATURES[j+jj][i] = featuresCV->FEATURES[j][i];
    }
  }
  jj += features->numFeatures;
  delete features;
  if (isCV) {
    delete featuresCV;
  }

  if (jj == 0) // use predefined features
  {
#ifdef USE_MATLAB
    ii = 0;
    for (int i = 0; i < meshes.size(); i++) {
      features = exportHKSandLaplacianFeatures( meshes[i], features, ii, numFaces, matlabEng, DEBUG_FEATURES );
      ii += meshes[i]->faces.size();
    }
    ii = 0;
    for (int i = 0; i < CVmeshes.size(); i++) {
      featuresCV = exportHKSandLaplacianFeatures( CVmeshes[i], featuresCV, ii, numCVFaces, matlabEng, DEBUG_FEATURES );
      ii += CVmeshes[i]->faces.size();
    }
    for (int j = 0; j < features->numFeatures; j++) {
      for (int i = 0; i < numFaces; i++) {
        featuresALL->FEATURES[j+jj][i] = features->FEATURES[j][i];
      }
      for (int i = 0; i < numCVFaces; i++) {
        featuresALLCV->FEATURES[j+jj][i] = featuresCV->FEATURES[j][i];
      }
    }
    jj += features->numFeatures;
    delete features;
    if (isCV) {
      delete featuresCV;
    }
#endif

    ii = 0;
    for (int i = 0; i < meshes.size(); i++) {
      features = exportCurvatureFeatures(meshes[i], features, ii, numFaces, DEBUG_FEATURES);
      ii += meshes[i]->faces.size();
    }
    ii = 0;
    for (int i = 0; i < CVmeshes.size(); i++) {
      featuresCV = exportCurvatureFeatures(CVmeshes[i], featuresCV, ii, numCVFaces, DEBUG_FEATURES);
      ii += CVmeshes[i]->faces.size();
    }
    for (int j = 0; j < features->numFeatures; j++) {
      for (int i = 0; i < numFaces; i++) {
        featuresALL->FEATURES[j + jj][i] = features->FEATURES[j][i];
      }
      for (int i = 0; i < numCVFaces; i++) {
        featuresALLCV->FEATURES[j + jj][i] = featuresCV->FEATURES[j][i];
      }
    }
    jj += features->numFeatures;
    delete features;
    if (isCV) {
      delete featuresCV;
    }

    ii = 0;
    for (int i = 0; i < meshes.size(); i++) {
      features = exportPCAFeatures(meshes[i], features, ii, numFaces, DEBUG_FEATURES);
      ii += meshes[i]->faces.size();
    }
    ii = 0;
    for (int i = 0; i < CVmeshes.size(); i++) {
      featuresCV = exportPCAFeatures(CVmeshes[i], featuresCV, ii, numCVFaces, DEBUG_FEATURES);
      ii += CVmeshes[i]->faces.size();
    }
    for (int j = 0; j < features->numFeatures; j++) {
      for (int i = 0; i < numFaces; i++) {
        featuresALL->FEATURES[j + jj][i] = features->FEATURES[j][i];
      }
      for (int i = 0; i < numCVFaces; i++) {
        featuresALLCV->FEATURES[j + jj][i] = featuresCV->FEATURES[j][i];
      }
    }
    jj += features->numFeatures;
    delete features;
    if (isCV) {
      delete featuresCV;
    }

    ii = 0;
    for (int i = 0; i < meshes.size(); i++) {
      features = exportSCFeatures(meshes[i], features, ii, numFaces, -90.0f, 90.0f, DEBUG_FEATURES);
      ii += meshes[i]->faces.size();
    }
    ii = 0;
    for (int i = 0; i < CVmeshes.size(); i++) {
      featuresCV = exportSCFeatures(CVmeshes[i], featuresCV, ii, numCVFaces, -90.0f, 90.0f, DEBUG_FEATURES);
      ii += CVmeshes[i]->faces.size();
    }
    for (int j = 0; j < features->numFeatures; j++) {
      for (int i = 0; i < numFaces; i++) {
        featuresALL->FEATURES[j + jj][i] = features->FEATURES[j][i];
      }
      for (int i = 0; i < numCVFaces; i++) {
        featuresALLCV->FEATURES[j + jj][i] = featuresCV->FEATURES[j][i];
      }
    }
    jj += features->numFeatures;
    meanGDfeatureID = jj;
    for (int j = 0; j < features->numFeatures2; j++) {
      for (int i = 0; i < numFaces; i++) {
        featuresALL->FEATURES[j + jj][i] = features->FEATURES2[j][i];
      }
      for (int i = 0; i < numCVFaces; i++) {
        featuresALLCV->FEATURES[j + jj][i] = featuresCV->FEATURES2[j][i];
      }
    }
    jj += features->numFeatures2;
    delete features;
    if (isCV) {
      delete featuresCV;
    }

    ii = 0;
    for (int i = 0; i < meshes.size(); i++) {
      features = exportSDFFeatures(meshes[i], features, ii, numFaces, DEBUG_FEATURES);
      ii += meshes[i]->faces.size();
    }
    ii = 0;
    for (int i = 0; i < CVmeshes.size(); i++) {
      featuresCV = exportSDFFeatures(CVmeshes[i], featuresCV, ii, numCVFaces, DEBUG_FEATURES);
      ii += CVmeshes[i]->faces.size();
    }
    for (int j = 0; j < features->numFeatures; j++) {
      for (int i = 0; i < numFaces; i++) {
        featuresALL->FEATURES[j + jj][i] = features->FEATURES[j][i];
      }
      for (int i = 0; i < numCVFaces; i++) {
        featuresALLCV->FEATURES[j + jj][i] = featuresCV->FEATURES[j][i];
      }
    }
    jj += features->numFeatures;
    for (int j = 0; j < features->numFeatures2; j++) {
      for (int i = 0; i < numFaces; i++) {
        featuresALL->FEATURES[j + jj][i] = features->FEATURES2[j][i];
      }
      for (int i = 0; i < numCVFaces; i++) {
        featuresALLCV->FEATURES[j + jj][i] = featuresCV->FEATURES2[j][i];
      }
    }
    jj += features->numFeatures2;
    delete features;
    if (isCV) {
      delete featuresCV;
    }

    ii = 0;
    for (int i = 0; i < meshes.size(); i++) {
      features = exportSpinImageFeatures(meshes[i], features, ii, numFaces, DEBUG_FEATURES);
      ii += meshes[i]->faces.size();
    }
    ii = 0;
    for (int i = 0; i < CVmeshes.size(); i++) {
      featuresCV = exportSpinImageFeatures(CVmeshes[i], featuresCV, ii, numCVFaces, DEBUG_FEATURES);
      ii += CVmeshes[i]->faces.size();
    }
    for (int j = 0; j < features->numFeatures; j++) {
      for (int i = 0; i < numFaces; i++) {
        featuresALL->FEATURES[j + jj][i] = features->FEATURES[j][i];
      }
      for (int i = 0; i < numCVFaces; i++) {
        featuresALLCV->FEATURES[j + jj][i] = featuresCV->FEATURES[j][i];
      }
    }
    jj += features->numFeatures;
    delete features;
    if (isCV) {
      delete featuresCV;
    }
  } // end of if for predefined features


  if (numlabels == 1)
  {
    int kk = 0;
    for (int i = 0; i < meshes.size(); i++)
    {
      std::string out_filename = std::string(meshes[i]->filename) + ".txt";
      std::cout << "Exporting " << featuresALL->numFeatures << " features to " << out_filename << std::endl;
      std::ofstream out_file(out_filename.c_str());
      for (int k = 0; k < meshes[i]->faces.size(); k++)
      {
        for (int j = 0; j < featuresALL->numFeatures; j++)
        {
          out_file << featuresALL->FEATURES[j][kk + k] << " ";
        }
        out_file << std::endl;
      }

      kk += meshes[i]->faces.size();
      out_file.close();
    }
    return 0;
  }

	///// CASCADE 
	float minCVerror = FLT_MAX;
	for (int em_iter = 0; em_iter < ADABOOST_CASCADES; em_iter++) {
		if (isCV) {
			JointBoost A(featuresALL->FEATURES, LABELS, INITIALWEIGHTS, numFaces, numlabels, featuresALL->numFeatures, featuresALLCV->FEATURES, CVLABELS, INITIALCVWEIGHTS, numCVFaces);
			if ( A.getminCVError() <= minCVerror || em_iter <= 1) {
				minCVerror = A.getminCVError();
				A.writeLearnedParameters( fout_pwparms );
				writeNLOGW(NLOGW_FILE, A, meshes, numlabels);
			} else {
				std::cout << std::endl << "Validation error is larger. Ignoring previous JointBoost and stopping cascade..." << std::endl;
				break;
			}
			if (em_iter < ADABOOST_CASCADES-1) {
				ii = 0;
				for (int i = 0; i < meshes.size(); i++) {
					features = exportSCClassProbFeatures( meshes[i], features, ii, numFaces, A.PH, numlabels, DEBUG_FEATURES );
					ii += meshes[i]->faces.size();
				}
			}

			A.test(featuresALLCV->FEATURES, CVLABELS, INITIALCVWEIGHTS, numCVFaces);
			writeNLOGW(CVNLOGW_FILE, A, CVmeshes, numlabels);
			if (em_iter < ADABOOST_CASCADES-1) {
				ii = 0;
				for (int i = 0; i < CVmeshes.size(); i++) {
					featuresCV = exportSCClassProbFeatures( CVmeshes[i], featuresCV, ii, numCVFaces, A.PH, numlabels, DEBUG_FEATURES );
					ii += CVmeshes[i]->faces.size();
				}
				for (int j = 0; j < features->numFeatures; j++) {
					for (int i = 0; i < numFaces; i++) {
						featuresALL->FEATURES[j+jj][i] = features->FEATURES[j][i];
					}
					for (int i = 0; i < numCVFaces; i++) {
						featuresALLCV->FEATURES[j+jj][i] = featuresCV->FEATURES[j][i];
					}
				}
				delete features;
				delete featuresCV;
			}

		} else {
			JointBoost A;
			if ( A.readLearnedParameters( fin_pwparms ) == true) {
				std::cout << "Read JointBoost" << em_iter << " parameters." << std::endl;
				A.test(featuresALL->FEATURES, LABELS, INITIALWEIGHTS, numFaces);
				writeNLOGW(NLOGW_FILE, A, meshes, numlabels);
			} else {
				break;
			}
			if (em_iter < ADABOOST_CASCADES-1) {
				ii = 0;
				for (int i = 0; i < meshes.size(); i++) {
					features = exportSCClassProbFeatures( meshes[i], features, ii, numFaces, A.PH, numlabels, DEBUG_FEATURES );
					ii += meshes[i]->faces.size();
				}
				for (int j = 0; j < features->numFeatures; j++) {
					for (int i = 0; i < numFaces; i++) {
						featuresALL->FEATURES[j+jj][i] = features->FEATURES[j][i];
					}
				}
				delete features;
			}
		}
	}
	delete featuresALL;
	if (isCV) {
		delete featuresALLCV;
	}



	//PIECEWISE TRAINING FOR EDGE FEATURES
	delete[] INITIALWEIGHTS;
	delete[] INITIALCVWEIGHTS;
	delete[] LABELS;
	delete[] CVLABELS;

	ii = 0;
	for (int i = 0; i < meshes.size(); i++) {
		features = exportEdgeFeatures( meshes[i], features, 3*ii, 3*numFaces, labels[i], 1.0f / (boundariesPercentage+EPSILON), DEBUG_FEATURES );
		ii += meshes[i]->faces.size();
	}
	ii = 0;
	for (int i = 0; i < CVmeshes.size(); i++) {
		featuresCV = exportEdgeFeatures( CVmeshes[i], featuresCV, 3*ii, 3*numCVFaces, CVlabels[i], 1.0f / (boundariesPercentage+EPSILON), DEBUG_FEATURES );
		ii += CVmeshes[i]->faces.size();
	}


	///// CASCADE 
	minCVerror = FLT_MAX;
	for (int em_iter = 0; em_iter < ADABOOST_CASCADES; em_iter++) {
		if (isCV) {
			JointBoost A(features->FEATURES, features->LABELS, features->WEIGHTS, 3*numFaces, 2, features->numFeatures, featuresCV->FEATURES, featuresCV->LABELS, featuresCV->WEIGHTS, 3*numCVFaces);			
			if ( A.getminCVError() <= minCVerror ) {
				minCVerror = A.getminCVError();
				A.writeLearnedParameters( fout_pwparmsedges );
				writeEDGESNLOGW(EDGESNLOGW_FILE, A, meshes);
			} else {
				std::cout << std::endl << "Validation error is larger. Ignoring previous JointBoost and stopping cascade..." << std::endl;
				break;
			}
			if (em_iter < ADABOOST_CASCADES-1) {
				ii = 0;
				for (int i = 0; i < meshes.size(); i++) {
					features = exportSCEdgeFeatures( meshes[i], features, 3*ii, NUM_EDGE_FEATURES, A.PH, DEBUG_FEATURES );
					ii += meshes[i]->faces.size();
				}
			}

			A.test(featuresCV->FEATURES, featuresCV->LABELS, featuresCV->WEIGHTS, 3*numCVFaces);
			writeEDGESNLOGW(CVEDGESNLOGW_FILE, A, CVmeshes);
			if (em_iter < ADABOOST_CASCADES-1) {
				ii = 0;
				for (int i = 0; i < CVmeshes.size(); i++) {
					featuresCV = exportSCEdgeFeatures( CVmeshes[i], featuresCV, 3*ii, NUM_EDGE_FEATURES, A.PH, DEBUG_FEATURES );
					ii += CVmeshes[i]->faces.size();
				}
			}
		} else {
			JointBoost A;
			if ( A.readLearnedParameters( fin_pwparmsedges ) == true) {
				std::cout << "Read JointBoost" << em_iter << " edge parameters." << std::endl;
				A.test(features->FEATURES, features->LABELS, features->WEIGHTS, 3*numFaces);
				writeEDGESNLOGW(EDGESNLOGW_FILE, A, meshes);
      } else {
        break;
      }
			if (em_iter < ADABOOST_CASCADES-1) {
				ii = 0;
				for (int i = 0; i < meshes.size(); i++) {
					features = exportSCEdgeFeatures( meshes[i], features, 3*ii, NUM_EDGE_FEATURES, A.PH, DEBUG_FEATURES );
					ii += meshes[i]->faces.size();
				}
			}
		}
	}
	delete features;
	if (isCV) {
		delete featuresCV;
	}


	// FINALIZATION
	if (isCV) {
		fout_pwparms.close();
		fout_pwparmsedges.close();
	} else {
		fin_pwparms.close();
		fin_pwparmsedges.close();
	}
	for (int i = 0; i < meshes.size(); i++) {
		delete meshes[i];
		delete[] labels[i];
	}
	for (int i = 0; i < CVmeshes.size(); i++) {
		delete CVmeshes[i];
		delete[] CVlabels[i];
	}

	return 0;
}


/////////////////////////////////////////////////////
///////////////// READ LABELS   /////////////////////
/////////////////////////////////////////////////////
int* readLabels( char *filename, int numfaces ) {
  if ( (strlen(filename) == 0) || ( strcmp(filename, "*") == 0) ) {
		int* labels = new int[ numfaces ];
		for (int i = 0; i < numfaces; i++) {
			labels[i] = 0;
		}
		return labels;
	}

	std::ifstream fin(filename);
	if ( !fin.good() ) {
		return NULL;
	}
	int* labels = new int[ numfaces ];
	for (int i = 0; i < numfaces; i++) {
		if ( !fin.good() ) {
			delete[] labels;
			return NULL;
		}
		fin >> labels[i];
	}

	return labels;
}



/////////////////////////////////////////////////////
///////////////// WRITE NLOGW DATA   /////////////////////
/////////////////////////////////////////////////////
void writeNLOGW( const char* nlogw_filename, JointBoost& A, vector<TriMesh*>& meshes, int numlabels  ) {
	std::ofstream fout(nlogw_filename); 
	fout.precision(10); 
	fout.setf( std::ios::scientific );
	int ii = 0; 

	for (int i = 0; i < meshes.size(); i++) {
		for (int j = 0; j < meshes[i]->faces.size(); j++) {
			fout << i << ' ' << j;
			for (int k = 0; k < numlabels; k++) {
				fout << ' ' << -log( A.PH[ii][k] );
			}
			ii++;
			fout << std::endl;
		}
	}

	fout.close();
}


void writeEDGESNLOGW( const char* nlogw_filename, JointBoost& A, vector<TriMesh*>& meshes  ) {
	std::ofstream fout(nlogw_filename); 
	fout.precision(10); 
	fout.setf( std::ios::scientific );

	int ii = 0; 

	for (int i = 0; i < meshes.size(); i++) {
		for (int j = 0; j < meshes[i]->faces.size(); j++) {
			for (int k = 0; k < 3; k++) {
				fout << i << ' ' << j << ' ' << meshes[i]->across_edge[j][k];
				fout << ' ' << -log( A.PH[ii][0] ) << ' ' << -log( A.PH[ii][1] );
				ii++;
				fout << std::endl;
			}
		}
	}

	fout.close();
}



/////////////////////////////////////////////////////
///////// create mxArray for best Features   ///////////////
/////////////////////////////////////////////////////
#ifdef USE_MATLAB
mxArray* createMxArrayBestFeatures(float** featuresALL, int numFaces, std::vector<int> bestFeatures  ) {
	mxArray *mxfeaturesBest = mxCreateDoubleMatrix(bestFeatures.size(), numFaces, mxREAL );
	double* mxfeaturesBestData = mxGetPr( mxfeaturesBest );
	for (int j = 0; j < bestFeatures.size(); j++) {
		for (int i = 0; i < numFaces; i++) {
			mxfeaturesBestData[ i*bestFeatures.size() + j ] = featuresALL[ bestFeatures[j] ][i];
		}
	}
	return mxfeaturesBest;
}
#endif
