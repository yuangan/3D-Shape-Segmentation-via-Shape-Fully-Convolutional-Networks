// This file is part of "LearningMeshSegmentation".

// Copyright (c) 2010 Evangelos Kalogerakis
// All rights reserved. 
// If you use this code or its parts, please also cite 
// "Learning 3D Mesh Segmentation and Labeling, 
// E. Kalogerakis, A. Hertzmann, K. Singh, ACM Transactions on Graphics, 
// Vol. 29, No. 3, July 2010"
// AND (Jointboost was introduced by)
// Antonio Torralba , Kevin P. Murphy , William T. Freeman, Sharing Visual Features 
// for Multiclass and Multiview Object Detection, IEEE Transactions on Pattern Analysis 
// and Machine Intelligence, v.29 n.5, p.854-869, May 2007

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
#include "MeshSegmentationFeatures.h"

class SharedStump {
public:
	float a;    // parameter a
	float b;	// parameter b
	float *k;   // parameters k
	float th;// parameter theta
	unsigned int thPos;
	int  f;		// selected feature
	bool *S;
	float err;
	int numClasses;
	bool consistent;

	SharedStump() {}
	SharedStump(int _numClasses): numClasses(_numClasses) { 
		S = new bool[ numClasses ];
		k = new float[ numClasses ];
		consistent = false;
	}
	SharedStump(int _numClasses, bool *_S, int _f): numClasses(_numClasses) { 
		S = new bool[ numClasses ];
		k = new float[ numClasses ];
		for (int c = 0; c < numClasses; c++) {
			S[c] = _S[c];
			k[c] = 0.0f;
		}
		f = _f;
		consistent = false;
	}
	SharedStump( SharedStump* stump ) {
		S = new bool[ stump->numClasses ];
		k = new float[ stump->numClasses ];
		for (int c = 0; c < stump->numClasses; c++) {
			S[c] = stump->S[c];
			k[c] = stump->k[c];
		}
		a = stump->a;
		b = stump->b;
		f = stump->f;
		th = stump->th;
		thPos = stump->thPos;
		err = stump->err;
		numClasses = stump->numClasses;
		consistent = stump->consistent;
	}
	~SharedStump() {
		delete[] S;
		delete[] k;
	}

	inline void update(bool *_S, int _f, float *_k) {
		for (int c = 0; c < numClasses; c++) {
			S[c] = _S[c];
			k[c] = _k[c];
		}
		f = _f;
		consistent = false;
	}

	float getResponse(float* X, int c) {
		if (S[c] == true) {
			if (X[f] > th) {
				return a;
			} else {
				return b;
			}
		}
		return k[c];
	}
};

class JointBoost {
private:
	bool trainMode; 
	bool testMode; 
	int numClasses;
	int numExamples;		// number of training examples (n)
	int numCVExamples;      // number of validation examples (n')
	int numFeatures;		// number of features

	float **W;   // weights of training examples n x numClasses
	float *WI;   // initial weights of training examples n x 1
	float **H;	// jointboost output - strong learner n x numClasses
	float **X;  // input features / training examples d x n
	int   *Y;   // input labels n x 1
	unsigned int  **SI;   // sorting indices d x n
	float ***cumsumYW;
	float ***cumsumW;
	unsigned int cumsumSubSample;
	unsigned int cumsumSize;
	float minCVerror;

	SharedStump** bestStumps;

	float **XC;  // test/validation input features / validation examples d x n'
	int   *YC;   // test/validation input labels n' x 1
	float *WCI;   // initial weights of validation examples n' x 1
	float **HC;	// jointboost output - strong learner n' x numClasses;


	void train( int maxrounds = ADABOOST_TRAINING_MAX_ROUNDS );
	void train2( int maxrounds = ADABOOST_TRAINING_MAX_ROUNDS );
	void train3( int maxrounds = ADABOOST_TRAINING_MAX_ROUNDS );
	void selfDestruct();
	bool* int2bin(bool *, int );
	void fitStump( SharedStump*, float );
	float getError( SharedStump*, float err = FLT_MAX );
	void updateLearner(SharedStump*);
	void updateCVLearner(SharedStump*);
	float getCVError();
	float getTrainingError();
	void predict();
	inline float labelToBin(int y, int l) {
		return ( (y == l)? 1.0f: -1.0f );
	}

public:
	float **PH;	// jointboost sigmoid output n x numClasses
	int   *YP;   // output labels n x 1
	int maxRounds;

	JointBoost(): trainMode(false), testMode(false) {
		bestStumps = NULL;
	}
	JointBoost(float **FEATURES, int* LABELS, float *INITIALWEIGHTS, int numfaces, int numLabels, int numfeatures,  float **CVFEATURES = NULL, int* CVLABELS = NULL, float *INITIALCVWEIGHTS = NULL, int numCVfaces = 0): 
	X(FEATURES), Y(LABELS), WI(INITIALWEIGHTS), numClasses(numLabels), numExamples(numfaces),  numFeatures(numfeatures), XC(CVFEATURES), YC(CVLABELS), WCI(INITIALCVWEIGHTS), numCVExamples( numCVfaces ), trainMode(false), testMode(false) {
		bestStumps = NULL;
		float sumW = 0.0f;
		for (int i = 0; i < numExamples; i++) {
			sumW += WI[i];
		}
		for (int i = 0; i < numExamples; i++) {
			WI[i] /= sumW;
		}
		if (XC != NULL) {
			float sumW = 0.0f;
			for (int i = 0; i < numCVExamples; i++) {
				sumW += WCI[i];
			}
			for (int i = 0; i < numCVExamples; i++) {
				WCI[i] /= sumW;
			}
		}

		int roundLimit = ADABOOST_TRAINING_MAX_ROUNDS;
		//int roundLimit = min( max(2*STOP_AFTER_N_ROUNDS_OF_CVERROR_INCREASING, 2*numClasses*numClasses), ADABOOST_TRAINING_MAX_ROUNDS );
		if (numLabels <= 8)
			train(roundLimit);
		else if (numLabels > 8 ) // && numLabels <= 25)
			train2(roundLimit);
//		else if (numLabels > 25)
//			train3(roundLimit);
	}; 

	~JointBoost() {
		selfDestruct();
		if (bestStumps != NULL) {
			//for (int i = 0; i < maxRounds; i++) {  
			//		delete bestStumps[i];
			//}
			delete[] bestStumps;
		}
	};

	float getminCVError() {
		return minCVerror;
	}

	void writeLearnedParameters(std::ofstream& fout) {
		fout.precision(10);
		fout.setf( std::ios::scientific );
		fout << maxRounds << ' ' << numFeatures << ' ' << numClasses << ' ';
		for (int i = 0; i < 1 + 2*numClasses; i++) {
			fout << 0 << ' ';
		}
		fout << std::endl;
		for (int round=1; round <= maxRounds; round++) {
			fout << bestStumps[round-1]->f << ' ';
			fout << bestStumps[round-1]->a << ' ';
			fout << bestStumps[round-1]->b << ' ';
			fout << bestStumps[round-1]->th << ' ';
			for (int c = 0; c < numClasses; c++) {
				fout << bestStumps[round-1]->k[c] << ' ';
			}
			for (int c = 0; c < numClasses; c++) {
				fout << bestStumps[round-1]->S[c] << ' ';
			}
			fout << std::endl;
		}
	}

	bool readLearnedParameters(std::ifstream& fin) {
		float tmp;
		fin >> tmp;
		if ( !fin.good() ) {
			return false;
		}
		maxRounds = (int)tmp;
		fin >> tmp;
		if ( !fin.good() ) {
			return false;
		}
		numFeatures = (int)tmp;
		fin >> tmp;
		if ( !fin.good() ) {
			return false;
		}
		numClasses = (int)tmp;
		for (int i = 0; i < 1 + 2*numClasses; i++) {
			fin >> tmp;
		}
		if ( !fin.good() ) {
			return false;
		}
		bestStumps = new SharedStump*[ maxRounds ];

		for (int round=1; round <= maxRounds; round++) {
			bestStumps[round-1] = new SharedStump( numClasses );
			fin >> tmp;
			bestStumps[round-1]->f = (int)tmp;
			fin >> bestStumps[round-1]->a;
			fin >> bestStumps[round-1]->b;
			fin >> bestStumps[round-1]->th;
			for (int c = 0; c < numClasses; c++) {
				fin >> bestStumps[round-1]->k[c];
			}
			for (int c = 0; c < numClasses; c++) {
				fin >> tmp;
				bestStumps[round-1]->S[c] = (bool)tmp;
			}
		}
		return true;
	};

	std::vector<int> getBestFeatures() {
		std::vector<int> bestFeatures;
		for (int round=1; round <= maxRounds; round++) {
			bestFeatures.push_back( bestStumps[round-1]->f );
		}
		return bestFeatures;
	}

	float test( float **FEATURES, int *GTLABELS, float *INITIALWEIGHTS, int numfaces );


	static bool selfTest() {
		std::cout << "Running jointboost self-test..." << std::endl;
		float** X;
		int*   Y;
		float* W;
		X = new float*[2];
		X[0] = new float[1000];
		X[1] = new float[1000];
		Y = new int[1000];
		W = new float[1000];

		for (int i = 0; i < 1000; i++) {
			X[0][i] = 2.0f * (i >= 0 & i < 300) + 4.0f * (i >= 300 & i < 650) + 6.0f * (i >= 650) + 2.0f * ( (float)rand() / (float)RAND_MAX );
			X[1][i] = ( (float)rand() / (float)RAND_MAX );
			Y[i] = 1 * (i >= 0 & i < 300) + 2 * (i >= 300 & i < 650) + 3 * (i >= 650);
			W[i] = 1.0f;
		}

		bool success = true;
		JointBoost A(X, Y, W, 1000, 3, 2, X, Y, W, 1000);
		for (int i = 0; i < 1000; i++) {
			if (Y[i] != A.YP[i]) {
				success = false;
			}
//			std::cout << Y[i] <<  ' ' << A.PH[i][0] << ' ' << A.PH[i][1] << ' ' << A.PH[i][2] << std::endl;;
		}
		if ( A.test( X, Y, W, 1000) > 0 )
			success = false;
		std::ofstream fout("___jointboost___tmp.txt");
		A.writeLearnedParameters(fout);
		fout.close();
		std::ifstream fin("___jointboost___tmp.txt");
		A.readLearnedParameters(fin);
		fin.close();
		if ( A.test( X, Y, W, 1000) > 0 )
			success = false;
		remove ("___jointboost___tmp.txt");

		delete[] X[0]; delete[] X[1]; delete[] X; delete[] Y; delete[] W;

		if (success) {
			std::cout << "Jointboost self-test passed... Proceeding with mesh processing." << std::endl;
		} else {
			std::cout << "Jointboost self-test failed... Internal Error - Exiting..." << std::endl;
		}
		return success;
	}


	static bool selfTest2() {
		std::cout << "Running Jointboost self-test 2..." << std::endl;
		float** X;
		int*   Y;
		float* W;
		float** XC;
		int*   YC;
		float* WC;

		X = new float*[64];
		Y = new int[20000];
		W = new float[20000];
		for (int i = 0; i < 64; i++) {
			X[i] = new float[20000];
		}
		XC = new float*[64];
		YC = new int[66691];
		WC = new float[66691];
		for (int i = 0; i < 64; i++) {
			XC[i] = new float[66691];
		}

		std::ifstream f1("data\\set1_toyCAD\\train\\rocker-arm_curvatureFeatures.txt");
		std::ifstream f2("data\\set1_toyCAD\\cv\\flange_curvatureFeatures.txt");
		std::ifstream fl1("data\\set1_toyCAD\\train\\rocker-arm_labelsN.txt");
		std::ifstream fl2("data\\set1_toyCAD\\cv\\flange_labelsN.txt");
		char tmp[1024];
		f1.getline(tmp, 1024);
		f2.getline(tmp, 1024);

		for (int i = 0; i < 20000; i++) {
			fl1 >> Y[i]; 
			W[i] = 1;
			for (int j = 0; j < 64; j++) {
				f1 >> X[j][i]; 
			}
		}
		for (int i = 0; i < 66691; i++) {
			fl2 >> YC[i];
			WC[i] = 1;
			for (int j = 0; j < 64; j++) {
				f2 >> XC[j][i]; 
			}
		}

		//std::cout << X[0][0] << ' ' << X[63][0] << ' ' << X[0][3] << ' ' << X[63][3] << ' ' << X[1][19999] << ' ' << X[63][19999] << std::endl;
		//std::cout << Y[0] << ' ' << Y[10] << ' ' << Y[19999] << std::endl;
		//std::cout << XC[0][0] << ' ' << XC[63][0] << ' ' << XC[0][3] << ' ' << XC[63][3] << ' ' << XC[1][66690] << ' ' << XC[63][66690] << std::endl;
		//std::cout << YC[0] << ' ' << YC[10] << ' ' << YC[66690] << std::endl;
		//system("pause");

		float err = 0;
		bool success = false;
		JointBoost A(X, Y, W, 20000, 2, 64, XC, YC, WC, 66691);
		for (int i = 0; i < 20000; i++) {
			err += ( Y[i] != A.YP[i] );
		}
		success = ( (err / 20000.0f) < .1);		

		std::ofstream fout("___jointboost___tmp.txt");
		A.writeLearnedParameters(fout);
		fout.close();
		std::cout << err << ' ' << A.test( X, Y, W, 20000) << std::endl;

		f1.close(); f2.close(); fl1.close(); fl2.close();
		if (success) {
			std::cout << "Jointboost self-test passed... Proceeding with mesh processing." << std::endl;
		} else {
			std::cout << "Jointboost self-test failed... Internal Error - Exiting..." << std::endl;
		}
		system("pause");
		exit(0);
		return success;
	}

};


