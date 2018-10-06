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

#include "TriMesh.h"
#include "TriMesh_algo.h"
#include "DijkstraForGeodesics.h"
#include "MeshSegmentationFeatures.h"
#include "waPCA.h"

void writeDebugInfoToFile( TriMesh *m, const char *featureType, FeatureSet* features, int numFaces, int pos, int fid = 1  );
#ifdef USE_MATLAB
extern meshLaplacianReturnData meshLaplacian(TriMesh*, Engine*, float, int, float**, int**, int);
#endif

void writeDebugInfoToFile( TriMesh *m, const char *featureType, FeatureSet* features, int numFaces, int pos, int fid ) {
	char featureOutputfilename[1024];
	strncpy(featureOutputfilename, m->filename, strlen(m->filename)-4);
	featureOutputfilename[ strlen(m->filename)-4 ] = '\0'; 
	strcat(featureOutputfilename, featureType);
	std::ofstream fout( featureOutputfilename );
	if( !fout.good() ) {
		std::cerr << "Failed to open file " << featureOutputfilename << std::endl;
	} else {
		fout.precision(10);
		fout.setf( std::ios::scientific );
		if (fid == 1) { 
			fout << features->numFeatures << ' ' << numFaces << ' ' << pos << std::endl;
			for( int i = 0; i < numFaces; i++ ) {
				for( int j = 0; j < features->numFeatures; j++ ) {
					fout << features->FEATURES[j][i+pos] << ' ';
				}
				fout << std::endl;
			}
		} else {
			fout << features->numFeatures2 << ' ' << numFaces << ' ' << pos << std::endl;
			for( int i = 0; i < numFaces; i++ ) {
				for( int j = 0; j < features->numFeatures2; j++ ) {
					fout << features->FEATURES2[j][i+pos] << ' ';
				}
				fout << std::endl;
			}
		}
		fout.close();
	}
}


TriMesh* processMesh( TriMesh *m, bool writeDebugInfo ) {
	m->need_neighbors();
	m->need_across_edge();
	m->need_adjacentfaces();

	// EXTRACTING GENERIC MESH FEATURES
	std::cout << std::endl << "Getting generic mesh information (max geodesic distance, mass center, face normals)" << std::endl;	
	std::cout.flush();
	if (USE_NORMALIZATION_FEATURE == MAX_GEOD_DISTANCE) {
		m->geodesicDistance = GeodesicTraversal(*m).getMaxGeodesicDistance( (int)ceil( (float)m->vertices.size() / SAMPLING_RATE_FOR_MESH_NORMALIZATION ) );
	} else if (USE_NORMALIZATION_FEATURE == MEAN_MAX_GEOD_DISTANCE) {
		m->geodesicDistance  = GeodesicTraversal(*m).getMeanMaxGeodesicDistance( (int)ceil( (float)m->vertices.size() / SAMPLING_RATE_FOR_MESH_NORMALIZATION ) );
	} else if (USE_NORMALIZATION_FEATURE == MEAN_MEAN_GEOD_DISTANCE) {
		m->geodesicDistance  = GeodesicTraversal(*m).getMeanGeodesicDistance( (int)ceil( (float)m->vertices.size() / SAMPLING_RATE_FOR_MESH_NORMALIZATION ) );
	} else if (USE_NORMALIZATION_FEATURE == MEDIAN_MEDIAN_GEOD_DISTANCE) {
		m->geodesicDistance  = GeodesicTraversal(*m).getMedianGeodesicDistance( (int)ceil( (float)m->vertices.size() / SAMPLING_RATE_FOR_MESH_NORMALIZATION ) );
	} else if (USE_NORMALIZATION_FEATURE == PERCENTILE_GEOD_DISTANCE) {
		m->geodesicDistance  = GeodesicTraversal(*m).getPercentileGeodesicDistance( (int)ceil( (float)m->vertices.size() / SAMPLING_RATE_FOR_MESH_NORMALIZATION ), .33f, true );
	} else if (USE_NORMALIZATION_FEATURE == BSPHERE_RADIUS) {
		m->need_bsphere();
		m->geodesicDistance  = m->bsphere.r;
	} else if (USE_NORMALIZATION_FEATURE == BBOX_DIAGONAL) {
		m->need_bbox();
		m->geodesicDistance  = len(m->bbox.max - m->bbox.min);
  } else if (USE_NORMALIZATION_FEATURE == MEDIAN_DISTANCE || USE_NORMALIZATION_FEATURE == PERCENTILE_DISTANCE) {
    point massCenter(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < m->faces.size(); i++) {
      massCenter = massCenter + m->faces[i].faceCenter * m->faces[i].faceArea;
    }
    massCenter /= m->totalFaceArea;

    std::vector<float> medianEuclideanDistances;
    medianEuclideanDistances.reserve(m->vertices.size());
    for (int i = 0; i < m->faces.size(); i++) {
      medianEuclideanDistances.push_back(len(m->faces[i].faceCenter - massCenter));
    }
    if (USE_NORMALIZATION_FEATURE == PERCENTILE_DISTANCE)
      m->geodesicDistance = percentile(medianEuclideanDistances, .33f) * (0.5f / .33f);
    else
      m->geodesicDistance = percentile(medianEuclideanDistances, .5f);
  }


	point massCenter(0.0f, 0.0f, 0.0f);
	for( int i = 0; i < m->faces.size(); i++ ) {
		massCenter = massCenter + m->faces[i].faceCenter * m->faces[i].faceArea;
	}
	massCenter /= m->totalFaceArea;
	for( int i = 0; i < m->vertices.size(); i++ ) {
		m->vertices[i] = (m->vertices[i] - massCenter) / m->geodesicDistance;
	}
	m->normals.clear(); m->need_normals();
	m->need_curvatures();
	m->need_dcurv();
	m->bsphere.valid = false; m->need_bsphere();
	m->bbox.valid = false; m->need_bbox();
	m->need_pointareas();

	for( int i = 0; i < m->vertices.size(); i++ ) {
		if ((m->pointareas[i] != m->pointareas[i]) || (m->pointareas[i] - m->pointareas[i] != m->pointareas[i] - m->pointareas[i])) {
			m->pointareas[i] = 0.0f;
			std::cout << "corrected" << std::endl;
		}
	}

	std::cout << "**********bounding sphere radius =" << m->bsphere.r << " ************" << std::endl;
	m->across_edge.clear(); m->dist_faces_across_edge.clear(); m->need_across_edge();
	m->geodesicDistance  = GeodesicTraversal(*m).getPercentileGeodesicDistance( (int)ceil( (float)m->vertices.size() / SAMPLING_RATE_FOR_MESH_NORMALIZATION ), .95f );


	if (writeDebugInfo) {
//		m->write( strcat(m->filename, ".off") );
		char featureOutputfilename[1024];
		strncpy(featureOutputfilename, m->filename, strlen(m->filename)-4);
		featureOutputfilename[ strlen(m->filename)-4 ] = '\0'; 
		strcat(featureOutputfilename,"_genericFeatures.txt");
		std::ofstream fout( featureOutputfilename );
		if( !fout.good() ) {
			std::cerr << "Failed to open file " << featureOutputfilename << std::endl;
			return m;
		}
		fout.precision(10);
		fout << m->geodesicDistance << ' ' << 0.0f << ' ' << 0.0f << std::endl;
		fout << massCenter[0] << ' ' << massCenter[1] << ' ' << massCenter[2] << std::endl;
		for( int i = 0; i < m->faces.size(); i++ ) {
			fout << m->faces[i].facenormal[0] << ' ' << m->faces[i].facenormal[1] << ' ' << m->faces[i].facenormal[2] << std::endl;
		}
		fout.close();
	}

	return m;
}




FeatureSet* importFeatures(TriMesh* m, FeatureSet* features, int pos, int numFaces, bool writeDebugInfo, bool returnNumFeaturesOnly) {
  if (pos == 0) {
    features = new FeatureSet();
    std::string features_filename = std::string(m->filename).substr(0, strlen(m->filename) - 4) + "_features.txt";
    std::cout << "Attempting to import: " << features_filename << std::endl;
    std::ifstream features_file(features_filename.c_str());

    if (!features_file.good())
    {
      std::cout << "No imported features will be used." << std::endl;
      features->numFeatures = 0;
      return features;
    }
    std::string line;    
    getline(features_file, line);
    std::istringstream is(line);
    std::vector< float > first_line_floats = std::vector< float >(std::istream_iterator<float>(is), std::istream_iterator<float>());
    features_file.close();    
    features->numFeatures = first_line_floats.size();
    if (returnNumFeaturesOnly) {
      return features;
    }

    features->FEATURES = new float*[features->numFeatures];
    for (int i = 0; i < features->numFeatures; i++) {
      features->FEATURES[i] = new float[numFaces];
    }
  }

  std::string features_filename = std::string(m->filename).substr(0, strlen(m->filename) - 4) + "_features.txt";
  std::ifstream features_file(features_filename.c_str());
  if (!features_file.good())
  {
    if (pos != 0)
      std::cout << "No imported features will be used." << std::endl;
    return features;
  }
  std::string line;
  std::vector< std::vector<float> > all_floats;
  while (getline(features_file, line)) {
    std::istringstream is(line);
    all_floats.push_back(std::vector<float>(std::istream_iterator<float>(is), std::istream_iterator<float>()));
  }

  std::cout << std::endl << "Imported features, #num faces from mesh = " << m->faces.size() << ", #num faces from imported file = " << all_floats.size() << ", #num features = " << features->numFeatures << std::endl;

  for (int i = 0; i < m->faces.size(); i++)
    for (int j = 0; j < features->numFeatures; j++)
      features->FEATURES[j][i + pos] = all_floats[i][j];

  if (writeDebugInfo)
    writeDebugInfoToFile(m, "_importedFeatures.txt", features, m->faces.size(), pos);

  return features;
}




FeatureSet* exportCurvatureFeatures(TriMesh* m, FeatureSet* features, int pos, int numFaces, bool writeDebugInfo, bool returnNumFeaturesOnly ) {
	if (pos == 0) {
		features = new FeatureSet(); 
		features->numFeatures = NUM_CURVATURE_FEATURES_PER_SCALE * NUM_SCALES;
		if (returnNumFeaturesOnly) {
			return features;
		}
		features->FEATURES = new float*[ features->numFeatures ];
		for( int i = 0; i < features->numFeatures; i++ ) {
			features->FEATURES[i] = new float[ numFaces ];
		}		
	} 
	GeodesicTraversal MeshTraversal(*m);
	std::vector< std::vector<float> > curvatureData( NUM_CURVATURE_FEATURES_PER_SCALE );
	for (int j = 0; j < NUM_CURVATURE_FEATURES_PER_SCALE; j++) {
		curvatureData[j].resize( m->faces.size() );
	}

	for (int scale = 0; scale < NUM_SCALES; scale++) {
		std::cout << std::endl << "Computing curvatures at scale " << scale << " (" << SCALES[scale]/2.0f << ") for mesh " << m->filename << std::endl;

		for( int i = 0; i < m->faces.size(); i++ ) {
			if (i % PRINT_EVERY_N == 0) {
				std::cout << 100.0f * (float)i / (float)m->faces.size() << "% complete\t\t\t\r";
				std::cout.flush();
			}
			vector<int> adjacentFaces = MeshTraversal.traverseFaces(i, SCALES[scale]/2.0f, 3, 0.0 );
			vec n = m->faces[i].facenormal;
			vec t = _normalize( ( m->vertices[ m->faces[i][0] ] - m->vertices[ m->faces[i][1] ] ) CROSS n );
			vec s = t CROSS n;
			double C[9] = {t[0], t[1], t[2], s[0], s[1], s[2], n[0], n[1], n[2]};

			char NORM = '1';
			integer M = (adjacentFaces.size()-1)*3;  // number of rows (data points+normals)
			integer N = 3; // number of unknowns, change to 7 for cubic fit and uncomment the corresponding lines
			integer NRHS = 1;
			integer JPVT[3] = {0, 0, 0};
			integer RANK;
			integer LWORK = 3 * M * N;
			double *WORK = new double[ LWORK ];
			double *B = new double[ M ];
			double *A = new double[ M*N ];
			integer INFO;

			for( int j = 1; j < adjacentFaces.size(); j++ ) { // adjacent face 0 is the source face i
				vec diff = m->faces[i].faceCenter - m->faces[adjacentFaces[j]].faceCenter;
				double x = C[0] * diff[0] + C[1] * diff[1] + C[2] * diff[2];
				double y = C[3] * diff[0] + C[4] * diff[1] + C[5] * diff[2];
				double z = C[6] * diff[0] + C[7] * diff[1] + C[8] * diff[2];
				double norm = m->faces[adjacentFaces[j]].faceArea; // (sqrt(x*x + y*y) + EPSILON);
				double xn = C[0] * m->faces[adjacentFaces[j]].facenormal[0] + C[1] * m->faces[adjacentFaces[j]].facenormal[1] + C[2] * m->faces[adjacentFaces[j]].facenormal[2];
				double yn = C[3] * m->faces[adjacentFaces[j]].facenormal[0] + C[4] * m->faces[adjacentFaces[j]].facenormal[1] + C[5] * m->faces[adjacentFaces[j]].facenormal[2];
				double zn = C[6] * m->faces[adjacentFaces[j]].facenormal[0] + C[7] * m->faces[adjacentFaces[j]].facenormal[1] + C[8] * m->faces[adjacentFaces[j]].facenormal[2];
				double u = -xn / (zn+EPSILON);
				double v = -yn / (zn+EPSILON);
				//std::cout << std::endl << x << ' ' << y << ' ' << z << std::endl;
				//std::cout << std::endl << adjacentFaces[j] << std::endl;

				A[      3*(j - 1)   ] = 0.5f * x*x * norm;
				A[  M + 3*(j - 1)   ] =        x*y * norm;
				A[2*M + 3*(j - 1)   ] = 0.5f * y*y * norm;
				//A[3*M + 3*(j - 1)   ] =      x*x*x * norm;
				//A[4*M + 3*(j - 1)   ] =		 x*x*y * norm;
				//A[5*M + 3*(j - 1)   ] =      x*y*y * norm;
				//A[6*M + 3*(j - 1)   ] =      y*y*y * norm;
				B[      3*(j - 1)	] = z * norm;

				A[      3*(j - 1)+1 ] =          x * norm;
				A[  M + 3*(j - 1)+1 ] =          y * norm;
				A[2*M + 3*(j - 1)+1 ] = 0;
				//A[3*M + 3*(j - 1)+1 ] = 3.0f * x*x * norm;
				//A[4*M + 3*(j - 1)+1 ] =	2.0f * x*y * norm;
				//A[5*M + 3*(j - 1)+1 ] =        y*y * norm;
				//A[6*M + 3*(j - 1)+1 ] = 0;
				B[      3*(j - 1)+1 ] = u * norm;

				A[      3*(j - 1)+2 ] = 0;
				A[  M + 3*(j - 1)+2 ] =          x * norm;
				A[2*M + 3*(j - 1)+2 ] =          y * norm;
				//A[3*M + 3*(j - 1)+2 ] = 0;
				//A[4*M + 3*(j - 1)+2 ] =	       x*x * norm;
				//A[5*M + 3*(j - 1)+2 ] = 2.0f * x*y * norm;
				//A[6*M + 3*(j - 1)+2 ] = 3.0f * y*y * norm;
				B[      3*(j - 1)+2 ] = v * norm;
			}

			double RCOND = dlange_( &NORM, &M, &N, A, &M, WORK );
			dgelsy_( &M, &N, &NRHS, A, &M, B, &M, JPVT, &RCOND, &RANK, WORK, &LWORK, &INFO );
			if (INFO != 0) {
				std::cerr << std::endl << "Face " << i << "possibly did not acquire right curvature values during dgelsy_ call! INFO = " << INFO << std::endl;
			}

			double trace = B[0] + B[2];
			double det = B[0] * B[2] - B[1] * B[1];
			double k1 = .5f * ( trace + sqrt( trace*trace - 4.0f * det ) );
			double k2 = .5f * ( trace - sqrt( trace*trace - 4.0f * det ) );

			delete[] A;
			delete[] B;
			delete[] WORK;

			curvatureData[0][i] = k1;
			curvatureData[1][i] = k2;
			curvatureData[2][i] = k1-k2;
			if ( fabs(k1) < fabs(k2) ) {
				std::swap( k1, k2 );
			}
			curvatureData[3][i] = k1;
			curvatureData[4][i] = k2;
			curvatureData[5][i] = k1-k2;
			curvatureData[6][i] = (k1 + k2) / 2;
			curvatureData[7][i] = k1 * k2;
			//curvatureData[8][i] = k2 / (k1 + EPSILON);
			//curvatureData[9][i] = k2 / (k1 + k2 + EPSILON);
			for (int j = 0; j < 8; j++) {			// don't forget to change this as well, if you change the features
				curvatureData[8+j][i] = fabs( curvatureData[j][i] );
			}
		}

		for (int j = 0; j < NUM_CURVATURE_FEATURES_PER_SCALE; j++) {
			if (j <= 7) {
				curvatureData[j] = removeOutliers( curvatureData[j], .95f );
			} else {
				curvatureData[j] = removePositiveOutliers( curvatureData[j], .95f );
			}
			//float meancurv = mean(curvatureData[j]);
			//float stdcurv = sqrt( variance( curvatureData[j], meancurv ) );
			for( int i = 0; i < m->faces.size(); i++ ) {
				//curvatureData[j][i] /= (stdcurv+EPSILON);
				features->FEATURES[NUM_CURVATURE_FEATURES_PER_SCALE*scale + j][i+pos] = (curvatureData[j][i] + curvatureData[j][m->across_edge[i][0]] + curvatureData[j][m->across_edge[i][1]] + curvatureData[j][m->across_edge[i][2]])/4.0f;
			}
		}

		std::cout << 100.0f << "% complete\t\t\t\r";
	}


	if (writeDebugInfo) 
		writeDebugInfoToFile( m, "_curvatureFeatures.txt", features, m->faces.size(), pos  );
	
	for (int j = 0; j < NUM_CURVATURE_FEATURES_PER_SCALE; j++) {
		curvatureData[j].clear();
	}
	curvatureData.clear();

	return features;
}


FeatureSet* exportPCAFeatures(TriMesh* m,  FeatureSet* features, int pos, int numFaces, bool writeDebugInfo, bool returnNumFeaturesOnly ) {
	if (pos == 0) {
		features = new FeatureSet(); 
		features->numFeatures = NUM_PCA_FEATURES_PER_SCALE * NUM_SCALES;
		if (returnNumFeaturesOnly) {
			return features;
		}
		features->FEATURES = new float*[ features->numFeatures ];
		for( int i = 0; i < features->numFeatures; i++ ) {
			features->FEATURES[i] = new float[ numFaces ];
		}		
	} 
	GeodesicTraversal MeshTraversal(*m);
	char JOBZ = 'N';
	char UPLO = 'L';
	integer N = 3; 
	double EIG[3]; // eigenvalues in ascending order
	double WORK[5 * 8];
	integer LWORK = 5 * 8;
	integer INFO;

	vector< vector<float> > pcaData(NUM_PCA_FEATURES_PER_SCALE);
	for (int j = 0; j < NUM_PCA_FEATURES_PER_SCALE; j++) {
		pcaData[j].resize( m->faces.size() );
	}

	for (int scale = 0; scale < NUM_SCALES; scale++) {
		std::cout << std::endl << "Computing pca eigenvalues at scale " << scale << " (" << SCALES[scale] * 3.0f << ") for mesh " << m->filename << std::endl;

		for( int i = 0; i < m->faces.size(); i++ ) {
			if (i % PRINT_EVERY_N == 0) {
				std::cout << 100.0f * (float)i / (float)m->faces.size() << "% complete\t\t\t\r";
				std::cout.flush();
			}
			vector<int> adjacentFaces = MeshTraversal.traverseFaces(i, SCALES[scale] * 3.0f, 9 );
			double *V = new double[ adjacentFaces.size() * 3 ];
			float M[3] = {0.0f, 0.0f, 0.0f};
			for (int j = 0; j < adjacentFaces.size(); j++) {
				float faceArea = m->faces[adjacentFaces[j]].faceArea;
				float x = m->faces[adjacentFaces[j]].faceCenter[0] * faceArea;
				float y = m->faces[adjacentFaces[j]].faceCenter[1] * faceArea;
				float z = m->faces[adjacentFaces[j]].faceCenter[2] * faceArea;
				M[ 0 ] += x;
				M[ 1 ] += y;
				M[ 2 ] += z;
			}
			M[0] /= m->totalFaceArea;
			M[1] /= m->totalFaceArea;
			M[2] /= m->totalFaceArea;

			for (int j = 0; j < adjacentFaces.size(); j++) {
				float faceArea = sqrt( m->faces[adjacentFaces[j]].faceArea );
				float x = (m->faces[adjacentFaces[j]].faceCenter[0] - M[0]) * faceArea;
				float y = (m->faces[adjacentFaces[j]].faceCenter[1] - M[1]) * faceArea;
				float z = (m->faces[adjacentFaces[j]].faceCenter[2] - M[2]) * faceArea;
				V[ j * 3 ] = x;
				V[ 1 + j * 3 ] = y;
				V[ 2 + j * 3 ] = z;
			}

			double* COV = covar(V, adjacentFaces.size());
			for (int j = 0; j < 9; j++) {
				COV[j] /= m->totalFaceArea;
			}
			dsyev_( &JOBZ, &UPLO, &N, COV, &N, EIG, WORK, &LWORK, &INFO );
			if (INFO != 0) {
				std::cerr << std::endl << "Face " << i << "possibly did not acquire right pca vales during dsyev_ call! INFO = " << INFO << std::endl;
			}
			EIG[0] = sqrt( max<double>(EIG[0], 0.0f) );
			EIG[1] = sqrt( max<double>(EIG[1], 0.0f) );
			EIG[2] = sqrt( max<double>(EIG[2], 0.0f) );
			double sumEIG = 1.0f; //EIG[0] + EIG[1] + EIG[2] + EPSILON;


			pcaData[0][i] = EIG[0] / sumEIG;
			pcaData[1][i] = EIG[1] / sumEIG;
			pcaData[2][i] = EIG[2] / sumEIG;
			pcaData[3][i] = (EIG[0]+EIG[1]) / sumEIG;
			pcaData[4][i] = (EIG[0]+EIG[2]) / sumEIG;
			pcaData[5][i] = (EIG[1]+EIG[2]) / sumEIG;
//			pcaData[6][i] = EIG[0] + EIG[1] + EIG[2];
			pcaData[6][i] = EIG[0]/(EIG[1]+EPSILON);
			pcaData[7][i] = EIG[0]/(EIG[2]+EPSILON);
			pcaData[8][i] = EIG[1]/(EIG[2]+EPSILON);
			pcaData[9][i] = pcaData[6][i]+pcaData[7][i];
			pcaData[10][i] = pcaData[6][i]+pcaData[8][i];
			pcaData[11][i] = pcaData[7][i]+pcaData[8][i];

			delete[] V;
		}


		for (int j = 0; j < NUM_PCA_FEATURES_PER_SCALE; j++) {
			pcaData[j] = removePositiveOutliers( pcaData[j], .95f );
			//float meanpca = mean(pcaData[j]);
			//float stdpca = sqrt( variance( pcaData[j], meanpca ) );
			for( int i = 0; i < m->faces.size(); i++ ) {
				//pcaData[j][i] /= (stdpca+EPSILON);
				features->FEATURES[NUM_PCA_FEATURES_PER_SCALE*scale + j][i+pos] = pcaData[j][i];
			}
		}

		std::cout << 100.0f << "% complete\t\t\t\r";
	}

	if (writeDebugInfo) 
		writeDebugInfoToFile( m, "_pcaFeatures.txt", features, m->faces.size(), pos  );

	for (int j = 0; j < NUM_PCA_FEATURES_PER_SCALE; j++) {
		pcaData[j].clear();
	}
	pcaData.clear();

	return features;
}


// call this with minf = -90.0f, maxf = 90.0f (degrees) always	
FeatureSet* exportSCFeatures(TriMesh* m, FeatureSet* features, int pos, int numFaces, float minf, float maxf, bool writeDebugInfo, bool returnNumFeaturesOnly) {
	if (pos == 0) {
		features = new FeatureSet(); 
		for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
			for( int x = 0; x < NUM_GD_BINS[j]; x++ ) {
				for( int y = 0; y < NUM_ANGLE_BINS[j]; y++ ) {
					features->numFeatures++;
				}
			}
		}
		features->numFeatures2 = NUM_FEATURES_GD;
		if (returnNumFeaturesOnly) {
			return features;
		}

		features->FEATURES = new float*[ features->numFeatures ];
		for( int i = 0; i < features->numFeatures; i++ ) {
			features->FEATURES[i] = new float[ numFaces ];
		}		

		features->FEATURES2 = new float*[ features->numFeatures2 ];
		for( int i = 0; i < features->numFeatures2; i++ ) {
			features->FEATURES2[i] = new float[ numFaces ];
		}	
	} 

	//// initializations of bins & corresponding features
	Bin2D*** bins = new Bin2D**[ NUM_BINNING_TYPES_GD ];
	for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
		bins[j] = new Bin2D*[ NUM_GD_BINS[j] ];
		for( int x = 0; x < NUM_GD_BINS[j]; x++ ) {
			bins[j][x] = new Bin2D[  NUM_ANGLE_BINS[j] ];
		}
	}

	for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
		float s = (float)NUM_GD_BINS[j]+1;
		float D = GD_MAX[j] * m->geodesicDistance;
		for( int x = 0; x < s-1; x++ ) {
			float startx = 0.0f;
			float endx = 0.0f;
			if (GD_LOGP[j] <= 0.0f) {
				startx = D * (float)x/(float)NUM_GD_BINS[j];
				endx = D * (float)(x+1)/(float)NUM_GD_BINS[j];
			} else {
				startx = pow( (-log((s-x)/s) ), GD_LOGP[j] ) * (D / pow( log(s), GD_LOGP[j]) );
				endx = pow( (-log((s-x-1)/s) ), GD_LOGP[j] ) * (D / pow( log(s), GD_LOGP[j]) ); 
			}
			for( int y = 0; y < NUM_ANGLE_BINS[j]; y++ ) {
				float starty = minf + (maxf-minf) * (float)y/(float)NUM_ANGLE_BINS[j];
				float endy =  minf + (maxf-minf) * ((float)y+1)/(float)NUM_ANGLE_BINS[j];
				bins[j][x][y].x1 = startx;
				bins[j][x][y].x2 = endx;
				bins[j][x][y].y1 = starty;
				bins[j][x][y].y2 = endy;
			}
		}
	}
	//std::cout << m->geodesicDistance << std::endl;
	//for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
	//	for( int x = 0; x < NUM_GD_BINS[j]; x++ ) {
	//		for( int y = 0; y < NUM_ANGLE_BINS; y++ ) {
	//			std::cout << bins[j][x][y];
	//		}
	//	}
	//}
	//system("pause");


	GeodesicTraversal MeshTraversal(*m);
	std::cout << std::endl << "Computing geodesic features and shape context for mesh " << m->filename << std::endl;
	bool* done = new bool[m->faces.size()];
	for( int i = 0; i < m->faces.size(); i++ ) {
		done[i] = false;
	}
	for( int i = 0; i < m->faces.size(); i++ ) {
		if (i % PRINT_EVERY_N == 0) {
			std::cout << 100.0f * (float)i / (float)m->faces.size() << "% complete\t\t\t\r";
			std::cout.flush();
		}

		if (ACCELERATE_N2_COMPUTATIONS) {
			for (int k = 0; k < 3; k++) {
				if ( done[m->across_edge[i][k]] == true ) {
					int ii = 0;
					for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
						for( int x = 0; x < NUM_GD_BINS[j]; x++ ) {
							for( int y = 0; y < NUM_ANGLE_BINS[j]; y++ ) {
								features->FEATURES[ii][i+pos] = features->FEATURES[ii][m->across_edge[i][k]+pos];
								ii++;
							}
						}
					}
					for (int j = 0; j < NUM_FEATURES_GD; j++) {
						features->FEATURES2[j][i+pos] = features->FEATURES2[j][m->across_edge[i][k]+pos];
					}
					done[i] = true;
					break;
				}
			}
			if (done[i] == true) {
				done[i] = false;
				continue;
			}
		}
		done[i] = true;


		for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
			for( int x = 0; x < NUM_GD_BINS[j]; x++ ) {
				for( int y = 0; y < NUM_ANGLE_BINS[j]; y++ ) {
					bins[j][x][y].reset();
				}
			}
		}
		vector<int> visitedFaces = MeshTraversal.traverseFaces(i);
		float *GD = MeshTraversal.getGeodDistances();
		float GEODESIC_FEATURES[ NUM_FEATURES_GD ];
		for (int k = 0; k < m->faces.size(); k++) {
			if (i == k) {
				continue;
			}
			float angle = getAngleFromUnitVectors( _normalize(m->faces[k].faceCenter - m->faces[i].faceCenter), (-m->faces[k].facenormal) ) - 90.0f;
			for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
				for( int x = 0; x < NUM_GD_BINS[j]; x++ ) {
					for( int y = 0; y < NUM_ANGLE_BINS[j]; y++ ) {
						if (bins[j][x][y].contains( GD[k], angle)) {
							bins[j][x][y].increaseValue( (m->faces[k].faceArea / m->totalFaceArea) * (1.0f / ( (GD[k]/m->geodesicDistance) + .1f)) );
//							bins[j][x][y].increaseValue( m->faces[k].faceArea * (1.0f / (GD[k]+.1f)) );
						}
					}
				}
			}
		}

		//for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
		//	for( int x = 0; x < NUM_GD_BINS[j]; x++ ) {
		//		float sumRowValues = EPSILON;
		//		for( int y = 0; y < NUM_ANGLE_BINS; y++ ) {
		//			sumRowValues += bins[j][x][y].value;
		//		}
		//		for( int y = 0; y < NUM_ANGLE_BINS; y++ ) {
		//			bins[j][x][y].value /= (sumRowValues + EPSILON);
		//		}
		//	}
		//}

		for (int j = 0; j < NUM_FEATURES_GD; j++) {
			GEODESIC_FEATURES[j] = -0.0f;
		}
		for (int k = 0; k < m->faces.size(); k++) {
			GEODESIC_FEATURES[0] += GD[k] * (m->faces[k].faceArea/m->totalFaceArea);
			GEODESIC_FEATURES[1] += GD[k] * GD[k] * (m->faces[k].faceArea/m->totalFaceArea);
			GEODESIC_FEATURES[2] += GD[k] * GD[k] * GD[k] * GD[k] * (m->faces[k].faceArea/m->totalFaceArea);
			GEODESIC_FEATURES[3] += pow( GD[k], 8.0f) * (m->faces[k].faceArea/m->totalFaceArea);
			GEODESIC_FEATURES[4] += pow( GD[k], 0.5f) * (m->faces[k].faceArea/m->totalFaceArea);
			GEODESIC_FEATURES[5] += pow( GD[k], 0.25f ) * (m->faces[k].faceArea/m->totalFaceArea);
		}
		for (int k = 0; k < m->faces.size(); k++) {
			GEODESIC_FEATURES[6] += ( GD[k] - GEODESIC_FEATURES[0] ) * ( GD[k] - GEODESIC_FEATURES[0] ) * (m->faces[k].faceArea/m->totalFaceArea);
		}
		for (int k = 0; k < m->faces.size(); k++) {
			float gdkm = GD[k] - GEODESIC_FEATURES[0];
			GEODESIC_FEATURES[7] +=  gdkm*gdkm*gdkm * (m->faces[k].faceArea/m->totalFaceArea);
			GEODESIC_FEATURES[8] +=  gdkm*gdkm*gdkm*gdkm * (m->faces[k].faceArea/m->totalFaceArea);
		}
		GEODESIC_FEATURES[7] /= sqrt(GEODESIC_FEATURES[6]*GEODESIC_FEATURES[6]*GEODESIC_FEATURES[6]+EPSILON);
		GEODESIC_FEATURES[8] /= (GEODESIC_FEATURES[6]*GEODESIC_FEATURES[6]+EPSILON);
		float currentSumFaceAreas = 0.0f;
		for (int k = 0; k < m->faces.size(); k++) {
			currentSumFaceAreas += m->faces[ visitedFaces[k] ].faceArea / m->totalFaceArea;
			if (currentSumFaceAreas >= .1 && GEODESIC_FEATURES[9] == -0.0f) {
				GEODESIC_FEATURES[9] = GD[ visitedFaces[k] ];
			}
			if (currentSumFaceAreas >= .2 && GEODESIC_FEATURES[10] == -0.0f) {
				GEODESIC_FEATURES[10] = GD[ visitedFaces[k] ];
			}
			if (currentSumFaceAreas >= .3 && GEODESIC_FEATURES[11] == -0.0f) {
				GEODESIC_FEATURES[11] = GD[ visitedFaces[k] ];
			}
			if (currentSumFaceAreas >= .4 && GEODESIC_FEATURES[12] == -0.0f) {
				GEODESIC_FEATURES[12] = GD[ visitedFaces[k] ];
			}
			if (currentSumFaceAreas >= .5 && GEODESIC_FEATURES[13] == -0.0f) {
				GEODESIC_FEATURES[13] = GD[ visitedFaces[k] ];
				break;
			}
		}

    //GEODESIC_FEATURES[14] = m->faces[i].faceCenter[1];  // USE ONLY IF DATASET IS UPRIGHT ORIENTED!

		int ii = 0;
		for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
			for( int x = 0; x < NUM_GD_BINS[j]; x++ ) {
				for( int y = 0; y < NUM_ANGLE_BINS[j]; y++ ) {
					features->FEATURES[ii][i+pos] = bins[j][x][y].value;
					ii++;
				}
			}
		}
		for (int j = 0; j < NUM_FEATURES_GD; j++) {
			features->FEATURES2[j][i+pos] = GEODESIC_FEATURES[j];
		}
	}
	std::cout << 100.0f << "% complete\t\t\t\r";

	for (int j = 0; j < NUM_FEATURES_GD; j++) {
		float minValue = FLT_MAX;
		for( int i = 0; i < m->faces.size(); i++ ) {
			if ( features->FEATURES2[j][i+pos] < minValue  )
				minValue = features->FEATURES2[j][i+pos];
		}
		for( int i = 0; i < m->faces.size(); i++ ) {
			features->FEATURES2[j][i+pos] -= minValue;
		}
	}


	if (writeDebugInfo) {
		writeDebugInfoToFile( m, "_scAngleFeatures.txt", features, m->faces.size(), pos, 1  );
		writeDebugInfoToFile( m, "_geodFeatures.txt", features, m->faces.size(), pos, 2 );
	}

	for( int j = 0; j < NUM_BINNING_TYPES_GD; j++ ) {
		for( int x = 0; x < NUM_GD_BINS[j]; x++ ) {
			delete[] bins[j][x];
		}
		delete[] bins[j];
	}
	delete[] bins;
	delete[] done;
	return features;
}



FeatureSet* exportSDFFeatures(TriMesh* m, FeatureSet* features, int pos, int numFaces, bool writeDebugInfo, bool returnNumFeaturesOnly) {
	if (pos == 0) {
		features = new FeatureSet(); 
		features->numFeatures = NUM_SDF_FEATURES_PER_BASE * NUM_SDF_FEATURES_PER_ANGLE * NUM_ANGLES_SDF;
		features->numFeatures2 = NUM_SDF_FEATURES_PER_BASE * NUM_VSI_FEATURES;
		if (returnNumFeaturesOnly) {
			return features;
		}

		features->FEATURES = new float*[ features->numFeatures ];
		for( int i = 0; i < features->numFeatures; i++ ) {
			features->FEATURES[i] = new float[ numFaces ];
		}		

		features->FEATURES2 = new float*[ features->numFeatures2 ];
		for( int i = 0; i < features->numFeatures2; i++ ) {
			features->FEATURES2[i] = new float[ numFaces ];
		}
	} 

	m->sdfData.resize(NUM_SDF_FEATURES_PER_ANGLE * NUM_ANGLES_SDF);
	m->vsiData.resize(NUM_VSI_FEATURES);
	for (int j = 0; j < NUM_SDF_FEATURES_PER_ANGLE * NUM_ANGLES_SDF; j++) {
		m->sdfData[j].resize( m->faces.size() );
	}
	for (int j = 0; j < NUM_VSI_FEATURES; j++) {
		m->vsiData[j].resize( m->faces.size() );
	}

	std::cout << std::endl << "Computing SDF and VSI features for mesh " << m->filename << std::endl;
	bool* done = new bool[m->faces.size()];
	bool* done2 = new bool[m->faces.size()];
	for( int i = 0; i < m->faces.size(); i++ ) {
		done[i] = false;
	}
	for( int i = 0; i < m->faces.size(); i++ ) {
		if (i % PRINT_EVERY_N == 0) {
			std::cout << 100.0f * (float)i / (float)m->faces.size() << "% complete\t\t\t\r";
			std::cout.flush();
		}
		if (ACCELERATE_N2_COMPUTATIONS) {
			for (int k = 0; k < 3; k++) {
				if ( done[m->across_edge[i][k]] == true ) {
					for (int j = 0; j < NUM_SDF_FEATURES_PER_ANGLE * NUM_ANGLES_SDF; j++) {
						m->sdfData[j][i] = m->sdfData[j][m->across_edge[i][k]];
					}
					for (int j = 0; j < NUM_VSI_FEATURES; j++) {
						m->vsiData[j][i] = m->vsiData[j][m->across_edge[i][k]];
					}
					done[i] = true;
					break;
				}
			}
			if (done[i] == true) {
				done[i] = false;
				continue;
			}
		}
		done[i] = true;

		int angleid = 0;
		vector<float> sdf;
		vector<float> vsi;
		float dist2center = 0.0f;
		vec v = -m->faces[i].facenormal;
		point o = m->faces[i].faceCenter;
		point rcenter;
		vec u = m->pdir1[ m->faces[i][0] ] + sgn( m->pdir1[ m->faces[i][0] ] DOT m->pdir1[ m->faces[i][1] ] ) *  m->pdir1[ m->faces[i][1] ];
		u = u + sgn( u DOT m->pdir1[ m->faces[i][2] ] ) * m->pdir1[ m->faces[i][2] ];
		u = normalize(u);
		u = _normalize(u CROSS v);

		// sdf computation
		for (float angle = 0; angle <= ANGLES_SDF[NUM_ANGLES_SDF-1]; angle+=ANGLE_STEP) {
			vec tv = _normalize( cosd(angle)*v + (1 - cosd(angle))*(v DOT u)*u + sind(angle)*(u CROSS v) );
			for (float rotangle = 0; rotangle < 360; rotangle += ROT_ANGLE_STEP) {
				vec ttv = _normalize( cosd(rotangle)*tv + (1 - cosd(rotangle))*(tv DOT v)*v + sind(rotangle)*(v CROSS tv) );
				sdf.push_back( checkClosestLineIntersection(ttv, o, i, m, done2) );
			}

			if (angle == ANGLES_SDF[angleid]) {
				dist2center = 0.5 * minimum(sdf); 
				rcenter = o + dist2center * v; 
				m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + 0][i] = mean(sdf);
				m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + 1][i] = mean2(sdf);
				m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + 2][i] = mean4(sdf);
				m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + 3][i] = meansqrt(sdf);
				m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + 4][i] = meansqrtsqrt(sdf);
				m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + 5][i] = median(sdf);
				angleid++;
			}
		}
		sdf.clear();

		// vsi computation
		vsi.push_back( dist2center ); //vsi.push_back( dist2center );
		for (float angle = ANGLE_VSI; angle < 180.0f; angle += ANGLE_VSI) {
			vec tv = _normalize( cosd(angle)*v + (1 - cosd(angle))*(v DOT u)*u + sind(angle)*(u CROSS v) );
			for (float rotangle = 0; rotangle < 360; rotangle += ANGLE_VSI) {
				vec ttv = _normalize( cosd(rotangle)*tv + (1 - cosd(rotangle))*(tv DOT v)*v + sind(rotangle)*(v CROSS tv) );
				vsi.push_back( checkClosestLineIntersection(ttv, rcenter, i, m, done2, false) );
			}
		}
		m->vsiData[0][i] = mean(vsi);
		m->vsiData[1][i] = mean2(vsi);
		m->vsiData[2][i] = mean4(vsi);
		m->vsiData[3][i] = meansqrt(vsi);
		m->vsiData[4][i] = meansqrtsqrt(vsi);
		m->vsiData[5][i] = median(vsi);
		vsi.clear();
	}

	// sdf renormalization
	for (int angleid = 0; angleid < NUM_ANGLES_SDF; angleid++) {
		for (int j = 0; j < NUM_SDF_FEATURES_PER_ANGLE; j++) {
			m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j] = removePositiveOutliers( m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j], .95f );
			//float meansdf = mean( sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j] );
			//float stdsdf = sqrt( variance( sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j], meansdf ) );
			//for( int i = 0; i < m->faces.size(); i++ ) {
			//	sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i] /= (stdsdf+EPSILON);
			//	features->FEATURES[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i+pos] = sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i];
			//}

			//float minsdf = minimum( m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j] );
			//float maxsdf = maximum( m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j] );
			//for( int i = 0; i < m->faces.size(); i++ ) {
			//	m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i] = (m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i]-minsdf) / (maxsdf-minsdf+EPSILON);
			//}

			float maxsdf = maximum( m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j] );
			for( int i = 0; i < m->faces.size(); i++ ) {
				m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i] /= maxsdf;
			}

			for( int i = 0; i < m->faces.size(); i++ ) {
				features->FEATURES[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i+pos] = m->sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i]; //(sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i] + sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][m->across_edge[i][0]] + sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][m->across_edge[i][1]] + sdfData[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][m->across_edge[i][2]]) / 4.0f;
			}

			for( int i = 0; i < m->faces.size(); i++ ) {
				for ( int base = 1; base < NUM_SDF_FEATURES_PER_BASE; base++) { // always start from 1, make sure the fiest element of BASES_SDF is 1 (no use of log then).
					int basepos = base * NUM_ANGLES_SDF*NUM_SDF_FEATURES_PER_ANGLE;
					features->FEATURES[basepos + NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i+pos] = log( features->FEATURES[NUM_SDF_FEATURES_PER_ANGLE*angleid + j][i+pos] * BASES_SDF[base] + 1.0f) / log(BASES_SDF[base] + 1.0f);
				}
			}
		}
	}

	// vsi renormalization
	for (int j = 0; j < NUM_VSI_FEATURES; j++) {
		m->vsiData[j] = removePositiveOutliers( m->vsiData[j], .95f );

		//float minvsi = minimum( m->vsiData[j] );
		//float maxvsi = maximum( m->vsiData[j] );
		//for( int i = 0; i < m->faces.size(); i++ ) {
		//	m->vsiData[j][i] = (m->vsiData[j][i]-minvsi) / (maxvsi-minvsi+EPSILON);
		//}

		float maxvsi = maximum( m->vsiData[j] );
		for( int i = 0; i < m->faces.size(); i++ ) {
			m->vsiData[j][i] /= maxvsi;
		}

		for( int i = 0; i < m->faces.size(); i++ ) {
			features->FEATURES2[j][i+pos] = m->vsiData[j][i]; //(vsiData[j][i] + vsiData[j][m->across_edge[i][0]] + vsiData[j][m->across_edge[i][1]] + vsiData[j][m->across_edge[i][2]])/4.0f;
		}

		for( int i = 0; i < m->faces.size(); i++ ) {
			for ( int base = 1; base < NUM_SDF_FEATURES_PER_BASE; base++) { // always start from 1, make sure the fiest element of BASES_SDF is 1 (no use of log then).
				int basepos = base * NUM_VSI_FEATURES;
				features->FEATURES2[basepos + j][i+pos] = log( features->FEATURES2[j][i+pos] * BASES_SDF[base] + 1.0f) / log(BASES_SDF[base] + 1.0f);
			}
		}
	}


	std::cout << 100.0f << "% complete\t\t\t\r";

	if (writeDebugInfo) {
		writeDebugInfoToFile( m, "_sdfFeatures.txt", features, m->faces.size(), pos, 1  );
		writeDebugInfoToFile( m, "_vsi.txt", features, m->faces.size(), pos, 2  );
	}


	delete[] done;
	delete[] done2;
	//for (int j = 0; j < NUM_SDF_FEATURES_PER_ANGLE * NUM_ANGLES_SDF; j++) {
	//	sdfData[j].clear();
	//}
	//for (int j = 0; j < NUM_VSI_FEATURES; j++) {
	//	vsiData[j].clear();
	//}
	//sdfData.clear();
	//vsiData.clear();

	return features;
}





FeatureSet* exportSpinImageFeatures(TriMesh* m, FeatureSet* features, int pos, int numFaces, bool writeDebugInfo, bool returnNumFeaturesOnly) {
	int numbins = (int)SPIN_RESOLUTION;
	if (pos == 0) {
		features = new FeatureSet(); 
		features->numFeatures = numbins * numbins;
		if (returnNumFeaturesOnly) {
			return features;
		}
		features->FEATURES = new float*[ features->numFeatures ];
		for( int i = 0; i < features->numFeatures; i++ ) {
			features->FEATURES[i] = new float[ numFaces ];
		}		
	} 

	//float SPIN_DISTANCE_SUPPORT = m->bsphere.r * 2.0f;

	//// initializations of spin image bins
	Bin2D** spin = new Bin2D*[ numbins ];
	for( int x = 0; x < SPIN_RESOLUTION; x++ ) {
		spin[x] = new Bin2D[ numbins ];
	}

	vector<float> vertexFaceAreas; vertexFaceAreas.resize( m->vertices.size() );
	float totalVertexFaceArea = 0.0f;
	for (int k = 0; k < m->vertices.size(); k++) {
		float vertexFaceArea = 0.0f;
		for (int kf = 0; kf < m->adjacentfaces[k].size(); kf++) {
			vertexFaceArea +=  m->faces[ m->adjacentfaces[k][kf] ].faceArea;
		}
		vertexFaceAreas[k] = vertexFaceArea;
		totalVertexFaceArea += vertexFaceArea;
	}
	float* binWeights = new float[ numbins * numbins ];


	std::cout << std::endl << "Computing spin images for mesh " << m->filename << std::endl;
	bool* done = new bool[m->faces.size()];
	for( int i = 0; i < m->faces.size(); i++ ) {
		done[i] = false;
	}

	for( int i = 0; i < m->faces.size(); i++ ) {
		if (i % PRINT_EVERY_N == 0) {
			std::cout << 100.0f * (float)i / (float)m->faces.size() << "% complete\t\t\t\r";
			std::cout.flush();
		}

		if (ACCELERATE_N2_COMPUTATIONS) {
			for (int k = 0; k < 3; k++) {
				if ( done[m->across_edge[i][k]] == true ) {
					int ii = 0;
					for (int j = 0; j < features->numFeatures; j++) {
						features->FEATURES[j][i+pos] = features->FEATURES[j][m->across_edge[i][k]+pos];
					}
					done[i] = true;
					break;
				}
			}
			if (done[i] == true) {
				done[i] = false;
				continue;
			}
		}
		done[i] = true;


		for( int x = 0; x < numbins; x++ ) {
			for( int y = 0; y < numbins; y++ ) {
				spin[x][y].reset();
			}
		}

		for (int k = 0; k < m->vertices.size(); k++) {
			if ( (m->normals[k] DOT m->faces[i].facenormal) < 0.0 )
				continue;

			vec diff = m->vertices[k] - m->faces[i].faceCenter;
			float beta = m->faces[i].facenormal DOT diff;
			float alpha = sqrt( len2(diff) - beta*beta );
			float ic = (SPIN_DISTANCE_SUPPORT/2.0f - beta/2.0f) / SPIN_BIN_SIZE;
			float jc = alpha / SPIN_BIN_SIZE;

			float sumw = 0.0f;
			int ii = 0;
			for (int x = 0; x < numbins; x++) {
				for (int y = 0; y < numbins; y++) {
					float cx = x + .5f;
					float cy = y + .5f;
					float w = 0.0f;
					if ( (fabs(ic - cx) <= 2.5f) && (fabs(jc - cy) <= 2.5f) ) {
						w = exp( -( (ic - cx)*(ic - cx) + (jc - cy)*(jc - cy) ) / .2f );
						if (w < .02)
							w = 0.0f;
					}
					binWeights[ii] = w;
					ii++;
					sumw += w;
				}
			}

			float areaProportion = vertexFaceAreas[k] / (totalVertexFaceArea + EPSILON);
			ii = 0;
			for (int x = 0; x < numbins; x++) {
				for (int y = 0; y < numbins; y++) {
					spin[x][y].increaseValue( (binWeights[ii] / (sumw+EPSILON)) * areaProportion  );
					ii++;
					//if ( _isnan( spin[x][y].value ) ) {
					//	std::cout << x << ' ' << y << ' ' << i << ' ' << ic << ' ' << jc << ' ' << binWeights[ii-1] << std::endl;
					//}
				}
			}
		}

		int ii = 0;
		for( int x = 0; x < numbins; x++ ) {
			for( int y = 0; y < numbins; y++ ) {
				features->FEATURES[ii][i+pos] = spin[x][y].value;
				ii++;
			}
		}
	}
	vertexFaceAreas.clear();
	std::cout << 100.0f << "% complete\t\t\t\r";


	if (writeDebugInfo) {
		writeDebugInfoToFile( m, "_spinImageFeatures.txt", features, m->faces.size(), pos, 1  );
	}

	for( int x = 0; x < numbins; x++ ) {
		delete[] spin[x];
	}
	delete[] spin;
	delete[] done;
	delete[] binWeights;
	return features;
}


FeatureSet* exportSCClassProbFeatures(TriMesh* m, FeatureSet* features, int pos, int numFaces, float** PH, int numlabels, bool writeDebugInfo, bool returnNumFeaturesOnly) {
	int numbins = NUM_GD_BINS_SCCLASSPROB;
	int numusepcaparts = min(numlabels, USE_PCADIR_PART_LABELS_MAX);

	if (pos == 0) {
		features = new FeatureSet(); 
		features->numFeatures = numbins * numlabels + numbins * numlabels * 3 * (numusepcaparts + 1);
		if (returnNumFeaturesOnly) {
			return features;
		}
		features->FEATURES = new float*[ features->numFeatures ];
		for( int i = 0; i < features->numFeatures; i++ ) {
			features->FEATURES[i] = new float[ numFaces ];
		}		
	}
	

	/// initialization of pca on part labels
	std::cout << "\nDetermining limitPH for faces of the mesh " << m->filename << std::endl;
	float limitPH = (1.0f / (float)numlabels); // could be 1.5
	std::vector<float> limitPHi( m->faces.size() );
	for( int i = 0; i < m->faces.size(); i++ ) {
		float maxprob = 0.0f;
		for (int c = 0; c < numlabels; c++) {
			if ( PH[ i+pos ][ c ] > maxprob) {
				maxprob = PH[ i+pos ][ c ];
			}
		}
		limitPHi[i] = min( limitPH, maxprob );
	}

	std::cout << "Determining parts pca for the mesh " << m->filename << std::endl;
	std::vector< std::vector<vec> > directions;
	std::vector< std::vector<float> > maxAxisDistances;
	waPCA wapcaGlobal(m->vertices, m->pointareas, m->vertices);

	directions.push_back( wapcaGlobal.alignedDirections );
	maxAxisDistances.push_back( wapcaGlobal.maxDistance );
	for (int c = 0; c < numusepcaparts; c++) {		
		std::vector<point> points;
		std::vector<float> weights;
		for( int i = 0; i < m->faces.size(); i++ ) {			
			if ( PH[ i+pos ][ c ] >= limitPHi[i] ) {
				points.push_back( m->faces[i].faceCenter );
				//weights.push_back( PH[ i+pos ][ c ]  );
				weights.push_back( m->faces[i].faceArea * PH[ i+pos ][ c ]  );
			}
		}
		waPCA wapca(points, weights, m->vertices);
		if ( len(wapca.alignedDirections[0]) <= EPSILON ) {
			directions.push_back( wapcaGlobal.alignedDirections );
			maxAxisDistances.push_back( wapcaGlobal.maxDistance );
		} else {
			directions.push_back( wapca.alignedDirections );
			maxAxisDistances.push_back( wapca.maxDistance );
		}
		points.clear();
		weights.clear();
	}

	//// initializations of bins & corresponding features
	Bin2D** bins = new Bin2D*[ numbins ];
	for( int x = 0; x < numbins; x++ ) {
		bins[x] = new Bin2D[ numlabels ];
	}
	Bin2D** ebins = new Bin2D*[ numbins ];
	for( int x = 0; x < numbins; x++ ) {
		ebins[x] = new Bin2D[ numlabels * 3 * (numusepcaparts + 1) ];
	}
	GeodesicTraversal MeshTraversal(*m);
	float s = (float)numbins+1;
	float D = GD_MAX_SCCLASSPROB * m->geodesicDistance;
	for( int x = 0; x < s-1; x++ ) {
		float startx = 0.0f;
		float endx = 0.0f;
		if (GD_LOGP_SCCLASSPROB <= 0.0f) {
			startx = D * (float)x/(float)numbins;
			endx = D * (float)(x+1)/(float)numbins;
		} else {
			startx = pow( (-log((s-x)/s) ), GD_LOGP_SCCLASSPROB ) * (D / pow( log(s), GD_LOGP_SCCLASSPROB) );
			endx = pow( (-log((s-x-1)/s) ), GD_LOGP_SCCLASSPROB ) * (D / pow( log(s), GD_LOGP_SCCLASSPROB) ); 
		}
		for( int y = 0; y < numlabels; y++ ) {
			bins[x][y].x1 = startx;
			bins[x][y].x2 = endx;
			bins[x][y].y1 = 0.0f;
			bins[x][y].y2 = 1.0f;
		}
	}
	for( int x = 0; x < s-1; x++ ) {
		float startx = 0.0f;
		float endx = 0.0f;
		for (int p = 0; p < numusepcaparts + 1; p++) {
			for (int d = 0; d < 3; d++) {
				D = ED_MAX_SCCLASSPROB * maxAxisDistances[p][d];
				if (ED_LOGP_SCCLASSPROB <= 0.0f) {
					startx = D * (float)x/(float)numbins;
					endx = D * (float)(x+1)/(float)numbins;
				} else {
					startx = pow( (-log((s-x)/s) ), ED_LOGP_SCCLASSPROB ) * (D / pow( log(s), ED_LOGP_SCCLASSPROB) );
					endx = pow( (-log((s-x-1)/s) ), ED_LOGP_SCCLASSPROB ) * (D / pow( log(s), ED_LOGP_SCCLASSPROB) ); 
				}
				for( int y = 0; y < numlabels; y++ ) {
					ebins[x][(p*3+d)*numlabels + y].x1 = startx;
					ebins[x][(p*3+d)*numlabels + y].x2 = endx;
					ebins[x][(p*3+d)*numlabels + y].y1 = 0.0f;
					ebins[x][(p*3+d)*numlabels + y].y2 = 1.0f;
				}
			}
		}
	}

	std::cout << "Computing shape context on probabilities for mesh " << m->filename << std::endl;
	bool* done = new bool[m->faces.size()];
	for( int i = 0; i < m->faces.size(); i++ ) {
		done[i] = false;
	}
	for( int i = 0; i < m->faces.size(); i++ ) {
		if (i % PRINT_EVERY_N == 0) {
			std::cout << 100.0f * (float)i / (float)m->faces.size() << "% complete\t\t\t\r";
			std::cout.flush();
		}

		if (ACCELERATE_N2_COMPUTATIONS) {
			for (int k = 0; k < 3; k++) {
				if ( done[m->across_edge[i][k]] == true ) {
					int ii = 0;
					for (int j = 0; j < features->numFeatures; j++) {
						features->FEATURES[j][i+pos] = features->FEATURES[j][m->across_edge[i][k]+pos];
					}
					done[i] = true;
					break;
				}
			}
			if (done[i] == true) {
				done[i] = false;
				continue;
			}
		}
		done[i] = true;


		for( int x = 0; x < numbins; x++ ) {
			for( int y = 0; y < numlabels; y++ ) {
				bins[x][y].reset();
			}
		}
		for( int x = 0; x < numbins; x++ ) {
			for( int y = 0; y < numlabels * 3 * (numusepcaparts + 1); y++ ) {
				ebins[x][y].reset();
			}
		}

		vector<int> visitedFaces = MeshTraversal.traverseFaces(i);
		float *GD = MeshTraversal.getGeodDistances();
		for (int k = 0; k < m->faces.size(); k++) {
			for( int x = 0; x < numbins; x++ ) {
				for( int y = 0; y < numlabels; y++ ) {
//					if ( PH[k+pos][y] < limitPHi[k] ) {
//						continue;
//					}
					if ( bins[x][y].contains( GD[k], PH[k+pos][y] ) ) {
						bins[x][y].increaseValue( (m->faces[k].faceArea / m->totalFaceArea) * PH[k+pos][y] * (1.0f / ((GD[k]/m->geodesicDistance)+.1f)) ); 
//						bins[x][y].increaseValue( m->faces[k].faceArea * PH[k+pos][y] * (1.0f / (GD[k] + .1f)) ); 
					}
				}				

				for ( int p = 0; p < 1 + numusepcaparts; p++ ) {
					if (directions[p].size() == 0) {
						continue;
					}
					for (int d = 0; d < 3; d++) {
						float odist = fabs( (m->faces[k].faceCenter - m->faces[i].faceCenter) DOT directions[p][d] );
						for( int y = 0; y < numlabels; y++ ) {
//							if ( PH[k+pos][y] < limitPHi[k] ) {
//								continue;
//							}
							if ( ebins[x][(p*3+d)*numlabels + y].contains( odist, PH[k+pos][y] ) ) {
								ebins[x][(p*3+d)*numlabels + y].increaseValue( (m->faces[k].faceArea / m->totalFaceArea)  * PH[k+pos][y] * (1.0f / ((odist/maxAxisDistances[p][d]) + .1f)) );
//								ebins[x][(p*3+d)*numlabels + y].increaseValue( m->faces[k].faceArea * PH[k+pos][y] * (1.0f / (odist+ .1f)) );
							}
						}
					}
				}
			}
		}

		//for( int x = 0; x < numbins; x++ ) {
		//	float sumRowValues = EPSILON;
		//	for( int y = 0; y < numlabels; y++ ) {
		//		sumRowValues += bins[x][y].value;
		//	}
		//	for( int y = 0; y < numlabels; y++ ) {
		//		bins[x][y].value /= sumRowValues;
		//	}
		//	
		//	for ( int p = 0; p < 1 + numusepcaparts; p++ ) {
		//		for (int d = 0; d < 3; d++) {
		//			sumRowValues = EPSILON;
		//			for( int y = 0; y < numlabels; y++ ) {
		//				sumRowValues += ebins[x][(p*3+d)*numlabels + y].value;
		//			}
		//			for( int y = 0; y < numlabels; y++ ) {
		//				ebins[x][(p*3+d)*numlabels + y].value /= sumRowValues;
		//			}
		//		}
		//	}
		//}

		int ii = 0;
		for( int x = 0; x < numbins; x++ ) {
			for( int y = 0; y < numlabels; y++ ) {
				features->FEATURES[ii][i+pos] = bins[x][y].value;
				ii++;
			}
		}
		for( int x = 0; x < numbins; x++ ) {
			for( int y = 0; y < numlabels * 3 * (numusepcaparts + 1); y++ ) {
				features->FEATURES[ii][i+pos] = ebins[x][y].value;
				ii++;
			}
		}
	}
	std::cout << 100.0f << "% complete\t\t\t\r";

	if (writeDebugInfo) {
		writeDebugInfoToFile( m, "_scClassProbFeatures.txt", features, m->faces.size(), pos, 1  );
	}

	for( int x = 0; x < numbins; x++ ) {
		delete[] bins[x];
		delete[] ebins[x];
	}
	delete[] bins;
	delete[] ebins;
	delete[] done;
	return features;
}




// revised a few times after submission.
FeatureSet* exportEdgeFeatures(TriMesh* m,  FeatureSet* features, int pos, int numEdges, int* labels, float boundaryWeight, bool writeDebugInfo, bool returnNumFeaturesOnly ) {
	if (pos == 0) {
		features = new FeatureSet(); 
		features->numFeatures = NUM_EDGE_FEATURES + NUM_EDGE_BINS_SCCLASSPROB * 2;
		if (returnNumFeaturesOnly) {
			return features;
		}
		features->FEATURES = new float*[ features->numFeatures ];
		for( int i = 0; i < features->numFeatures; i++ ) {
			features->FEATURES[i] = new float[ numEdges ];
		}		
		features->WEIGHTS = new float[ numEdges ];
		features->LABELS = new int[ numEdges ];
		features->numFeatures2 = 1;
		features->FEATURES2 = new float*[1]; // this will store the labels for debugging reasons
		features->FEATURES2[0] = new float[ numEdges ];
	} 

	std::cout << std::endl << "Computing edge structures for mesh " << m->filename << std::endl;
	m->edges.resize( 3 * m->faces.size() );
	m->edgesVertices.resize( 3 * m->faces.size() );
	m->verticesToEdges.resize( m->vertices.size() );
	for( int i = 0; i < m->faces.size(); i++ ) {
		for (int j = 0; j < 3; j++) {
			int fj = m->across_edge[i][j];
			m->edges[ 3*i + j ].push_back( i );
			m->edges[ 3*i + j ].push_back( fj );
			for (int vi = 0; vi < 3; vi++) {
				for (int vj = 0; vj < 3; vj++) {
					if (m->faces[i][vi] == m->faces[fj][vj]) {
						m->edgesVertices[ 3*i + j ].push_back(  m->faces[i][vi] );
						m->verticesToEdges[ m->faces[i][vi] ].push_back(  3*i + j );
					}
				}
			}
			if ( (m->edgesVertices[ 3*i + j ][0] == m->faces[i][0]) && (m->edgesVertices[ 3*i + j ][1] == m->faces[i][2]) ) {
				 m->edgesVertices[ 3*i + j ][0] = m->faces[i][2];
				 m->edgesVertices[ 3*i + j ][1] = m->faces[i][0];
			}
		}
	}
	std::vector< std::vector<float> > faceDihedralAngleChange(NUM_EDGE_SCALES*NUM_FACEDIHEDRALANGLE_FEATURES_PERSCALE);
	for (int j = 0; j < NUM_EDGE_SCALES*NUM_FACEDIHEDRALANGLE_FEATURES_PERSCALE; j++) 
		faceDihedralAngleChange[j].resize( m->faces.size() );

	std::cout << std::endl << "Computing dihedral angles for mesh " << m->filename << std::endl;
	for( int i = 0; i < m->edges.size(); i++ ) {
		int fi =  m->edges[i][0];
		int fj =  m->edges[i][1];
		int vi =  m->edgesVertices[i][0];
		int vj =  m->edgesVertices[i][1];

		features->LABELS[pos + i] = ( labels[fi] != labels[fj] ) + 1;
		features->WEIGHTS[pos + i] = (m->faces[fi].faceArea / m->totalFaceArea) + (m->faces[fj].faceArea / m->totalFaceArea);  //len( m->vertices[vi] - m->vertices[vj] );
		if ( features->LABELS[pos + i] == 2 )
			features->WEIGHTS[pos + i] *= boundaryWeight;
		features->FEATURES2[0][pos + i] = float(features->LABELS[pos + i]);

		// 0-4: dihedral angles
		vec faceNormali = _normalize( (m->vertices[m->faces[fi][0]] - m->vertices[m->faces[fi][1]]) CROSS (m->vertices[m->faces[fi][1]] - m->vertices[m->faces[fi][2]]) );  //m->faces[fi].facenormal; 
		vec faceNormalj = _normalize( (m->vertices[m->faces[fj][0]] - m->vertices[m->faces[fj][1]]) CROSS (m->vertices[m->faces[fj][1]] - m->vertices[m->faces[fj][2]]) );  //m->faces[fj].facenormal;		
		float cosphi = faceNormali DOT faceNormalj;
		vec   sinphi = faceNormali CROSS faceNormalj;
		float dihedralAngle = atan2( len(sinphi),  cosphi );
		vec   edge   = _normalize(m->vertices[vj] - m->vertices[vi]);
		dihedralAngle = dihedralAngle * sgn( sinphi DOT edge );
		dihedralAngle = (M_PI + dihedralAngle) / M_PI;
		features->FEATURES[0][pos + i] = dihedralAngle;
		features->FEATURES[1][pos + i] = dihedralAngle * dihedralAngle;
		features->FEATURES[2][pos + i] = dihedralAngle * dihedralAngle * dihedralAngle;
		features->FEATURES[3][pos + i] = pow(dihedralAngle, 5.0f);
		features->FEATURES[4][pos + i] = pow(dihedralAngle, 10.0f);
	}

	std::cout << std::endl << "Computing edge features for mesh " << m->filename << std::endl;
	GeodesicTraversal MeshTraversal(*m);
	for( int i = 0; i < m->edges.size(); i++ ) {
		if (i % (PRINT_EVERY_N*3) == 0) {
			std::cout << 100.0f * (float)i / (float)m->edges.size() << "% complete\t\t\t\r";
			std::cout.flush();
		}
		int fi =  m->edges[i][0];
		int fj =  m->edges[i][1];
		int vi =  m->edgesVertices[i][0];
		int vj =  m->edgesVertices[i][1];

		// 5-14, 15-24, 25-34, 35-44: average and median dihedral angle in neighborhood
		int ii = 0;
		int fii = 0;
		for (float scale = 0.005f; ii < NUM_EDGE_SCALES*NUM_DIHEDRALANGLE_FEATURES_PERSCALE; scale *= 2.5f) {
			vector<float> dihedralAngles;
			vector<float> weights;
			vector<int> visitedVertices = MeshTraversal.traverse(vi, scale);
			float *GD = MeshTraversal.getGeodDistances();
			for (int k = 0; k < visitedVertices.size(); k++) {
				for (int l = 0; l < m->verticesToEdges[ visitedVertices[k] ].size(); l++) {
					int visitedEdge = m->verticesToEdges[ visitedVertices[k] ][l];
					dihedralAngles.push_back( features->FEATURES[0][pos + visitedEdge] );
					weights.push_back( features->WEIGHTS[pos + visitedEdge] / (GD[ visitedVertices[k] ] + 0.01f) );
				}
			}
			visitedVertices = MeshTraversal.traverse(vj, EDGE_MAX_SCCLASSPROB);
			GD = MeshTraversal.getGeodDistances();
			for (int k = 0; k < visitedVertices.size(); k++) {
				for (int l = 0; l < m->verticesToEdges[ visitedVertices[k] ].size(); l++) {
					int visitedEdge = m->verticesToEdges[ visitedVertices[k] ][l];
					dihedralAngles.push_back( features->FEATURES[0][pos + visitedEdge] );
					weights.push_back( features->WEIGHTS[pos + visitedEdge] / (GD[ visitedVertices[k] ] + 0.01f) );
				}
			}

			float meanDihedralAngle = weightedmean(dihedralAngles, weights);
			float medianDihedralAngle = weightedmedian(dihedralAngles, weights);
			faceDihedralAngleChange[fii + 0][fi] = meanDihedralAngle;
			faceDihedralAngleChange[fii + 1][fi] = medianDihedralAngle;
			fii += NUM_FACEDIHEDRALANGLE_FEATURES_PERSCALE;
			
			features->FEATURES[5 + ii][pos + i] = meanDihedralAngle;
			features->FEATURES[6 + ii][pos + i] = meanDihedralAngle * meanDihedralAngle;
			features->FEATURES[7 + ii][pos + i] = meanDihedralAngle * meanDihedralAngle * meanDihedralAngle;
			features->FEATURES[8 + ii][pos + i] = pow(meanDihedralAngle, 5.0f);
			features->FEATURES[9 + ii][pos + i] = pow(meanDihedralAngle, 10.0f);
			features->FEATURES[10 + ii][pos + i] = medianDihedralAngle;
			features->FEATURES[11 + ii][pos + i] = medianDihedralAngle * medianDihedralAngle;
			features->FEATURES[12 + ii][pos + i] = medianDihedralAngle * medianDihedralAngle * medianDihedralAngle;
			features->FEATURES[13 + ii][pos + i] = pow(medianDihedralAngle, 5.0f);
			features->FEATURES[14 + ii][pos + i] = pow(medianDihedralAngle, 10.0f);

			ii += NUM_DIHEDRALANGLE_FEATURES_PERSCALE;

			dihedralAngles.clear();
			weights.clear();
		}

		//// 45-49 mean/gauss curvature and derivatives
		//float k2vi = m->curv2[vi], k2vj = m->curv2[vj];
		//float gausskvi = m->curv2[vi]*m->curv1[vi], gausskvj = m->curv2[vj]*m->curv1[vj];
		//float dk2vi = m->dcurv[vi][3], dk2vj = m->dcurv[vj][3];
		//float k2absvi, k2absvj, dk2absvi, dk2absvj;
		//if ( fabs( m->curv2[vi] ) > fabs( m->curv1[vi] ) ) {
		//	k2absvi = m->curv1[vi];
		//	dk2absvi = m->dcurv[vi][0];
		//} else {
		//	k2absvi = m->curv2[vi];
		//	dk2absvi = m->dcurv[vi][3];
		//}
		//if ( fabs( m->curv2[vj] ) > fabs( m->curv1[vj] ) ) {
		//	k2absvj = m->curv1[vj];
		//	dk2absvj = m->dcurv[vj][0];
		//} else {
		//	k2absvj = m->curv2[vj];
		//	dk2absvj = m->dcurv[vj][3];
		//}
		//features->FEATURES[45][pos + i] = .5f * (k2vi + k2vj);
		//features->FEATURES[46][pos + i] = .5f * (gausskvi + gausskvj);
		//features->FEATURES[47][pos + i] = .5f * (k2absvi + k2absvj);
		//features->FEATURES[48][pos + i] = .5f * (dk2vi + dk2vj);
		//features->FEATURES[49][pos + i] = .5f * (dk2absvi + dk2absvj);

		//// 45-59 mean/gauss curvature and derivatives
		float k2vi = m->curv2[vi], k2vj = m->curv2[vj];
		float gausskvi = m->curv2[vi]*m->curv1[vi], gausskvj = m->curv2[vj]*m->curv1[vj];
		float dk2vi = m->dcurv[vi][3], dk2vj = m->dcurv[vj][3];
		float dk1vi = m->dcurv[vi][0], dk1vj = m->dcurv[vj][0];
		float k2absvi, k2absvj, dk2absvi, dk2absvj;
		if ( fabs( m->curv2[vi] ) > fabs( m->curv1[vi] ) ) {
			k2absvi = m->curv1[vi];
			dk2absvi = m->dcurv[vi][0];
		} else {
			k2absvi = m->curv2[vi];
			dk2absvi = m->dcurv[vi][3];
		}
		if ( fabs( m->curv2[vj] ) > fabs( m->curv1[vj] ) ) {
			k2absvj = m->curv1[vj];
			dk2absvj = m->dcurv[vj][0];
		} else {
			k2absvj = m->curv2[vj];
			dk2absvj = m->dcurv[vj][3];
		}
		float kdiffvi = m->curv1[vi] - fabs( m->curv2[vi] ), kdiffvj = m->curv1[vj] - fabs( m->curv2[vj] );
		float ksumvi = m->curv2[vi] + fabs( m->curv1[vi] ),  ksumvj = m->curv2[vj] + fabs( m->curv1[vj] );		
		float dk1diff = (dk1vj - dk1vi)* sgn( _normalize( m->vertices[vj] -  m->vertices[vi] ) DOT m->pdir1[vi] );
		float dk2diff = (dk2vj - dk2vi)* sgn( _normalize( m->vertices[vj] -  m->vertices[vi] ) DOT m->pdir2[vi] );

		features->FEATURES[45][pos + i] = .5f * (k2vi + k2vj);
		features->FEATURES[46][pos + i] = .5f * (k2absvi + k2absvj);
		features->FEATURES[47][pos + i] = .5f * (dk1vi + dk1vj);
		features->FEATURES[48][pos + i] = .5f * (dk2vi + dk2vj);
		features->FEATURES[49][pos + i] = dk1vi * dk1vj;
		features->FEATURES[50][pos + i] = dk2vi * dk2vj;
		features->FEATURES[51][pos + i] = .5f * (kdiffvi + kdiffvj);
		features->FEATURES[52][pos + i] = .5f * (ksumvi + ksumvj);
		features->FEATURES[53][pos + i] = dk1diff;
		features->FEATURES[54][pos + i] = dk2diff;
		features->FEATURES[55][pos + i] = .5f * (dk2absvi + dk2absvj); 
		features->FEATURES[56][pos + i] = .5f * (gausskvi + gausskvj);
		features->FEATURES[57][pos + i] = .5f * (m->dcurv[vi][1] + m->dcurv[vj][1]);
		features->FEATURES[58][pos + i] = .5f * (m->dcurv[vi][2] + m->dcurv[vj][2]);
		features->FEATURES[59][pos + i] = .5f * (m->curv1[vi] + m->curv1[vj]);


    //// 60-69 difference of median sdf
    float diffsdf = 0.0f;
    if (!m->sdfData.empty())
      diffsdf = fabs(m->sdfData[m->sdfData.size() - 1][fi] - m->sdfData[m->sdfData.size() - 1][fj]);
    features->FEATURES[60][pos + i] = diffsdf;
    features->FEATURES[61][pos + i] = diffsdf * diffsdf;
    features->FEATURES[62][pos + i] = diffsdf * diffsdf * diffsdf;
    features->FEATURES[63][pos + i] = pow(diffsdf, 5.0f);
    features->FEATURES[64][pos + i] = pow(diffsdf, 10.0f);
    if (!m->sdfData.empty())
      diffsdf = fabs(m->sdfData[m->sdfData.size() - 6][fi] - m->sdfData[m->sdfData.size() - 6][fj]);
    features->FEATURES[65][pos + i] = diffsdf;
    features->FEATURES[66][pos + i] = diffsdf * diffsdf;
    features->FEATURES[67][pos + i] = diffsdf * diffsdf * diffsdf;
    features->FEATURES[68][pos + i] = pow(diffsdf, 5.0f);
    features->FEATURES[69][pos + i] = pow(diffsdf, 10.0f);

    ////70-79 difference of mean vsi
    float diffvsi = 0.0f;
    if (!m->vsiData.empty())
      diffvsi = fabs(m->vsiData[m->vsiData.size() - 1][fi] - m->vsiData[m->vsiData.size() - 1][fj]);
    features->FEATURES[70][pos + i] = diffvsi;
    features->FEATURES[71][pos + i] = diffvsi * diffvsi;
    features->FEATURES[72][pos + i] = diffvsi * diffvsi * diffvsi;
    features->FEATURES[73][pos + i] = pow(diffvsi, 5.0f);
    features->FEATURES[74][pos + i] = pow(diffvsi, 10.0f);
    if (!m->vsiData.empty())
      diffvsi = fabs(m->vsiData[m->vsiData.size() - 6][fi] - m->vsiData[m->vsiData.size() - 6][fj]);
    features->FEATURES[75][pos + i] = diffvsi;
    features->FEATURES[76][pos + i] = diffvsi * diffvsi;
    features->FEATURES[77][pos + i] = diffvsi * diffvsi * diffvsi;
    features->FEATURES[78][pos + i] = pow(diffvsi, 5.0f);
    features->FEATURES[79][pos + i] = pow(diffvsi, 10.0f);
  }

	for( int i = 0; i < m->edges.size(); i++ ) {
		int fi =  m->edges[i][0];
		int fj =  m->edges[i][1];

		for (int j = 0; j < NUM_FACEDIHEDRALANGLE_FEATURES_PERSCALE*NUM_EDGE_SCALES; j++) {
			features->FEATURES[80+j*5][pos + i] = fabs( faceDihedralAngleChange[j][fi] - faceDihedralAngleChange[j][fj] );
			features->FEATURES[81+j*5][pos + i] = fabs( faceDihedralAngleChange[j][fi]*faceDihedralAngleChange[j][fi] - faceDihedralAngleChange[j][fj]*faceDihedralAngleChange[j][fj] );
			features->FEATURES[82+j*5][pos + i] = fabs( faceDihedralAngleChange[j][fi]*faceDihedralAngleChange[j][fi]*faceDihedralAngleChange[j][fi] - faceDihedralAngleChange[j][fj]*faceDihedralAngleChange[j][fj]*faceDihedralAngleChange[j][fj] );
			features->FEATURES[83+j*5][pos + i] = fabs( pow(faceDihedralAngleChange[j][fi], 5.0f) - pow( faceDihedralAngleChange[j][fj], 5.0f ) );
			features->FEATURES[84+j*5][pos + i] = fabs( pow(faceDihedralAngleChange[j][fi], 10.0f) - pow( faceDihedralAngleChange[j][fj], 10.0f ) );
		}

		for (int j = NUM_EDGE_FEATURES; j < features->numFeatures; j++) {
			features->FEATURES[j][pos + i] = 0.0f;
		}
	}
	std::cout << 100.0f << "% complete\t\t\t\r";

	if (writeDebugInfo) {
		writeDebugInfoToFile( m, "_edgeLabels.txt", features, m->faces.size()*3, pos, 2  );
		writeDebugInfoToFile( m, "_edgeFeatures.txt", features, m->faces.size()*3, pos, 1  );
	}

	for (int j = 0; j < faceDihedralAngleChange.size(); j++) {
		faceDihedralAngleChange[j].clear();
	}
	faceDihedralAngleChange.clear();

	return features;
}


FeatureSet* exportSCEdgeFeatures(TriMesh* m, FeatureSet* features, int pos, int fpos, float** PH, bool writeDebugInfo ) {
	//// initializations of bins & corresponding features
	float limitPH = .5f;
	int numbins = NUM_EDGE_BINS_SCCLASSPROB;
	Bin2D* bins = new Bin2D[ numbins ];
	Bin2D* pbins = new Bin2D[ numbins ];
	float s = (float)numbins+1;
	float D = EDGE_MAX_SCCLASSPROB;
	for( int x = 0; x < s-1; x++ ) {
		float startx = 0.0f;
		float endx = 0.0f;
		if (EDGE_LOGP_SCCLASSPROB <= 0.0f) {
			startx = D * (float)x/(float)numbins;
			endx = D * (float)(x+1)/(float)numbins;
		} else {
			startx = pow( (-log((s-x)/s) ), EDGE_LOGP_SCCLASSPROB ) * (D / pow( log(s), EDGE_LOGP_SCCLASSPROB) );
			endx = pow( (-log((s-x-1)/s) ), EDGE_LOGP_SCCLASSPROB ) * (D / pow( log(s), EDGE_LOGP_SCCLASSPROB) ); 
		}
		bins[x].x1 = 0.0f; //startx;
		bins[x].x2 = endx;
		bins[x].y1 = limitPH;
		bins[x].y2 = 1.0f;
		pbins[x].x1 = 0.0f; //startx;
		pbins[x].x2 = endx;
		pbins[x].y1 = limitPH;
		pbins[x].y2 = 1.0f;
	}

	GeodesicTraversal MeshTraversal(*m);
	std::cout << "Computing context on edge probabilities for mesh " << m->filename << std::endl;
	for( int i = 0; i < m->edges.size(); i++ ) {
		if (i % (PRINT_EVERY_N*3) == 0) {
			std::cout << 100.0f * (float)i / (float)m->edges.size() << "% complete\t\t\t\r";
			std::cout.flush();
		}

		for( int x = 0; x < numbins; x++ ) {
			bins[x].reset();
			pbins[x].reset();
		}
		int fi =  m->edges[i][0];
		int fj =  m->edges[i][1];
		int vi =  m->edgesVertices[i][0];
		int vj =  m->edgesVertices[i][1];
		point edgeMidPoint = 0.5f * (m->vertices[vi] + m->vertices[vj]);
		vec edgeNormal = _normalize( m->faces[fi].facenormal * m->faces[fi].faceArea +  m->faces[fj].facenormal * m->faces[fj].faceArea );
		std::vector<int> visitedEdges;
		std::vector<float> edgeGeodDistances;

		vector<int> visitedVertices = MeshTraversal.traverse(vi, EDGE_MAX_SCCLASSPROB);
		float *GD = MeshTraversal.getGeodDistances();
		for (int k = 0; k < visitedVertices.size(); k++) {
			for (int l = 0; l < m->verticesToEdges[ visitedVertices[k] ].size(); l++) {
				int visitedEdge = m->verticesToEdges[ visitedVertices[k] ][l];
				visitedEdges.push_back(visitedEdge);
				edgeGeodDistances.push_back( GD[ visitedVertices[k] ] );
			}
		}
		visitedVertices = MeshTraversal.traverse(vj, EDGE_MAX_SCCLASSPROB);
		GD = MeshTraversal.getGeodDistances();
		for (int k = 0; k < visitedVertices.size(); k++) {
			for (int l = 0; l < m->verticesToEdges[ visitedVertices[k] ].size(); l++) {
				int visitedEdge = m->verticesToEdges[ visitedVertices[k] ][l];
				visitedEdges.push_back(visitedEdge);
				edgeGeodDistances.push_back( GD[ visitedVertices[k] ] );
			}
		}

		for (int k = 0; k < visitedEdges.size(); k++) {
			int ek = visitedEdges[k];
			int fik = m->edges[ek][0];
			int fjk = m->edges[ek][1];
			int vik = m->edgesVertices[ek][0];
			int vjk = m->edgesVertices[ek][1];
			vec edgekNormal = _normalize( m->faces[fik].facenormal * m->faces[fik].faceArea +  m->faces[fjk].facenormal * m->faces[fjk].faceArea );
			point edgekMidPoint = 0.5f * (m->vertices[vik] + m->vertices[vjk]);
			vec eek = _normalize( edgekMidPoint - edgeMidPoint );
			float coplanarity = (1.0f - fabs(_normalize(eek CROSS edgeNormal) DOT edgekNormal));

			for( int x = 0; x < numbins; x++ ) {
				if ( bins[x].contains( edgeGeodDistances[k], PH[ek+pos][1] ) ) {
					bins[x].increaseValue( ((m->faces[fik].faceArea / m->totalFaceArea) + (m->faces[fjk].faceArea / m->totalFaceArea)) * PH[ek+pos][1] * (1.0f / (edgeGeodDistances[k]+.01f) ) );
					pbins[x].increaseValue( ((m->faces[fik].faceArea / m->totalFaceArea) + (m->faces[fjk].faceArea / m->totalFaceArea)) * PH[ek+pos][1] * coplanarity * coplanarity );
				}
			}
		}

		int ii = 0;
		for( int x = 0; x < numbins; x++ ) {
			features->FEATURES[ii+fpos][i+pos] = bins[x].value;
			ii++;
		}
		for( int x = 0; x < numbins; x++ ) {
			features->FEATURES[ii+fpos][i+pos] = pbins[x].value;
			ii++;
		}

		visitedEdges.clear();
		edgeGeodDistances.clear();
		visitedVertices.clear();
	}
	std::cout << 100.0f << "% complete\t\t\t\r";

	if (writeDebugInfo) {
		writeDebugInfoToFile( m, "_edgeFeatures.txt", features, m->faces.size()*3, pos, 1  );
	}

	delete[] bins;
	delete[] pbins;
	return features;
}
