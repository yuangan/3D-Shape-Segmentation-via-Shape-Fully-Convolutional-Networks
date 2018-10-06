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
#ifndef __DIJKSTRA_FOR_GEODESICS_H
#define __DIJKSTRA_FOR_GEODESICS_H

#include "TriMesh.h"
#include "MeshSegmentationFeatures.h"
#include <queue>

class vertexForDijkstra {
private:
	float _di;
	int _vi;

public:
  vertexForDijkstra(int vi, float di): _vi(vi), _di(di) { }
  bool operator < (const vertexForDijkstra& v) const {
    return _di > v._di;
  }
  float getGeodDistance() { 
	  return _di;
  }
  int getVertexIndex() { 
	  return _vi;
  }
};

class GeodesicTraversal {
private: 
	TriMesh	*_mesh;
	bool* _visited;
	float* _currentGeodesicDistance;
	std::priority_queue<vertexForDijkstra> _pq;
public:
	GeodesicTraversal(TriMesh* mesh);
	float traverse(int sourceVertex);
	vector<int>  traverse(int sourceVertex, float maxGeodesicDistance);
	vector<int>  traverseFaces(int sourceFace, float maxGeodesicDistance = FLT_MAX, int minFaces = 0, float checkNormalCompatibility = -FLT_MAX);
	float getMaxGeodesicDistance(int subsample = 1);	
	float getMeanMaxGeodesicDistance(int subsample = 1);	
	float getMeanGeodesicDistance(int subsample = 1);
	float getMedianGeodesicDistance(int subsample = 1);
	float getPercentileGeodesicDistance(int subsample = 1, float k = 0.3f, bool normalize = false);
	float* getGeodDistances();
	~GeodesicTraversal();
};


# endif