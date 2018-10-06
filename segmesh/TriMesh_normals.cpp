/*
Szymon Rusinkiewicz
Princeton University

TriMesh_normals.cc
Compute per-vertex normals for TriMeshes

For meshes, uses average of per-face normals, weighted according to:
  Max, N.
  "Weights for Computing Vertex Normals from Facet Normals,"
  Journal of Graphics Tools, Vol. 4, No. 2, 1999.

For raw point clouds, fits plane to k nearest neighbors.
*/

#include <stdio.h>
#include "TriMesh.h"
#include "KDtree.h"
#include "lineqn.h"
#include <set>
using std::set;


// Helper class for finding k-nearest-neighbors: returns true iff
// a point is not in the given set of points
class NotInSet : public KDtree::CompatFunc {
private:
	const float *plist;
	const set<int> &s;
public:
	NotInSet(const float *plist_, const set<int> &s_) :
			plist(plist_), s(s_)
		{}
	virtual bool operator () (const float *p) const
	{
		int ind = p - plist; 
		return (s.find(ind) == s.end());
	}
};


// Compute per-vertex normals
void TriMesh::need_normals()
{
	if (normals.size() == vertices.size())
		return;

	need_faces();
	int nf = faces.size(), nv = vertices.size();

	dprintf("Computing normals... ");
	normals.clear();
	normals.resize(nv);

	if (nf != 0) {
		// Find normals of a mesh
#pragma omp parallel for
		for (int i = 0; i < nf; i++) {
			const point &p0 = vertices[faces[i][0]];
			const point &p1 = vertices[faces[i][1]];
			const point &p2 = vertices[faces[i][2]];
			vec a = p0-p1, b = p1-p2, c = p2-p0;
			float l2a = len2(a), l2b = len2(b), l2c = len2(c);
			vec bias(1e-37f, 1e-37f, 1e-37f);
			faces[i].facenormal = (a CROSS b) + bias;
			faces[i].faceArea = .5f * len( faces[i].facenormal );
			normals[faces[i][0]] += faces[i].facenormal * (1.0f / (l2a * l2c + 1e-27f));
			normals[faces[i][1]] += faces[i].facenormal * (1.0f / (l2b * l2a + 1e-27f));
			normals[faces[i][2]] += faces[i].facenormal * (1.0f / (l2c * l2b + 1e-27f));
			faces[i].faceCenter = (p0+p1+p2)/3.0f;
		}
	} else {
		// Find normals of a point cloud
		const int k = 12;
		const vec ref(0, 0, 1);
		const float *v0 = &vertices[0][0];
		KDtree *kd = new KDtree(v0, nv);
		for (int i = 0; i < nv; i++) {
			const float *vi = &vertices[i][0];
			set<int> s;
			s.insert(vi - v0);
			for (int j = 0; j < k; j++) {
				NotInSet ns(v0, s);
				const float *match =
					kd->closest_to_pt(vi, 0.0f, &ns);
				if (!match)
					break;
				s.insert(match - v0);
			}
			if (s.size() < 4) {
				printf("Warning: not enough points for vertex %d\n", i);
				normals[i] = ref;
				continue;
			}
			// Compute covariance
			float C[3][3] = { {0,0,0}, {0,0,0}, {0,0,0} };
			for (set<int>::iterator it = s.begin(); it != s.end(); it++) {
				int ind = *it / 3;
				if (ind == i)
					continue;
				vec d = vertices[ind] - vertices[i];
				for (int l = 0; l < 3; l++)
					for (int m = 0; m < 3; m++)
						C[l][m] += d[l] * d[m];
			}
			float e[3];
			eigdc<float,3>(C, e);
			normals[i] = vec(C[0][0], C[1][0], C[2][0]);
			if ((normals[i] DOT ref) < 0.0f)
				normals[i] = -normals[i];
		}
		delete kd;
	}

#pragma omp parallel for
	for (int i = 0; i < nv; i++)
		normalize(normals[i]);

	totalFaceArea = 0;
#pragma omp parallel for
	for (int i = 0; i < nf; i++) {
		faces[i].facenormal = _normalize( normals[faces[i][0]] + normals[faces[i][1]] + normals[faces[i][2]] );
		totalFaceArea += faces[i].faceArea;
	}

	dprintf("Done.\n");
}

