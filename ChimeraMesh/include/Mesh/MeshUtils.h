
//  Copyright (c) 2013, Vinicius Costa Azevedo
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met: 
//
//1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer. 
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution. 
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//	The views and conclusions contained in the software and documentation are those
//	of the authors and should not be interpreted as representing official policies, 
//	either expressed or implied, of the FreeBSD Project.

#ifndef _CHIMERA_DATA_MESH_UTILS_H_
#define _CHIMERA_DATA_MESH_UTILS_H_

#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "Mesh/Face.h"

//Cross reference
//#include "CutCells/CutCells3D.h"
//#include "Mesh/TriangleMesh3D.h"

namespace Chimera {
	namespace Meshes {
		namespace MeshUtils {

			//class TriangleAdjacencyGraph {
			//	public:
			//	typedef struct triangleAdj_t {
			//		triangleAdj_t *neighbors[3];
			//		int degree;
			//		int indices[3];
			//		int ID;

			//		triangleAdj_t() {
			//			neighbors[0] = neighbors[1] = neighbors[2] = NULL;
			//			degree = 0;
			//			indices[0] = indices[1] = indices[2] = -1;
			//			ID = -1;
			//		}
			//	} triangleAdj_t;
			//
			//	TriangleAdjacencyGraph(const vector<TriangleMesh3D::triangle_t> &triangles, int numVertices);

			//	void checkTriangleNormal(int currNode, map<int, bool> &initialEdgeMap);
			//	void orientNormals(map<int, bool> &initialEdgeMap);
			//	static int hash(int i, int j, int numVertices) {
			//		return numVertices*j + i;
			//	}


			//	const vector<triangleAdj_t> & getTriangles() const {
			//		return m_triangleAdjacencyVector;
			//	}
			//private:
			//	
			//	vector<triangleAdj_t> m_triangleAdjacencyVector;
			//	vector<bool> m_visitedTriangles;
			//	vector<bool> m_flippedTriangles;
			//	//Necessary for hash calculation
			//	int m_numVertices;

			//	void findAdjacentTriangles(int ithTriangle);
			//	bool alreadyConnected(int t1, int t2) {
			//		for (int i = 0; i < m_triangleAdjacencyVector[t1].degree; i++) {
			//			if (m_triangleAdjacencyVector[t1].neighbors[i] == &m_triangleAdjacencyVector[t2]) {
			//				return true;
			//			}
			//		}
			//		return false;
			//	}
			//	int getBorderNode(int initialIndex = 0) {
			//		for (int i = initialIndex; i < m_triangleAdjacencyVector.size(); i++) {
			//			if (m_triangleAdjacencyVector[i].degree < 3)
			//				return i;
			//		}
			//		return -1;
			//	}

			//	void flipTriangleEdges(int ithTriangle) {
			//		triangleAdj_t &triangle = m_triangleAdjacencyVector[ithTriangle];
			//		swap(triangle.indices[0], triangle.indices[2]);
			//	}
			//};

			int findDiscontinuity(const vector<pair<unsigned int, unsigned int>> &edges, int initialIndex = 0);

			bool comparePairs(pair<Vector3, unsigned int *> a, pair<Vector3, unsigned int *> b);

			/** Simplifies redundant vertices and faces representation. This redundant structure is created from the assembly 
			/** of different cut-faces of a cut-cell*/
			pair < vector<Vector3>, vector<vector<unsigned int>> > simplifyMesh(const vector<Vector3> &vertices, const vector<vector<unsigned int>> & faces);


			/** This function converts a cut-cell to a oriented outward-normal-oriented triangleMesh. It also fixes the normals 
			/** of the original cut-cell to be pointing outward. It makes easier to do this two tasks here since the linking between
			/** cut-cells and the triangle mesh triangles is not trivial */
			//TriangleMesh3D convertCutCell(CutVoxel &cutVoxel, CutCells3D *pCutCells);

			/** Calculates the centroid of a line. Works for non-convex cells. The line must be closed.*/
			//Vector2 calculateCentroid(const vector<Vector2> &points);

			/** Calculates the centroid of a line when its points are lying on the same plane (3-D). Works for non-convex cells.
				The line must be closed. */
			template <class VectorT>
			VectorT calculateCentroid(const vector<VectorT> &points);

			/** Calculates the signed distance function of a line when its points are lying on the same plane (3-D). 
			Works for non-convex cells. The line must be closed. */
			template <class VectorT>
			DoubleScalar signedDistanceFunction(const VectorT &position, const vector<VectorT> &points, faceLocation_t faceLocation) {
				DoubleScalar minDistance = FLT_MAX;
				int ithEdge = -1;
				for (int i = 0; i < points.size(); i++) {
					int nextI = roundClamp<int>(i + 1, 0, points.size());
					DoubleScalar tempDistance = distanceToLineSegment(position, points[i], points[nextI]);
					VectorT edgeCentroid = (points[i] + points[nextI])*0.5;

					if (tempDistance < minDistance) {
						minDistance = tempDistance;
						ithEdge = i;
					}
				}
				int j = 0;
				int nextIthEdge = roundClamp<int>(ithEdge + 1, 0, points.size());
				VectorT edgeNormal = -getEdgeNormal(points[ithEdge], points[nextIthEdge], faceLocation);
				VectorT edgeCentroid = (points[nextIthEdge] + points[ithEdge])*0.5;
				if (edgeNormal.dot(position - edgeCentroid) < 0) {
					minDistance = -minDistance;
				}

				return minDistance;
			}

			template <class VectorT>
			VectorT getEdgeNormal(const VectorT &e1, const VectorT &e2, faceLocation_t faceLocation) {
				VectorT edgeVec = e2 - e1;
				switch (faceLocation) {
				case leftFace:
					return VectorT(0, edgeVec.z, -edgeVec.y);
					break;
				case bottomFace:
					return VectorT(-edgeVec.z, 0, edgeVec.x);
					break;
				case backFace:
					return VectorT(-edgeVec.y, edgeVec.x, 0);
					break;

				default:
					throw exception("Invalid face location");
					return VectorT(0, 0, 0);
				break;
				}
			}
		}
	}

	template<class VectorT>
	VectorT Chimera::Meshes::MeshUtils::calculateCentroid(const vector<VectorT>& points) {
		Scalar signedAreaXY = 0, signedAreaXZ = 0, signedAreaYZ = 0;
		Vector2 centroidXY, centroidXZ, centroidYZ;
		VectorT centroid;

		for (int i = 0; i < points.size(); i++) {
			int nextI = roundClamp<int>(i + 1, 0, points.size());
			VectorT initialEdge = points[i];
			VectorT finalEdge = points[nextI];
			VectorT edgeCentroid = (points[i] + points[nextI])*0.5;

			if (initialEdge == finalEdge)
				continue;

			//Projecting onto the XY plane
			Vector2 initialEdgeXY(initialEdge.x, initialEdge.y);
			Vector2 finalEdgeXY(finalEdge.x, finalEdge.y);

			Scalar currAreaXY = initialEdgeXY.cross(finalEdgeXY);
			signedAreaXY += currAreaXY;
			centroidXY.x += (edgeCentroid.x)*currAreaXY;
			centroidXY.y += (edgeCentroid.y)*currAreaXY;

			//Projecting onto the XZ plane
			Vector2 initialEdgeXZ(initialEdge.x, initialEdge.z);
			Vector2 finalEdgeXZ(finalEdge.x, finalEdge.z);

			Scalar currAreaXZ = initialEdgeXZ.cross(finalEdgeXZ);
			signedAreaXZ += currAreaXZ;
			centroidXZ.x += (edgeCentroid.x)*currAreaXZ;
			centroidXZ.y += (edgeCentroid.z)*currAreaXZ;

			//Projecting onto the YZ plane
			Vector2 initialEdgeYZ(initialEdge.y, initialEdge.z);
			Vector2 finalEdgeYZ(finalEdge.y, finalEdge.z);

			Scalar currAreaYZ = initialEdgeYZ.cross(finalEdgeYZ);
			signedAreaYZ += currAreaYZ;
			centroidYZ.x += (edgeCentroid.y)*currAreaYZ;
			centroidYZ.y += (edgeCentroid.z)*currAreaYZ;
		}

		signedAreaXZ *= 0.5f;
		signedAreaXY *= 0.5f;
		signedAreaYZ *= 0.5f;

		if (abs(signedAreaXZ) < 1e-4  && abs(signedAreaXY) < 1e-4) {
			centroid.x = points[0].x;
		}
		else if (abs(signedAreaXZ) > abs(signedAreaXY)) {
			centroid.x = centroidXZ.x / (3.0f*signedAreaXZ);
		}
		else {
			centroid.x = centroidXY.x / (3.0f*signedAreaXY);
		}

		if (abs(signedAreaXY) < 1e-4  && abs(signedAreaYZ) < 1e-4) {
			centroid.y = points[0].y;
		}
		else  if (abs(signedAreaXY) > abs(signedAreaYZ)) {
			centroid.y = centroidXY.y / (3.0f*signedAreaXY);
		}
		else {
			centroid.y = centroidYZ.x / (3.0f*signedAreaYZ);
		}

		if (abs(signedAreaXZ) < 1e-4  && abs(signedAreaYZ) < 1e-4) {
			centroid.z = points[0].z;
		}
		else if (abs(signedAreaXZ) > abs(signedAreaYZ)) {
			centroid.z = centroidXZ.y / (3.0f*signedAreaXZ);
		}
		else {
			centroid.z = centroidYZ.y / (3.0f*signedAreaYZ);
		}

		return centroid;
	}
}

#endif