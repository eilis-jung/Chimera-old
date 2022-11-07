////  Copyright (c) 2013, Vinicius Costa Azevedo
////	All rights reserved.
////
////	Redistribution and use in source and binary forms, with or without
////	modification, are permitted provided that the following conditions are met: 
////
////1. Redistributions of source code must retain the above copyright notice, this
////	list of conditions and the following disclaimer. 
////	2. Redistributions in binary form must reproduce the above copyright notice,
////	this list of conditions and the following disclaimer in the documentation
////	and/or other materials provided with the distribution. 
////
////	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
////	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
////	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
////	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
////	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
////	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
////LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
////	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
////	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
////	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////
////	The views and conclusions contained in the software and documentation are those
////	of the authors and should not be interpreted as representing official policies, 
////	either expressed or implied, of the FreeBSD Project.
//
//#ifndef __CHIMERA_TRIANGLE_MESH_H_
//#define __CHIMERA_TRIANGLE_MESH_H_
//
//#include "ChimeraCore.h"
////Cross reference, careful
//#include "CutCells/CutVoxel.h"
//
//#pragma once
//
//namespace Chimera {
//	using namespace Core;
//
//	namespace Rendering {
//		class PolygonSurface;
//	}
//	namespace Meshes {
//
//		class Edge {
//			int initialPointIndex;
//			int finalPointIndex;
//		};
//
//		namespace CutCells {
//			class CutCells3D;
//		}
//		
//
//		class TriangleMesh3D {
//		public:
//			typedef enum nodeType_t {
//				gridNode,
//				geometryNode,
//				centroidNode,
//				mixedNode
//			}nodeType_t;
//
//			typedef enum triangleType_t {
//				gridFaceTriangle,
//				geometryTriangle
//			};
//
//			typedef struct triangle_t {
//				int pointsIndexes[3];
//				Vector3D normal;
//				Vector3D centroid;
//				//Links the geometry triangle to the cut-voxel's geometry face
//				int cutFaceID;
//				Rendering::PolygonSurface *pPolygonSurface;
//
//				triangleType_t triangleType;
//
//				triangle_t() : normal(0, 0, 0){
//					pointsIndexes[0] = pointsIndexes[1] = pointsIndexes[2] = NULL;
//					triangleType = gridFaceTriangle;
//					pPolygonSurface = NULL;
//					cutFaceID = -1;
//				}
//			};
//
//			//Special trignale mesh for boundary cells interpolation. Only has its vertices initialized
//			TriangleMesh3D(const vector<Vector3D> &vertices, Scalar dx);
//
//			TriangleMesh3D(const CutCells::CutVoxel &cutVoxel, CutCells::CutCells3D *pCutCells);
//
//			TriangleMesh3D(const vector<Vector3D> &points, const vector<triangle_t> & triangles, Scalar dx, const CutCells::CutVoxel &cutVoxel);
//			TriangleMesh3D(const vector<Vector3D> &points, const vector<triangle_t> & triangles, Scalar dx);
//
//
//			const vector<Vector3D> & getPoints() const {
//				return m_points;
//			}
//
//			const vector<Vector3D> & getNormals() const {
//				return m_normals;
//			}
//
//			const vector<triangle_t> & getTriangles() const {
//				return m_triangles;
//			}
//
//			const vector<nodeType_t> & getNodeTypes() const {
//				return m_nodeTypes;
//			}
//
//			const Vector3D & getCentroid() const {
//				return m_centroid;
//			}
//
//
//			const vector<vector<int>> & getFaceLineIndices() const {
//				return m_faceLinesIndices;
//			}
//			const vector<Scalar> & getFaceLinesTotalLengths() const {
//				return m_faceLinesTotalSizes;
//			}
//
//			bool isRectangularFace(const CutCells::CutFace<Vector3D> *pCutFace);
//
//			bool isInsideMesh(const Vector3D & position);
//			bool isInsideMesh2(const Vector3D & position);
//			bool isInsideMesh3(const Vector3D & position);
//
//			Scalar signedDistanceFunction(const Vector3D & point);
//
//			/** Face lines are auxiliary structures that are used to interpolate velocities on the free-slip case*/
//			void initializeFaceLines(const CutVoxel &cutVoxel);
//			void clearFaceLines();
//
//			void addCutFaceToMixedNode(int localMixedNodeID, CutFace<Vector3D> *pFace) {
//				m_mixedNodeFaces[localMixedNodeID].push_back(pFace);
//			}
//
//			map<int, vector<CutFace<Vector3D> *>> & getCutFaceToMixedNodeMap() {
//				return m_mixedNodeFaces;
//			}
//
//			map<int, vector<Vector3D>> & getCutFaceNormalsToMixedNodeMap() {
//				return m_mixedNodeFacesNormals;
//			}
//
//			//Returns if the triangle mesh is only a container for its vertices used on inverse distance interpolation
//			bool isEmpty() const {
//				return m_triangles.size() == 0;
//			}
//		private:
//			#pragma region ComparisionFunctions
//			static bool comparePairs(pair<Vector3D, int *> a, pair<Vector3D, int *> b);
//			static bool comparePairs_b(pair<Vector3D, int *> a, pair<Vector3D, int *> b);
//			static bool uniqueVectors(Vector3D a, Vector3D b);
//			#pragma endregion
//
//			#pragma region PrivateFunctionalities
//			void initializeVertexNormals(const CutVoxel &cutVoxel);
//
//			void alignTriangleNormals();
//			/** Since the triangle mesh class uses points from the Non-Manifold mesh it has unused points
//			** that have to be removed in order to keep the data structures tight */
//			void removeUnusedPoints();
//
//			void followNextPointOnFace(const map<int, vector<int>> & map, int currID, int currFaceLocationID);
//			#pragma endregion
//
//			typedef struct trianglePointers_t {
//				int *pointsIndexes[3];
//
//				trianglePointers_t() {
//					pointsIndexes[0] = pointsIndexes[1] = pointsIndexes[2] = NULL;
//				}
//			};
//
//			vector<Vector3D> m_normals;
//			vector<Vector3D> m_points;
//			vector<triangle_t> m_triangles;
//			vector<nodeType_t> m_nodeTypes;
//			vector<vector<int>> m_faceLinesIndices;
//			vector<Scalar> m_faceLinesTotalSizes;
//			Vector3D m_centroid;
//			Scalar m_gridSpacing;
//			map<int, bool> m_tempVisitedNodes;
//			map<int, vector<CutFace<Vector3D> *>> m_mixedNodeFaces;
//			map<int, vector<Vector3D>> m_mixedNodeFacesNormals;
//		};
//	}
//
//}
//#endif