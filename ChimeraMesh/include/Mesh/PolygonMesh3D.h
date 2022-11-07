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
//#ifndef __CHIMERA_POLYGON_MESH_3D_H_
//#define __CHIMERA_POLYGON_MESH_3D_H_
//
//#pragma once
//
//#include "ChimeraCore.h"
//#include "ChimeraCGALWrapper.h"
//
//
//#include "Mesh/TriangleMesh3D.h"
//
//namespace Chimera {
//	using namespace Math;
//
//	namespace Meshes {
//		
//		class PolygonMesh3D {
//		public:
//
//			typedef enum nodeType_t {
//				gridNode,
//				geometryNode,
//				centroidNode,
//				mixedNode
//			} nodeType_t;
//
//			typedef enum polygonType_t {
//				gridFacePolygon,
//				geometryPolygon
//			};
//
//			typedef struct polygon_t {
//				vector<pair<unsigned int, unsigned int>> edges;
//				Vector3D centroid;
//				Vector3D normal;
//				polygonType_t polygonType;
//			};
//
//			PolygonMesh3D(const vector<Vector3D > &points, const vector<Vector3D> &normals, const vector<faceLocation_t> &faceLocations, vector<vector<pair<unsigned int, unsigned int>>> &polygons, Scalar dx);
//			PolygonMesh3D(const vector<Vector3D > &points, const vector<polygon_t> &polygons, Scalar dx);
//
//			const vector<Vector3D> & getPoints() const {
//				return m_points;
//			}
//			
//			const vector<Vector3D> & getPointsNormals() const {
//				return m_pointsNormals;
//			}
//
//			
//
//			const vector<polygon_t> & getPolygons() const {
//				return m_polygons;
//			}
//
//			const vector<nodeType_t> & getNodeTypes() const {
//				return m_nodeTypes;
//			}
//
//			unsigned int edgeHash(unsigned int i, unsigned int j) const {
//				if (i < j)
//					return m_points.size()*i + j;
//				return m_points.size()*j + i;
//			}
//
//			unsigned int edgeHash2(unsigned int i, unsigned int j) const {
//				return m_points.size()*j + i;
//			}
//
//			bool isInsideMesh(const Vector3D & position);
//
//			Vector3D getFaceNormal(unsigned int faceIndex);
//
//			const dimensions_t & getCutCellLocation() const {
//				return m_regularGridDimensions;
//			}
//
//			TriangleMesh3D * getTriangleMesh() {
//				return m_pTriangleMesh;
//			}
//
//			const Vector3D & getEdgeNormal(unsigned int i, unsigned int j) const {
//				return m_edgeNormals.at(edgeHash(i, j));
//			}
//
//		private:
//
//			typedef struct polygonPointers_t {
//				vector<int *> pointsIndexes;
//			};
//
//			vector<Vector3D> m_points;
//			vector<Vector3D> m_pointsNormals;
//			map<unsigned int, Vector3D> m_edgeNormals;
//			vector<polygon_t> m_polygons;
//			vector<nodeType_t> m_nodeTypes;
//			CGALWrapper::CgalPolyhedron m_cgalPoly;
//			CGALWrapper::AABBTree *m_pAABBTree;
//			dimensions_t m_regularGridDimensions;
//			TriangleMesh3D *m_pTriangleMesh;
//			map<unsigned int, pair<unsigned int, unsigned int>> m_edgesPolygonMap;
//			map<unsigned int, int> m_edgesPolygonCount;
//
//			bool removedUnusedVertices();
//			bool revertGeometryNormals();
//
//			void initializeVerticesNormals();
//			void initializeEdgesNormals();
//		};
//	}
//
//}
//#endif