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
//
//#ifndef __RENDERING_POLYGON_SURFACE_H_
//#define __RENDERING_POLYGON_SURFACE_H_
//
//#include "ChimeraData.h"
//#include "ChimeraCGALWrapper.h"
//
///************************************************************************/
///* Rendering                                                            */
///************************************************************************/
//#include "ChimeraRenderingCore.h"
//#include "Primitives/Object3D.h"
//#include "Resources/Shaders/GLSLShader.h"
//
//namespace Chimera {
//	namespace Rendering {
//
//
//
		//class PolygonSurface : public Object3D {
//
//			typedef pair<unsigned int, unsigned int> polyEdge;
//
//			//This hash doesn't depend on the orientation of the edge. Thus its not a halfedge hash
//			unsigned int edgeHash(unsigned int i, unsigned int j) const {
//				if (i < j)
//					return m_vertices.size()*i + j;
//				return m_vertices.size()*j + i;
//			}
//
//		public:
//			/************************************************************************/
//			/* ctors                                                                */
//			/************************************************************************/
//			PolygonSurface(int polygonID, const Vector3 &position, const Vector3 &scale, const string &surfaceFilename, Scalar dx);
//
//			PolygonSurface(int polygonID, const pair<vector<CGALWrapper::CgalPolyhedron::Point_3>, vector<vector<polyEdge>>> &polygonInfo);
//
//			/************************************************************************/
//			/* Functionalities                                                      */
//			/************************************************************************/
//			void draw();
//			void update(Scalar dt);
//			void updateInitialRotation();
//			void updateLocalDataStructures();
//			void reinitializeVBOBuffers();
//			void fixDuplicatedVertices();
//			bool isInside(const Vector3D &point);
//			void treatVerticesOnGridPoints();
//			void treatVerticesOnGridEdges();
//			void simplifyMesh();
//
//			bool doesEdgeIntersect(const Vector3 &e1, const Vector3 &e2);
//			bool doesEdgeIntersect(const Vector3 &e1, const Vector3 &e2, Vector3 &intersectionPoint);
//
//			vector<unsigned int> getListOfPossibleTrianglesCollision(const Vector3 &initialPoint, const Vector3 &finalPoint, Scalar distanceThreshold);
//
//			#pragma region AccessFunctions
//			int getPolygonID() const {
//				return m_ID;
//			}
//
//			CGALWrapper::CgalPolyhedron * getCGALPolyehedron() {
//				return &m_cgalPoly;
//			}
//			CGALWrapper::CgalPolyhedron *getInitialiCGALPolyhedron() {
//				return &m_initialCgalPoly;
//			}
//
//			const vector<Vector3D> & getVertices() const {
//				return m_vertices;
//			}
//			const vector <simpleFace_t> & getFaces() const {
//				return m_faces;
//			}
//			const vector<Vector3D> & getFacesCentroids() const {
//				return m_facesCentroids;
//			}
//			const vector<Vector3D> & getNormals() const {
//				return m_facesNormals;
//			}
//
//			const vector<Vector3D> & getVerticesNormals() const {
//				return m_verticesNormals;
//			}
//
//			const Vector3D & getCentroid() const {
//				return m_centroid;
//			}
//			const Vector3D & getEdgeNormal(unsigned int edge1, unsigned int edge2)  {
//				unsigned int edgeHashVar = edgeHash(edge1, edge2);
//				if (m_edgeNormals.find(edgeHashVar) == m_edgeNormals.end()) {
//					throw exception("PolygonSurface: edge normal not found");
//				}
//				return m_edgeNormals[edgeHashVar];
//			}
//
//			Vector3D getClosestPoint(const Vector3D &point);
//
//			void getClosestPoint(const Vector3D &queryPoint, Vector3D &resultingPoint, Vector3D &resultingNormal);
//
//			#pragma endregion
//		private:
//			#pragma region ClassMembers
//			int m_ID;
//			CGALWrapper::CgalPolyhedron m_cgalPoly;
//			CGALWrapper::CgalPolyhedron m_initialCgalPoly;
//			CGALWrapper::AABBTree *m_pInitialCgalPolyTree;
//			CGALWrapper::AABBTree *m_pCgalPolyTree;
//			Scalar m_dx;
//
//			//Wireframe shader
//			shared_ptr<GLSLShader> m_pWireframeShader;
//			shared_ptr<GLSLShader> m_pPhongShader;
//			shared_ptr<GLSLShader> m_pPhongWireframeShader;
//			GLint m_lightPosLoc;
//			GLint m_lightPosLocWire;
//
//			/** Vertex buffers */
//			GLuint *m_pVertexVBO;
//			GLuint *m_pIndicesVBO;
//			GLuint *m_pPolygonVAO;
//
//			vector<Vector3D> m_vertices;
//			vector<Vector3D> m_verticesNormals;
//			vector<DoubleScalar> m_verticesWeights;
//			vector<simpleFace_t> m_faces;
//			vector<Vector3D> m_facesCentroids;
//			vector<Vector3D> m_facesNormals;
//			map<unsigned int, Vector3D> m_edgeNormals;
//			map<unsigned int, DoubleScalar> m_edgeWeights;
//			map<int, int> m_oldVertexMap;
//			/** Total number of vertices passed to glDrawElements */
//			int m_totalNumberOfVertices;
//			bool m_triangleMesh;
//			DoubleScalar m_rotationAngle;
//			Vector3D m_centroid;
//			Vector3D m_initialCentroid;
//
//			map<int, int> m_edgesRefCount;
//			
//			#pragma endregion
//
//			#pragma region PrivateFunctionalities
//			void updateEdgesCount();
//			#pragma endregion
//			
//			#pragma region InitializationFunctions
//			void initializeFaceCentroids();
//			void initializeFaceNormals();
//			/** Vertex buffer initialization*/
//			void initializeVBOs();
//			void initializeVAOs();
//			/** Initialize vertices normals using weighted pseudonormals */
//			void initializeVerticesNormals();
//			/** Initialize edges normals using weighted pseudonormals */
//			void initializeEdgesNormals();
//			#pragma endregion
//
//		};
//
//		#pragma region InternalClasses
//		typedef struct MeshPatchMap {
//			vector<unsigned int> faces;
//			bool danglingPatch;
//			bool visited;
//			PolygonSurface *pPolySurface;
//
//			MeshPatchMap() {
//				danglingPatch = false;
//				visited = false;
//			}
//		};
//
//		
//		#pragma endregion
//	}
//}
//#endif