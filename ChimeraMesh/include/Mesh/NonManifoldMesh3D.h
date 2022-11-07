
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

#ifndef _CHIMERA_DATA_NON_MANIFOLD_MESH_H_
#define _CHIMERA_DATA_NON_MANIFOLD_MESH_H_

#pragma once

#include "ChimeraCore.h"
#include "ChimeraMath.h"

#include "Grids/CutCells3D.h"
#include "Mesh/TriangleMesh3D.h"
#include "ChimeraCGALWrapper.h"

namespace Chimera {
	namespace Rendering {
		class PolygonSurface;
		typedef struct MeshPatchMap;
	}

	namespace Meshes {

		class NonManifoldMesh3D {
		public:

			typedef struct face_t {
				vector<face_t *> neighbors;
				//Explicitly storing a pair of edge indices, since adjacent edges may be disconnected
				vector<pair<unsigned int, unsigned int>> edgeIndices;
				unsigned int ID;
				unsigned int degree;
				CutFace<Vector3D> *pCutFace;
				//Valid Interior point used for isLeft comparisons
				Vector3D interiorPoint;
				bool m_smalledgeFace;
				unsigned int meshPatchID;
				Rendering::PolygonSurface *pPolygonSurface; //If it is a geometry face, it has a polygonsurface attached

				faceLocation_t faceLocation;

				face_t(unsigned int g_ID) {
					degree = 0;
					ID = g_ID;
					pCutFace = NULL;
					m_smalledgeFace = false;
					pPolygonSurface = NULL;
					meshPatchID = -1;
				}
			} face_t;


			NonManifoldMesh3D(dimensions_t cellDim, const vector<Rendering::MeshPatchMap *> &patchMaps, CutCells3D *pCutCells);


			vector<CutVoxel> split(unsigned int totalCutVoxels, bool onEdgeMixedNodes);

			const vector<face_t> & getFaces() const {
				return m_faces;
			}


		private:
			vector<Vector3D> m_vertices;
			vector<face_t> m_faces;
			vector<bool> m_visitedFaces;
			vector<bool> m_flippedTriangles;
			CutCells3D *m_pCutCells;
			dimensions_t m_voxelDim;
			unsigned int m_initialVertexOffset;
			Scalar m_distanceProximity;
			vector<Rendering::MeshPatchMap *> m_patchMaps;

			map<unsigned int, vector<face_t *>> m_edgeMap;

			void breadthFirstSearch(face_t &face, CutVoxel &nCutVoxel, int regionTag);

			//This hash doesnt depend on the orientation of the edge. Thus its not a halfedge hash
			unsigned int edgeHash(unsigned int i, unsigned int j) {
				if (i < j)
					return m_vertices.size()*i + j;
				return m_vertices.size()*j + i;
			}

			// Calculates the vertex position inside vertices vector of a given grid point dimension
			// gridPointDim must be greater than m_voxelDim
			unsigned int gridPointHash(const dimensions_t &gridPointDim) {
				return (gridPointDim.z - m_voxelDim.z) * 4 + (gridPointDim.y - m_voxelDim.y) * 2 + (gridPointDim.x - m_voxelDim.x);
			}

			//Used to add vertices that are not on top of grid points. Returns the index of the newly added vertex
			unsigned int addVertex(const Vector3D &newVertex);


			//Add current face to the faces structure
			void addFace(CutFace<Vector3D> *pFace, faceLocation_t faceLocation);

			//Finds a non-visited face from the faces vector
			int findNonVisitedFace() {
				for (int i = 0; i < m_faces.size(); i++) {
					if (!m_visitedFaces[i] && m_faces[i].faceLocation != geometryFace)
						return i;
				}
				//All faces are visited
				return -1;
			}

			bool isInsideFace(const Vector3D & point, const face_t &face);

			DoubleScalar computeSignedDistanceFunction(const Vector3D & point, Rendering::PolygonSurface *pPolySurface);
			int computeSignedDistanceFunctionTag(const Vector3D & point, Rendering::PolygonSurface *pPolySurface);

			void addDoubleSidedGeometryFaces(CutVoxel &cutVoxel, Rendering::MeshPatchMap *pPatchMap, int patchMapID, Scalar dx);

			Vector3D getFaceNormal(faceLocation_t facelocation);
			Vector2D convertToVec2(const Vector3D &vec, faceLocation_t facelocation);
			Vector2D projectIntoPlane(const Vector3D &vec, faceLocation_t facelocation);

			void removeFaceFromEdgeMap(unsigned int i, unsigned int j, int faceID);
		};
	}
}


#endif