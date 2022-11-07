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

#ifndef __CHIMERA_CUT_VOXEL__
#define __CHIMERA_CUT_VOXEL__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraMesh.h"
#include "CutCells/CutFace.h"


namespace Chimera {
	namespace Data {

		class CutVoxel {
		public:
			/** Cell ID: Usually it is the position on the specialCells vector; however this can be used in order to 
			 ** re-organize special cells*/
			int ID;

			/** Faces pointers. Each cell face has a representation on the cutfaces location vector too*/
			vector<CutFace<Vector3D> *> cutFaces;
			
			vector<faceLocation_t> cutFacesLocations;

			/** Cell centroid */
			Vector3D centroid;

			/** Regular grid index */
			dimensions_t regularGridIndex;

			/** Vertices and edges. Its seems duplicated, since edges and vertices can be accessed through cutfaces,
			** but having this explicit storage helps creating polygons ready to be converted to CGAL platform. 
			** Also edges are separated by faces (hence the double vector), which also simplifies the conversion
			** to CGAL plataform. */
			vector<Vector3D> m_vertices;
			vector<vector<pair<unsigned int, unsigned int>>> m_edgeIndices;

			/** This structure represents the correspondence map between points on the geometry and the triangles that it belongs to.
			** Currently this structure represents only the triangles that are on the geometry, but it can also represent grid triangles. */
			map<int, vector<int>> pointsToTriangleMeshMap;
			
			/** This structure represents the correspondence map between points faces edges and the triangles mesh edges that are on the same.*/
			map<int, vector<int>> facesEdgesToMeshMap;

			/** This structure represents the indices correspondence between the ordered list of geometry faces of the cutvoxel and geometry faces
			/** of the triangle mesh.*/
			vector<int> geometryFacesToMesh;

			Mesh<Vector3D> *m_pMesh;

			void initializeMesh(Scalar dx, const vector<Vector3D> &normals, bool onEdgeMixedNode = false);
			void insertCutFace(CutFace<Vector3D> *pCutFace, faceLocation_t faceLocation) {
				cutFaces.push_back(pCutFace);
				cutFacesLocations.push_back(faceLocation);
			}

			void initializePointsToMeshMap(const Mesh<Vector3D> &mesh);
			void initializeFacesEdgeMap(const Mesh<Vector3D> &mesh);
			int findOnMesh(const Vector3D &point, const Mesh<Vector3D> &mesh);

			const map<int, vector<int>> & getPointsToMeshMap() const {
				return pointsToTriangleMeshMap;
			}

			Mesh<Vector3D> * getMesh() const {
				return m_pMesh;
			}

			bool danglingVoxel;

			CutVoxel() {
				danglingVoxel = false;
				ID = -1;
				m_pMesh = NULL;
			}
		};
	}
}
#endif