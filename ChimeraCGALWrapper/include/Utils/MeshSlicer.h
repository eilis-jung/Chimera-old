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


#ifndef __CHIMERA_CGAL_WRAPPER_MESH_SLICER_H_
#define __CHIMERA_CGAL_WRAPPER_MESH_SLICER_H_

#pragma once
#include "ChimeraCore.h"
#include "CGALConfig.h"
#include "Mesh/MeshesCoreDefs.h"
#include "Mesh/LineMesh.h"
//#include "ChimeraMesh.h"

namespace Chimera {

	using namespace Meshes;

	namespace CGALWrapper {

		typedef std::vector<Kernel::Point_3> Polyline_type;
		typedef std::list< Polyline_type > Polylines;

		template <class VectorType> 
		class MeshSlicer : public Core::Singleton<MeshSlicer<VectorType>> {

		public:

			#pragma region Constructors
			MeshSlicer(CgalPolyhedron *pPolyhedron, const dimensions_t gridDimensions, Scalar dx) : m_gridDimensions(gridDimensions) {
				m_pCgalPoly = pPolyhedron;
				m_gridDx = dx;
			}
			#pragma endregion

			#pragma region Functionalities
			/** Slices the mesh working only on the CGAL polyhedron structure*/
			void sliceMesh();

			/** Initialize internal LineMesh slices structures, uses vertices already initialized */
			void initializeLineMeshes(vector<Vertex<VectorType> *> vertices);
			#pragma endregion

			#pragma region AccessFunctions
			const map<uint, vector<LineMesh<VectorType> *>> & getXYLineMeshes() const {
				return m_XYLineMeshes;
			}

			const map<uint, vector<LineMesh<VectorType> *>> & getXZLineMeshes() const {
				return m_XZLineMeshes;
			}

			const map<uint, vector<LineMesh<VectorType> *>> & getYZLineMeshes() const {
				return m_YZLineMeshes;
			}
			#pragma endregion

		private:
			#pragma region ClassMembers
			CgalPolyhedron *m_pCgalPoly;

			/** Line meshes organized by slice index. Since multiple poly meshes are possible, we have to store the ones
			* that are on the same slice on the same structure, so cut-cells can be properly initialized */
			map<uint, vector<LineMesh<VectorType> *>> m_XYLineMeshes;
			map<uint, vector<LineMesh<VectorType> *>> m_YZLineMeshes;
			map<uint, vector<LineMesh<VectorType> *>> m_XZLineMeshes;

			map<uint, bool> m_visitedVertices;

			vector<Vertex<VectorType> *> m_vertices;

			/** Grid Spacing and dimensions */
			Scalar m_gridDx;
			dimensions_t m_gridDimensions;
			#pragma endregion

			#pragma region PrivateFunctionalities
			void sliceMesh(const VectorType &normal, const VectorType &origin, Scalar increment, uint numSlices);
			void followThroughVerticesOnPlane(CgalPolyhedron::Vertex *pCurrVertex, CgalPolyhedron::Vertex *pInitialVertex, uint ithPlane, faceLocation_t planeType, vector<VectorType> &linePoints);
			void followThroughVerticesOnPlane(Vertex<VectorType> *pCurrVertex, Vertex<VectorType> *pInitialVertex, uint ithPlane, faceLocation_t planeType, vector<Vertex<VectorType> *> &verticesVec, vector<Edge<VectorType> *> &edgesVec);
			void initializePlaneLineMeshes(uint ithPlane, faceLocation_t planeType);
			#pragma endregion

		};
	}
}

#endif