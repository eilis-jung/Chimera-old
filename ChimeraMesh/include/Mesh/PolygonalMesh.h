//  Copyright (c) 2017, Vinicius Costa Azevedo
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

#ifndef _CHIMERA_OBJ_MESH_IMPORTER_
#define _CHIMERA_OBJ_MESH_IMPORTER_

#pragma once
#include "Mesh.h"
#include "Mesh/LineMesh.h"
#include "Mesh/Volume.h"
#include "Mesh/MeshUtils.h"
#include "ChimeraCGALWrapper.h"

namespace Chimera {

	namespace Meshes {

		/** A polygonal mesh has faces with two attached half-faces each */
		template <class VectorType>
		class PolygonalMesh : public Mesh<VectorType, Face> {

		public:
	
			#pragma region Constructors
				PolygonalMesh(const VectorType &position, const string &objFilename, dimensions_t gridDimensions = dimensions_t(0, 0, 0), Scalar gridDx = 0.0f, bool perturbPoints = true);
			#pragma endregion

			#pragma region AccessFunctions
				CGALWrapper::CgalPolyhedron * getCGALMesh() {
					return m_pCgalPolyhedron;
				}

				CGALWrapper::MeshSlicer<VectorType> * getMeshSlicer() {
					return m_pMeshSlicer;
				}

				const vector<uint> & getPatchesIndices(uint i, uint j, uint k) const {
					return m_regularGridPatches(i, j, k);
				}

			#pragma endregion	
		protected:

			#pragma region ClassMembers
			CGALWrapper::CgalPolyhedron *m_pCgalPolyhedron;
			
			/** Mesh slicer: responsible for slicing the mesh :) */
			CGALWrapper::MeshSlicer<VectorType> *m_pMeshSlicer;


			/** Auxiliary structure used for organizing poly-patches inside a regular grid. It stores local elements that are
			inside a regular grid that this polymesh is embedded. If it stores a greater than zero vector inside the
			array, it means that regular-grid location contains a part of the polymesh (the specific part is indicated by
			the indices) */
			Array3D<vector<uint>> m_regularGridPatches;


			/** Ghost vertices map: auxiliary structure that will be used to initialize half-edges propertly. This structure
				allows us to have mismatching velocities for free-slip */
			map<uint, Vertex<VectorType> *> m_ghostVerticesMap;
			#pragma endregion

			#pragma region PrivateFunctionalities
			void classifyVertices(Scalar gridDX);

			void computeVerticesNormals();

			void initializeRegularGridPatches(Scalar dx);

			void initializeGhostVertices();
			void createHalfFaces();
			#pragma endregion
		};
	}
}
#endif