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

#ifndef __CHIMERA_CUTCELLS_3D_H__
#define __CHIMERA_CUTCELLS_3D_H__

#pragma once

#include "ChimeraCore.h"
#include "CutCells/CutCellsBase.h"
#include "ChimeraMesh.h"

namespace Chimera {
	using namespace Meshes;

	namespace CutCells {
		
		template <class VectorType>
		class CutCells3D : public CutCellsBase<VectorType> {
		
		public:
			
			#pragma endregion
			#pragma region Constructors
			/** This constructor builds a planar mesh with multiple initialized line meshes. */
			CutCells3D(extStructures_t extStructures, const vector<LineMesh<VectorType> *> &lineMeshes, Scalar gridSpacing, const dimensions_t &gridDimensions, faceLocation_t faceLocation);
			
			#pragma endregion

			#pragma region AccessFunctions
			/** Gets 3-D cut-cell by position */
			uint getCutCellIndex(const VectorType &position) override;
			#pragma endregion

			#pragma region Functionalities
			void initialize() override;
			#pragma endregion
		protected:
			#pragma region PrivateFunctionalities
			void mirrorFaces();
			
			#pragma endregion
			#pragma region PureVirtualFunctions
			Vertex<VectorType> *createVertex(uint i, uint j) {
				//Since we are replacing a dummy pointer, delete the dummy pointer in the location that we are creating
				//the vertex
				if (m_nodeVertices(i, j) != nullptr) {
					delete m_nodeVertices(i, j);
				}

				if (m_faceLocation == XYFace) {
					VectorType vertexPosition(i*m_gridSpacing, j*m_gridSpacing, m_sliceIndex*m_gridSpacing);
					return new Vertex<VectorType>(vertexPosition, gridVertex);
				}
				else if (m_faceLocation == XZFace) {
					VectorType vertexPosition(i*m_gridSpacing, m_sliceIndex*m_gridSpacing, j*m_gridSpacing);
					return new Vertex<VectorType>(vertexPosition, gridVertex);
				}
				else if (m_faceLocation == YZFace) {
					VectorType vertexPosition(m_sliceIndex*m_gridSpacing, j*m_gridSpacing, i*m_gridSpacing);
					return new Vertex<VectorType>(vertexPosition, gridVertex);
				}
			}

			Edge<VectorType> * createGridEdge(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2, halfEdgeLocation_t halfEdgeLocation) override;

			/** Sorts vertices based on their xyz positions. Has to consider the plane orientation that these cut-cells
				are in*/
			void sortVertices(vector<Vertex<VectorType> *> &bottomVertices, vector<Vertex<VectorType> *> &leftVertices, vector<Vertex<VectorType> *> &topVertices, vector<Vertex<VectorType> *> &rightVertices) override;
			

			void classifyVertex(const dimensions_t &gridDim, Vertex<VectorType> *pVertex, vector<Vertex<VectorType> *> &bottomVertices, vector<Vertex<VectorType> *> &leftVertices,
																						  vector<Vertex<VectorType> *> &topVertices, vector<Vertex<VectorType> *> &rightVertices) override;
			#pragma endregion
		};
	}

}
#endif