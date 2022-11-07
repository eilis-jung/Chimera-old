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

#ifndef __CHIMERA_CUT_SLICE_3D__
#define __CHIMERA_CUT_SLICE_3D__
#pragma once

/************************************************************************/
/* Other packages														*/
/************************************************************************/
#include "ChimeraCore.h"
#include "ChimeraMesh.h"
#include "ChimeraGrids.h"

/************************************************************************/
/* Data                                                                 */
/************************************************************************/
#include "CutCells/Crossing.h"
#include "CutCells/CutFace.h"
#include "CutCells/CutVoxel.h"
#include "Mesh/NonManifoldMesh2D.h"


namespace Chimera {

	namespace CutCells {

		class CutSlice3D {

			#pragma region ClassMembers
			faceLocation_t m_sliceLocation;
			dimensions_t m_sliceDimensions;
			vector<bool> m_lineMeshesCrossGrid;
			vector<LineMesh<Vector3D> *> m_lineMeshes;
			Array2D <vector<lineSegment_t<Vector3D>>> m_lineSegments;
			Scalar m_dx;
			Scalar m_proximityThreshold;
			
			Scalar m_orthogonalDimensionalPosition;
			unsigned int m_orthogonalDimensionalIndex;

			vector<NonManifoldMesh2D *> m_manifoldMeshes;
			#pragma endregion 

			#pragma region PrivateFunctionalities
			void remeshLines();
			#pragma endregion 

			#pragma region InitilizationFunctions
			void initializeLineSegments();
			void initializeManifoldMeshes();
			#pragma endregion 

			#pragma region ConversionFunctions
			Vector2D convertVectorTo2D(const Vector3D &position);
			Vector3D convertVectorTo3D(const Vector2D &vec);
			dimensions_t convertVectorToDimensions2D(const Vector3D &position);
			#pragma endregion 

			public:
			#pragma region Constructors
			CutSlice3D(dimensions_t sliceDimensions, faceLocation_t sliceLocation, unsigned int orthogonalDimension, Scalar gridSpacing, 
						vector<LineMesh<Vector3D> *> lineMeshes);
			#pragma endregion 

			#pragma region AccessFunctions
			const vector<NonManifoldMesh2D *> & getNonManifoldMeshes() const {
				return m_manifoldMeshes;
			}
			const vector<LineMesh<Vector3D> *> & getLineMeshes() const {
				return m_lineMeshes;
			}
			#pragma endregion 
		};
	}
}

#endif