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

#ifndef __CHIMERA_CUT_VOXELS_VELOCITIES_3D_H_
#define __CHIMERA_CUT_VOXELS_VELOCITIES_3D_H_
#pragma once

#include "CutCells/CutCellsVelocities.h"
#include "CutCells/CutVoxels3D.h"

namespace Chimera {

	using namespace Meshes;

	namespace CutCells {

		//class CutCells2D;
		template <class VectorT>
		class CutFace;

		class CutVoxelsVelocities3D : public CutCellsVelocities<Vector3, Volume> {
		
		public:
			#pragma region Constructors
			CutVoxelsVelocities3D(CutVoxels3D<Vector3> *pCutVoxels, solidBoundaryType_t solidBoundaryType);
			#pragma endregion
			
			#pragma region Functionalities
			/** Updates cut-cells velocities considering an underlying nodal-based regular grid */
			void update(const Array3D<Vector3> &nodalVelocities, bool useAuxiliaryVelocities = false);
			#pragma endregion
			

		protected:
			CutVoxels3D<Vector3> *m_pCutVoxels;
			mixedNodeInterpolationType_t m_mixedNodeInterpolationType;
			#pragma region PrivateFunctionalities
			void processNoSlipVelocities(const Array3D<Vector3> &nodalVelocities, bool useAuxiliaryVelocities = false);
			void processFreeSlipVelocities(const Array3D<Vector3> &nodalVelocities, bool useAuxiliaryVelocities = false);

			Vector3 interpolateMixedNodeVelocitiesUnweighted(Vertex<Vector3> *pVertex, bool useAuxiliaryVelocities = false);
			#pragma endregion

		};
	}
}


#endif
