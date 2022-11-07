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

#ifndef __CHIMERA_CUT_CELLS_VELOCITIES_2D_H_
#define __CHIMERA_CUT_CELLS_VELOCITIES_2D_H_
#pragma once

#include "CutCells/CutCellsVelocities.h"
#include "CutCells/CutCells2D.h"

namespace Chimera {

	using namespace Meshes;

	namespace CutCells {
		//class CutCells2D;
		template <class VectorT>
		class CutFace;

		class CutCellsVelocities2D : public CutCellsVelocities<Vector2, Face> {
		
		public:
			#pragma region Constructors
			CutCellsVelocities2D(CutCells2D<Vector2> *pCutCells, solidBoundaryType_t solidBoundaryType);
			#pragma endregion
			
			#pragma region Functionalities
			/** Updates cut-cells velocities considering an underlying nodal-based regular grid */
			void update(const Array2D<Vector2> &nodalVelocities, bool useAuxiliaryVelocities = false);
			void projectMixedNodeVelocities();
			#pragma endregion
			

		protected:
			CutCells2D<Vector2> *m_pCutCells2D;

			#pragma region PrivateFunctionalities
			Vector2 interpolateMixedNodeVelocity(Scalar fluidFlux, Scalar thinObjectFlux, const Vector2 &faceNormal, const Vector2 &thinObjectNormal);
			Vector2 projectFreeSlipMixedNodes(const Vector2 &nodalVelocity, const Vector2 &faceVelocity, const Vector2 &faceNormal);

			Vector2 getNextMixedNodeVelocity(const CutFace<Vector2> &cutFace, int ithThinObjectPoint);

			void processNoSlipVelocities(const Array2D<Vector2> &nodalVelocities, bool useAuxiliaryVelocities = false);
			void processFreeSlipVelocities(const Array2D<Vector2> &nodalVelocities, bool useAuxiliaryVelocities = false);

			#pragma endregion

		};
	}
}


#endif
