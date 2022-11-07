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
//	

#ifndef __CHIMERA_THIN_OBJECT_2D__
#define __CHIMERA_THIN_OBJECT_2D__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraMesh.h"
#include "ChimeraCutCells.h"

namespace Chimera {

	using namespace Meshes;
	using namespace CutCells;

	namespace Solids {

		template <class VectorT>
		class RigidObject2D : public PhysicalObject<VectorT> {

		public:

			#pragma region Constructosr
			RigidObject2D(LineMesh<VectorT> *pLineMesh, positionUpdate_t positionUpdate = positionUpdate_t(),
														rotationUpdate_t rotationUpdate = rotationUpdate_t(),
														couplingType_t couplingType = oneWayCouplingSolidToFluid);
			#pragma endregion

			#pragma region AccessFunctions
			LineMesh<VectorT> * getLineMesh() {
				return m_pLineMesh;
			}

			couplingType_t getCouplingType() const {
				return m_couplingType;
			}
			#pragma endregion

			#pragma region Functionalities
			void update(Scalar dt);

			/** Returns the first edge velocity */
			void updateCutEdgesVelocities(int timeOffset, Scalar dx, bool useAuxiliaryVelocities = false);
			#pragma endregion

		private:

			#pragma region ClassMembers
			/** ID vars */
			uint m_ID;
			static uint m_currID;

			couplingType_t m_couplingType;

			typename LineMesh<VectorT>::params_t m_initialParams;
			typename LineMesh<VectorT>::params_t m_updatedParams;

			LineMesh<VectorT> *m_pLineMesh;
			#pragma endregion
		};
	}
	
}

#endif
