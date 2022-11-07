#pragma once
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

#ifndef __CHIMERA_TURBULENT_PARTICLES_TO_STAGGERED_GRID_H__
#define __CHIMERA_TURBULENT_PARTICLES_TO_STAGGERED_GRID_H__
#pragma once

#include "ChimeraCore.h"
#include "ParticleBased/ParticlesToStaggeredGrid2D.h"

namespace Chimera {

	using namespace Core;

	namespace Advection {

		class TurbulentParticlesGrid2D : public ParticlesToStaggeredGrid2D {

		public:
			//Will only reserve sizes inside vectors, explicit position/velocity initialization is on user-side
			TurbulentParticlesGrid2D(const dimensions_t &gridDimensions, TransferKernel<Vector2> *pTransferKernel)
				: ParticlesToStaggeredGrid2D(gridDimensions, pTransferKernel) {

			}

			#pragma region Functionalities
			/**Nothing to override here*/
			#pragma endregion 		


			#pragma region AccessFunctions
			/**Nothing to override here*/
			#pragma endregion 		
		protected:

			#pragma region PrivateFunctionalities
		
			/** Streamfunction transfer from particles to finer grid. This function only updates scalar fields that are
			  * named 'streamfunctionFine'. This function transfers to node based locations.*/
			virtual void accumulateScalarFieldValues(string scalarFieldName, ParticlesData<Vector2> *pParticlesData, Scalar dx);
			#pragma endregion 		
		};
	}

	

}

#endif