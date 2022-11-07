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

#ifndef __CHIMERA_PARTICLES_TO_STAGGERED_GRID_H_
#define __CHIMERA_PARTICLES_TO_STAGGERED_GRID_H_
#pragma once

#include "ChimeraCore.h"
#include "ParticleBased/ParticlesToGrid.h"

namespace Chimera {

	using namespace Core;

	namespace Advection {

		class ParticlesToStaggeredGrid2D : public ParticlesToGrid<Vector2, Array2D> {

		public:
			//Will only reserve sizes inside vectors, explicit position/velocity initialization is on user-side
			ParticlesToStaggeredGrid2D(const dimensions_t &gridDimensions, TransferKernel<Vector2> *pTransferKernel)
				: ParticlesToGrid(gridDimensions, pTransferKernel) {

			}

			#pragma region Functionalities
			/** Velocity transfer from particles to grid. All subclasses must implement this. */
			virtual void transferVelocityToGrid(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData);

			/** Velocity transfer from particles to grid. All subclasses must implement this. */
			virtual void transferScalarAttributesToGrid(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData);
			#pragma endregion 		


			#pragma region AccessFunctions
			#pragma endregion 		
		protected:

			#pragma region PrivateFunctionalities
			/** Velocity transfer from particles to grid. All subclasses must implement this. */
			virtual void accumulateVelocities(int ithParticle, ParticlesData<Vector2> *pParticlesData, Scalar dx, velocityComponent_t velocityComponent);

			/** Scalar transfer from particles to grid. Assuming that scalar values are stored in cell centers */
			virtual void accumulateScalarFieldValues(string scalarFieldName, ParticlesData<Vector2> *pParticlesData, Scalar dx);
			#pragma endregion 		
		};
	}

	

}

#endif