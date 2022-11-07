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

#ifndef __CHIMERA_PARTICLES_TO_STAGGERED_APIC_GRID_H_
#define __CHIMERA_PARTICLES_TO_STAGGERED_APIC_GRID_H_
#pragma once

#include "ParticleBased/ParticlesToStaggeredGrid2D.h"

namespace Chimera {
	
	namespace Advection {

		class ParticlesToStaggeredGridAPIC2D : public ParticlesToStaggeredGrid2D {

		public:
			//Will only reserve sizes inside vectors, explicit position/velocity initialization is on user-side
			ParticlesToStaggeredGridAPIC2D(const dimensions_t &gridDimensions, TransferKernel<Vector2> *pTransferKernel, Interpolant<Vector2, Array2D, Vector2> *pInterpolant = NULL)
				: ParticlesToStaggeredGrid2D(gridDimensions, pTransferKernel) {}

			#pragma region Functionalities
			/** Velocity transfer from particles to grid. All subclasses must implement this. */
			virtual void transferVelocityToGrid(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {
				ParticlesToStaggeredGrid2D::transferVelocityToGrid(pGridData, pParticlesData);
			}
			#pragma endregion 		


			#pragma region AccessFunctions
			#pragma endregion 		
		protected:

			#pragma region PrivateFunctionalities
		

			/** Rigid Velocity transfer from particles to grid. All subclasses must implement this. */
			virtual void accumulateVelocities(int ithParticle, ParticlesData<Vector2> *pParticlesData, Scalar dx, velocityComponent_t velocityComponent);

			/** Accumulates a particle velocity into the accumulated vector buffer with a given weight. If the velocity
			* component is not specified (fullVector), accumulates all possible velocities. */
			virtual void accumulateVelocity(int ithParticle, Vector2 weight, const vector<Vector2> &particleVelocities, Vector2 &auxVelocity,
														const dimensions_t & gridNodeIndex, velocityComponent_t velocityComponent = fullVector);
			#pragma endregion
		};
	}
	

}

#endif