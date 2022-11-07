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

#ifndef _MATH_FORWARD_EULER_PARTICLE_POSITION_INTEGRATOR_H_
#define _MATH_FORWARD_EULER_PARTICLE_POSITION_INTEGRATOR_H_
#pragma once

#include "ChimeraCore.h"

#include "Integration/PositionIntegrator.h"

namespace Chimera {

	using namespace Core;

	namespace Advection {

		/** Integrates particles position from a vector of VectorType */
		template<class VectorType, template <class> class ArrayType>
		class ForwardEulerIntegrator : public PositionIntegrator<VectorType, ArrayType> {
		public:

			#pragma region Constructors
				ForwardEulerIntegrator(ParticlesData<VectorType> *pParticlesData, Interpolant<VectorType, ArrayType, VectorType> *pInterpolant, Scalar gridDx);
			#pragma endregion

			#pragma region Functionalities
				virtual void integratePosition(uint particleID, Scalar dt) override;

				virtual VectorType integrate(const VectorType &position, const VectorType &velocity, Scalar dt, Interpolant<VectorType, ArrayType, VectorType> *pCustomInterpolant = nullptr) override;
			#pragma endregion

		};


		template<class VectorType, template <class> class ArrayType>
		inline ForwardEulerIntegrator<VectorType, ArrayType>::ForwardEulerIntegrator(ParticlesData<VectorType> *pParticlesData, Interpolant<VectorType, ArrayType, VectorType> *pInterpolant, Scalar gridDx) :
			PositionIntegrator(pParticlesData, pInterpolant, gridDx)
		{
		}

	}
}

#endif