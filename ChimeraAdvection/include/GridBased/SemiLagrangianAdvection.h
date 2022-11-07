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

#ifndef _ADVECTION_SEMI_LAGRANGIAN_H_
#define _ADVECTION_SEMI_LAGRANGIAN_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
//
#include "Integration/PositionIntegrator.h"
#include "AdvectionBase.h"

namespace Chimera {
	using namespace Interpolation;
	namespace Advection {

		template <class VectorType, template <class> class ArrayType>
		class SemiLagrangianAdvection : public AdvectionBase {
		public:

			#pragma region Constructors
			SemiLagrangianAdvection(const baseParams_t & baseParams, GridData<VectorType> *pGridData,
									PositionIntegrator<VectorType, ArrayType> *pPositionIntegrator,
									Interpolant<VectorType, ArrayType, VectorType> *pVelocityInterpolation,
									Interpolant<Scalar, ArrayType, VectorType> *pScalarInterpolation, 
									Interpolant<Scalar, ArrayType, VectorType> *pTemperatureInterpolation = nullptr) 
									: AdvectionBase(baseParams) {
				m_pGridData = pGridData;
				m_pPositionIntegrator = pPositionIntegrator;
				m_pVelocityInterpolant = pVelocityInterpolation;
				m_pDensityInterpolant = pScalarInterpolation;
				m_pTemperatureInterpolant = pTemperatureInterpolation;
			}
			#pragma endregion

			#pragma region AccessFunctions
			Interpolant<VectorType, ArrayType, VectorType> * getVelocityInterpolant() {
				return m_pVelocityInterpolant;
			}
			
			Interpolant<Scalar, ArrayType, VectorType> * getDensityInterpolant() {
				return m_pDensityInterpolant;
			}

			Interpolant<Scalar, ArrayType, VectorType> * getTemperatureInterpolant() {
				return m_pTemperatureInterpolant;
			}

			void setVelocityInterpolant(Interpolant<VectorType, ArrayType, VectorType> *pVelocityInterpolant) {
				m_pVelocityInterpolant = pVelocityInterpolant;
			}

			void setDensityInterpolant(Interpolant<Scalar, ArrayType, VectorType> *pDensityInterpolant) {
				m_pDensityInterpolant = pDensityInterpolant;
			}

			void setTemperatureInterpolant(Interpolant<Scalar, ArrayType, VectorType> *pTemperatureInterpolant) {
				m_pTemperatureInterpolant = pTemperatureInterpolant;
			}
			#pragma endregion

			#pragma region UpdateFunctioons
			virtual void advect(Scalar dt) override;

			/* Optional post projection update. Particle based advection schemes will use this*/
			virtual void postProjectionUpdate(Scalar dt) override;
			#pragma endregion

			#pragma region AdvectionFunctions
			/** Advects saving velocity into velocity field buffer using customizable interpolant*/
			void advect(Scalar dt, ArrayType<VectorType> *pVelocityField, Interpolant<VectorType, ArrayType, VectorType> *pVelocityInterpolant);

			/** Advects density saving into density field buffer */
			void advectScalarField(Scalar dt, ArrayType<Scalar> &scalarField, Interpolant<Scalar, ArrayType, VectorType> *pScalarInterpolant);
			#pragma endregion
		protected:

			#pragma region ClassMembers
			GridData<VectorType> *m_pGridData;
			Interpolant<VectorType, ArrayType, VectorType> *m_pVelocityInterpolant;
			Interpolant<Scalar, ArrayType, VectorType> *m_pDensityInterpolant;
			Interpolant<Scalar, ArrayType, VectorType> *m_pTemperatureInterpolant;
			PositionIntegrator<VectorType, ArrayType> *m_pPositionIntegrator;
			#pragma endregion
		};
	}
}

#endif

