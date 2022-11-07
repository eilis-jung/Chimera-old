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

#ifndef _ADVECTION_MAC_CORMACK_H_
#define _ADVECTION_MAC_CORMACK_H_
#pragma once

#include "GridBased/SemiLagrangianAdvection.h"

namespace Chimera {

	namespace Advection {

		template <class VectorType, template <class> class ArrayType>
		class MacCormackAdvection : public SemiLagrangianAdvection<VectorType, ArrayType> {
		public:
			
			#pragma region Constructors
			MacCormackAdvection(const baseParams_t &params, GridData<VectorType> *pGridData, PositionIntegrator<VectorType, ArrayType> *pPositionIntegrator,
												Interpolant<VectorType, ArrayType, VectorType> *pVelocityInterpolation, 
												Interpolant<Scalar, ArrayType, VectorType> *pScalarInterpolation, 
												Interpolant<Scalar, ArrayType, VectorType> *pTemperatureInterpolation = nullptr) 
				: SemiLagrangianAdvection(params, pGridData, pPositionIntegrator, pVelocityInterpolation, pScalarInterpolation, pTemperatureInterpolation),
					m_auxiliaryVelocityField(pGridData->getDimensions()), m_velocityMinLimiters(pGridData->getDimensions()), 
					m_velocityMaxLimiters(pGridData->getDimensions()), m_scalarFieldMaxLimiters(pGridData->getDimensions()),
					m_scalarFieldMinLimiters(pGridData->getDimensions()), m_auxiliaryScalarField(pGridData->getDimensions()) {
				
			}
			#pragma endregion

			#pragma region UpdateFunctioons
			virtual void advect(Scalar dt) override;

			/* Optional post projection update. Particle based advection schemes will use this*/
			virtual void postProjectionUpdate(Scalar dt) override;

		protected:

			#pragma region ClassMembers
			ArrayType<VectorType> m_auxiliaryVelocityField;
			ArrayType<VectorType> m_velocityMaxLimiters;
			ArrayType<VectorType> m_velocityMinLimiters;

			ArrayType<Scalar> m_auxiliaryScalarField;
			ArrayType<Scalar> m_scalarFieldMaxLimiters;
			ArrayType<Scalar> m_scalarFieldMinLimiters;
			#pragma endregion

			#pragma PrivateFunctionalities
			/** 2-D Limiters*/
			Scalar getMinLimiterX(uint i, uint j);
			Scalar getMaxLimiterX(uint i, uint j);
			Scalar getMinLimiterY(uint i, uint j);
			Scalar getMaxLimiterY(uint i, uint j);

			/** 3-D Limiters*/
			Scalar getMinLimiterX(uint i, uint j, uint k);
			Scalar getMaxLimiterX(uint i, uint j, uint k);
			Scalar getMinLimiterY(uint i, uint j, uint k);
			Scalar getMaxLimiterY(uint i, uint j, uint k);
			Scalar getMinLimiterZ(uint i, uint j, uint k);
			Scalar getMaxLimiterZ(uint i, uint j, uint k);

			/*Scalar limiters*/
			Scalar getMinLimiter(uint i, uint j, uint k, const Array3D<Scalar> &scalarField);
			Scalar getMaxLimiter(uint i, uint j, uint k, const Array3D<Scalar> &scalarField);

			Scalar getMinLimiter(uint i, uint j, const Array2D<Scalar> &scalarField);
			Scalar getMaxLimiter(uint i, uint j, const Array2D<Scalar> &scalarField);
			#pragma endregion
		};
	}
}

#endif

