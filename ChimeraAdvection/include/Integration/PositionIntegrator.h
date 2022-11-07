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

#ifndef _MATH_PARTICLE_POSITION_INTEGRATOR_H_
#define _MATH_PARTICLE_POSITION_INTEGRATOR_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraCutCells.h"
#include "ChimeraInterpolation.h"
#include "ChimeraMesh.h"
#include "ChimeraParticles.h"

namespace Chimera {

	using namespace Core;
	using namespace Interpolation;
	using namespace Particles;

	namespace Advection {

		/** Integrates particles position from a vector of VectorType */
		template<class VectorType, template <class> class ArrayType>
		class PositionIntegrator {
		public:
			
			#pragma region Constructors
			PositionIntegrator(ParticlesData<VectorType> *pParticlesData, Interpolant<VectorType, ArrayType, VectorType> *pInterpolant, Scalar gridDx);
			#pragma endregion

			#pragma region Functionalities
			FORCE_INLINE virtual void integratePositions(Scalar dt) {
				for (uint i = 0; i < m_pParticlesData->getPositions().size(); i++) {
					integratePosition(i, dt);
				}
			};

			/** Virtual function that each sub-class has to implement */
			virtual void integratePosition(uint particleID, Scalar dt) = 0;

			/** Integrate a single position with an initial velocity */
			virtual VectorType integrate(const VectorType &position, const VectorType &velocity, Scalar dt, Interpolant<VectorType, ArrayType, VectorType> *pCustomInterpolant = nullptr) = 0;

			/** Clamps the position to be contained inside the interpolation domain */
			void clampPosition(VectorType &position);
			#pragma endregion

			#pragma region AccessFunctions
			Interpolant<VectorType, ArrayType, VectorType> * getInterpolant() {
				return m_pInterpolant;
			}

			void setCutCells(CutCellsBase<VectorType> *pCutCellBase) {
				m_pCutCell = pCutCellBase;
			}

			void setCutVoxels(CutVoxels3D<VectorType> *pCutVoxels) {
				m_pCutVoxels = pCutVoxels;
			}
			#pragma endregion


		protected:

			#pragma region PrivateFunctionalities
			bool checkCollision(const VectorType &v1, const VectorType &v2);
			#pragma endregion

			#pragma region ClassMembers
			Interpolant<VectorType, ArrayType, VectorType> *m_pInterpolant;
			Scalar m_dx;

			ParticlesData<VectorType> *m_pParticlesData;

			CutCellsBase<VectorType> *m_pCutCell;
			CutVoxels3D<VectorType> *m_pCutVoxels;
			#pragma endregion
		
		};

		template<class VectorType, template <class> class ArrayType>
		inline PositionIntegrator<VectorType, ArrayType>::PositionIntegrator(ParticlesData<VectorType>* pParticlesData, Interpolant<VectorType, ArrayType, VectorType>* pInterpolant, Scalar gridDx)
		{
			m_pInterpolant = pInterpolant;
			m_dx = gridDx;
			m_pCutCell = NULL;
			m_pCutVoxels = nullptr;
			m_pParticlesData = pParticlesData;
		}
	}
}

#endif