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

#ifndef __CHIMERA_ADVECTION_H__
#define __CHIMERA_ADVECTION_H__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraCutCells.h"

namespace Chimera {

	namespace Advection {

		class AdvectionBase {

		public:
			#pragma region InternalStructures
			typedef struct baseParams_t {
				integrationMethod_t integrationMethod;
				plataform_t platform;

				advectionCategory_t advectionCategory;
				gridBasedAdvectionMethod_t gridBasedAdvectionMethod;
				particleBasedAdvectionMethod_t particleBasedAdvectionMethod;

				baseParams_t() {
					integrationMethod = forwardEuler;
					platform = PlataformCPU;

					gridBasedAdvectionMethod = SemiLagrangian;
					particleBasedAdvectionMethod = FLIP;
					advectionCategory = EulerianAdvection;
				}
				virtual ~baseParams_t() {

				}
			} baseParams_t;
			#pragma endregion

			#pragma region Constructors
			AdvectionBase(const baseParams_t &params) {
				m_baseParams = params;
			}
			#pragma endregion

			#pragma region AccessFunctions
			const baseParams_t & getParams() {
				return m_baseParams;
			}

			baseParams_t * getParamsPtr() {
				return &m_baseParams;
			}
			#pragma endregion

			#pragma region Functionalities
			/* Virtual advection method that base classes have to implement */
			virtual void advect(Scalar dt) = 0;
			
			/* Optional post projection update. Particle based advection schemes will use this*/
			virtual void postProjectionUpdate(Scalar dt) { };
			#pragma endregion

		protected:
			#pragma region ClassMembers
			baseParams_t m_baseParams;
			#pragma endregion
		};
	}
}


#endif