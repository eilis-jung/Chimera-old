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

#ifndef __CHIMERA_GRID_TO_PARTICLES_H_
#define __CHIMERA_GRID_TO_PARTICLES_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraCutCells.h"

namespace Chimera {

	using namespace Core;
	using namespace Particles;
	using namespace Interpolation;
	using namespace CutCells;

	namespace Grids {

		template<class VectorType, template <class> class ArrayType>
		class GridToParticles {
		protected:
			typedef Interpolant <VectorType, ArrayType, VectorType> velocityInterpolant;
			typedef Interpolant <Scalar, ArrayType, VectorType> scalarInterpolant;

		public:
			#pragma region Constructors
			//Will only reserve sizes inside vectors, explicit position/velocity initialization is on user-side
			GridToParticles(Interpolant <VectorType, ArrayType, VectorType> *pInterpolant) {
				m_pInterpolant = pInterpolant;
			}
			#pragma endregion

			#pragma region CustomAttributesFunctions
			FORCE_INLINE void addScalarAttribute(string name, Interpolant<Scalar, ArrayType, VectorType> *pScalarInterpolant) {
				m_scalarAttributes[name] = pScalarInterpolant;
			}

			FORCE_INLINE typename map<string, Interpolant<Scalar, ArrayType, VectorType> *>::const_iterator getScalarBasedAttribute(const string	&scalarAttribute) const {
				return m_scalarAttributes.find(scalarAttribute);
			}

			FORCE_INLINE typename map<string, Interpolant<Scalar, ArrayType, VectorType> *>::iterator getScalarBasedAttribute(const string	&scalarAttribute) {
				return m_scalarAttributes.find(scalarAttribute);
			}

			FORCE_INLINE void addVectorAttribute(string name, Interpolant<VectorType, ArrayType, VectorType> *pVectorInterpolant) {
				m_vectorAttributes[name] = pVectorInterpolant;
			}

			FORCE_INLINE typename map<string, Interpolant<VectorType, ArrayType, VectorType> *>::const_iterator getVectorBasedAttribute(const string	&vectorAttribute) const {
				return m_vectorAttributes.find(vectorAttribute);
			}

			FORCE_INLINE typename map<string, Interpolant<VectorType, ArrayType, VectorType> *>::iterator  getVectorBasedAttribute(const string	&vectorAttribute) {
				return m_vectorAttributes.find(vectorAttribute);
			}

			#pragma endregion
			#pragma region Functionalities
			/** Velocity transfer from grid to particles. All subclasses must implement this. */
			virtual void transferVelocityToParticles(GridData<VectorType> *pGridData, ParticlesData<VectorType> *pParticlesData) = 0;

			/** Custom attribute transfers. Implementation by subclasses is optional. */
			virtual void transferVelocityAttributesToParticles(GridData<VectorType> *pGridData, ParticlesData<VectorType> *pParticlesData) { };
			virtual void transferScalarAttributesToParticles(GridData<VectorType> *pGridData, ParticlesData<VectorType> *pParticlesData) { };
			virtual void transferIntegerAttributesToParticles() { };

			#pragma endregion

			#pragma region AccessFunctions
			FORCE_INLINE velocityInterpolant * getVelocityInterpolant() {
				return m_pInterpolant;
			}
			#pragma endregion

		protected:
			#pragma region ClassMembers
			/** Standard velocity interpolant */
			velocityInterpolant *m_pInterpolant;

			/** Along with custom attributes for particles, we store the interpolants to retrieve those from a regular
			grid. */
			map<string, Interpolant<VectorType, ArrayType, VectorType> *> m_vectorAttributes;
			map<string, Interpolant<Scalar, ArrayType, VectorType> *> m_scalarAttributes;
		};
	}
}

#endif