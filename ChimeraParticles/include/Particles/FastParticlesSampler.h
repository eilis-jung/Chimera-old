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

#ifndef __CHIMERA_FAST_PARTICLES_SAMPLER_H_
#define __CHIMERA_FAST_PARTICLES_SAMPLER_H_
#pragma once
#include "Particles/ParticlesSampler.h"

namespace Chimera {

	namespace Particles {

		/** Samples and resamples particles as fast as possible */
		template <class VectorType, template <class> class ArrayType>
		class FastParticlesSampler : public ParticlesSampler<VectorType, ArrayType> {

		public:
			FastParticlesSampler(GridData<VectorType> *pGridData, Scalar particlesPerCell) : ParticlesSampler(pGridData, particlesPerCell) {
				m_pParticlesData = createSampledParticles();
			};

			/** Particles sampling*/
			virtual ParticlesData<VectorType> * createSampledParticles() override;

			/** Particles resampling*/
			virtual void resampleParticles(ParticlesData<VectorType> *pParticlesData) override;
		protected:

			FORCE_INLINE Scalar safeRandom() const {
				return (rand() / (float)RAND_MAX) - 1e-7;
			}

		};
	}
}

#endif