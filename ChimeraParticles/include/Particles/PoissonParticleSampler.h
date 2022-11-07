#pragma once

#include "Particles/ParticlesSampler.h"

namespace Chimera {

	namespace Particles {

		template <class VectorType, template <class> class ArrayType>
		class PoissonParticleSampler : public ParticlesSampler<VectorType, ArrayType> {
		
			public:
			PoissonParticleSampler(GridData<VectorType> *pGridData, Scalar particlesPerCell) : ParticlesSampler(pGridData, particlesPerCell) {
				m_pParticlesData = createSampledParticles();
			};

			virtual ParticlesData<VectorType> * createSampledParticles() override;
			
		protected:

		};
	}
}