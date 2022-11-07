#include "ParticleBased/GridToParticlesRPIC2D.h"

namespace Chimera {

	namespace Advection {
		
		void GridToParticlesRPIC2D::transferVelocityToParticles(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {

			GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);

			const vector<Vector2> &particlesPosition = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<Vector2> &particlesAngularMomentums = pParticlesData->getVectorBasedAttribute("angularMomentum");

			BilinearAPICStaggeredInterpolant2D *pInterpolant = dynamic_cast<BilinearAPICStaggeredInterpolant2D*>(m_pInterpolant);

			for (int i = 0; i < particlesPosition.size(); i++) {
				particlesAngularMomentums[i] = pInterpolant->angualrMomentemInterpolate(particlesPosition[i]);
				particlesVelocities[i] = m_pInterpolant->interpolate(particlesPosition[i]);
			}
		}
	}
}