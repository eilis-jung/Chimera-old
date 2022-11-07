#include "ParticleBased/GridToParticlesAPIC2D.h"

namespace Chimera {

	namespace Advection {

		void GridToParticlesAPIC2D::transferVelocityToParticles(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {

			GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);

			const vector<Vector2> &particlesPosition = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<Vector2> &particlesVelocityDerivativesX = pParticlesData->getVectorBasedAttribute("velocityDerivativeX");
			vector<Vector2> &particlesVelocityDerivativesY = pParticlesData->getVectorBasedAttribute("velocityDerivativeY");

			BilinearAPICStaggeredInterpolant2D *pInterpolant = dynamic_cast<BilinearAPICStaggeredInterpolant2D*>(m_pInterpolant);

			for (int i = 0; i < particlesPosition.size(); i++) {
				Matrix2x2 particleVelocityDerivative = pInterpolant->velocityDerivativeInterpolate(particlesPosition[i]);
				particlesVelocityDerivativesX[i] = particleVelocityDerivative[0];
				particlesVelocityDerivativesY[i] = particleVelocityDerivative[1];
				particlesVelocities[i] = pInterpolant->interpolate(particlesPosition[i]);
			}
		}
	}
}