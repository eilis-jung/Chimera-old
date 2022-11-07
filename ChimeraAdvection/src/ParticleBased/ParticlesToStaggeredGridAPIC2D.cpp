#include "ParticleBased/ParticlesToStaggeredGridAPIC2D.h"

namespace Chimera {


	namespace Advection {
		#pragma region PrivateFunctionalities
		void ParticlesToStaggeredGridAPIC2D::accumulateVelocities(int ithParticle, ParticlesData<Vector2> *pParticlesData, Scalar dx, velocityComponent_t velocityComponent) {
			int i, j;
			Vector2 relativeFractions;

			Vector2 position = pParticlesData->getPositions()[ithParticle]/dx;
			Vector2 positionTemp = pParticlesData->getPositions()[ithParticle];

			Vector2 velocity = pParticlesData->getVelocities()[ithParticle];
			Vector2 cellOrigin;
			if (velocityComponent == velocityComponent_t::xComponent) { //Staggered grid centered on x velocity components
				i = floor(position.x);
				j = floor(position.y - 0.5f);
				cellOrigin.x = i;
				cellOrigin.y = j + 0.5;

				relativeFractions.x = position.x - i;
				if (j < 0) { j = 0; relativeFractions.y = 0.0; }
				else if (j > m_gridDimensions.y - 1) { j = m_gridDimensions.y - 1; relativeFractions.y = 1.0; }
				else { relativeFractions.y = position.y - (j + 0.5); }
			}
			else if (velocityComponent == velocityComponent_t::yComponent) { //Staggered grid centered on y velocity components
				i = floor(position.x - 0.5f);
				j = floor(position.y);
				cellOrigin.x = i + 0.5;
				cellOrigin.y = j;

				relativeFractions.y = position.y - j;
				if (i < 0) { i = 0; relativeFractions.x = 0.0; }
				else if (i > m_gridDimensions.x - 1) { i = m_gridDimensions.x - 1; relativeFractions.x = 1.0; }
				else { relativeFractions.x = position.x - (i + 0.5); }
			}

			vector<Vector2> &particlesVelocityDerivativeX = pParticlesData->getVectorBasedAttribute("velocityDerivativeX");
			vector<Vector2> &particlesVelocityDerivativeY = pParticlesData->getVectorBasedAttribute("velocityDerivativeY");

			Vector2 weight;
			Vector2 p2n;
			Vector2 auxVelocity;
			
			weight[velocityComponent] = (1 - relativeFractions.x)*(1 - relativeFractions.y);
			//p2n = Vector2(-relativeFractions.x, -relativeFractions.y) * dx;
			p2n = Vector2(cellOrigin.x*dx - positionTemp.x, cellOrigin.y*dx - positionTemp.y);
			auxVelocity = Vector2(particlesVelocityDerivativeX[ithParticle].dot(p2n), particlesVelocityDerivativeY[ithParticle].dot(p2n));
			accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), auxVelocity, dimensions_t(i, j), velocityComponent);

			weight[velocityComponent] = relativeFractions.x*(1 - relativeFractions.y);
			if (i < m_gridDimensions.x - 1) {
				//p2n = Vector2(1 - relativeFractions.x, -relativeFractions.y) * dx;
				p2n = Vector2((cellOrigin.x + 1)*dx - positionTemp.x, cellOrigin.y*dx - positionTemp.y);
				auxVelocity = Vector2(particlesVelocityDerivativeX[ithParticle].dot(p2n), particlesVelocityDerivativeY[ithParticle].dot(p2n));
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), auxVelocity, dimensions_t(i + 1, j), velocityComponent);
			}

			weight[velocityComponent] = (1 - relativeFractions.x)*relativeFractions.y;
			if (j < m_gridDimensions.y - 1) {
				//p2n = Vector2(-relativeFractions.x, 1 - relativeFractions.y) * dx;
				p2n = Vector2(cellOrigin.x*dx - positionTemp.x, (cellOrigin.y + 1)*dx - positionTemp.y);
				auxVelocity = Vector2(particlesVelocityDerivativeX[ithParticle].dot(p2n), particlesVelocityDerivativeY[ithParticle].dot(p2n));
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), auxVelocity, dimensions_t(i, j + 1), velocityComponent);
			}

			weight[velocityComponent] = relativeFractions.x*relativeFractions.y;
			if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
				//p2n = Vector2(1 - relativeFractions.x, 1 - relativeFractions.y) * dx;
				p2n = Vector2((cellOrigin.x + 1)*dx - positionTemp.x, (cellOrigin.y + 1)*dx - positionTemp.y);
				auxVelocity = Vector2(particlesVelocityDerivativeX[ithParticle].dot(p2n), particlesVelocityDerivativeY[ithParticle].dot(p2n));
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), auxVelocity, dimensions_t(i + 1, j + 1), velocityComponent);
			}
		}

		void ParticlesToStaggeredGridAPIC2D::accumulateVelocity(int ithParticle, Vector2 weight, const vector<Vector2> &particleVelocities, Vector2 &auxVelocity, const dimensions_t & gridNodeIndex, velocityComponent_t velocityComponent /* = -1 */) {
			if (velocityComponent != fullVector) {
				m_accVelocityField(gridNodeIndex).entry[velocityComponent] += (particleVelocities[ithParticle][velocityComponent] + auxVelocity[velocityComponent]) * weight[velocityComponent];
				m_accVelocityField(gridNodeIndex).weight[velocityComponent] += weight[velocityComponent];
			}
			else {
				m_accVelocityField(gridNodeIndex).entry += (particleVelocities[ithParticle] + auxVelocity) * weight[velocityComponent];
				m_accVelocityField(gridNodeIndex).weight[velocityComponent] += weight[velocityComponent];
			}
		}

		#pragma endregion
	}

	
}