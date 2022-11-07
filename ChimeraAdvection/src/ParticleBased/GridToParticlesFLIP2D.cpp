#include "ParticleBased/GridToParticlesFLIP2D.h"

namespace Chimera {

	namespace Advection {
		
		GridToParticlesFLIP2D::GridToParticlesFLIP2D(velocityInterpolant * pVelocityInterpolant, Scalar mixPic /*= 0.0f*/)
			: GridToParticles(pVelocityInterpolant) {

			m_mixPIC = mixPic;
		}

		void GridToParticlesFLIP2D::transferVelocityToParticles(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {

			GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);
			Scalar dx = pGridData2D->getGridSpacing();

			const vector<Vector2> &particlesPosition = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			

			for (int i = 0; i < particlesPosition.size(); i++) {
				Vector2 gridSpacePosition = particlesPosition[i] / dx;
				Vector2 currentVel, previousVel;

				currentVel = m_pInterpolant->interpolate(particlesPosition[i]);
				previousVel = m_pInterpolant->getSibilingInterpolant()->interpolate(particlesPosition[i]);

				const vector<bool> &resampledParticles = pParticlesData->getResampledParticles();
				if (resampledParticles.size() > 0) {
					//Particle was resampled, use PIC
					if (resampledParticles[i]) {
						particlesVelocities[i] = currentVel;
						continue; //Go to next particle
					}
				}

				/** Check for Cutcells */
				MeanValueInterpolant2D<Vector2> *pMeanValueInterpolant = dynamic_cast<MeanValueInterpolant2D<Vector2> *>(m_pInterpolant);
				bool isCutCell = false;
				if (pMeanValueInterpolant && pMeanValueInterpolant->getCutCells2D()) {
					if (pMeanValueInterpolant->getCutCells2D()->isCutCellAt(gridSpacePosition.x, gridSpacePosition.y)) {
						isCutCell = true;
						isCutCell = false;
					}
				}

				/** Safe distance from boundaries */
				if (floor(gridSpacePosition.x) > 1 && floor(gridSpacePosition.x) < pGridData2D->getDimensions().x - 1 &&
					floor(gridSpacePosition.y) > 1 && floor(gridSpacePosition.y) < pGridData2D->getDimensions().y - 1 && !isCutCell) {
					particlesVelocities[i] = (particlesVelocities[i] + currentVel - previousVel)*(1 - m_mixPIC) + currentVel*m_mixPIC;
				}
				else { //Too close from boundaries, use PIC
					particlesVelocities[i] = currentVel;
				}
			}
		}


		void GridToParticlesFLIP2D::transferScalarAttributesToParticles(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {
			GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);

			for (auto iter = m_scalarAttributes.begin(); iter != m_scalarAttributes.end(); iter++) {
				Scalar dx = pGridData2D->getGridSpacing();

				const vector<Vector2> &particlesPosition = pParticlesData->getPositions();
				vector<Scalar> &particlesScalarAttributes = pParticlesData->getScalarBasedAttribute(iter->first);

				for (int i = 0; i < particlesScalarAttributes.size(); i++) {
					Vector2 gridSpacePosition = particlesPosition[i] / dx;
					Scalar currentVal, previousVal;

					currentVal = iter->second->interpolate(particlesPosition[i]);
					previousVal = iter->second->getSibilingInterpolant()->interpolate(particlesPosition[i]);

					const vector<bool> &resampledParticles = pParticlesData->getResampledParticles();
					if (resampledParticles.size() > 0) {
						//Particle was resampled, use PIC
						if (resampledParticles[i]) {
							particlesScalarAttributes[i] = currentVal;
							continue; //Go to next particle
						}
					}

					//particlesScalarAttributes[i] = currentVal;

					/** Safe distance from boundaries */
					if (floor(gridSpacePosition.x) > 1 && floor(gridSpacePosition.x) < pGridData2D->getDimensions().x - 1 &&
						floor(gridSpacePosition.y) > 1 && floor(gridSpacePosition.y) < pGridData2D->getDimensions().y - 1) {
						particlesScalarAttributes[i] = (particlesScalarAttributes[i] + currentVal - previousVal)*(1 - m_mixPIC) + currentVal*m_mixPIC;
					}
					else { //Too close from boundaries, use PIC
						particlesScalarAttributes[i] = currentVal;
					}

					//LS - TODO: check if this is vorticity - do not clamp if it is (lets remind us of developing clamping strategies)
					//iter->first gives you the name of the buffer (check if the name is == to "vorticity")
					particlesScalarAttributes[i] = clamp<Scalar>(particlesScalarAttributes[i], 0.f, 1.f);
				}
			}
		}
	}
	
}