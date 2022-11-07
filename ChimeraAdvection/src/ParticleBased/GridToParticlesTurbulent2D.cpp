#include "ParticleBased/GridToParticlesTurbulent2D.h"

namespace Chimera {

	namespace Advection {
		GridToParticlesTurbulent2D::GridToParticlesTurbulent2D(	const pair<VectorInterpolant*, VectorInterpolant*> &velocityInterpolants,
																const pair<ScalarInterpolant*, ScalarInterpolant*> &scalarInterpolants,
																const pair<VectorInterpolant*, VectorInterpolant*> &fineVelocityInterpolant,
																const pair<ScalarInterpolant*, ScalarInterpolant*> &streamfunctionInterpolant,
																Scalar mixPic) : 
			GridToParticlesFLIP2D(velocityInterpolants.first, mixPic), 
			m_fineVelocitiesInterpolants(fineVelocityInterpolant), m_streamfunctionInterpolants(streamfunctionInterpolant) {
			
		}

		void GridToParticlesTurbulent2D::transferVelocityToParticles(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {
			GridToParticlesFLIP2D::transferVelocityToParticles(pGridData, pParticlesData);

			if (m_fineVelocitiesInterpolants.first == NULL) //Velocity should not be updated, return
				return;

			GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);
			Scalar dx = pGridData2D->getGridSpacing();

			const vector<Vector2> &particlesPosition = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();

			for (int i = 0; i < particlesPosition.size(); i++) {
				Vector2 gridSpacePosition = particlesPosition[i] / dx;
				Vector2 currentFineVel, previousFineVel, velocityIncrement;

				currentFineVel = m_fineVelocitiesInterpolants.first->interpolate(particlesPosition[i]);
				previousFineVel = m_fineVelocitiesInterpolants.second->interpolate(particlesPosition[i]);

				const vector<bool> &resampledParticles = pParticlesData->getResampledParticles();
				if (resampledParticles.size() > 0) {
					//Particle was resampled, use PIC
					if (resampledParticles[i]) {
						velocityIncrement = currentFineVel;
						continue; //Go to next particle
					}
				}

				/** Safe distance from boundaries */
				if (floor(gridSpacePosition.x) > 1 && floor(gridSpacePosition.x) < pGridData2D->getDimensions().x - 1 &&
					floor(gridSpacePosition.y) > 1 && floor(gridSpacePosition.y) < pGridData2D->getDimensions().y - 1) {
					velocityIncrement = (particlesVelocities[i] + currentFineVel - previousFineVel)*(1 - m_mixPIC) + currentFineVel*m_mixPIC;
				}
				else { //Too close from boundaries, use PIC
					velocityIncrement = currentFineVel;
				}
				velocityIncrement = currentFineVel;


				particlesVelocities[i] += velocityIncrement;
			}
		}


		void GridToParticlesTurbulent2D::transferScalarAttributesToParticles(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {
			if (m_streamfunctionInterpolants.first == NULL) //Velocity should not be updated, return
				return;

			GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);
			Scalar dx = pGridData2D->getGridSpacing();

			const vector<Vector2> &particlesPosition = pParticlesData->getPositions();
			vector<Scalar> &particlesStreamfunctions = pParticlesData->getScalarBasedAttribute("streamfunctionFine");

			for (int i = 0; i < particlesStreamfunctions.size(); i++) {
				Vector2 gridSpacePosition = particlesPosition[i] / dx;
				Scalar currentVal, previousVal;

				currentVal = m_streamfunctionInterpolants.first->interpolate(particlesPosition[i]);
				previousVal = m_streamfunctionInterpolants.second->interpolate(particlesPosition[i]);

				const vector<bool> &resampledParticles = pParticlesData->getResampledParticles();
				if (resampledParticles.size() > 0) {
					//Particle was resampled, use PIC
					if (resampledParticles[i]) {
						particlesStreamfunctions[i] = currentVal;
						continue; //Go to next particle
					}
				}

				/** Safe distance from boundaries */
				if (floor(gridSpacePosition.x) > 1 && floor(gridSpacePosition.x) < pGridData2D->getDimensions().x - 1 &&
					floor(gridSpacePosition.y) > 1 && floor(gridSpacePosition.y) < pGridData2D->getDimensions().y - 1) {
					particlesStreamfunctions[i] = (particlesStreamfunctions[i] + currentVal - previousVal)*(1 - m_mixPIC) + currentVal*m_mixPIC;
				}
				else { //Too close from boundaries, use PIC
					particlesStreamfunctions[i] = currentVal;
				}
			}
		}
	}

}