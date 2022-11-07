#include "ParticleBased/GridToParticlesFLIP3D.h"

namespace Chimera {

	namespace Advection {

		GridToParticlesFLIP3D::GridToParticlesFLIP3D(velocityInterpolant * pVelocityInterpolant, Scalar mixPic)
			: GridToParticles(pVelocityInterpolant) {
			m_mixPIC = mixPic;
		}

		void GridToParticlesFLIP3D::transferVelocityToParticles(GridData<Vector3> *pGridData, ParticlesData<Vector3> *pParticlesData) {
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(pGridData);
			Scalar dx = pGridData3D->getGridSpacing();

			const vector<Vector3> &particlesPosition = pParticlesData->getPositions();
			vector<Vector3> &particlesVelocities = pParticlesData->getVelocities();

			for (int i = 0; i < particlesPosition.size(); i++) {
				Vector3 gridSpacePosition = particlesPosition[i] / dx;
				Vector3 currentVel, previousVel;

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
				Vector3 diff = (currentVel - previousVel);
				//MeanValueInterpolant3D<Vector3> *pMeanValueInterpolant = dynamic_cast<MeanValueInterpolant3D<Vector3> *>(m_pInterpolant);
				
				/** Safe distance from boundaries */
				if (floor(gridSpacePosition.x) > 1 && floor(gridSpacePosition.x) < pGridData3D->getDimensions().x - 1 &&
					floor(gridSpacePosition.y) > 1 && floor(gridSpacePosition.y) < pGridData3D->getDimensions().y - 1 &&
					floor(gridSpacePosition.z) > 1 && floor(gridSpacePosition.z) < pGridData3D->getDimensions().z - 1) {
					particlesVelocities[i] = (particlesVelocities[i] + currentVel - previousVel)*(1 - m_mixPIC) + currentVel*m_mixPIC;
				}
				else { //Too close from boundaries, use PIC
					particlesVelocities[i] = currentVel;
				}

				//particlesVelocities[i] = currentVel;
			}
		}


		void GridToParticlesFLIP3D::transferScalarAttributesToParticles(GridData<Vector3> *pGridData, ParticlesData<Vector3> *pParticlesData) {
			GridData3D *pGridData3D = dynamic_cast<GridData3D*>(pGridData);
			
			for (auto iter = m_scalarAttributes.begin(); iter != m_scalarAttributes.end(); iter++) {
				Scalar dx = pGridData3D->getGridSpacing();

				const vector<Vector3> &particlesPosition = pParticlesData->getPositions();
				vector<Scalar> &particlesScalarAttributes = pParticlesData->getScalarBasedAttribute(iter->first);

				for (int i = 0; i < particlesScalarAttributes.size(); i++) {
					Vector3 gridSpacePosition = particlesPosition[i] / dx;
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

					//** Safe distance from boundaries */
					if (floor(gridSpacePosition.x) > 1 && floor(gridSpacePosition.x) < pGridData3D->getDimensions().x - 1 &&
						floor(gridSpacePosition.y) > 1 && floor(gridSpacePosition.y) < pGridData3D->getDimensions().y - 1 &&
						floor(gridSpacePosition.z) > 1 && floor(gridSpacePosition.z) < pGridData3D->getDimensions().z - 1) {
						particlesScalarAttributes[i] = (particlesScalarAttributes[i] + currentVal - previousVal)*(1 - m_mixPIC) + currentVal*m_mixPIC;
					}
					else { //Too close from boundaries, use PIC
						particlesScalarAttributes[i] = currentVal;
					}

					particlesScalarAttributes[i] =clamp(particlesScalarAttributes[i], 0.f, 1.f);
				}
			}
		}
	}

}