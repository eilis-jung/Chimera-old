#include "Physics/ParticlesSampler.h"

namespace Chimera {

	namespace Grids {

		template<>
		ParticlesData<Vector2> * SimpleParticlesSampler<Vector2, Array2D>::createSampledParticles() {
			//We are not instantiating particles at the boundaries, that why dim - 2
			int totalNumberOfCells = (m_pGridData->getDimensions().x - 2)*(m_pGridData->getDimensions().y - 2);
			
			ParticlesData<Vector2> *pParticlesData = new ParticlesData<Vector2>(totalNumberOfCells*m_particlesPerCell);

			Scalar dx = m_pGridData->getGridSpacing();
			//This will subdivide the cells according with number of particles per cell to provide better sampling
			int refactorScale = floor(sqrt(m_particlesPerCell));

			vector<Vector2> &particlesPositions = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();

			dimensions_t newGridDimensions(m_pGridData->getDimensions().x*refactorScale, m_pGridData->getDimensions().y*refactorScale);
			for (int i = refactorScale; i < newGridDimensions.x - refactorScale; i++) {
				for (int j = refactorScale; j < newGridDimensions.y  refactorScale; j++) {
					Vector2 particlePosition = Vector2(i + (rand() / (float)RAND_MAX), j + (rand() / (float)RAND_MAX));
					particlePosition /= refactorScale;
					particlesPositions.push_back(particlePosition);
					particlesVelocities.push_back(Vector2());
					particlesResampled.push_back(false);

					/*m_resampledParticles.push_back(false);
					m_particlesTemperatures.push_back(interpolateScalar(particlePosition, *m_pGridData->getTemperatureBuffer().getBufferArray2()));
					m_particlesDensities.push_back(interpolateScalar(particlePosition, *m_pGridData->getDensityBuffer().getBufferArray2()));
					m_particlesTags.push_back(0);*/
				}
			}

			return pParticlesData;
		}

		
		template<> 
		void SimpleParticlesSampler<Vector2, Array2D>::boundaryResample(int ithParticle, ParticlesData<Vector2> *pParticlesData) {
			vector<Vector2> &particlesPositions = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();


			if (particlesPositions[ithParticle].x > m_pGridData->getDimensions().x - 1) {
				particlesPositions[ithParticle].x = (rand() / (float)RAND_MAX) + 1;
				particlesPositions[ithParticle].y = clamp<Scalar>(particlesPositions[ithParticle].y, 0, m_pGridData->getDimensions().y - 1);
				particlesVelocities[ithParticle] = m_pInterpolant->interpolate(particlesPositions[ithParticle]);
				particlesResampled[ithParticle] = true;
			}
			else if (particlesPositions[ithParticle].x < 1) {
				particlesPositions[ithParticle].x = (rand() / (float)RAND_MAX) + m_pGridData->getDimensions().x - 2;
				particlesPositions[ithParticle].y = clamp<Scalar>(particlesPositions[ithParticle].y, 0, m_pGridData->getDimensions().y - 1);
				particlesVelocities[ithParticle] = m_pInterpolant->interpolate(particlesPositions[ithParticle]);
				particlesResampled[ithParticle] = true;
			}

			if (particlesPositions[ithParticle].y > m_pGridData->getDimensions().y - 1) {
				particlesPositions[ithParticle].y = (rand() / (float)RAND_MAX) + 1;

				particlesPositions[ithParticle].x = clamp<Scalar>(particlesPositions[ithParticle].x, 0, m_pGridData->getDimensions().x - 1);
				//particlesPositions[ithParticle].x = (rand() / ((float)RAND_MAX))*(37 - 25) + 25;;
				particlesVelocities[ithParticle] = m_pInterpolant->interpolate(particlesPositions[ithParticle]);
				particlesResampled[ithParticle] = true;
			}
			else if (particlesPositions[ithParticle].y < 1) {
				particlesPositions[ithParticle].y = (rand() / (float)RAND_MAX) + m_pGridData->getDimensions().x - 2;
				particlesPositions[ithParticle].x = clamp<Scalar>(particlesPositions[ithParticle].x, 0, m_pGridData->getDimensions().x - 1);
				particlesVelocities[ithParticle] = m_pInterpolant->interpolate(particlesPositions[ithParticle]);
				particlesResampled[ithParticle] = true;
			}
		}
		template<>
		void SimpleParticlesSampler<Vector2, Array2D>::resampleParticles(ParticlesData<Vector2> *pParticlesData) {

			vector<Vector2> &particlesPositions = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();
			
			Scalar dx = m_pGridData->getGridSpacing();
			Array2D<Scalar> m_particlesCount(m_pGridData->getDimensions());

			m_particlesCount.assign(0);
			for (int i = 0; i < particlesPositions.size(); i++) {
				particlesResampled[i] = false;
				boundaryResample(i, pParticlesData);
				
				int indexI = particlesPositions[i].x, indexJ = particlesPositions[i].y;
				m_particlesCount(indexI, indexJ) += 1;
			}

			priority_queue<cellParticleCount_t, vector<cellParticleCount_t>, CountCompareNode> pq;
			for (int i = 1; i < m_particlesCount.getDimensions().x - 1; i++) {
				for (int j = 1; j < m_particlesCount.getDimensions().y - 1; j++) {
					pq.push(cellParticleCount_t(dimensions_t(i, j), m_particlesCount(i, j)));
				}
			}

			for (int i = 0; i < particlesPositions.size(); i++) {
				int maxParticles = ceil(m_particlesPerCell*1.05f);

				if (m_particlesCount(particlesPositions[i].x, particlesPositions[i].y) > maxParticles) {
					cellParticleCount_t top = pq.top();
					m_particlesCount(particlesPositions[i].x, particlesPositions[i].y) -= 1;
					pq.pop();
					particlesPositions[i] = Vector2(top.cellIndex.x + (rand() / (float)RAND_MAX), top.cellIndex.y + (rand() / (float)RAND_MAX));
					particlesResampled[i] = true;
					particlesVelocities[i] = m_pInterpolant->interpolate(particlesPositions[i]);
					top.count++;
					pq.push(top);
				}
			}
		}
	}
}