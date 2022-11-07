#include "Particles/FastParticlesSampler.h"

namespace Chimera {

	namespace Particles {

		#pragma region Functionalities

		template<>
		ParticlesData<Vector2> * FastParticlesSampler<Vector2, Array2D>::createSampledParticles() {
			int totalNumberOfCells = (m_pGridData->getDimensions().x - 2)*(m_pGridData->getDimensions().y - 2);
			ParticlesData<Vector2> *pParticlesData = new ParticlesData<Vector2>(totalNumberOfCells*m_particlesPerCell);

			Scalar dx = m_pGridData->getGridSpacing();

			vector<Vector2> &particlesPositions = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();

			for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
					for (int l = 0; l < m_particlesPerCell; l++) {
						Vector2 particlePosition = Vector2(i + safeRandom(), j + safeRandom());
						particlesPositions.push_back(particlePosition*dx);
						particlesVelocities.push_back(Vector2());
						particlesResampled.push_back(false);
					}
				}
			}

			return pParticlesData;
		}

		template<>
		ParticlesData<Vector3> * FastParticlesSampler<Vector3, Array3D>::createSampledParticles() {
			int totalNumberOfCells = (m_pGridData->getDimensions().x - 2)*(m_pGridData->getDimensions().y - 2)*(m_pGridData->getDimensions().z - 2);
			ParticlesData<Vector3> *pParticlesData = new ParticlesData<Vector3>(totalNumberOfCells*m_particlesPerCell);

			Scalar dx = m_pGridData->getGridSpacing();
			
			vector<Vector3> &particlesPositions = pParticlesData->getPositions();
			vector<Vector3> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();

			for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
					for (int k = 1; k < m_pGridData->getDimensions().z - 1; k++) {
						for (int l = 0; l < m_particlesPerCell; l++) {
							Vector3 particlePosition = Vector3(i + safeRandom(), j + safeRandom(), k + safeRandom());
							particlesPositions.push_back(particlePosition*dx);
							particlesVelocities.push_back(Vector3(0, 0, 0));

							particlesResampled.push_back(false);
						}
					}
				}
			}

			return pParticlesData;
		}


		template<>
		void FastParticlesSampler<Vector2, Array2D>::resampleParticles(ParticlesData<Vector2> *pParticlesData) {
			vector<Vector2> &particlesPositions = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();

			Scalar dx = m_pGridData->getGridSpacing();
			Array2D<int> m_particlesCount(m_pGridData->getDimensions());

			m_particlesCount.assign(0);
			for (int i = 0; i < particlesPositions.size(); i++) {
				particlesResampled[i] = false;
				boundaryResample(i, pParticlesData);

				int indexI = particlesPositions[i].x / dx, indexJ = particlesPositions[i].y / dx;
				m_particlesCount(indexI, indexJ) += 1;
			}

			vector<uint> particlesResampledIndex;
			particlesResampledIndex.reserve(floor(0.1*particlesPositions.size()));

			/** First pass: find removable particles */
			for (int i = 0; i < particlesPositions.size(); i++) {
				/** Look if particles are too close to objects boundaries */
				Vector2 gridSpacePosition = particlesPositions[i] / dx;
				if (m_pCutCells) {
					CutCells2D<Vector2> *pCutCells2D = dynamic_cast<CutCells2D<Vector2> *>(m_pCutCells);
					if (pCutCells2D->isCutCellAt(gridSpacePosition.x, gridSpacePosition.y)) {
						int cutCellIndex = pCutCells2D->getCutCellIndex(gridSpacePosition);
						if (pCutCells2D->getCutCell(cutCellIndex).getDistanceToBoundary(particlesPositions[i]) < 0.001) {
							particlesResampledIndex.push_back(i);
							m_particlesCount(particlesPositions[i].x / dx, particlesPositions[i].y / dx)--;
							continue; //Follow to next particle
						}
					}
				}

				if (m_particlesCount(particlesPositions[i].x / dx, particlesPositions[i].y / dx) > m_particlesPerCell) {
					/** Use particles resampled as flag to resample particles */
					particlesResampledIndex.push_back(i);
					m_particlesCount(particlesPositions[i].x / dx, particlesPositions[i].y / dx)--;
				}
			}

			/** Second pass: add particles to places with not enough of them */
			for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
					Vector2 gridSpaceParticle(i, j);
					while (m_particlesCount(i, j) < m_particlesPerCell) {
						if (particlesResampledIndex.size() > 0) {
							uint particleIndex = particlesResampledIndex.back();
							particlesResampledIndex.pop_back();
							pParticlesData->resampleParticle(particleIndex, Vector2(i + safeRandom(),
																					j + safeRandom())*dx);
							particlesResampled[particleIndex] = true;
						}
						else {
							pParticlesData->addParticle(Vector2(i + safeRandom(), j + safeRandom())*dx);
						}
						m_particlesCount(i, j)++;
					}
				}
			}
		}

		template<>
		void FastParticlesSampler<Vector3, Array3D>::resampleParticles(ParticlesData<Vector3> *pParticlesData) {
			vector<Vector3> &particlesPositions = pParticlesData->getPositions();
			vector<Vector3> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();

			Scalar dx = m_pGridData->getGridSpacing();
			Array3D<int> m_particlesCount(m_pGridData->getDimensions());

			m_particlesCount.assign(0);
			for (int i = 0; i < particlesPositions.size(); i++) {
				particlesResampled[i] = false;
				boundaryResample(i, pParticlesData);

				int indexI = particlesPositions[i].x / dx, 
					indexJ = particlesPositions[i].y / dx,
					indexK = particlesPositions[i].z / dx;
				m_particlesCount(indexI, indexJ, indexK) += 1;
			}

			vector<uint> particlesResampledIndex;
			particlesResampledIndex.reserve(floor(0.1*particlesPositions.size()));

			/** First pass: find removable particles */
			for (int i = 0; i < particlesPositions.size(); i++) {
				if (m_particlesCount(particlesPositions[i].x / dx, particlesPositions[i].y / dx, particlesPositions[i].z / dx) > m_particlesPerCell) {
					/** Use particles resampled as flag to resample particles */
					particlesResampledIndex.push_back(i);
					m_particlesCount(particlesPositions[i].x / dx, particlesPositions[i].y / dx, particlesPositions[i].z / dx)--;
				}
			}

			
			/** Second pass: add particles to places with not enough of them */
			for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
					for (int k = 1; k < m_pGridData->getDimensions().z - 1; k++) {
						Vector3 gridSpaceParticle(i, j, k);
						while (m_particlesCount(i, j, k) < m_particlesPerCell) {
							if (particlesResampledIndex.size() > 0) {
								uint particleIndex = particlesResampledIndex.back();
								particlesResampledIndex.pop_back();
								pParticlesData->resampleParticle(particleIndex, Vector3(i + safeRandom(),
																						j + safeRandom(),
																						k + safeRandom())*dx);
								particlesResampled[particleIndex] = true;
							}
							else {
								pParticlesData->addParticle(Vector3(i + safeRandom(), j + safeRandom(), k + safeRandom())*dx);
							}
							m_particlesCount(i, j, k)++;
						}
					}
				}
			}
		}
		#pragma endregion

		template class FastParticlesSampler<Vector2, Array2D>;
		template class FastParticlesSampler<Vector3, Array3D>;
	}


}