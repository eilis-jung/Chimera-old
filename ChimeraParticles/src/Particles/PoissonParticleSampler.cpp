#include "Particles/PoissonParticleSampler.h"

namespace Chimera {

	namespace Particles {

		template<>
		ParticlesData<Vector2>* PoissonParticleSampler<Vector2, Array2D>::createSampledParticles() {
			//We are not instantiating particles at the boundaries, that why dim - 2
			int totalNumberOfCells = (m_pGridData->getDimensions().x - 2)*(m_pGridData->getDimensions().y - 2);
			ParticlesData<Vector2> *pParticlesData = new ParticlesData<Vector2>(totalNumberOfCells*m_particlesPerCell);
			vector<Vector2> &particlesPositions = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();

			Scalar dx = m_pGridData->getGridSpacing();
			int particlesPerCell = m_particlesPerCell;
			int resamplingFactor = sqrt(particlesPerCell);
			Scalar radius = (dx / resamplingFactor);
			Vector2 initialBoundary = Vector2(1, 1)*dx;
			Vector2 finalBoundary = Vector2(m_pGridData->getDimensions().x + 1, m_pGridData->getDimensions().y + 1)*dx;
			dimensions_t newGridDimensions((finalBoundary.x - initialBoundary.x)/radius, (finalBoundary.y - initialBoundary.y) / radius);

			Array2D<int> particleIndices(newGridDimensions);
			particleIndices.assign(-1);
			for (int i = resamplingFactor; i < newGridDimensions.x - resamplingFactor; i++) {
				for (int j = resamplingFactor; j < newGridDimensions.y - resamplingFactor; j++) {
					bool closeToOtherParticles = true;
					Vector2 particlePos;
					if (closeToOtherParticles) {
						int k = 0;
						do
						{
							particlePos = Vector2(i + (rand() / (float)RAND_MAX), j + (rand() / (float)RAND_MAX)) *(dx/resamplingFactor);

							if (i == resamplingFactor || i == newGridDimensions.x - resamplingFactor - 1) {
								closeToOtherParticles = false; continue; k++;
							}
							else if (j == resamplingFactor || j == newGridDimensions.y - resamplingFactor - 1) {
								closeToOtherParticles = false; continue; k++;
							}

							if (particleIndices(i - 1, j) != -1) {
								Vector2 leftParticlePosition = particlesPositions[particleIndices(i - 1, j)];
								Scalar neighborParticleDistance = (particlePos - leftParticlePosition).length();
								if (neighborParticleDistance < radius) {
									k++;
									continue;
								}
							}
							if (particleIndices(i, j - 1) != -1) {
								Vector2 bottomParticlePosition = particlesPositions[particleIndices(i, j - 1)];
								Scalar neighborParticleDistance = (particlePos - bottomParticlePosition).length();
								if (neighborParticleDistance < radius) {
									k++;
									continue;
								}
							}
							if (particleIndices(i - 1, j - 1) != -1) {
								Vector2 bottomLeftParticlePosition = particlesPositions[particleIndices(i - 1, j - 1)];
								Scalar neighborParticleDistance = (particlePos - bottomLeftParticlePosition).length();
								if (neighborParticleDistance < radius) {
									k++;
									continue;
								}
							}

							closeToOtherParticles = false;
							k++;
						} while (closeToOtherParticles && k < 30);
					}

					if (!closeToOtherParticles) {
						particleIndices(i, j) = particlesPositions.size();

						particlesPositions.push_back(particlePos);
						particlesResampled.push_back(false);
						particlesVelocities.push_back(Vector2(0, 0));
					}

				}
			}

			return pParticlesData;
		}


		template<>
		ParticlesData<Vector3>* PoissonParticleSampler<Vector3, Array3D>::createSampledParticles() {
			//We are not instantiating particles at the boundaries, that why dim - 2
			int totalNumberOfCells = (m_pGridData->getDimensions().x - 2)*(m_pGridData->getDimensions().y - 2)*(m_pGridData->getDimensions().z - 2);
			ParticlesData<Vector3> *pParticlesData = new ParticlesData<Vector3>(totalNumberOfCells*m_particlesPerCell);
			vector<Vector3> &particlesPositions = pParticlesData->getPositions();
			vector<Vector3> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();

			Scalar dx = m_pGridData->getGridSpacing();
			int particlesPerCell = m_particlesPerCell;
			int resamplingFactor = sqrt(particlesPerCell);
			Scalar radius = (dx / resamplingFactor);
			Vector3 initialBoundary = Vector3(1, 1, 1)*dx;
			Vector3 finalBoundary = Vector3(m_pGridData->getDimensions().x + 1, m_pGridData->getDimensions().y + 1, m_pGridData->getDimensions().z + 1)*dx;
			dimensions_t newGridDimensions(	(finalBoundary.x - initialBoundary.x) / radius, 
											(finalBoundary.y - initialBoundary.y) / radius, 
											(finalBoundary.z - initialBoundary.z) / radius);

			Array3D<int> particleIndices(newGridDimensions);
			particleIndices.assign(-1);
			for (int i = resamplingFactor; i < newGridDimensions.x - resamplingFactor; i++) {
				for (int j = resamplingFactor; j < newGridDimensions.y - resamplingFactor; j++) {
					for (int k = resamplingFactor; k < newGridDimensions.z - resamplingFactor; k++) {
						bool closeToOtherParticles = true;
						Vector2 particlePos;
						if (closeToOtherParticles) {
							int k = 0;
							do
							{
								particlePos = Vector2(i + (rand() / (float)RAND_MAX), j + (rand() / (float)RAND_MAX)) *(dx / resamplingFactor);

								if (i == resamplingFactor || i == newGridDimensions.x - resamplingFactor - 1) {
									closeToOtherParticles = false; continue; k++;
								}
								else if (j == resamplingFactor || j == newGridDimensions.y - resamplingFactor - 1) {
									closeToOtherParticles = false; continue; k++;
								}

								if (particleIndices(i - 1, j, k) != -1) {
									Vector3 leftParticlePosition = particlesPositions[particleIndices(i - 1, j, k)];
									Scalar neighborParticleDistance = (particlePos - leftParticlePosition).length();
									if (neighborParticleDistance < radius) {
										k++;
										continue;
									}
								}
								if (particleIndices(i, j - 1, k) != -1) {
									Vector3 bottomParticlePosition = particlesPositions[particleIndices(i, j - 1, k)];
									Scalar neighborParticleDistance = (particlePos - bottomParticlePosition).length();
									if (neighborParticleDistance < radius) {
										k++;
										continue;
									}
								}
								if (particleIndices(i - 1, j - 1, k) != -1) {
									Vector3 bottomLeftParticlePosition = particlesPositions[particleIndices(i - 1, j - 1, k)];
									Scalar neighborParticleDistance = (particlePos - bottomLeftParticlePosition).length();
									if (neighborParticleDistance < radius) {
										k++;
										continue;
									}
								}

								//k + 1 index
								if (particleIndices(i - 1, j, k + 1) != -1) {
									Vector3 leftParticlePosition = particlesPositions[particleIndices(i - 1, j, k + 1)];
									Scalar neighborParticleDistance = (particlePos - leftParticlePosition).length();
									if (neighborParticleDistance < radius) {
										k++;
										continue;
									}
								}
								if (particleIndices(i, j - 1, k + 1) != -1) {
									Vector3 bottomParticlePosition = particlesPositions[particleIndices(i, j - 1, k + 1)];
									Scalar neighborParticleDistance = (particlePos - bottomParticlePosition).length();
									if (neighborParticleDistance < radius) {
										k++;
										continue;
									}
								}
								if (particleIndices(i - 1, j - 1, k + 1) != -1) {
									Vector3 bottomLeftParticlePosition = particlesPositions[particleIndices(i - 1, j - 1, k + 1)];
									Scalar neighborParticleDistance = (particlePos - bottomLeftParticlePosition).length();
									if (neighborParticleDistance < radius) {
										k++;
										continue;
									}
								}



								closeToOtherParticles = false;
								k++;
							} while (closeToOtherParticles && k < 30);
						}

						if (!closeToOtherParticles) {
							particleIndices(i, j, k) = particlesPositions.size();

							particlesPositions.push_back(particlePos);
							particlesResampled.push_back(false);
							particlesVelocities.push_back(Vector3(0, 0, 0));
						}
					}
				}
			}

			return pParticlesData;
		}
	}
}

