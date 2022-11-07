#include "Particles/ParticlesSampler.h"

namespace Chimera {

	namespace Particles {

		#pragma region Functionalities
		template <class VectorType, template <class> class ArrayType>
		void ParticlesSampler<VectorType, ArrayType>::interpolateVelocities(Interpolation::Interpolant<VectorType, ArrayType, VectorType> *pInterpolant, ParticlesData<VectorType> *pParticlesData) {
			for (int i = 0; i < pParticlesData->getVelocities().size(); i++) {
				pParticlesData->getVelocities()[i] = pInterpolant->interpolate(pParticlesData->getPositions()[i]);
			}
		}

		template<>
		void ParticlesSampler<Vector2, Array2D>::resampleParticles(ParticlesData<Vector2> *pParticlesData) {

			vector<Vector2> &particlesPositions = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();
			
			Scalar dx = m_pGridData->getGridSpacing();
			Array2D<int> m_particlesCount(m_pGridData->getDimensions());

			m_particlesCount.assign(0);
			for (int i = 0; i < particlesPositions.size(); i++) {
				particlesResampled[i] = false;
				boundaryResample(i, pParticlesData);
				
				int indexI = particlesPositions[i].x/dx, indexJ = particlesPositions[i].y/dx;
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
				Vector2 gridSpacePosition = particlesPositions[i]/dx;
				if (m_pCutCells) {
					CutCells2D<Vector2> *pCutCells2D = dynamic_cast<CutCells2D<Vector2> *>(m_pCutCells);
					if (pCutCells2D->isCutCellAt(gridSpacePosition.x, gridSpacePosition.y)) {
						int cutCellIndex = pCutCells2D->getCutCellIndex(gridSpacePosition);
						if (pCutCells2D->getCutCell(cutCellIndex).getDistanceToBoundary(particlesPositions[i]) < 0.001) {
							cellParticleCount_t top = pq.top();
							m_particlesCount(gridSpacePosition.x, gridSpacePosition.y) -= 1;
							pq.pop();
							particlesPositions[i] = Vector2(top.cellIndex.x + (rand() / (float)RAND_MAX), top.cellIndex.y + (rand() / (float)RAND_MAX))*dx;
							particlesResampled[i] = true;
							top.count++;
							pq.push(top);
						}
					}
				}
				if (m_particlesCount(gridSpacePosition.x, gridSpacePosition.y) > maxParticles) {
					cellParticleCount_t top = pq.top();
					m_particlesCount(gridSpacePosition.x, gridSpacePosition.y) -= 1;
					pq.pop();
					particlesPositions[i] = Vector2(top.cellIndex.x + (rand() / (float)RAND_MAX), top.cellIndex.y + (rand() / (float)RAND_MAX))*dx;
					particlesResampled[i] = true;
					top.count++;
					pq.push(top);
				}
			}
		}

		template<>
		void ParticlesSampler<Vector3, Array3D>::resampleParticles(ParticlesData<Vector3> *pParticlesData) {

			vector<Vector3> &particlesPositions = pParticlesData->getPositions();
			vector<Vector3> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();

			Scalar dx = m_pGridData->getGridSpacing();
			Array3D<int> m_particlesCount(m_pGridData->getDimensions());

			m_particlesCount.assign(0);
			for (int i = 0; i < particlesPositions.size(); i++) {
				particlesResampled[i] = false;
				boundaryResample(i, pParticlesData);

				int indexI = particlesPositions[i].x / dx, indexJ = particlesPositions[i].y / dx, indexK = particlesPositions[i].z/dx;
				m_particlesCount(indexI, indexJ, indexK) += 1;
			}

			priority_queue<cellParticleCount_t, vector<cellParticleCount_t>, CountCompareNode> pq;
			for (int i = 1; i < m_particlesCount.getDimensions().x - 1; i++) {
				for (int j = 1; j < m_particlesCount.getDimensions().y - 1; j++) {
					for (int k = 1; k < m_particlesCount.getDimensions().z - 1; k++) {
						pq.push(cellParticleCount_t(dimensions_t(i, j, k), m_particlesCount(i, j, k)));
					}
				}
			}

			for (int i = 0; i < particlesPositions.size(); i++) {
				int maxParticles = ceil(m_particlesPerCell*1.05f);
				Vector3 gridSpacePosition = particlesPositions[i] / dx;
				int currParCount = m_particlesCount(gridSpacePosition.x, gridSpacePosition.y, gridSpacePosition.z);
				if (m_particlesCount(gridSpacePosition.x, gridSpacePosition.y, gridSpacePosition.z) > maxParticles) {
					cellParticleCount_t top = pq.top();
					m_particlesCount(gridSpacePosition.x, gridSpacePosition.y, gridSpacePosition.z) -= 1;
					pq.pop();
					particlesPositions[i] = Vector3(top.cellIndex.x + (rand() / (float)RAND_MAX), 
													top.cellIndex.y + (rand() / (float)RAND_MAX),
													top.cellIndex.z + (rand() / (float)RAND_MAX))*dx;
					particlesResampled[i] = true;
					top.count++;
					pq.push(top);
				}
			}
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template<> 
		void ParticlesSampler<Vector2, Array2D>::boundaryResample(int ithParticle, ParticlesData<Vector2> *pParticlesData) {
			vector<Vector2> &particlesPositions = pParticlesData->getPositions();
			vector<Vector2> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();
			Scalar dx = m_pGridData->getGridSpacing();

			Vector2 gridSpaceParticle = particlesPositions[ithParticle] / dx;

			Scalar maxBoundsX = (m_pGridData->getDimensions().x - 1)*dx;
			Scalar maxBoundsY = (m_pGridData->getDimensions().y - 1)*dx;

			if (gridSpaceParticle.x > m_pGridData->getDimensions().x - 1) {
				particlesPositions[ithParticle].x = ((rand() / (float)RAND_MAX) + 1) * dx;
				particlesPositions[ithParticle].y = clamp<Scalar>(particlesPositions[ithParticle].y, 0, maxBoundsY);
				particlesResampled[ithParticle] = true;
			}
			else if (gridSpaceParticle.x < 1) {
				particlesPositions[ithParticle].x = ((rand() / (float)RAND_MAX) + m_pGridData->getDimensions().x - 2) * dx;
				particlesPositions[ithParticle].y = clamp<Scalar>(particlesPositions[ithParticle].y, 0, maxBoundsY);
				particlesResampled[ithParticle] = true;
			}

			if (gridSpaceParticle.y > m_pGridData->getDimensions().y - 1) {
				JetBC<Vector2> *pSouthJetBoundary = nullptr;
				for (int i = 0; i < m_boundaryConditions.size(); i++) {
					if (m_boundaryConditions[i]->getLocation() == South && m_boundaryConditions[i]->getType() == Jet) {
						pSouthJetBoundary = dynamic_cast<JetBC<Vector2> *>(m_boundaryConditions[i]);
					}
				}
				if (pSouthJetBoundary) {
					Scalar centralPointX = (m_pGridData->getDimensions().x + 1)*dx*0.5;
					//particlesPositions[ithParticle].y = ((rand() / (float)RAND_MAX) + 1) * dx;
					particlesPositions[ithParticle].y = dx*(0.05 + 1);
					particlesPositions[ithParticle].x = centralPointX + ((rand() / (float)RAND_MAX) - 0.5)*2*pSouthJetBoundary->getSize();
					particlesResampled[ithParticle] = true;
				}
				else {
					particlesPositions[ithParticle].y = ((rand() / (float)RAND_MAX) + 1) * dx;
					particlesPositions[ithParticle].x = clamp<Scalar>(particlesPositions[ithParticle].x, 0, maxBoundsX);
					particlesResampled[ithParticle] = true;
				}	
			}
			else if (gridSpaceParticle.y < 1) {
				particlesPositions[ithParticle].y = ((rand() / (float)RAND_MAX) + m_pGridData->getDimensions().x - 2) * dx;
				particlesPositions[ithParticle].x = clamp<Scalar>(particlesPositions[ithParticle].x, 0, maxBoundsX);
				particlesResampled[ithParticle] = true;
			}

			particlesPositions[ithParticle].x = clamp<Scalar>(particlesPositions[ithParticle].x, 0, maxBoundsX);
			particlesPositions[ithParticle].y = clamp<Scalar>(particlesPositions[ithParticle].y, 0, maxBoundsY);
		}

		template<>
		void ParticlesSampler<Vector3, Array3D>::boundaryResample(int ithParticle, ParticlesData<Vector3> *pParticlesData) {
			vector<Vector3> &particlesPositions = pParticlesData->getPositions();
			vector<Vector3> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();
			Scalar dx = m_pGridData->getGridSpacing();

			Vector3 gridSpaceParticle = particlesPositions[ithParticle] / dx;

			Scalar maxBoundsX = (m_pGridData->getDimensions().x - 1)*dx;
			Scalar maxBoundsY = (m_pGridData->getDimensions().y - 1)*dx;
			Scalar maxBoundsZ = (m_pGridData->getDimensions().z - 1)*dx;

			if (gridSpaceParticle.x > m_pGridData->getDimensions().x - 1) {
				particlesPositions[ithParticle].x = ((rand() / (float)RAND_MAX) + 1) * dx;
				particlesResampled[ithParticle] = true;
			}
			else if (gridSpaceParticle.x < 1) {
				particlesPositions[ithParticle].x = ((rand() / (float)RAND_MAX) + m_pGridData->getDimensions().x - 2) * dx;
				particlesResampled[ithParticle] = true;
			}

			if (gridSpaceParticle.y > m_pGridData->getDimensions().y - 1) {
				particlesPositions[ithParticle].y = ((rand() / (float)RAND_MAX) + 1) * dx;
				particlesResampled[ithParticle] = true;
			}
			else if (gridSpaceParticle.y < 1) {
				particlesPositions[ithParticle].y = ((rand() / (float)RAND_MAX) + m_pGridData->getDimensions().y - 2) * dx;
				particlesResampled[ithParticle] = true;
			}

			if (gridSpaceParticle.z > m_pGridData->getDimensions().z - 1) {
				particlesPositions[ithParticle].z = ((rand() / (float)RAND_MAX) + 1) * dx;
				particlesResampled[ithParticle] = true;
			}
			else if (gridSpaceParticle.z < 1) {
				particlesPositions[ithParticle].z = ((rand() / (float)RAND_MAX) + m_pGridData->getDimensions().z - 2) * dx;
				particlesResampled[ithParticle] = true;
			}
		}

		template<>
		ParticlesData<Vector2> * ParticlesSampler<Vector2, Array2D>::createSampledParticles() {
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
				for (int j = refactorScale; j < newGridDimensions.y - refactorScale; j++) {
					Vector2 particlePosition = Vector2(i + (rand() / (float)RAND_MAX), j + (rand() / (float)RAND_MAX));
					particlePosition /= refactorScale;
					particlesPositions.push_back(particlePosition*dx);
					particlesVelocities.push_back(Vector2());

					particlesResampled.push_back(false);
				}
			}

			return pParticlesData;
		}

		template<>
		ParticlesData<Vector3> * ParticlesSampler<Vector3, Array3D>::createSampledParticles() {
			//We are not instantiating particles at the boundaries, that why dim - 2
			int totalNumberOfCells = (m_pGridData->getDimensions().x - 2)*(m_pGridData->getDimensions().y - 2)*(m_pGridData->getDimensions().z - 2);

			ParticlesData<Vector3> *pParticlesData = new ParticlesData<Vector3>(totalNumberOfCells*m_particlesPerCell);

			Scalar dx = m_pGridData->getGridSpacing();
			//This will subdivide the cells according with number of particles per cell to provide better sampling
			int refactorScale = floor(cbrt(m_particlesPerCell));

			vector<Vector3> &particlesPositions = pParticlesData->getPositions();
			vector<Vector3> &particlesVelocities = pParticlesData->getVelocities();
			vector<bool> &particlesResampled = pParticlesData->getResampledParticles();

			dimensions_t newGridDimensions(m_pGridData->getDimensions().x*refactorScale, m_pGridData->getDimensions().y*refactorScale, m_pGridData->getDimensions().z*refactorScale);
			for (int i = refactorScale; i < newGridDimensions.x - refactorScale; i++) {
				for (int j = refactorScale; j < newGridDimensions.y - refactorScale; j++) {
					for (int k = refactorScale; k < newGridDimensions.z - refactorScale; k++) {
						Vector3 particlePosition = Vector3(i + (rand() / (float)RAND_MAX), j + (rand() / (float)RAND_MAX), k + (rand() / (float)RAND_MAX));
						particlePosition /= refactorScale;
						particlesPositions.push_back(particlePosition*dx);
						particlesVelocities.push_back(Vector3(0, 0, 0));

						particlesResampled.push_back(false);
					}

				}
			}

			return pParticlesData;
		}

		#pragma endregion

		template class ParticlesSampler<Vector2, Array2D>;
		template class ParticlesSampler<Vector3, Array3D>;
	}


}