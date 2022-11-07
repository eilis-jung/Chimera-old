#include "ParticleBased/ParticlesToNodalGrid3D.h"


namespace Chimera {


	namespace Advection {

		#pragma region Functionalities
		void ParticlesToNodalGrid3D::transferVelocityToGrid(GridData<Vector3> *pGridData, ParticlesData<Vector3> *pParticlesData) {
			flushAccumulatedVelocities();

			for (int i = 0; i < pParticlesData->getPositions().size(); i++) {
				if (pParticlesData->getResampledParticles()[i] != true) {
					accumulateVelocities(i, pParticlesData, pGridData->getGridSpacing());
				}
			}

			for (int i = 1; i < pGridData->getDimensions().x; i++) {
				for (int j = 1; j < pGridData->getDimensions().y; j++) {
					for (int k = 1; k < pGridData->getDimensions().z; k++) {
						if (m_accVelocityField(i, j, k).weight.x > 0) {
							m_accVelocityField(i, j, k).entry.x /= m_accVelocityField(i, j, k).weight.x;
						}
						if (m_accVelocityField(i, j, k).weight.y > 0) {
							m_accVelocityField(i, j, k).entry.y /= m_accVelocityField(i, j, k).weight.y;
						}
						if (m_accVelocityField(i, j, k).weight.z > 0) {
							m_accVelocityField(i, j, k).entry.z /= m_accVelocityField(i, j, k).weight.z;
						}
					}
					
				}
			}

			if (m_pCutVoxels) {
				Scalar dx = m_pCutVoxels->getGridSpacing();
				for (uint i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {
					auto currCell = m_pCutVoxels->getCutVoxel(i);
					auto verticesMap = currCell.getVerticesMap();
					for (auto iter = verticesMap.begin(); iter != verticesMap.end(); iter++) {
						if (iter->second->getVertexType() == gridVertex) {
							dimensions_t nodeLocation(	iter->second->getPosition().x/dx, 
														iter->second->getPosition().y/dx,
														iter->second->getPosition().z/dx);
							iter->second->setAuxiliaryVelocity(m_accVelocityField(nodeLocation).entry);
						}
						/*DoubleScalar currWeight = iter->second->getWeight();
						if (currWeight > 0.0f) {
							Vector3 currVel = iter->second->getAuxiliaryVelocity();
							iter->second->setAuxiliaryVelocity(currVel / currWeight);
							iter->second->setWeight(0.0f);
						}*/
					}
				}
			}

			/** Setting staggered face intermediate velocities to be the average of nodal-based ones */
			GridData3D *pGridData3D = dynamic_cast<GridData3D*>(pGridData);
			for (int i = 1; i < pGridData3D->getDimensions().x - 1; i++) {
				for (int j = 1; j < pGridData3D->getDimensions().y - 1; j++) {
					for (int k = 1; k < pGridData3D->getDimensions().z - 1; k++) {
						if (m_pCutVoxels == NULL || !m_pCutVoxels->isCutVoxel(i, j, k)) {
							/** Summing up left-velocities */
							DoubleScalar xVelocity = m_accVelocityField(i, j, k).entry.x;
							xVelocity += m_accVelocityField(i, j, k + 1).entry.x;
							xVelocity += m_accVelocityField(i, j + 1, k + 1).entry.x;
							xVelocity += m_accVelocityField(i, j + 1, k).entry.x;
							xVelocity *= 0.25;
							pGridData3D->setAuxiliaryVelocity(xVelocity, i, j, k, xComponent);

							DoubleScalar yVelocity = m_accVelocityField(i, j, k).entry.y;
							yVelocity += m_accVelocityField(i, j, k + 1).entry.y;
							yVelocity += m_accVelocityField(i + 1, j, k + 1).entry.y;
							yVelocity += m_accVelocityField(i + 1, j, k).entry.y;
							yVelocity *= 0.25;
							pGridData3D->setAuxiliaryVelocity(yVelocity, i, j, k, yComponent);

							DoubleScalar zVelocity = m_accVelocityField(i, j, k).entry.z;
							zVelocity += m_accVelocityField(i + 1, j, k).entry.z;
							zVelocity += m_accVelocityField(i + 1, j + 1, k).entry.z;
							zVelocity += m_accVelocityField(i, j + 1, k).entry.z;
							zVelocity *= 0.25;
							pGridData3D->setAuxiliaryVelocity(zVelocity, i, j, k, zComponent);
						}
					}
				}
			}

			Scalar dx = pGridData3D->getGridSpacing();
			if (m_pCutVoxels) {
				for (int i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {
					auto cutVoxel = m_pCutVoxels->getCutVoxel(i);
					auto halfFaces = cutVoxel.getHalfFaces();
					for (int j = 0; j < halfFaces.size(); j++) {
						auto halfEdges = halfFaces[j]->getHalfEdges();
						Vector3 avgVelocity;
						for (int k = 0; k < halfEdges.size(); k++) {
							avgVelocity += halfEdges[k]->getVertices().first->getAuxiliaryVelocity();
						}
						avgVelocity /= halfEdges.size(); 

						halfFaces[j]->getFace()->setAuxiliaryVelocity(avgVelocity);
					}
				}
			}
		}

		void ParticlesToNodalGrid3D::transferScalarAttributesToGrid(GridData<Vector3> *pGridData, ParticlesData<Vector3> *pParticlesData) {
			flushAccumulatedScalarAttributes();

			for (auto it = m_accScalarFields.begin(); it != m_accScalarFields.end(); ++it) {
				accumulateScalarFieldValues(it->first, pParticlesData, pGridData->getGridSpacing());

				for (int i = 1; i < it->second.getDimensions().x - 1; i++) {
					for (int j = 1; j < it->second.getDimensions().y - 1; j++) {
						for (int k = 1; k < it->second.getDimensions().z - 1; k++) {
							if (it->second(i, j, k).weight > 0) {
								it->second(i, j, k).entry /= it->second(i, j, k).weight;
							}
						}
					}
				}

				if (it->first == "density") {
					//TODO: link user-defined fields in pGridData3D
					GridData3D *pGridData3D = dynamic_cast<GridData3D*>(pGridData);
					for (int i = 1; i < it->second.getDimensions().x - 1; i++) {
						for (int j = 1; j < it->second.getDimensions().y - 1; j++) {
							for (int k = 1; k < it->second.getDimensions().z - 1; k++) {
								pGridData3D->getDensityBuffer().setValue(it->second(i, j, k).entry, i, j, k);
							}
						}
					}
				}	
			}
		}

		#pragma endregion

		#pragma region PrivateFunctionalities
		void ParticlesToNodalGrid3D::flushAccumulatedVelocities() {
			ParticlesToGrid<Vector3, Array3D>::flushAccumulatedVelocities();
			if (m_pCutVoxelsVelocities) {
				m_pCutVoxelsVelocities->zeroVelocities();
				m_pCutVoxelsVelocities->zeroWeights();
			}
			

			Vector3 zeroVector(0, 0, 0);
			if (m_pCutVoxels) {
				for (uint i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {
					auto currCell = m_pCutVoxels->getCutVoxel(i);
					auto verticesMap = currCell.getVerticesMap();
					for (auto iter = verticesMap.begin(); iter != verticesMap.end(); iter++) {
						if (iter->second->getVertexType() == gridVertex) {
							Vector3 currVel = iter->second->getAuxiliaryVelocity();
							iter->second->setAuxiliaryVelocity(zeroVector);
						}
					}
				}
			}
		}

		void ParticlesToNodalGrid3D::accumulateVelocities(int ithParticle, int ithCutVoxel, ParticlesData<Vector3> *pParticlesData, Scalar dx) {
			Vector3 particlePosition = pParticlesData->getPositions()[ithParticle];

			auto currCell = m_pCutVoxels->getCutVoxel(ithCutVoxel);
			auto verticesMap = currCell.getVerticesMap();
			for (auto iter = verticesMap.begin(); iter != verticesMap.end(); iter++) {
				if (iter->second->getVertexType() == gridVertex) {
					Vector3 targetNode = iter->second->getPosition();
					dimensions_t cellIndex(targetNode.x / dx, targetNode.y / dx, targetNode.z / dx);
					Vector3 weight;
					weight.x = weight.y = weight.z = m_pKernel->calculateKernel(particlePosition, targetNode, (particlePosition - targetNode).length());
					accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), cellIndex, fullVector);
				}
				// else if () { treat free-slip here

				//}
			}
		}

		void ParticlesToNodalGrid3D::accumulateVelocities(int ithParticle, ParticlesData<Vector3> *pParticlesData, Scalar dx) {
			Vector3 position = pParticlesData->getPositions()[ithParticle] / dx;
			int i, j, k;
			i = floor(position.x);
			j = floor(position.y);
			k = floor(position.z);

			if (m_pCutVoxels) {
				if (m_pCutVoxels->isCutVoxel(i, j, k)) {
					accumulateVelocities(ithParticle, m_pCutVoxels->getCutVoxelIndex(position), pParticlesData, dx);
					return;
				}
			}

			Vector3 cellOrigin(i, j, k);

			Vector3 weight;
			weight[0] = weight[1] = weight[2] = m_pKernel->calculateKernel(position, cellOrigin, (position - cellOrigin).length()*dx);
			accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j, k), fullVector);

			Vector3 targetNode = cellOrigin + Vector3(1, 0, 0);
			weight[0] = weight[1] = weight[2] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (i < m_gridDimensions.x - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j, k), fullVector);
			}

			targetNode = cellOrigin + Vector3(0, 1, 0);
			weight[0] = weight[1] = weight[2] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (j < m_gridDimensions.y - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j + 1, k), fullVector);
			}

			targetNode = cellOrigin + Vector3(1, 1, 0);
			weight[0] = weight[1] = weight[2] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j + 1, k), fullVector);
			}

			//k + 1 indices
			targetNode = cellOrigin + Vector3(0, 0, 1);
			weight[0] = weight[1] = weight[2] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (k < m_gridDimensions.z - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j, k + 1), fullVector);
			}
			
			targetNode = cellOrigin + Vector3(1, 0, 1);
			weight[0] = weight[1] = weight[2] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (i < m_gridDimensions.x - 1 && k < m_gridDimensions.z - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j, k + 1), fullVector);
			}

			targetNode = cellOrigin + Vector3(0, 1, 1);
			weight[0] = weight[1] = weight[2] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (j < m_gridDimensions.y - 1 && k < m_gridDimensions.z - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j + 1, k + 1), fullVector);
			}

			targetNode = cellOrigin + Vector3(1, 1, 1);
			weight[0] = weight[1] = weight[2] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1 && k < m_gridDimensions.z - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j + 1, k + 1), fullVector);
			}
		}

		//TODO: bring back scalar-field accumulation
		void ParticlesToNodalGrid3D::accumulateScalarFieldValues(string scalarFieldName, ParticlesData<Vector3> *pParticlesData, Scalar dx) {

			ScalarArray &scalarFieldAccumulator = getScalarAttributeArray(scalarFieldName);
			const vector<Scalar> &particlesScalarAttribute = pParticlesData->getScalarBasedAttribute(scalarFieldName);

			for (int ithParticle = 0; ithParticle < pParticlesData->getPositions().size(); ithParticle++) {

				int i, j, k;
				//Vector2 relativeFractions;

				if (pParticlesData->getResampledParticles()[ithParticle])
					continue;

				Vector3 position = pParticlesData->getPositions()[ithParticle] / dx;

				i = floor(position.x - 0.5f);
				j = floor(position.y - 0.5f);
				k = floor(position.z - 0.5f);



				/*
				relativeFractions.x = position.x - (i + 0.5);
				relativeFractions.y = position.y - (j + 0.5);


				Scalar totalWeight = 0;
				Scalar weight = (1 - relativeFractions.x)*(1 - relativeFractions.y);
				totalWeight += weight;
				accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i, j));

				weight = relativeFractions.x*(1 - relativeFractions.y);
				totalWeight += weight;
				if (i < m_gridDimensions.x - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i + 1, j));
				}

				weight = (1 - relativeFractions.x)*relativeFractions.y;
				totalWeight += weight;
				if (j < m_gridDimensions.y - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i, j + 1));
				}

				weight = relativeFractions.x*relativeFractions.y;
				totalWeight += weight;
				if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i + 1, j + 1));
				}*/
			}

		}
		#pragma endregion
	}
}
