#include "ParticleBased/ParticlesToStaggeredGrid3D.h"


namespace Chimera {


	namespace Advection {

		#pragma region Functionalities
		void ParticlesToStaggeredGrid3D::transferVelocityToGrid(GridData<Vector3> *pGridData, ParticlesData<Vector3> *pParticlesData) {
			flushAccumulatedVelocities();

			for (int i = 0; i < pParticlesData->getPositions().size(); i++) {
				if (pParticlesData->getResampledParticles()[i] != true) {
					accumulateVelocities(i, pParticlesData, pGridData->getGridSpacing(), xComponent);
					accumulateVelocities(i, pParticlesData, pGridData->getGridSpacing(), yComponent);
					accumulateVelocities(i, pParticlesData, pGridData->getGridSpacing(), zComponent);
				}
			}

			for (int i = 1; i < pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < pGridData->getDimensions().y - 1; j++) {
					for (int k = 1; k < pGridData->getDimensions().z - 1; k++) {
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

			GridData3D *pGridData3D = dynamic_cast<GridData3D*>(pGridData);
			for (int i = 1; i < pGridData3D->getDimensions().x - 1; i++) {
				for (int j = 1; j < pGridData3D->getDimensions().y - 1; j++) {
					for (int k = 1; k < pGridData->getDimensions().z - 1; k++) {
						pGridData3D->setAuxiliaryVelocity(m_accVelocityField(i, j, k).entry, i, j, k);
					}
				}
			}
		}

		void ParticlesToStaggeredGrid3D::transferScalarAttributesToGrid(GridData<Vector3> *pGridData, ParticlesData<Vector3> *pParticlesData) {
			flushAccumulatedScalarAttributes();

			for (auto it = m_accScalarFields.begin(); it != m_accScalarFields.end(); ++it) {
				accumulateScalarFieldValues(it->first, pParticlesData, pGridData->getGridSpacing());

				for (int i = 1; i < it->second.getDimensions().x - 1; i++) {
					for (int j = 1; j < it->second.getDimensions().y - 1; j++) {
						for (int k = 1; k < it->second.getDimensions().z - 1; k++) {
							if (it->second(i, j, k).weight > 0) {
								Scalar weight = it->second(i, j, k).weight;
								Scalar entry = it->second(i, j, k).entry / weight;
								it->second(i, j, k).entry /= it->second(i, j, k).weight;
							}
						}
					}
				}

				if (it->first == "density") {
					//TODO: link user-defined fields in pGridData2D
					GridData3D *pGridData3D = dynamic_cast<GridData3D*>(pGridData);
					for (int i = 1; i < it->second.getDimensions().x - 1; i++) {
						for (int j = 1; j < it->second.getDimensions().y - 1; j++) {
							for (int k = 1; k < it->second.getDimensions().z - 1; k++) {
								pGridData3D->getDensityBuffer().setValue(it->second(i, j, k).entry, i, j, k);

							}
						}
					}
				}
				else if (it->first == "temperature") {
					GridData3D *pGridData3D = dynamic_cast<GridData3D*>(pGridData);
					for (int i = 1; i < it->second.getDimensions().x - 1; i++) {
						for (int j = 1; j < it->second.getDimensions().y - 1; j++) {
							for (int k = 1; k < it->second.getDimensions().z - 1; k++) {
								pGridData3D->getTemperatureBuffer().setValue(it->second(i, j, k).entry, i, j, k);
							}
						}
					}
				}
			}


		}

		#pragma endregion

		#pragma region PrivateFunctionalities
		void ParticlesToStaggeredGrid3D::accumulateVelocities(int ithParticle, ParticlesData<Vector3> *pParticlesData, Scalar dx, velocityComponent_t velocityComponent) {
			int i, j, k;
			Vector3 position = pParticlesData->getPositions()[ithParticle] / dx;
			Vector3 cellOrigin;
			if (velocityComponent == velocityComponent_t::xComponent) { //Staggered grid centered on x velocity components
				i = floor(position.x);
				j = floor(position.y - 0.5f);
				k = floor(position.z - 0.5f);
				i = clamp(i, 0, m_gridDimensions.x - 1);
				j = clamp(j, 0, m_gridDimensions.y - 1);
				k = clamp(k, 0, m_gridDimensions.z - 1);

				cellOrigin.x = i;
				cellOrigin.y = j + 0.5;
				cellOrigin.z = k + 0.5;
			}
			else if (velocityComponent == velocityComponent_t::yComponent) { //Staggered grid centered on y velocity components
				i = floor(position.x - 0.5f);
				j = floor(position.y);
				k = floor(position.z - 0.5f);
				i = clamp(i, 0, m_gridDimensions.x - 1);
				j = clamp(j, 0, m_gridDimensions.y - 1);
				k = clamp(k, 0, m_gridDimensions.z - 1);
				cellOrigin.x = i + 0.5;
				cellOrigin.y = j;
				cellOrigin.z = k + 0.5;
			}
			else if (velocityComponent == velocityComponent_t::zComponent) {//Staggered grid centered on z velocity components
				i = floor(position.x - 0.5f);
				j = floor(position.y - 0.5f);
				k = floor(position.z);
				i = clamp(i, 0, m_gridDimensions.x - 1);
				j = clamp(j, 0, m_gridDimensions.y - 1);
				k = clamp(k, 0, m_gridDimensions.z - 1);
				cellOrigin.x = i + 0.5;
				cellOrigin.y = j + 0.5;
				cellOrigin.z = k;
			}

			Vector3 positionParticle = pParticlesData->getPositions()[ithParticle];
			Vector3 velocityParticle = pParticlesData->getVelocities()[ithParticle];
			Vector3 relativeFractions;
			relativeFractions.x = position.x - cellOrigin.x;
			relativeFractions.y = position.y - cellOrigin.y;
			relativeFractions.z = position.z - cellOrigin.z;

			Scalar totalWeight = 0;
			Vector3 weight;
			weight[velocityComponent] = m_pKernel->calculateKernel(position, cellOrigin, (position - cellOrigin).length()*dx);
			Scalar weightTemp = (position - cellOrigin).length()*dx;
			weight[velocityComponent] = 1 / (weightTemp*weightTemp);
			weight[velocityComponent] = (1 - relativeFractions.x)*(1 - relativeFractions.y)*(1 - relativeFractions.z);
			accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j, k), velocityComponent);

			Vector3 targetNode = cellOrigin + Vector3(1, 0, 0);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			weightTemp = (position - targetNode).length()*dx;
			weight[velocityComponent] = 1 / (weightTemp*weightTemp);
			weight[velocityComponent] = (relativeFractions.x)*(1 - relativeFractions.y)*(1 - relativeFractions.z);
			if (i < m_gridDimensions.x - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j, k), velocityComponent);
			}

			targetNode = cellOrigin + Vector3(0, 1, 0);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			weightTemp = (position - targetNode).length()*dx;
			weight[velocityComponent] = 1 / (weightTemp*weightTemp);
			weight[velocityComponent] = (1 - relativeFractions.x)*(relativeFractions.y)*(1 - relativeFractions.z);
			if (j < m_gridDimensions.y - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j + 1, k), velocityComponent);
			}

			targetNode = cellOrigin + Vector3(1, 1, 0);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			weightTemp = (position - targetNode).length()*dx;
			weight[velocityComponent] = 1 / (weightTemp*weightTemp);
			weight[velocityComponent] = (relativeFractions.x)*(relativeFractions.y)*(1 - relativeFractions.z);
			if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j + 1, k), velocityComponent);
			}

			targetNode = cellOrigin + Vector3(0, 0, 1);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, cellOrigin, (position - targetNode).length()*dx);
			weightTemp = (position - targetNode).length()*dx;
			weight[velocityComponent] = 1 / (weightTemp*weightTemp);
			weight[velocityComponent] = (1 - relativeFractions.x)*(1 - relativeFractions.y)*(relativeFractions.z);
			if (k < m_gridDimensions.z - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j, k + 1), velocityComponent);
			}

			targetNode = cellOrigin + Vector3(1, 0, 1);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			weightTemp = (position - targetNode).length()*dx;
			weight[velocityComponent] = 1 / (weightTemp*weightTemp);
			weight[velocityComponent] = (relativeFractions.x)*(1 - relativeFractions.y)*(relativeFractions.z);
			if (i < m_gridDimensions.x - 1 && k < m_gridDimensions.z - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j, k + 1), velocityComponent);
			}

			targetNode = cellOrigin + Vector3(0, 1, 1);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			weightTemp = (position - targetNode).length()*dx;
			weight[velocityComponent] = 1 / (weightTemp*weightTemp);
			weight[velocityComponent] = (1 - relativeFractions.x)*(relativeFractions.y)*(relativeFractions.z);
			if (j < m_gridDimensions.y - 1 && k < m_gridDimensions.z - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j + 1, k + 1), velocityComponent);
			}

			targetNode = cellOrigin + Vector3(1, 1, 1);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			weightTemp = (position - targetNode).length()*dx;
			weight[velocityComponent] = 1 / (weightTemp*weightTemp);
			weight[velocityComponent] = (relativeFractions.x)*(relativeFractions.y)*(relativeFractions.z);
			if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1 && k < m_gridDimensions.z - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j + 1, k + 1), velocityComponent);
			}
		}

		void ParticlesToStaggeredGrid3D::accumulateScalarFieldValues(string scalarFieldName, ParticlesData<Vector3> *pParticlesData, Scalar dx) {

			ScalarArray &scalarFieldAccumulator = getScalarAttributeArray(scalarFieldName);
			const vector<Scalar> &particlesScalarAttribute = pParticlesData->getScalarBasedAttribute(scalarFieldName);

			for (int ithParticle = 0; ithParticle < pParticlesData->getPositions().size(); ithParticle++) {
				int i, j, k;

				if(pParticlesData->getResampledParticles()[ithParticle])
					continue;
				
				Vector3 position = pParticlesData->getPositions()[ithParticle] / dx;

				i = floor(position.x - 0.5f);
				j = floor(position.y - 0.5f);
				k = floor(position.z - 0.5f);

				i = clamp(i, 0, m_gridDimensions.x - 1);
				j = clamp(j, 0, m_gridDimensions.y - 1);
				k = clamp(k, 0, m_gridDimensions.z - 1);

				
				Vector3 cellOrigin(i + 0.5f, j + 0.5, k + 0.5);
				Vector3 relativeFractions;
				relativeFractions.x = position.x - cellOrigin.x;
				relativeFractions.y = position.y - cellOrigin.y;
				relativeFractions.z = position.z - cellOrigin.z;
				Scalar weight = m_pKernel->calculateKernel(position, cellOrigin, (position - cellOrigin).length()*dx);
				weight = (1 - relativeFractions.x)*(1 - relativeFractions.y)*(1 - relativeFractions.z);
				Scalar weightTemp = (position - cellOrigin).length()*dx;
				//weight = 1 / (weightTemp*weightTemp);
				accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i, j, k));
				
				Vector3 targetNode = cellOrigin + Vector3(1, 0, 0);
				weight = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
				//weight = (relativeFractions.x)*(1 - relativeFractions.y)*(1 - relativeFractions.z);
				weightTemp = (position - targetNode).length()*dx;
				weight = 1 / (weightTemp*weightTemp);
				if (i < m_gridDimensions.x - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i + 1, j, k));
				}

				targetNode = cellOrigin + Vector3(0, 1, 0);
				weight = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
				//weight = (1 - relativeFractions.x)*(relativeFractions.y)*(1 - relativeFractions.z);
				weightTemp = (position - targetNode).length()*dx;
				weight = 1 / (weightTemp*weightTemp);
				if (j < m_gridDimensions.y - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i, j + 1, k));
				}

				targetNode = cellOrigin + Vector3(1, 1, 0);
				weight = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
				//weight = (relativeFractions.x)*(relativeFractions.y)*(1 - relativeFractions.z);
				weightTemp = (position - targetNode).length()*dx;
				weight = 1 / (weightTemp*weightTemp);
				if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i + 1, j + 1, k));
				}

				targetNode = cellOrigin + Vector3(0, 0, 1);
				weight = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
				//weight = (1 - relativeFractions.x)*(1 - relativeFractions.y)*(relativeFractions.z);
				weightTemp = (position - targetNode).length()*dx;
				weight = 1 / (weightTemp*weightTemp);
				if (k < m_gridDimensions.z - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i, j, k + 1));
				}
				
				targetNode = cellOrigin + Vector3(1, 0, 1);
				weight = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
				//weight = (relativeFractions.x)*(1 - relativeFractions.y)*( relativeFractions.z);
				weightTemp = (position - targetNode).length()*dx;
				weight = 1 / (weightTemp*weightTemp);
				if (i < m_gridDimensions.x - 1 && k < m_gridDimensions.z - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i + 1, j, k + 1));
				}

				targetNode = cellOrigin + Vector3(0, 1, 1);
				weight = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
				//weight = (1 - relativeFractions.x)*(relativeFractions.y)*(relativeFractions.z);
				weightTemp = (position - targetNode).length()*dx;
				weight = 1 / (weightTemp*weightTemp);
				if (j < m_gridDimensions.y - 1 && k < m_gridDimensions.z - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i, j + 1, k + 1));
				}

				targetNode = cellOrigin + Vector3(1, 1, 1);
				weight = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
				//weight = (relativeFractions.x)*(relativeFractions.y)*(relativeFractions.z);
				weightTemp = (position - targetNode).length()*dx;
				weight = 1 / (weightTemp*weightTemp);
				if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1 && k < m_gridDimensions.z - 1) {
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i + 1, j + 1, k + 1));
				}
			}

		}
		#pragma endregion
	}
}
