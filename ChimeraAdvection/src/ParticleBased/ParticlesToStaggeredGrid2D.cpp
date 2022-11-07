#include "ParticleBased/ParticlesToStaggeredGrid2D.h"


namespace Chimera {


	namespace Advection {

		#pragma region Functionalities
		void ParticlesToStaggeredGrid2D::transferVelocityToGrid(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {
			flushAccumulatedVelocities();

			for (int i = 0; i < pParticlesData->getPositions().size(); i++) {
				if (pParticlesData->getResampledParticles()[i] != true) {
					accumulateVelocities(i, pParticlesData, pGridData->getGridSpacing(), xComponent);
					accumulateVelocities(i, pParticlesData, pGridData->getGridSpacing(), yComponent);
				}
			}

			for (int i = 1; i < pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < pGridData->getDimensions().y - 1; j++) {
					if (m_accVelocityField(i, j).weight.x > 0) {
						m_accVelocityField(i, j).entry.x /= m_accVelocityField(i, j).weight.x;
					}
					if (m_accVelocityField(i, j).weight.y > 0) {
						m_accVelocityField(i, j).entry.y /= m_accVelocityField(i, j).weight.y;
					}
				}
			}

			GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);
			for (int i = 1; i < pGridData2D->getDimensions().x - 1; i++) {
				for (int j = 1; j < pGridData2D->getDimensions().y - 1; j++) {
					pGridData2D->setAuxiliaryVelocity(m_accVelocityField(i, j).entry, i, j);
				}
			}
		}

		void ParticlesToStaggeredGrid2D::transferScalarAttributesToGrid(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {
			flushAccumulatedScalarAttributes();

			for (auto it = m_accScalarFields.begin(); it != m_accScalarFields.end(); ++it) {
				accumulateScalarFieldValues(it->first, pParticlesData, pGridData->getGridSpacing());

				for (int i = 1; i < it->second.getDimensions().x - 1; i++) {
					for (int j = 1; j < it->second.getDimensions().y - 1; j++) {
						if (it->second(i, j).weight > 0) {
							it->second(i, j).entry /= it->second(i, j).weight;					
						}
					}
				}

				if (it->first == "density") {
					//TODO: link user-defined fields in pGridData2D
					GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);
					for (int i = 1; i < it->second.getDimensions().x - 1; i++) {
						for (int j = 1; j < it->second.getDimensions().y - 1; j++) {
							pGridData2D->getDensityBuffer().setValue(it->second(i, j).entry, i, j);
						}
					}
				}
				else if (it->first == "temperature") {
					//TODO: link user-defined fields in pGridData2D
					GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);
					for (int i = 1; i < it->second.getDimensions().x - 1; i++) {
						for (int j = 1; j < it->second.getDimensions().y - 1; j++) {
							pGridData2D->getTemperatureBuffer().setValue(it->second(i, j).entry, i, j);
						}
					}
				}
				else if (it->first == "vorticity") {
					//TODO: link user-defined fields in pGridData2D
					GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);
					for (int i = 1; i < it->second.getDimensions().x - 1; i++) {
						for (int j = 1; j < it->second.getDimensions().y - 1; j++) {
							//LS - TODO: change to auxiliary vorticities
							pGridData2D->setVorticity(it->second(i, j).entry, i, j);
						}
					}
				}
			}


		}

		#pragma endregion

		#pragma region PrivateFunctionalities
		void ParticlesToStaggeredGrid2D::accumulateVelocities(int ithParticle, ParticlesData<Vector2> *pParticlesData, Scalar dx, velocityComponent_t velocityComponent) {
			int i, j;
			Vector2 relativeFractions;

			Vector2 position = pParticlesData->getPositions()[ithParticle] / dx;
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


			Vector2 positionParticle = pParticlesData->getPositions()[ithParticle];
			Vector2 velocityParticle = pParticlesData->getVelocities()[ithParticle];

			Scalar totalWeight = 0;
			Vector2 weight;
			weight[velocityComponent] = m_pKernel->calculateKernel(position, cellOrigin, (position - cellOrigin).length()*dx);
			weight[velocityComponent] = (1 - relativeFractions.x)*(1 - relativeFractions.y);
			//totalWeight += weight;
			accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j), velocityComponent);

			Vector2 targetNode = cellOrigin + Vector2(1, 0);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			weight[velocityComponent] = relativeFractions.x*(1 - relativeFractions.y);
			//totalWeight += weight;
			if (i < m_gridDimensions.x - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j), velocityComponent);
			}

			targetNode = cellOrigin + Vector2(0, 1);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			weight[velocityComponent] = (1 - relativeFractions.x)*relativeFractions.y;
			//totalWeight += weight;
			if (j < m_gridDimensions.y - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j + 1), velocityComponent);
			}

			targetNode = cellOrigin + Vector2(1, 1);
			weight[velocityComponent] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			weight[velocityComponent] = relativeFractions.x*relativeFractions.y;
			//totalWeight += weight;
			if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j + 1), velocityComponent);
			}
		}

		void ParticlesToStaggeredGrid2D::accumulateScalarFieldValues(string scalarFieldName, ParticlesData<Vector2> *pParticlesData, Scalar dx) {

			ScalarArray &scalarFieldAccumulator = getScalarAttributeArray(scalarFieldName);
			const vector<Scalar> &particlesScalarAttribute = pParticlesData->getScalarBasedAttribute(scalarFieldName);

			for (int ithParticle = 0; ithParticle < pParticlesData->getPositions().size(); ithParticle++) {

				if (pParticlesData->getResampledParticles()[ithParticle])
					continue;

				int i, j;
				Vector2 relativeFractions;

				Vector2 position = pParticlesData->getPositions()[ithParticle] / dx;

				i = floor(position.x - 0.5f);
				j = floor(position.y - 0.5f);

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
				}
			}

		}
		#pragma endregion
	}
}
