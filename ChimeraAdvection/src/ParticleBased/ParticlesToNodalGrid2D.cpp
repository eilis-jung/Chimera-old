#include "ParticleBased/ParticlesToNodalGrid2D.h"


namespace Chimera {


	namespace Advection {

		#pragma region Functionalities
		void ParticlesToNodalGrid2D::transferVelocityToGrid(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {
			flushAccumulatedVelocities();

			for (int i = 0; i < pParticlesData->getPositions().size(); i++) {
				if (pParticlesData->getResampledParticles()[i] != true) {
					accumulateVelocities(i, pParticlesData, pGridData->getGridSpacing());
				}
			}

			for (int i = 0; i < pGridData->getDimensions().x; i++) {
				for (int j = 0; j < pGridData->getDimensions().y; j++) {
					if (m_accVelocityField(i, j).weight.x > 0) {
						m_accVelocityField(i, j).entry.x /= m_accVelocityField(i, j).weight.x;
					}
					if (m_accVelocityField(i, j).weight.y > 0) {
						m_accVelocityField(i, j).entry.y /= m_accVelocityField(i, j).weight.y;
					}
				}
			}
			if (m_pCutCells2D) {
				for (uint i = 0; i < m_pCutCells2D->getNumberCutCells(); i++) {
					auto currCell = m_pCutCells2D->getCutCell(i);
					for (uint j = 0; j < currCell.getHalfEdges().size(); j++) {
						DoubleScalar currWeight = currCell.getHalfEdges()[j]->getVertices().first->getWeight();
						if (currWeight > 0.0f) {
							Vector2 currVel = currCell.getHalfEdges()[j]->getVertices().first->getAuxiliaryVelocity();
							currCell.getHalfEdges()[j]->getVertices().first->setAuxiliaryVelocity(currVel / currWeight);
							currCell.getHalfEdges()[j]->getVertices().first->setWeight(0.0f); //Do not update its velocity anymore
						}						
					}
				}
			}

			GridData2D *pGridData2D = dynamic_cast<GridData2D*>(pGridData);
			for (int i = 1; i < pGridData2D->getDimensions().x - 1; i++) {
				for (int j = 1; j < pGridData2D->getDimensions().y - 1; j++) {
					if (m_pCutCells2D == NULL || !m_pCutCells2D->isCutCellAt(i, j)) {
						pGridData2D->setAuxiliaryVelocity((m_accVelocityField(i, j).entry.x + m_accVelocityField(i, j + 1).entry.x)*0.5, i, j, xComponent);
						pGridData2D->setAuxiliaryVelocity((m_accVelocityField(i, j).entry.y + m_accVelocityField(i + 1, j).entry.y)*0.5, i, j, yComponent);
					}
				}
			}

			Scalar dx = pGridData2D->getGridSpacing();
			if (m_pCutCells2D) {
				for (int i = 0; i < m_pCutCells2D->getNumberCutCells(); i++) {
					auto cutCell = m_pCutCells2D->getCutCell(i);
					for (int j = 0; j < cutCell.getHalfEdges().size(); j++) {
						auto currHalfEdge = cutCell.getHalfEdges()[j];
						auto currEdge = currHalfEdge->getEdge();
						Vector2 v1, v2;
						if (currHalfEdge->getLocation() != geometryHalfEdge) {
							int prevIndex = roundClamp<int>(j - 1, 0, cutCell.getHalfEdges().size());
							int nextIndex = roundClamp<int>(j + 1, 0, cutCell.getHalfEdges().size());

							if (currHalfEdge->getVertices().first->getVertexType() == gridVertex) {
								v1 = m_accVelocityField(currHalfEdge->getVertices().first->getPosition().x/dx, currHalfEdge->getVertices().first->getPosition().y/dx).entry;
							}
							else {
								v1 = currHalfEdge->getVertices().first->getAuxiliaryVelocity();
							}

							if (currHalfEdge->getVertices().second->getVertexType() == gridVertex) {
								v2 = m_accVelocityField(currHalfEdge->getVertices().second->getPosition().x/dx, currHalfEdge->getVertices().second->getPosition().y/dx).entry;
							}
							else {
								v2 = currHalfEdge->getVertices().second->getAuxiliaryVelocity();
							}
							
							Vector2 edgeVelocity = (v1 + v2)*0.5f;

							if (currHalfEdge->getLocation() == bottomHalfEdge || currHalfEdge->getLocation() == topHalfEdge)
								edgeVelocity.x = 0;
							else if (currHalfEdge->getLocation() == leftHalfEdge || currHalfEdge->getLocation() == rightHalfEdge)
								edgeVelocity.y = 0;
							
							currEdge->setAuxiliaryVelocity(edgeVelocity);
						}

					}
				}
			}
		}

		void ParticlesToNodalGrid2D::transferScalarAttributesToGrid(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData) {
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
		void ParticlesToNodalGrid2D::flushAccumulatedVelocities() {
			ParticlesToGrid<Vector2, Array2D>::flushAccumulatedVelocities();
			m_pCutCellsVelocities2D->zeroVelocities();
			m_pCutCellsVelocities2D->zeroWeights();
			if (m_pCutCells2D) {
				for (uint i = 0; i < m_pCutCells2D->getNumberCutCells(); i++) {
					auto currCell = m_pCutCells2D->getCutCell(i);
					for (uint j = 0; j < currCell.getHalfEdges().size(); j++) {
						currCell.getHalfEdges()[j]->getVertices().first->setWeight(0.0);
						if (currCell.getHalfEdges()[j]->getVertices().first->getVertexType() == gridVertex) {
							currCell.getHalfEdges()[j]->getVertices().first->setAuxiliaryVelocity(Vector2(0, 0));
						}
					}
				}
			}
		}

		void ParticlesToNodalGrid2D::accumulateVelocities(int ithParticle, int ithCutFace, ParticlesData<Vector2> *pParticlesData, Scalar dx) {
			Vector2 particlePosition = pParticlesData->getPositions()[ithParticle];

			auto currCell = m_pCutCells2D->getCutCell(ithCutFace);

			for (int i = 0; i < currCell.getHalfEdges().size(); i++) {
				auto currHalfEdge = currCell.getHalfEdges()[i];
				if (currHalfEdge->getVertices().first->getVertexType() == gridVertex) {
					Vector2 targetNode = currHalfEdge->getVertices().first->getPosition();
					dimensions_t cellIndex(targetNode.x/m_pCutCells2D->getGridSpacing(), targetNode.y / m_pCutCells2D->getGridSpacing());
					Vector2 weight;
					weight.x = weight.y = m_pKernel->calculateKernel(particlePosition, targetNode, (particlePosition - targetNode).length());
					accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), cellIndex, fullVector);
				}
				else if(m_pCutCellsVelocities2D->getSolidBoundaryType() == Solid_FreeSlip) { //only process other nodes for free slip
					Vector2 targetNode = currHalfEdge->getVertices().first->getPosition();
					Vector2 weight;
					weight.x = weight.y = m_pKernel->calculateKernel(particlePosition, targetNode, (particlePosition - targetNode).length());
					
					currHalfEdge->getVertices().first->addAuxiliaryVelocity(pParticlesData->getVelocities()[ithParticle] * weight);
					currHalfEdge->getVertices().first->addWeight(weight.x);
				}
			}
		}

		void ParticlesToNodalGrid2D::accumulateVelocities(int ithParticle, ParticlesData<Vector2> *pParticlesData, Scalar dx) {
			Vector2 position = pParticlesData->getPositions()[ithParticle] / dx;
			int i, j;
			i = floor(position.x);
			j = floor(position.y);

			if (m_pCutCells2D) {
				if (m_pCutCells2D->isCutCellAt(i, j)) {
					accumulateVelocities(ithParticle, m_pCutCells2D->getCutCellIndex(position), pParticlesData, dx);
					return;
				}
			}


			Vector2 relativeFractions;
			relativeFractions.x = position.x - i;
			relativeFractions.y = position.y - j;
			Vector2 cellOrigin(i, j);

			Vector2 weight;
			weight[0] = weight[1] = m_pKernel->calculateKernel(position, cellOrigin, (position - cellOrigin).length()*dx);
			accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j), fullVector);

			Vector2 targetNode = cellOrigin + Vector2(1, 0);
			weight[0] = weight[1] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (i < m_gridDimensions.x - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j), fullVector);
			}

			targetNode = cellOrigin + Vector2(0, 1);
			weight[0] = weight[1] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (j < m_gridDimensions.y - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i, j + 1), fullVector);
			}

			targetNode = cellOrigin + Vector2(1, 1);
			weight[0] = weight[1] = m_pKernel->calculateKernel(position, targetNode, (position - targetNode).length()*dx);
			if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
				accumulateVelocity(ithParticle, weight, pParticlesData->getVelocities(), dimensions_t(i + 1, j + 1), fullVector);
			}
		}

		void ParticlesToNodalGrid2D::accumulateScalarFieldValues(string scalarFieldName, ParticlesData<Vector2> *pParticlesData, Scalar dx) {

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
