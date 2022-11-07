#include "Solvers/2D/CutCellSolverSO2D.h"

namespace Chimera {

	namespace Solvers {
		#pragma region ConstructorsDestructors
		CutCellSolverSO2D::CutCellSolverSO2D(const params_t &params, StructuredGrid<Vector2> *pGrid,
												 const vector<BoundaryCondition<Vector2> *> &boundaryConditions, const vector<RigidObject2D<Vector2> *> &rigidObjects)
												: CutCellSolver2D(params, pGrid, boundaryConditions, rigidObjects) {
			CutCellSolverSO2D::updatePoissonThinSolidWalls();
			CutCellSolverSO2D::initializePressureGradients();
			m_toleranceIterations = 1;
		}
		#pragma endregion

		#pragma region PressureProjection
		void CutCellSolverSO2D::updatePoissonThinSolidWalls() {
			if (m_numIterations <= 1) {
				CutCellSolver2D::updatePoissonThinSolidWalls();
			} else {

				CutCellSolver2D::updatePoissonThinSolidWalls();
				//return;

				//Double check the initial offset number
				int initialOffset = this->getInitialOffset();

				//Regular cells that are now special cells are treated as solid cells 
				if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
					for (int j = 2; j < m_pGrid->getDimensions().y - 2; ++j) {
						for (int i = 2; i < m_pGrid->getDimensions().x - 2; ++i) {
							if (m_pCutCells->isCutCellAt(i, j)) {
								m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(i - 1, j - 1), 0, 0, 1, 0, 0);
							}
							else {
								int currRowIndex = m_pPoissonMatrix->getRowIndex(i - 1, j - 1);

								if (m_pCutCells->isCutCellAt(i + 1, j)) {
									m_pPoissonMatrix->setEastValue(currRowIndex, 0);
								}
								if (m_pCutCells->isCutCellAt(i - 1, j)) {
									m_pPoissonMatrix->setWestValue(currRowIndex, 0);
								}
								if (m_pCutCells->isCutCellAt(i, j + 1)) {
									m_pPoissonMatrix->setNorthValue(currRowIndex, 0);
								}
								if (m_pCutCells->isCutCellAt(i, j - 1)) {
									m_pPoissonMatrix->setSouthValue(currRowIndex, 0);
								}
							}
						}
					}
				}

				m_pPoissonMatrix->copyDIAtoCOO();
				int numEntries = m_pPoissonMatrix->getNumberOfEntriesCOO();
				m_pPoissonMatrix->resizeToFitCOO();

				int matrixInternalId = numEntries;

				for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
					int row = initialOffset + i; //compute the matrix index. after the last regular cell.
					Scalar pc = 0;
					const HalfFace<Vector2> &currCell = m_pCutCells->getCutCell(i);
					uint currFaceID = currCell.getID();

					Scalar xi = currCell.getCentroid().x;
					Scalar yi = currCell.getCentroid().y;
					Vector2 neighboringCentroid;

					for (uint j = 0; j < currCell.getHalfEdges().size(); j++) {
						Edge<Vector2> *pCurrEdge = currCell.getHalfEdges()[j]->getEdge();
						if (pCurrEdge->getConnectedHalfFaces().size() > 2) {
							throw(exception("Invalid number of faces connected to an edge"));
						}
						DoubleScalar pressureCoefficient = 0;
						if (pCurrEdge->getType() != geometricEdge) {
							uint otherPressure;
							if (pCurrEdge->getConnectedHalfFaces().size() == 1) { //Neighbor to a regular grid face
								otherPressure = getRowIndex(currCell.getFace()->getGridCellLocation(), currCell.getHalfEdges()[j]->getLocation());
								dimensions_t currDim = currCell.getFace()->getGridCellLocation();

								dimensions_t neighboringCellDim = getNeighborCellDim(currCell.getFace()->getGridCellLocation(), currCell.getHalfEdges()[j]->getLocation());
								neighboringCentroid = m_pGridData->getCenterPoint(neighboringCellDim.x, neighboringCellDim.y);
							}
							else {
								otherPressure = pCurrEdge->getConnectedHalfFaces()[0]->getID() == currFaceID ? pCurrEdge->getConnectedHalfFaces()[1]->getID() : pCurrEdge->getConnectedHalfFaces()[0]->getID();
								neighboringCentroid = m_pCutCells->getCutCell(otherPressure).getCentroid();
								otherPressure += initialOffset;
							}
							Scalar dx = m_pGridData->getGridSpacing();

							switch (currCell.getHalfEdges()[j]->getLocation())
							{
							case rightHalfEdge:
								// A_e
								pressureCoefficient = pCurrEdge->getRelativeFraction() * dx / (neighboringCentroid.x - xi);
								break;
							case leftHalfEdge:
								// A_w
								pressureCoefficient = -pCurrEdge->getRelativeFraction() * dx / (neighboringCentroid.x - xi);
								break;
							case topHalfEdge:
								// A_n
								pressureCoefficient = pCurrEdge->getRelativeFraction() * dx / (neighboringCentroid.y - yi);
								break;
							case bottomHalfEdge:
								// A_s
								pressureCoefficient = -pCurrEdge->getRelativeFraction() * dx / (neighboringCentroid.y - yi);
								break;
							}
							pc += pressureCoefficient;
							m_pPoissonMatrix->setValue(matrixInternalId++, row, otherPressure, -pressureCoefficient);
							if (pCurrEdge->getConnectedHalfFaces().size() == 1) {
								/*switch (currCell.getHalfEdges()[j]->getLocation()) {
								case rightHalfEdge:
									m_pPoissonMatrix->setWestValue(otherPressure, -pressureCoefficient);
								break;
								case leftHalfEdge:
									m_pPoissonMatrix->setEastValue(otherPressure, -pressureCoefficient);
								break;
								case topHalfEdge:
									m_pPoissonMatrix->setSouthValue(otherPressure, -pressureCoefficient);
								break;
								case bottomHalfEdge:
									m_pPoissonMatrix->setNorthValue(otherPressure, -pressureCoefficient);
								break;
								}*/
								//m_pPoissonMatrix->setValue(matrixInternalId++, otherPressure, row, -pressureCoefficient); //Guarantees matrix symmetry
								//But since pressure coefficient is not -1 anymore, we also have to change the main diagonal

								Scalar nbCentralValue = m_pPoissonMatrix->getValue(otherPressure, otherPressure);
								nbCentralValue += pressureCoefficient - 1;
								m_pPoissonMatrix->setCentralValue(otherPressure, nbCentralValue);
								//m_pPoissonMatrix->setValue(matrixInternalId++, otherPressure, otherPressure, nbCentralValue); //Guarantees matrix symmetry
							}

						}
					}
					m_pPoissonMatrix->setValue(matrixInternalId++, row, row, pc);
				}
				m_pPoissonMatrix->updateCudaData();
			}
		}

		void CutCellSolverSO2D::updateCutCellsDivergence(Scalar dt) {
			/* At 1st timestep, initialize with old version*/
			if (m_numIterations <= m_toleranceIterations) {
				CutCellSolver2D::updateCutCellsDivergence(dt);
				return;
			}
			//CutCellSolver2D::updateCutCellsDivergence(dt);
			//return;
			if (m_numIterations == 20) {
				Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
				logVelocity("velocity_so_" + to_string(dx));
				logPressure("pressure_so_" + to_string(dx));
				logVorticity("vorticity_so_" + to_string(dx));
				//logVelocityForCutCells("velocity_so");
			}
			clearPressureGradientOnNodes();
			updatePressureGradientOnGridNodes();
			updatePressureGradientsForCutCells();

			Scalar dx = m_pGridData->getGridSpacing();
			for (uint i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
				int initialOffset = this->getInitialOffset();
				const HalfFace<Vector2> * pCurrCell = &m_pCutCells->getCutCell(i);
				/** Calculates the divergent with the previous old version first*/
				DoubleScalar divergent = 0;
				auto halfEdges = m_pCutCells->getCutCell(i).getHalfEdges();
				for (uint j = 0; j < halfEdges.size(); j++) {
					auto pEdge = halfEdges[j]->getEdge();
					divergent += halfEdges[j]->getNormal().dot(pEdge->getAuxiliaryVelocity())*pEdge->getRelativeFraction()*dx;
				}
				DoubleScalar originalDiv = divergent;

				/** If the cut-cell has only 2 non-geometric neighbors, we replace its divergence calculation with 1st-order. **/
				int numAxisAlignedEdges = 0;
				for (uint j = 0; j < pCurrCell->getHalfEdges().size(); j++) {
					Edge<Vector2> *pCurrEdge = pCurrCell->getHalfEdges()[j]->getEdge();
					if (pCurrEdge->getConnectedHalfFaces().size() > 2) {
						throw(exception("Invalid number of faces connected to an edge"));
					}
					if (pCurrEdge->getType() != geometricEdge) {
						numAxisAlignedEdges++;
					}
				}
				//if (numAxisAlignedEdges == 2) {
				//	divergent = originalDiv;
				//	m_cutCellsDivergents[i] = -divergent / dt;
				//	continue;
				//}


				/** Pressure Correction phase */
				Vector2 currCentroid = pCurrCell->getCentroid();
				Vector2 curGradientOfPressure = getPressureGradient(pCurrCell);
				Vector2 neighboringCentroid;
				/* =================================================================
				* Tranverse all edges of one cut-cell. Here we ignore geometry edge.
				* ================================================================== */
				for (uint j = 0; j < pCurrCell->getHalfEdges().size(); j++) {
					Edge<Vector2> *pCurrEdge = pCurrCell->getHalfEdges()[j]->getEdge();
					if (pCurrEdge->getConnectedHalfFaces().size() > 2) {
						throw(exception("Invalid number of faces connected to an edge"));
					}
					DoubleScalar pressureCoefficient = 0;
					HalfFace<Vector2> *pNeighborCell = nullptr;
					if (pCurrEdge->getType() != geometricEdge) {
						uint nbInd;
						Vector2 nbGradientOfPressure;
						if (pCurrEdge->getConnectedHalfFaces().size() == 1) {
							// Neighboring to a regular grid face
							nbInd = getNeighborCellIndex(pCurrCell->getFace()->getGridCellLocation(), pCurrCell->getHalfEdges()[j]->getLocation());
							dimensions_t neighboringCellDim = getCellDimensions(nbInd);
							
							nbGradientOfPressure = getPressureGradient(neighboringCellDim);
							neighboringCentroid = m_pGridData->getCenterPoint(neighboringCellDim.x, neighboringCellDim.y);
						}
						else {
							// Neighboring to a cut-cell
							nbInd = pCurrEdge->getConnectedHalfFaces()[0]->getID() == i ? pCurrEdge->getConnectedHalfFaces()[1]->getID() : pCurrEdge->getConnectedHalfFaces()[0]->getID();
							const HalfFace<Vector2> * pNbCell = &m_pCutCells->getCutCell(nbInd);
							nbGradientOfPressure = getPressureGradient(pNbCell);
							neighboringCentroid = m_pCutCells->getCutCell(nbInd).getCentroid();
							
						}
						/* =================================================================
						* For each valid edge, compute corresponding divergent 2nd-order modification.
						* Here since at LHS we used negative coeffs, on RHS the expression should also
						* be negative of the original ones on paper.
						* ================================================================== */
						Scalar rhsCoeff = 0;
						switch (pCurrCell->getHalfEdges()[j]->getLocation()) {
						case rightHalfEdge: { // East
							rhsCoeff = -(pCurrEdge->getRelativeFraction() * dx / (neighboringCentroid - currCentroid).x);
							Scalar ye = pCurrEdge->getCentroid().y;
							Scalar addedContrib = dx * rhsCoeff * (nbGradientOfPressure.y * (ye - neighboringCentroid.y) - curGradientOfPressure.y*(ye - currCentroid.y));
							divergent += addedContrib;
							break;
						}
						case leftHalfEdge: { // West
							rhsCoeff = (pCurrEdge->getRelativeFraction() * dx / (neighboringCentroid - currCentroid).x);
							Scalar yw = pCurrEdge->getCentroid().y;
							Scalar addedContrib = dx * rhsCoeff* (nbGradientOfPressure.y*(yw - neighboringCentroid.y) - curGradientOfPressure.y*(yw - currCentroid.y));
							divergent += addedContrib;
							break;
						}
						case topHalfEdge: { // North
							rhsCoeff = -(pCurrEdge->getRelativeFraction() * dx / (neighboringCentroid - currCentroid).y);
							Scalar xn = pCurrEdge->getCentroid().x;
							Scalar addedContrib = dx * rhsCoeff* (nbGradientOfPressure.x*(xn - neighboringCentroid.x) - curGradientOfPressure.x*(xn - currCentroid.x));
							divergent += addedContrib;
							break;
						}
						case bottomHalfEdge: { // South
							rhsCoeff = (pCurrEdge->getRelativeFraction() * dx / (neighboringCentroid - currCentroid).y);
							Scalar xs = pCurrEdge->getCentroid().x;
							Scalar addedContrib = dx * rhsCoeff * (nbGradientOfPressure.x*(xs - neighboringCentroid.x) - curGradientOfPressure.x*(xs - currCentroid.x));
							divergent += addedContrib;
							break;
						}
						default:
							break;
						}
					}
				}
				
				m_cutCellsDivergents[i] = -divergent / dt;
			}

		}

		Scalar CutCellSolverSO2D::calculateFluxDivergent(int i, int j) {
			if (m_numIterations <= m_toleranceIterations) {
				return CutCellSolver2D::calculateFluxDivergent(i, j);
			}

			Scalar divergent = 0;
			Scalar dx, dy = 0;
			dimensions_t currDim(i, j);
			/* ========================================================
			* If current cell is split into several cut-cells,
			* tranverse all cut-cells in it, and for each cut-cell,
			* tranverse its 4 neighbours in all 4 directions.
			* Finally, add corresponding 4 modifications on divergent.
			* Thus, for each occasion, there are 2 loops.
			* ======================================================== */

			/* =======================================================================
			* Occasion 1: current cell is a regular cell split into several cut-cells
			* ======================================================================= */
			/** This is handled by the calculateSecondOrderDivergentOfCutCell on the update cut-cells divergent function, return 0 */
			if (m_pCutCells && m_pCutCells->isCutCell(currDim)) {
				return 0;
			}
			/* ==============================================
			* Occasion 2: current cell is a pure regular cell
			* =============================================== */
			else if (i > 1 && i < m_dimensions.x - 1 && j > 1 && j < m_dimensions.y - 1) {
				/* ============================================
				* First, compute original 1st-order divergence
				* This is same as original version.
				* ============================================ */
				divergent += CutCellSolver2D::calculateFluxDivergent(i, j);
				//divergent = 0;
				Vector2 curCentroid = m_pGridData->getCenterPoint(i, j);
				// Here the cell index is for assigning values in poisson matrix, thus it should be getCellIndexForMatrix
				Vector2 curGradientOfPressure = getPressureGradient(dimensions_t(i, j));
				curGradientOfPressure = Vector2(0, 0);
				/* ================================================
				* Then, compute new 2nd-order divergence correction
				* ================================================= */
				if (m_pCutCells->isCutCellAt(i + 1, j)) { // East
					HalfFace<Vector2> * pNbCell = getNeighborCutCell(dimensions_t(i, j), rightHalfEdge);
					Vector2 nbCentroid = pNbCell->getCentroid();
					Scalar ye = curCentroid.y;
					Vector2 nbGradientOfPressure = getPressureGradient(pNbCell);
					Scalar rhsCoeff = (1 / (nbCentroid - curCentroid).x)*m_pGridData->getGridSpacing();
					divergent += rhsCoeff * (nbGradientOfPressure.y * (ye - nbCentroid.y) - curGradientOfPressure.y*(ye - curCentroid.y));

					//Scalar specialCellVelocity = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), yAlignedEdge)[0]->getAuxiliaryVelocity().x
					//	* m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), yAlignedEdge)[0]->getRelativeFraction();
					//divergent += specialCellVelocity / m_pGridData->getScaleFactor(i, j).x;
				}
				else {
					//Scalar specialCellVelocity = m_pGridData->getAuxiliaryVelocity(i + 1, j).x;
					//divergent += specialCellVelocity / m_pGridData->getScaleFactor(i, j).x;
				}


				if (m_pCutCells->isCutCellAt(i - 1, j)) { // West
					HalfFace<Vector2> * pNbCell = getNeighborCutCell(dimensions_t(i, j), leftHalfEdge);
					Vector2 nbCentroid = pNbCell->getCentroid();
					Scalar yw = curCentroid.y;
					Vector2 nbGradientOfPressure = getPressureGradient(pNbCell);
					Scalar rhsCoeff = -(1 / (curCentroid - nbCentroid).x)*m_pGridData->getGridSpacing();
					divergent += rhsCoeff* (curGradientOfPressure.y*(yw - curCentroid.y) - nbGradientOfPressure.y*(yw - nbCentroid.y));

					//Scalar specialCellVelocity = m_pCutCells->getEdgeVector(dimensions_t(i, j), yAlignedEdge).back()->getAuxiliaryVelocity().x
					//	* m_pCutCells->getEdgeVector(dimensions_t(i, j), yAlignedEdge).back()->getRelativeFraction();
					//divergent -= specialCellVelocity / m_pGridData->getScaleFactor(i, j).x;
				}
				else {
					//Scalar specialCellVelocity = m_pGridData->getAuxiliaryVelocity(i, j).x;
					//divergent -= specialCellVelocity / m_pGridData->getScaleFactor(i, j).x;
				}


				if (m_pCutCells->isCutCellAt(i, j + 1)) { // North
					HalfFace<Vector2> * pNbCell = getNeighborCutCell(dimensions_t(i, j), topHalfEdge);
					Vector2 nbCentroid = pNbCell->getCentroid();
					Scalar xn = curCentroid.x;
					Vector2 nbGradientOfPressure = getPressureGradient(pNbCell);
					Scalar rhsCoeff = (1 / (nbCentroid - curCentroid).y)*m_pGridData->getGridSpacing();
					divergent += rhsCoeff * (nbGradientOfPressure.x*(xn - nbCentroid.x) - curGradientOfPressure.x*(xn - curCentroid.x));

					//Scalar specialCellVelocity = m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), xAlignedEdge)[0]->getAuxiliaryVelocity().y
					//	* m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), xAlignedEdge)[0]->getRelativeFraction();
					//divergent += specialCellVelocity / m_pGridData->getScaleFactor(i, j).y;
				}
				else {
					//Scalar specialCellVelocity = m_pGridData->getAuxiliaryVelocity(i, j + 1).y;
					//divergent += specialCellVelocity / m_pGridData->getScaleFactor(i, j).y;
				}


				if (m_pCutCells->isCutCellAt(i, j - 1)) { // South
					HalfFace<Vector2> * pNbCell = getNeighborCutCell(dimensions_t(i, j), bottomHalfEdge);
					Vector2 nbCentroid = pNbCell->getCentroid();
					Scalar xs = curCentroid.x;
					Vector2 nbGradientOfPressure = getPressureGradient(pNbCell);
					Scalar rhsCoeff = -(1 / (curCentroid - nbCentroid).y)*m_pGridData->getGridSpacing();
					divergent += rhsCoeff* (curGradientOfPressure.x*(xs - curCentroid.x) - nbGradientOfPressure.x*(xs - nbCentroid.x));

					//Scalar specialCellVelocity = m_pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge).back()->getAuxiliaryVelocity().y
					//	* m_pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge).back()->getRelativeFraction();
					//divergent -= specialCellVelocity / m_pGridData->getScaleFactor(i, j).y;
				}
				else {
					//Scalar specialCellVelocity = m_pGridData->getAuxiliaryVelocity(i, j).y;
					//divergent -= specialCellVelocity / m_pGridData->getScaleFactor(i, j).y;
				}
			}
			else {
				dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j).x - m_pGridData->getAuxiliaryVelocity(i, j).x) / m_pGridData->getScaleFactor(i, j).x;
				dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1).y - m_pGridData->getAuxiliaryVelocity(i, j).y) / m_pGridData->getScaleFactor(i, j).y;
				divergent = dx + dy;
			}
			return divergent;

		}

#pragma endregion

#pragma region ExactPressureGradientInterpolation
		void CutCellSolverSO2D::initializePressureGradients() {
			for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
				m_cutCellsPressureGradients.push_back(Vector2(0, 0));
			}
		}

		void CutCellSolverSO2D::clearPressureGradientOnNodes() {
			if (m_cutCellsPressureGradients.size() < m_pCutCells->getNumberCutCells()) {
				for (int i = m_cutCellsPressureGradients.size(); i < m_pCutCells->getNumberCutCells(); i++) {
					m_cutCellsPressureGradients.push_back(Vector2(0, 0));
				}
			}
			for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
				m_cutCellsPressureGradients[i] = Vector2(0, 0);
			}
			auto vertices = m_pCutCells->getVertices();
			for (auto & vertex : vertices) {
				vertex->setPressureGradient(Vector2(0, 0));
			}
		}

		Vector2 CutCellSolverSO2D::getPressureGradient(dimensions_t cellDim) {
			Vertex<Vector2> * vertices[4];
			/*
			     0 ------ 1
				 |        |
				 |        |
				 2 ------ 3
			*/
			// Get all 4 vertices in this cell.
			auto allVertices = m_pCutCells->getVertices();
			Vector2 pressureGradients[4];
			Vector2 locations[4];
			Scalar dx = m_pGridData->getGridSpacing();
			bool v0 = false, v1 = false, v2 = false, v3 = false;
			// If pressure gradient is already calculated, then directly use it
			for (auto & vertex : allVertices) {
				if (vertex->getVertexType() == gridVertex) {
					dimensions_t tempCellDim = dimensions_t(
						vertex->getPosition().x / m_pGridData->getGridSpacing(),
						vertex->getPosition().y / m_pGridData->getGridSpacing() - 1);
					if (tempCellDim == dimensions_t(cellDim.x, cellDim.y)) {
						v0 = true;
						vertices[0] = vertex;
						locations[0] = vertex->getPosition();
						pressureGradients[0] = vertex->getPressureGradient();
					}
					else if (tempCellDim == dimensions_t(cellDim.x + 1, cellDim.y)) {
						v1 = true;
						vertices[1] = vertex;
						locations[1] = vertex->getPosition();
						pressureGradients[1] = vertex->getPressureGradient();
					}
					else if (tempCellDim == dimensions_t(cellDim.x, cellDim.y - 1)) {
						v2 = true;
						vertices[2] = vertex;
						locations[2] = vertex->getPosition();
						pressureGradients[2] = vertex->getPressureGradient();
					}
					else if (tempCellDim == dimensions_t(cellDim.x + 1, cellDim.y - 1)) {
						v3 = true;
						vertices[3] = vertex;
						locations[3] = vertex->getPosition();
						pressureGradients[3] = vertex->getPressureGradient();
					}
					else 
						continue;
				}
			}
		
			// If not calculated, assemble. For grid nodes, it's simple an average between 2 gradients.
			if (!v0) {
				Scalar pressures[4];
				pressures[0] = m_pGridData->getPressure(cellDim.x - 1, cellDim.y + 1);
				pressures[1] = m_pGridData->getPressure(cellDim.x, cellDim.y + 1);
				pressures[2] = m_pGridData->getPressure(cellDim.x - 1, cellDim.y);
				pressures[3] = m_pGridData->getPressure(cellDim.x, cellDim.y);
				pressureGradients[0] = Vector2(
					((pressures[1] - pressures[0]) + (pressures[3] - pressures[2])) / (2 * dx),
					((pressures[1] - pressures[3]) + (pressures[0] - pressures[2])) / (2 * dx)
				);
				locations[0] = m_pGridData->getCenterPoint(cellDim.x, cellDim.y) + Vector2(-dx / 2, +dx / 2);
			}
			if (!v1) {
				Scalar pressures[4];
				pressures[0] = m_pGridData->getPressure(cellDim.x, cellDim.y + 1);
				pressures[1] = m_pGridData->getPressure(cellDim.x + 1, cellDim.y + 1);
				pressures[2] = m_pGridData->getPressure(cellDim.x, cellDim.y);
				pressures[3] = m_pGridData->getPressure(cellDim.x + 1, cellDim.y);
				pressureGradients[1] = Vector2(
					((pressures[1] - pressures[0]) + (pressures[3] - pressures[2])) / (2 * dx),
					((pressures[1] - pressures[3]) + (pressures[0] - pressures[2])) / (2 * dx)
				);
				locations[1] = m_pGridData->getCenterPoint(cellDim.x, cellDim.y) + Vector2(+dx / 2, +dx / 2);
			}
			if (!v2) {
				Scalar pressures[4];
				pressures[0] = m_pGridData->getPressure(cellDim.x - 1, cellDim.y);
				pressures[1] = m_pGridData->getPressure(cellDim.x, cellDim.y);
				pressures[2] = m_pGridData->getPressure(cellDim.x - 1, cellDim.y - 1);
				pressures[3] = m_pGridData->getPressure(cellDim.x, cellDim.y - 1);
				pressureGradients[2] = Vector2(
					((pressures[1] - pressures[0]) + (pressures[3] - pressures[2])) / (2 * dx),
					((pressures[1] - pressures[3]) + (pressures[0] - pressures[2])) / (2 * dx)
				);
				locations[2] = m_pGridData->getCenterPoint(cellDim.x, cellDim.y) + Vector2(-dx / 2, -dx / 2);
			}
			if (!v3) {
				Scalar pressures[4];
				pressures[0] = m_pGridData->getPressure(cellDim.x, cellDim.y);
				pressures[1] = m_pGridData->getPressure(cellDim.x + 1, cellDim.y);
				pressures[2] = m_pGridData->getPressure(cellDim.x, cellDim.y - 1);
				pressures[3] = m_pGridData->getPressure(cellDim.x + 1, cellDim.y - 1);
				pressureGradients[3] = Vector2(
					((pressures[1] - pressures[0]) + (pressures[3] - pressures[2])) / (2 * dx),
					((pressures[1] - pressures[3]) + (pressures[0] - pressures[2])) / (2 * dx)
				);
				locations[3] = m_pGridData->getCenterPoint(cellDim.x, cellDim.y) + Vector2(+dx / 2, -dx / 2);
			}
		
			vector<Vector2> values;
			vector<Vector2> locs;
			values.push_back(pressureGradients[0]);
			locs.push_back(locations[0]);
			values.push_back(pressureGradients[1]);
			locs.push_back(locations[1]);
			values.push_back(pressureGradients[3]);
			locs.push_back(locations[3]);
			values.push_back(pressureGradients[2]);
			locs.push_back(locations[2]);

			return MVCInterpolationFor2DVector(values, locs, m_pGridData->getCenterPoint(cellDim.x, cellDim.y));
			//return leastSquareInterpolationFor2DVector(values, locs, m_pGridData->getCenterPoint(cellDim.x, cellDim.y));
		}


		Vector2 CutCellSolverSO2D::getPressureGradient(const HalfFace<Vector2> *pCurrCell) {
			return m_cutCellsPressureGradients[pCurrCell->getID()];
		}
		void CutCellSolverSO2D::updatePressureGradientOnGridNodes2() {
			//weights = MeanValueInterpolant2D<Vector2>::calculateWeights(position, polygonPoints);
			auto vertices = m_pCutCells->getVertices();
			for (auto & vertex : vertices) {
				if (vertex->getVertexType() == gridVertex && vertex->getPressureGradient() == Vector2(0, 0)) {
					// For nodes on grid, do a weighted least-square from all faces connected.
					// dimensions of the cell where current vertex is on top-left corner.
					dimensions_t cellID = dimensions_t(
						vertex->getPosition().x / m_pGridData->getGridSpacing(),
						vertex->getPosition().y / m_pGridData->getGridSpacing() - 1);
					if (!(cellID.x < m_pGrid->getDimensions().x && cellID.y < m_pGrid->getDimensions().y && cellID.x > 0 && cellID.y > 0))
						continue;

					dimensions_t dims[4];
					dims[0] = dimensions_t(cellID.x - 1, cellID.y + 1);
					dims[1] = dimensions_t(cellID.x, cellID.y + 1);
					dims[2] = dimensions_t(cellID.x - 1, cellID.y);
					dims[3] = cellID;

					// Only nodes connecting a cutcell or connecting a regular cell neighboring cutcells are considered.
					// For other nodes, both least square or finite difference would do.
					vector<HalfFace<Vector2> * > connectedHalfFaces;
					bool isConnectingCutCells = false;
					for (int i = 0; i < 4; i++) {
						if (isNeighboringCutCell(dims[i])) {
							isConnectingCutCells = true;
						}
						if (m_pCutCells->isCutCellAt(dims[i].x, dims[i].y)) {
							isConnectingCutCells = true;
							auto edges = vertex->getConnectedEdges();
							for (auto & edge : edges) {
								auto halfFaces = edge->getConnectedHalfFaces();
								for (auto & halfFace : halfFaces) {
									bool isExisted = false;
									for (auto & halfFace2 : connectedHalfFaces) {
										if (halfFace2->getID() == halfFace->getID()) {
											isExisted = true;
											break;
										}
									}
									if (!isExisted) {
										connectedHalfFaces.push_back(halfFace);
									}
								}
							}
						}
					}
					if (!isConnectingCutCells)
						continue;

					Vector2 centroids[4];
					Scalar pressures[4];
					for (int i = 0; i < 4; i++) {
						centroids[i] = m_pGridData->getCenterPoint(dims[i].x, dims[i].y);
						pressures[i] = m_pGridData->getPressure(dims[i].x, dims[i].y);
					}
					for (auto & pHalfFace : connectedHalfFaces) {
						dimensions_t tempDim = pHalfFace->getFace()->getGridCellLocation();
						for (int i = 0; i < 4; i++) {
							if (tempDim == dims[i]) {
								centroids[i] = pHalfFace->getCentroid();
								pressures[i] = m_cutCellsPressures[pHalfFace->getID()];
								break;
							}
						}
					}

					EdgePressureGradient edgePG[4];

					// edges: 0-1, 1-3, 3-2, 2-0.
					edgePG[0].set(centroids[0], centroids[1], pressures[0], pressures[1], vertex->getPosition());
					edgePG[1].set(centroids[1], centroids[3], pressures[1], pressures[3], vertex->getPosition());
					edgePG[2].set(centroids[3], centroids[2], pressures[3], pressures[2], vertex->getPosition());
					edgePG[3].set(centroids[2], centroids[0], pressures[2], pressures[0], vertex->getPosition());

					// 1st order: a=3; 2nd: a=6; 3rd: a=10.
					// TODO: standardize first!
					Eigen::MatrixXd X(4, 2), W(4, 4), Y(4, 1);
					Eigen::RowVectorXd dirX(2), dirY(2);
					dirX << 1, 0;
					dirY << 0, 1;
					//Eigen::MatrixXd X(4, 6), W(4, 4), Y(4, 1);
					//Eigen::RowVectorXd dirX(6), dirY(6);
					//dirX << 1, 1, 0, 1, 0, 0;
					//dirY << 1, 0, 1, 0, 1, 0;

					X.setZero();
					W.setZero();
					for (int i = 0; i < 4; i++) {
						W(i, i) = edgePG[i].weight;
						X(i, 0) = 1;
						X(i, 0) = edgePG[i].direction.x;
						X(i, 1) = edgePG[i].direction.y;
						//X(i, 3) = edgePG[i].direction.x * edgePG[i].direction.x;
						//X(i, 4) = edgePG[i].direction.y * edgePG[i].direction.y;
						//X(i, 5) = edgePG[i].direction.x * edgePG[i].direction.y;
						Y(i, 0) = edgePG[i].gradAlongEdge;
					}

					Eigen::MatrixXd LHSInv = (X.transpose() * W * X).inverse();
					Eigen::MatrixXd RHS = X.transpose() * W * Y;
					Eigen::MatrixXd B = LHSInv * RHS;
					Scalar gradX = (dirX * B)(0, 0);
					Scalar gradY = (dirY * B)(0, 0);
					vertex->setPressureGradient(Vector2(gradX, gradY));
				}
			}

		}

		void CutCellSolverSO2D::updatePressureGradientOnGridNodes() {
			auto vertices = m_pCutCells->getVertices();

			/** A node always has a the following configuration for edges that are connected to it:
						pn
						|
						|
						|
			pw ------- * ------- pe
						|
						|
						|
						ps
						*/

			/* We are trying to find perpendicular pressure gradients to those edges. For cut-edges, the pressure gradients
			are calculated by the function calculatePressureGradients(pEdge). Regular edges will be handled here. */
			
			Scalar dx = m_pGridData->getGridSpacing();
			for (auto & vertex : vertices) {
				dimensions_t vertexLocation(vertex->getPosition().x / dx, (vertex->getPosition().y / dx) - 1);
				if (vertex->getVertexType() == gridVertex) {
					Vector2 pressureGradients[4];
					Vector2 locations[4]; // order: e, w, n, s
					pair<Vector2, Vector2> edgeLocations[4];
					bool pe = false, pn = false, ps = false, pw = false;
					for (int i = 0; i < vertex->getConnectedEdges().size(); i++) {
						Edge<Vector2> *pEdge = vertex->getConnectedEdges()[i];
						if (pEdge->getType() == xAlignedEdge) {
							if (pEdge->getCentroid().x > vertex->getPosition().x) { //east (right) edge
								pe = true;
								auto halfFaces = pEdge->getConnectedHalfFaces();
								if (halfFaces.size() == 2) {
									edgeLocations[0] = std::pair<Vector2, Vector2>(halfFaces[0]->getCentroid(), halfFaces[1]->getCentroid());
								}
								else {
									Vector2 centroid1 = pEdge->getConnectedHalfFaces()[0]->getCentroid();
									Vector2 centroid2;
									if (pEdge->getCentroid().y < centroid1.y)
										centroid2 = m_pGridData->getCenterPoint(vertexLocation.x, vertexLocation.y);
									else
										centroid2 = m_pGridData->getCenterPoint(vertexLocation.x, vertexLocation.y + 1);
									edgeLocations[0] = std::pair<Vector2, Vector2>(centroid1, centroid2);
								}
								pressureGradients[0] = calculateEdgePressureGradient(pEdge);
							}
							else {
								pw = true;
								auto halfFaces = pEdge->getConnectedHalfFaces();
								if (halfFaces.size() == 2) {
									edgeLocations[1] = std::pair<Vector2, Vector2>(halfFaces[0]->getCentroid(), halfFaces[1]->getCentroid());
								}
								else {
									Vector2 centroid1 = pEdge->getConnectedHalfFaces()[0]->getCentroid();
									Vector2 centroid2;
									if (pEdge->getCentroid().y < centroid1.y)
										centroid2 = m_pGridData->getCenterPoint(vertexLocation.x - 1, vertexLocation.y);
									else
										centroid2 = m_pGridData->getCenterPoint(vertexLocation.x - 1, vertexLocation.y + 1);
									edgeLocations[1] = std::pair<Vector2, Vector2>(centroid1, centroid2);
								}
								pressureGradients[1] = calculateEdgePressureGradient(pEdge);
							}
						}
						else if (pEdge->getType() == yAlignedEdge) {
							if (pEdge->getCentroid().y > vertex->getPosition().y) { //north (top) edge
								pn = true;
								auto halfFaces = pEdge->getConnectedHalfFaces();
								if (halfFaces.size() == 2) {
									edgeLocations[2] = std::pair<Vector2, Vector2>(halfFaces[0]->getCentroid(), halfFaces[1]->getCentroid());
								}
								else {
									Vector2 centroid1 = pEdge->getConnectedHalfFaces()[0]->getCentroid();
									Vector2 centroid2;
									if (pEdge->getCentroid().x < centroid1.x)
										centroid2 = m_pGridData->getCenterPoint(vertexLocation.x - 1, vertexLocation.y + 1);
									else
										centroid2 = m_pGridData->getCenterPoint(vertexLocation.x, vertexLocation.y + 1);
									edgeLocations[2] = std::pair<Vector2, Vector2>(centroid1, centroid2);
								}
								pressureGradients[2] = calculateEdgePressureGradient(pEdge);
							}
							else {
								ps = true;
								auto halfFaces = pEdge->getConnectedHalfFaces();
								if (halfFaces.size() == 2) {
									edgeLocations[3] = std::pair<Vector2, Vector2>(halfFaces[0]->getCentroid(), halfFaces[1]->getCentroid());
								}
								else {
									Vector2 centroid1 = pEdge->getConnectedHalfFaces()[0]->getCentroid();
									Vector2 centroid2;
									if (pEdge->getCentroid().x < centroid1.x)
										centroid2 = m_pGridData->getCenterPoint(vertexLocation.x - 1, vertexLocation.y);
									else
										centroid2 = m_pGridData->getCenterPoint(vertexLocation.x, vertexLocation.y);
									edgeLocations[3] = std::pair<Vector2, Vector2>(centroid1, centroid2);
								}
								pressureGradients[3] = calculateEdgePressureGradient(pEdge);
							}
						}
					}
					Scalar p1, p2;

					if (!pe) { //Calculate pressure gradient for the right location
						p2 = m_pGridData->getPressure(vertexLocation.x, vertexLocation.y + 1);
						p1 = m_pGridData->getPressure(vertexLocation.x, vertexLocation.y);
						pressureGradients[0] = (Vector2(0, 1)*(p2 - p1) / dx);
						edgeLocations[0] = std::pair<Vector2, Vector2>(
							m_pGridData->getCenterPoint(vertexLocation.x, vertexLocation.y + 1),
							m_pGridData->getCenterPoint(vertexLocation.x, vertexLocation.y));
					}
					if (!pw) { //Calculate pressure gradient for the left location
						p2 = m_pGridData->getPressure(vertexLocation.x - 1, vertexLocation.y + 1);
						p1 = m_pGridData->getPressure(vertexLocation.x - 1, vertexLocation.y);
						pressureGradients[1] = (Vector2(0, 1)*(p2 - p1) / dx);
						edgeLocations[1] = std::pair<Vector2, Vector2>(
							m_pGridData->getCenterPoint(vertexLocation.x - 1, vertexLocation.y + 1),
							m_pGridData->getCenterPoint(vertexLocation.x - 1, vertexLocation.y));
					}
					if (!pn) { //Calculate pressure gradient for the top location
						p2 = m_pGridData->getPressure(vertexLocation.x, vertexLocation.y + 1);
						p1 = m_pGridData->getPressure(vertexLocation.x - 1, vertexLocation.y + 1);
						pressureGradients[2] = (Vector2(1, 0)*(p2 - p1) / dx);
						edgeLocations[2] = std::pair<Vector2, Vector2>(
							m_pGridData->getCenterPoint(vertexLocation.x, vertexLocation.y + 1),
							m_pGridData->getCenterPoint(vertexLocation.x - 1, vertexLocation.y + 1));
					}
					if (!ps) { //Calculate pressure gradient for the bottom location
						p2 = m_pGridData->getPressure(vertexLocation.x, vertexLocation.y);
						p1 = m_pGridData->getPressure(vertexLocation.x - 1, vertexLocation.y);
						pressureGradients[3] = (Vector2(1, 0)*(p2 - p1) / dx);
						edgeLocations[3] = std::pair<Vector2, Vector2>(
							m_pGridData->getCenterPoint(vertexLocation.x, vertexLocation.y),
							m_pGridData->getCenterPoint(vertexLocation.x - 1, vertexLocation.y));
					}
					for (int i = 0; i < 4; i++) {
						locations[i] = getProjectionOfPointOntoLine(vertex->getPosition(), edgeLocations[i].first, edgeLocations[i].second);
					}

					vector<Vector2> values;
					vector<Vector2> locs;
					values.push_back(pressureGradients[0]);
					locs.push_back(locations[0]);
					values.push_back(pressureGradients[3]);
					locs.push_back(locations[3]);
					values.push_back(pressureGradients[1]);
					locs.push_back(locations[1]);
					values.push_back(pressureGradients[2]);
					locs.push_back(locations[2]);

					/* =================================================================
					* Interpolate pressure gradients to the vertex using MVC
					* ================================================================== */
					Vector2 res(0, 0);
					//res = leastSquareInterpolationFor2DVector(values, locs, vertex->getPosition());
					res = MVCInterpolationFor2DVector(values, locs, vertex->getPosition());
					vertex->setPressureGradient(res);
				}
			}

		}

		void CutCellSolverSO2D::updatePressureGradientsForCutCells() {
			// For cut-cells
			for (int cellID = 0; cellID < m_pCutCells->getNumberCutCells(); cellID++) {
				const HalfFace<Vector2> *pCurrCell = &m_pCutCells->getCutCell(cellID);

				// Do a MVC to the centroid.
				Vector2 res(0, 0);
				auto edges = pCurrCell->getHalfEdges();
				vector<Vertex<Vector2> *> vertices;
				vector<std::pair<Vector2, unsigned> > t;
				int i = 0;
				for (auto & edge : edges) {
					auto vertex = edge->getVertices().first;
					vertices.push_back(vertex);
					t.push_back(std::make_pair(vertex->getPosition(), i++));
				}
				CGALWrapper::Triangulator<Vector2> tri(t);
				auto coordinates = tri.getBarycentricCoordinates(pCurrCell->getCentroid());
				for (auto & coord : coordinates) {
					res += (vertices[coord.second]->getPressureGradient()) * coord.first;
				}

				m_cutCellsPressureGradients[cellID] = res;
				//m_cutCellsPressureGradients[cellID] = Vector2(0, 0);
			}
		}

		Vector2 CutCellSolverSO2D::calculateEdgePressureGradient(Edge<Vector2> *pEdge) {
			Scalar dx = m_pGridData->getGridSpacing();
			int x, y;
			if (pEdge->getType() == xAlignedEdge) {
				// left to right, edge is on top side of the cell
				x = floor(pEdge->getCentroid().x / dx);
				y = pEdge->getCentroid().y / dx - 1;
			}
			else {
				// bottom to top, edge is on left side of the cell
				x = pEdge->getCentroid().x / dx;
				y = floor(pEdge->getCentroid().y / dx);
			}			
			dimensions_t edgeLocationDim(x, y);
			Scalar p1 = m_cutCellsPressures[pEdge->getConnectedHalfFaces()[0]->getID()];
			Scalar p2;
			Vector2 centroid1 = pEdge->getConnectedHalfFaces()[0]->getCentroid();
			Vector2 centroid2;
			if (pEdge->getConnectedHalfFaces().size() == 2) {
				p2 = m_cutCellsPressures[pEdge->getConnectedHalfFaces()[1]->getID()];
				centroid2 = pEdge->getConnectedHalfFaces()[1]->getCentroid();
				if (pEdge->getType() == xAlignedEdge) {
					if (pEdge->getConnectedHalfFaces()[1]->getFace()->getGridCellLocation().x < pEdge->getConnectedHalfFaces()[0]->getFace()->getGridCellLocation().x) {
						swap(p1, p2);
						swap(centroid1, centroid2);
					}
				}
				else if (pEdge->getType() == yAlignedEdge) {
					if (pEdge->getConnectedHalfFaces()[1]->getFace()->getGridCellLocation().y < pEdge->getConnectedHalfFaces()[0]->getFace()->getGridCellLocation().y) {
						swap(p1, p2);
						swap(centroid1, centroid2);
					}
				}
			}
			else {
				if (pEdge->getType() == xAlignedEdge) {
					if (pEdge->getCentroid().y < centroid1.y) { // cutcell is above edge
						p2 = m_pGridData->getPressure(edgeLocationDim.x, edgeLocationDim.y);
						centroid2 = m_pGridData->getCenterPoint(edgeLocationDim.x, edgeLocationDim.y);
						swap(p1, p2);
						swap(centroid1, centroid2);
					}
					else { // cutcell is below edge
						p2 = m_pGridData->getPressure(edgeLocationDim.x, edgeLocationDim.y + 1);
						centroid2 = m_pGridData->getCenterPoint(edgeLocationDim.x, edgeLocationDim.y + 1);
					}
				}
				else if (pEdge->getType() == yAlignedEdge) {
					if (pEdge->getCentroid().x < centroid1.x) { // cutcell is at the right side of edge
						p2 = m_pGridData->getPressure(edgeLocationDim.x - 1, edgeLocationDim.y);
						centroid2 = m_pGridData->getCenterPoint(edgeLocationDim.x - 1, edgeLocationDim.y);
						swap(p1, p2);
						swap(centroid1, centroid2);
					}
					else { // cutcell is at the left side of edge
						p2 = m_pGridData->getPressure(edgeLocationDim.x, edgeLocationDim.y);
						centroid2 = m_pGridData->getCenterPoint(edgeLocationDim.x, edgeLocationDim.y);
					}
				}
			}

			Vector2 centroidDirection = (centroid2 - centroid1).normalized();
			return centroidDirection*(p2 - p1) / dx;
		}
		#pragma endregion


		#pragma region Misc

		void CutCellSolverSO2D::updateLeastSquare(dimensions_t nbDim, Eigen::Matrix2d & LSlhs, Eigen::Vector2d & LSrhs, Vector2 centroid, Scalar currPressure)
		{
			Vector2 centroidNeighbor;
			Scalar deltaPhi;
			centroidNeighbor = m_pGrid->getGridData2D()->getCenterPoint(nbDim.x, nbDim.y);
			deltaPhi = m_pGrid->getGridData2D()->getPressure(nbDim.x, nbDim.y) - currPressure;

			Scalar distance = (centroidNeighbor - centroid).length2();
			Scalar weight = 1.0 / distance;
			Scalar deltaX = centroidNeighbor.x - centroid.x;
			Scalar deltaY = centroidNeighbor.y - centroid.y;
			LSlhs(0, 0) += weight*deltaX*deltaX;
			LSlhs(0, 1) += weight*deltaX*deltaY;
			LSlhs(1, 0) += weight*deltaX*deltaY;
			LSlhs(1, 1) += weight*deltaY*deltaY;
			LSrhs(0) += weight*deltaX*deltaPhi;
			LSrhs(1) += weight*deltaY*deltaPhi;
		}

		void CutCellSolverSO2D::updateLeastSquare(const HalfFace<Vector2> *pNbCell, Eigen::Matrix2d & LSlhs, Eigen::Vector2d & LSrhs, Vector2 centroid, Scalar currPressure)
		{
			Vector2 centroidNeighbor;
			Scalar deltaPhi;
			uint nbCellInd = pNbCell->getID();
			centroidNeighbor = pNbCell->getCentroid();
			deltaPhi = m_cutCellsPressures[nbCellInd] - currPressure;

			Scalar distance = (centroidNeighbor - centroid).length2();
			Scalar weight = 1.0 / distance;
			Scalar deltaX = centroidNeighbor.x - centroid.x;
			Scalar deltaY = centroidNeighbor.y - centroid.y;
			LSlhs(0, 0) += weight*deltaX*deltaX;
			LSlhs(0, 1) += weight*deltaX*deltaY;
			LSlhs(1, 0) += weight*deltaX*deltaY;
			LSlhs(1, 1) += weight*deltaY*deltaY;
			LSrhs(0) += weight*deltaX*deltaPhi;
			LSrhs(1) += weight*deltaY*deltaPhi;
		}

		Vector2 CutCellSolverSO2D::getPressureGradientOld(dimensions_t cellDim) {
			/* ==========================================
			* Occasion 1: current cell is a regular cell
			* ========================================== */
			dimensions_t cellIndex;
			Eigen::Matrix2d LSlhs;
			Eigen::Vector2d gradPhi, LSrhs;
			Vector2 centroid;
			Scalar currPressure;
			LSlhs.setZero();
			gradPhi.setZero();
			LSrhs.setZero();

			/* ========================================================
			* For each cell (regular or cut-cell), find its neighbours
			* and assemble LS matrix for solving.
			* For regular cells there are 4 neighbours. For cut-cells
			* there could be 3 to 5 neighbours.
			* ======================================================== */

			/* E.g. for cell (9, 14), its cellIndForMatrix should be 9*29+14 = 275, while its cellInd should be 9*31+14 = 293.
			After getCellDimensionsForMatrix, it should be (9, 14) anyway. Here We need to check. */
			centroid = m_pGrid->getGridData2D()->getCenterPoint(cellDim.x, cellDim.y);
			currPressure = m_pGrid->getGridData2D()->getPressure(cellDim.x, cellDim.y);
			if (getNeighborCutCell(cellDim, leftHalfEdge) != nullptr)
				this->updateLeastSquare(getNeighborCutCell(cellDim, leftHalfEdge), LSlhs, LSrhs, centroid, currPressure);
			else
				this->updateLeastSquare(getNeighborCellDim(cellDim, leftHalfEdge), LSlhs, LSrhs, centroid, currPressure);

			if (getNeighborCutCell(cellDim, rightHalfEdge) != nullptr)
				this->updateLeastSquare(getNeighborCutCell(cellDim, rightHalfEdge), LSlhs, LSrhs, centroid, currPressure);
			else
				this->updateLeastSquare(getNeighborCellDim(cellDim, rightHalfEdge), LSlhs, LSrhs, centroid, currPressure);

			if (getNeighborCutCell(cellDim, topHalfEdge) != nullptr)
				this->updateLeastSquare(getNeighborCutCell(cellDim, topHalfEdge), LSlhs, LSrhs, centroid, currPressure);
			else
				this->updateLeastSquare(getNeighborCellDim(cellDim, topHalfEdge), LSlhs, LSrhs, centroid, currPressure);

			if (getNeighborCutCell(cellDim, bottomHalfEdge) != nullptr)
				this->updateLeastSquare(getNeighborCutCell(cellDim, bottomHalfEdge), LSlhs, LSrhs, centroid, currPressure);
			else
				this->updateLeastSquare(getNeighborCellDim(cellDim, bottomHalfEdge), LSlhs, LSrhs, centroid, currPressure);

			gradPhi = LSlhs.inverse()*LSrhs;
			return Vector2(gradPhi.x(), gradPhi.y());
		}

		//Vector2 CutCellSolverSO2D::getPressureGradient(const HalfFace<Vector2> *pCurrCell) {
		//	dimensions_t cellIndex;
		//	Eigen::Matrix2d LSlhs;
		//	Eigen::Vector2d gradPhi, LSrhs;
		//	Vector2 centroid;
		//	Scalar currPressure;
		//	LSlhs.setZero();
		//	gradPhi.setZero();
		//	LSrhs.setZero();

		//	uint cellIndForMatrix = pCurrCell->getID();
		//	centroid = pCurrCell->getCentroid();
		//	currPressure = m_cutCellsPressures[cellIndForMatrix];

		//	int numAxisAlignedEdges = 0;
		//	vector<dimensions_t> neighbors;
		//	vector< const HalfFace<Vector2> *> pNeighborCutCells;
		//	vector<dimensions_t> neighborRegularCells;
		//	/* ============================
		//	* Transverse through ALL EDGES
		//	* ============================ */
		//	for (uint edge = 0; edge < pCurrCell->getHalfEdges().size(); edge++) {
		//		Edge<Vector2> *pCurrEdge = pCurrCell->getHalfEdges()[edge]->getEdge();
		//		if (pCurrEdge->getConnectedHalfFaces().size() > 2) {
		//			throw(exception("Invalid number of faces connected to an edge"));
		//		}
		//		/* ===========================================================
		//		* For non-geometry edges, find corresponding neighboring cells
		//		* ============================================================ */
		//		if (pCurrEdge->getType() != geometricEdge) {
		//			numAxisAlignedEdges++;
		//			dimensions_t nbDim = getNeighborCellDim(pCurrCell->getFace()->getGridCellLocation(), pCurrCell->getHalfEdges()[edge]->getLocation());

		//			if ((pCurrEdge->getConnectedHalfFaces().size() == 1)) { // Neighbor to a regular grid face
		//				this->updateLeastSquare(nbDim, LSlhs, LSrhs, centroid, currPressure);
		//				neighbors.push_back(nbDim);
		//				neighborRegularCells.push_back(nbDim);
		//			}
		//			else {
		//				uint nbCellInd = pCurrEdge->getConnectedHalfFaces()[0]->getID() == cellIndForMatrix ? pCurrEdge->getConnectedHalfFaces()[1]->getID() : pCurrEdge->getConnectedHalfFaces()[0]->getID();
		//				const HalfFace<Vector2> * pNbCell = &m_pCutCells->getCutCell(nbCellInd);
		//				neighbors.push_back(pNbCell->getFace()->getGridCellLocation());
		//				pNeighborCutCells.push_back(pNbCell);
		//				this->updateLeastSquare(pNbCell, LSlhs, LSrhs, centroid, currPressure);
		//			}
		//		}
		//		/* ===================================================
		//		* For geometry edges, in 2D, neighboring cell is the
		//		* sibling cut-cell in same regular cell. Usually, its
		//		* ID is the next/previous number of current cell ID.
		//		* ==================================================== */
		//		else {
		//			// TODO: However in current sample, neighboring cut-cell is in another closed space. Thus we don't take it into account.
		//			/*int nbCellId = cutCellId;
		//			if (cutCellId + 1 >= 0 && cutCellId + 1 < m_pCutCells2D->getNumberOfCells()) {
		//			nbCellId = cutCellId + 1;
		//			if (m_pCutCells2D->getSpecialCell(nbCellId).m_regularGridIndex != dimensions_t(i, j) && cutCellId - 1 >= 0) {
		//			nbCellId = cutCellId - 1;
		//			}
		//			}
		//			else if (cutCellId - 1 >= 0) {
		//			nbCellId = cutCellId - 1;
		//			}
		//			if (m_pCutCells2D->getSpecialCell(nbCellId).m_regularGridIndex != dimensions_t(i, j)) // This cannot happen!
		//			return Vector2(0, 0);
		//			this->updateLeastSquare(i, j, LSlhs, LSrhs, centroid, curPressure, nbCellId);*/
		//		}
		//	}

		//	if (numAxisAlignedEdges == 2) {
		//		/* =================================================
		//		* If it's a "triangle" cut-cell, we need an additional 
		//		* neighbor to correctly interpolate.
		//		* ================================================= */
		//		float iTemp = (neighbors[0].x + neighbors[1].x) / 2;
		//		float jTemp = (neighbors[0].y + neighbors[1].y) / 2;
		//		dimensions_t additionalNb;
		//		if (iTemp < (pCurrCell->getFace()->getGridCellLocation().x))
		//			additionalNb.x = (pCurrCell->getFace()->getGridCellLocation().x) - 1;
		//		else
		//			additionalNb.x = (pCurrCell->getFace()->getGridCellLocation().x) + 1;
		//		if (jTemp < (pCurrCell->getFace()->getGridCellLocation().y))
		//			additionalNb.y = (pCurrCell->getFace()->getGridCellLocation().y) - 1;
		//		else
		//			additionalNb.y = (pCurrCell->getFace()->getGridCellLocation().y) + 1;

		//		if (m_pCutCells->isCutCellAt(additionalNb.x, additionalNb.y)) {
		//			/* =================================================
		//			* Find the diagonally neighboring cutcell. It should be 
		//			* connected to the original 2 neighbors.
		//			* ================================================= */
		//			auto candidates = m_pCutCells->getFace(additionalNb)->getHalfFaces();
		//			vector<vector<uint>> potentialNbCellIDs;
		//			for (auto & it : neighborRegularCells) {
		//				vector<uint> tempIDs;
		//				if (additionalNb == this->getNeighborCellDim(it, rightHalfEdge)) {
		//					tempIDs.push_back(getNeighborCutCell(it, rightHalfEdge)->getID());
		//				}
		//				if (additionalNb == this->getNeighborCellDim(it, leftHalfEdge)) {
		//					tempIDs.push_back(getNeighborCutCell(it, leftHalfEdge)->getID());
		//				}
		//				if (additionalNb == this->getNeighborCellDim(it, topHalfEdge)) {
		//					tempIDs.push_back(getNeighborCutCell(it, topHalfEdge)->getID());
		//				}
		//				if (additionalNb == this->getNeighborCellDim(it, bottomHalfEdge)) {
		//					tempIDs.push_back(getNeighborCutCell(it, bottomHalfEdge)->getID());
		//				}
		//				potentialNbCellIDs.push_back(tempIDs);
		//			}

		//			for (auto & it : pNeighborCutCells) {
		//				vector<uint> tempIDs;
		//				auto halfEdges = it->getHalfEdges();
		//				uint currID = it->getID();
		//				for (uint j = 0; j < halfEdges.size(); j++) {
		//					auto pEdge = halfEdges[j]->getEdge();
		//					if (pEdge->getConnectedHalfFaces().size() == 2) {
		//						uint nbCellID = pEdge->getConnectedHalfFaces()[0]->getID() == currID ? pEdge->getConnectedHalfFaces()[1]->getID() : pEdge->getConnectedHalfFaces()[0]->getID();
		//						tempIDs.push_back(nbCellID);
		//					}
		//				}
		//				potentialNbCellIDs.push_back(tempIDs);
		//			}
		//			int trueNbCutCell = -1;
		//			for (auto & candidate : candidates) {
		//				bool flag = false;
		//				for (auto & currVec : potentialNbCellIDs) {
		//					if (find(currVec.begin(), currVec.end(), candidate->getID()) == currVec.end()) {
		//						flag = true;
		//						break;
		//					}
		//				}
		//				if (flag == false) {
		//					this->updateLeastSquare(candidate, LSlhs, LSrhs, centroid, currPressure);
		//					trueNbCutCell = candidate->getID();
		//					break;
		//				}
		//			}
		//		}
		//		else {
		//			this->updateLeastSquare(additionalNb, LSlhs, LSrhs, centroid, currPressure);
		//		}
		//	}
		//	/* =================================================
		//	* Finally, Solve LS to get gradPhi for current cell
		//	* ================================================= */
		//	gradPhi = LSlhs.inverse()*LSrhs;
		//	return Vector2(gradPhi.x(), gradPhi.y());
		//}


		#pragma endregion

		#pragma region Triangulation
		//void CutCellSolverSO2D::initializeTriangulator()
		//{
		//	m_referencePoints = getExpandedBoundary();
		//	vector<pair<Vector2, dimensions_t> > bPShrinked = getShrinkedBoundary();

		//	m_referencePoints.insert(m_referencePoints.end(), bPShrinked.begin(), bPShrinked.end());

		//	vector<pair<Vector2, unsigned int>> boundaryPointsF;
		//	for (int i = 0; i < m_referencePoints.size(); i++)
		//	{
		//		boundaryPointsF.push_back(pair<Vector2, unsigned int>(m_referencePoints[i].first, i));
		//	}
		//	m_triangulator = CGALWrapper::Triangulator<Vector2>(boundaryPointsF);
		//	m_triangulator.getBoundaryVerticesList();
		//}
		//
		//vector<pair<Vector2, dimensions_t> > CutCellSolverSO2D::getExpandedBoundary() {
		//	/* ==================================================
		//	* This function is for getting all cells outside of an
		//	* object boundary, then connect them into a closed
		//	* polygon.
		//	* The algorithm is described as below:
		//	* Given a polygon already projected onto grids, its
		//	* boundary can be described as a series of special cells.
		//	* 1) record coords of all special cells
		//	* 2) move them in 4 directions by 1 cell, record coords
		//	* 3) if new coord overlays a special cell, or is in the
		//	*    region of original polygon, remove it
		//	* 4) the coords left in queue is expanded boundary
		//	* Limitation:
		//	* this method can only be applied to the situation where
		//	* ONLY ONE object exists. Also, error could happen when
		//	* an edge is exactly on grid edges.
		//	* ================================================== */
		//	using namespace std;
		//	set<dimensions_t> resSet;
		//	vector<pair<Vector2, dimensions_t> > res;
		//	set<dimensions_t>::iterator it;
		//	// Here I bet there's an easier way to get a list of all special cells.
		//	// Unfortunately I haven't discovered it yet.
		//	for (int i = 1; i < m_dimensions.x - 1; i++) {
		//		for (int j = 1; j < m_dimensions.y - 1; j++) {
		//			if (m_pCutCells->isCutCellAt(i, j))
		//			{
		//				resSet.insert(dimensions_t(i + 1, j));
		//				resSet.insert(dimensions_t(i - 1, j));
		//				resSet.insert(dimensions_t(i, j - 1));
		//				resSet.insert(dimensions_t(i, j + 1));
		//			}
		//		}
		//	}

		//	for (it = resSet.begin(); it != resSet.end(); it++) {
		//		int x = (*it).x;
		//		int y = (*it).y;
		//		if (!(m_pCutCells->isCutCellAt(x, y) || this->isInObject(*it))) {
		//			res.push_back(make_pair(m_pGridData->getCenterPoint(x, y), dimensions_t(x, y)));
		//		}
		//	}

		//	return res;
		//}

		//vector<pair<Vector2, dimensions_t> > CutCellSolverSO2D::getShrinkedBoundary() {
		//	set<dimensions_t> resSet;
		//	vector<pair<Vector2, dimensions_t> > res;
		//	set<dimensions_t>::iterator it;
		//	// Here I bet there's an easier way to get a list of all special cells.
		//	// Unfortunately I haven't discovered it yet.
		//	//for (int i = 1; i < m_dimensions.x - 1; i++) {
		//	//	for (int j = 1; j < m_dimensions.y - 1; j++) {
		//	//		if (m_pCutCells->isCutCellAt(i, j))
		//	//		{
		//	//			resSet.insert(dimensions_t(i + 1, j));
		//	//			resSet.insert(dimensions_t(i - 1, j));
		//	//			resSet.insert(dimensions_t(i, j - 1));
		//	//			resSet.insert(dimensions_t(i, j + 1));
		//	//		}
		//	//	}
		//	//}

		//	//for (it = resSet.begin(); it != resSet.end(); it++) {
		//	//	int x = (*it).x;
		//	//	int y = (*it).y;
		//	//	if ((!m_pCutCells->isCutCellAt(x, y)) && this->isInObject(*it)) {
		//	//		res.push_back(make_pair(m_pGridData->getCenterPoint(x, y), dimensions_t(x, y)));

		//	//	}
		//	//}

		//	return res;
		//}
		#pragma endregion

		#pragma region ExactInterpolation
		//Scalar CutCellSolverSO2D::getCutCellPressureExact(int index)
		//{
		//	HalfFace<Vector2> currCell = m_pCutCells->getCutCell(index);
		//	Vector2 centerCoord = currCell.getCentroid();
		//	vector<std::pair<double, unsigned>> weights = m_triangulator.getBarycentricCoordinates(centerCoord);
		//	vector<std::pair<double, unsigned>>::iterator it;
		//	double updatedPressure = 0;
		//	////cout << "======= FOR CELL " << index << " =======" << endl;
		//	////cout << "Original pressure: " << m_pCutCells2D->getPressure(index) << endl;
		//	//for (it = weights.begin(); it != weights.end(); it++)
		//	//{
		//	//	int ind = (*it).second;
		//	//	double weight = (*it).first;
		//	//	//cout << "Cell coordinate: " << m_referencePoints[ind].second.x << ", " << m_referencePoints[ind].second.y;
		//	//	//cout << ", Weight: " << weight << endl;

		//	//	double pressure = m_pGridData->getPressure(m_referencePoints[ind].second.x, m_referencePoints[ind].second.y);
		//	//	updatedPressure += pressure * weight;
		//	//}
		//	////cout << "Updated pressure: " << updatedPressure << endl;
		//	return updatedPressure;
		//}

		//void CutCellSolverSO2D::interpolateOnCutCellCenter()
		//{
		//	//for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++)
		//	//{
		//	//	HalfFace<Vector2> currCell = m_pCutCells->getCutCell(i);
		//	//	Vector2 centerCoord = currCell.getCentroid();
		//	//	vector<std::pair<double, unsigned>> weights = m_triangulator.getBarycentricCoordinates(centerCoord);
		//	//	vector<std::pair<double, unsigned>>::iterator it;
		//	//	double updatedPressure = 0;
		//	//	//cout << "======= FOR CELL " << i << " =======" << endl;

		//	//	//cout << "Original pressure: " << m_pCutCells2D->getPressure(i) << endl;
		//	//	for (it = weights.begin(); it != weights.end(); it++)
		//	//	{
		//	//		int ind = (*it).second;
		//	//		double weight = (*it).first;
		//	//		//cout << "Cell coordinate: " << m_referencePoints[ind].second.x << ", " << m_referencePoints[ind].second.y;
		//	//		//cout << ", Weight: " << weight << endl;

		//	//		double pressure = m_pGridData->getPressure(m_referencePoints[ind].second.x, m_referencePoints[ind].second.y);
		//	//		updatedPressure += pressure * weight;
		//	//	}
		//	//	//cout << "Updated pressure: " << updatedPressure << endl;
		//	//}
		//}
		#pragma endregion
	}	
}