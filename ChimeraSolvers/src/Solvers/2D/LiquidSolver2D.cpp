////  Copyright (c) 2013, Vinicius Costa Azevedo
////	All rights reserved.
////
////	Redistribution and use in source and binary forms, with or without
////	modification, are permitted provided that the following conditions are met: 
////
////1. Redistributions of source code must retain the above copyright notice, this
////	list of conditions and the following disclaimer. 
////	2. Redistributions in binary form must reproduce the above copyright notice,
////	this list of conditions and the following disclaimer in the documentation
////	and/or other materials provided with the distribution. 
////
////	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
////	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
////	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
////	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
////	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
////	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
////LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
////	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
////	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
////	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////
////	The views and conclusions contained in the software and documentation are those
////	of the authors and should not be interpreted as representing official policies, 
////	either expressed or implied, of the FreeBSD Project.
//
//#include "Solvers/2D/LiquidSolver2D.h"
//
//namespace Chimera {
//
//	namespace Solvers {
//		#pragma region Constructors
//		LiquidSolver2D::LiquidSolver2D(const FlowSolverParameters &params, StructuredGrid<Vector2> *pGrid, 
//										LiquidRepresentation2D *pLiquidRepresentation,
//										const vector<BoundaryCondition<Vector2> *> &boundaryConditions)
//			: FlowSolver(params, pGrid) {
//			m_pGrid = pGrid;
//			m_pLiquidRepresentation = pLiquidRepresentation;
//			m_dimensions = m_pGrid->getDimensions();
//			m_boundaryConditions = boundaryConditions;
//			m_pGridData = m_pGrid->getGridData2D();
//
//			
//
//			/** Boundary conditions for initialization */
//			enforceBoundaryConditions();
//
//			/** Standard integration params initialization */
//			initializeIntegrationParams();
//		}
//		#pragma endregion 
//
//		#pragma region UpdateFunctions
//		//Updates the current flow solver with the given time step
//		void LiquidSolver2D::update(Scalar dt) {
//			m_numIterations++;
//			m_params.m_totalSimulationTimer.start();
//
//			/** Advection */
//			enforceBoundaryConditions();
//
//			if (PhysicsCore<Vector2>::getInstance()->getElapsedTime() < dt) {
//				applyForces(dt);
//				enforceBoundaryConditions();
//				updateDivergents(dt);
//				enforceBoundaryConditions();
//				solvePressure();
//				enforceBoundaryConditions();
//				project(dt);
//				enforceBoundaryConditions();
//
//				if (m_pParticleBasedAdvection) {
//					m_pParticleBasedAdvection->getParticlesSampler()->interpolateVelocities(m_pVelocityInterpolant, m_pParticleBasedAdvection->getParticlesData());
//				}
//			}
//
//			m_params.m_advectionTimer.start();
//			m_pParticleBasedAdvection->updatePositions(dt);
//			m_pParticleBasedAdvection->updateGridAttributes();
//			m_params.m_advectionTimer.stop();
//			m_advectionTime = m_params.m_advectionTimer.secondsElapsed();
//		
//			enforceBoundaryConditions();
//			/** Solve pressure */
//			m_params.m_solvePressureTimer.start();
//			
//			updateDivergents(dt);
//			solvePressure();
//
//			enforceBoundaryConditions();
//			m_params.m_solvePressureTimer.stop();
//			m_solvePressureTime = m_params.m_solvePressureTimer.secondsElapsed();
//
//			/** Project velocity */
//			m_params.m_projectionTimer.start();
//			project(dt);
//			m_params.m_projectionTimer.stop();
//			m_projectionTime = m_params.m_projectionTimer.secondsElapsed();
//			enforceBoundaryConditions();
//
//			enforceBoundaryConditions();
//
//			/** Updating FLIP particles velocities */
//			if (m_pParticleBasedAdvection) {
//				m_pParticleBasedAdvection->updateParticleAttributes();
//				m_pGridData->getDensityBuffer().swapBuffers();
//				m_pGridData->getTemperatureBuffer().swapBuffers();
//			}
//
//			m_params.m_totalSimulationTimer.stop();
//			m_totalSimulationTime = m_params.m_totalSimulationTimer.secondsElapsed();
//		}
//
//		void LiquidSolver2D::updateFLIPParticleTags() {
//			Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
//			if (m_pFLIP) {
//				for (int i = 0; i < m_pFLIP->getParticlesPosition().size(); i++) {
//					for (int j = 0; j < m_liquidMeshes.size(); j++) {
//						Vector2 transformedPosition = m_pFLIP->getParticlesPosition()[i] * dx;
//						if (isInsidePolygon(transformedPosition, m_liquidMeshes[j]->getPoints())) {
//							m_pFLIP->getParticlesTagPtr()->at(i) = 1;
//						}
//						else {
//							m_pFLIP->getParticlesTagPtr()->at(i) = 0;
//						}
//					}
//				}
//			}
//
//		}
//		#pragma endregion UpdateFunctions
//
//		#pragma region ObjectsInitialization
//		void LiquidSolver2D::reinitializeLiquidBounds() {
//			/** Reinitializing Cut-cells data structure */
//			m_pCutCells2D->flushThinBounds();
//			m_params.m_cutCellGenerationTimer.start();
//			m_pCutCells2D->initializeThinBounds(m_liquidMeshes, false);
//			m_params.m_cutCellGenerationTimer.stop();
//
//			/** Updates poisson matrix with liquid representation, then updates conjugate gradient solver to accomodate
//			* these changes. */
//			updateConjugateGradientSolver(updatePoissonMatrix());
//		}
//		#pragma endregion ObjectsInitialization
//
//		#pragma region InitializationFunctions
//		PoissonMatrix * LiquidSolver2D::createPoissonMatrix() {
//			dimensions_t poissonMatrixDim = m_pGridData->getDimensions();
//
//			if (m_params.getPressureSolverParams().getMethodCategory() == Krylov && !m_pGrid->isPeriodic()) {
//				poissonMatrixDim.x += -2; poissonMatrixDim.y += -2;
//			}
//			else if (m_params.getPressureSolverParams().getMethodCategory() == Krylov && m_pGrid->isPeriodic()) { //X peridiocity
//				poissonMatrixDim.y += -2;
//			}
//
//
//			PoissonMatrix *pMatrix = NULL;
//			bool initializeGPU = m_params.getPressureSolverParams().getPressureSolverMethod() == GPU_CG;
//			pMatrix = new PoissonMatrix(poissonMatrixDim, initializeGPU);
//
//			return pMatrix;
//		}
//		#pragma endregion InitializationFunctions
//
//		#pragma region SimulationFunctions
//		void LiquidSolver2D::applyForces(Scalar dt) {
//			if (PhysicsCore::getInstance()->getElapsedTime() < dt) {
//				for (int i = 0; i < m_pFLIP->getParticlesTagPtr()->size(); i++) {
//					if (m_pFLIP->getParticlesTagPtr()->at(i) == 1) {
//						m_pFLIP->getParticlesVelocitiesPtr()[i] = Vector2(0, -4);
//					}
//				}
//			}
//		}
//
//		void LiquidSolver2D::flipAdvection(Scalar dt) {
//			//m_pFLIP->setThinObjectAcceleration(-(nextThinObjectVelocity - currThinObjectVelocity)/(dt));
//
//			m_pFLIP->updateParticlesPosition(dt);
//			m_pLiquidRepresentation->updateMeshes();
//			m_liquidMeshes = m_pLiquidRepresentation->getLineMeshes();
//
//			updatePoissonMatrix();
//
//			/*if (PhysicsCore::getInstance()->getElapsedTime() >= dt) {
//				return;
//			}*/
//			if(m_pCutCells2D)
//				reinitializeLiquidBounds();
//
//			if (m_pCutCells2D != NULL) { ///Updates free-slip velocities and node positions to test against cells  
//				m_pCutCells2D->preprocessVelocityData(m_pNodeVelocityField);
//			}
//
//			m_pFLIP->accumulateVelocitiesToGrid();
//			enforceBoundaryConditions();
//			m_pFLIP->backupVelocities();
//		}
//
//		Scalar LiquidSolver2D::calculateFluxDivergent(int i, int j) {
//			Scalar divergent = 0;
//
//			int row = 0;
//			if (m_params.getPressureSolverParams().getMethodCategory() == Krylov) {
//				row = m_pPoissonMatrix->getRowIndex(i - 1, j - 1);
//			}
//			else {
//				row = m_pPoissonMatrix->getRowIndex(i, j);
//			}
//
//			Scalar dx, dy = 0;
//			if (m_pCutCells2D && m_pCutCells2D->isSpecialCell(i, j)) {
//				return 0;
//			}
//			else if (i > 1 && i < m_dimensions.x - 1 && j > 1 && j < m_dimensions.y - 1) {
//				Scalar pn = abs(m_pPoissonMatrix->getNorthValue(row));
//				Scalar pe = abs(m_pPoissonMatrix->getEastValue(row));
//				Scalar ps = abs(m_pPoissonMatrix->getWestValue(row));
//				Scalar pw = abs(m_pPoissonMatrix->getSouthValue(row));
//
//				if (pn == 0.0f)
//					pn = 1;
//				if (pe == 0.0f)
//					pe = 1;
//				if (ps == 0.0f)
//					ps = 1;
//				if (pw == 0.0f)
//					pw = 1;
//
//				if (m_pCutCells2D->isSpecialCell(i + 1, j)) {
//					Scalar specialCellVelocity = m_pCutCells2D->getEdgeVector(dimensions_t(i + 1, j), leftEdge)[0].getVelocity().x
//						* m_pCutCells2D->getEdgeVector(dimensions_t(i + 1, j), leftEdge)[0].getLengthFraction();
//					dx = (specialCellVelocity - m_pGridData->getAuxiliaryVelocity(i, j).x*pw) / m_pGridData->getScaleFactor(i, j).x;
//				}
//				else if (m_pCutCells2D->isSpecialCell(i - 1, j)) { //Use the last element on the face vector, since this cell will always be on "top"
//					Scalar specialCellVelocity = m_pCutCells2D->getEdgeVector(dimensions_t(i - 1, j), rightEdge).back().getVelocity().x
//						* m_pCutCells2D->getEdgeVector(dimensions_t(i - 1, j), rightEdge).back().getLengthFraction();
//					dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j).x*pe - specialCellVelocity) / m_pGridData->getScaleFactor(i, j).x;
//				}
//				else {
//					dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j).x*pe
//						- m_pGridData->getAuxiliaryVelocity(i, j).x*pw) / m_pGridData->getScaleFactor(i, j).x;
//				}
//
//				if (m_pCutCells2D->isSpecialCell(i, j + 1)) {
//					Scalar specialCellVelocity = m_pCutCells2D->getEdgeVector(dimensions_t(i, j + 1), bottomEdge)[0].getVelocity().y
//						* m_pCutCells2D->getEdgeVector(dimensions_t(i, j + 1), bottomEdge)[0].getLengthFraction();
//					dy = (specialCellVelocity - m_pGridData->getAuxiliaryVelocity(i, j).y*ps) / m_pGridData->getScaleFactor(i, j).y;
//				}
//				else if (m_pCutCells2D->isSpecialCell(i, j - 1)) { //Use the last element on the face vector, since this cell will always be on "top"
//					Scalar specialCellVelocity = m_pCutCells2D->getEdgeVector(dimensions_t(i, j - 1), topEdge).back().getVelocity().y
//						* m_pCutCells2D->getEdgeVector(dimensions_t(i, j - 1), topEdge).back().getLengthFraction();
//					dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1).y*pn - specialCellVelocity) / m_pGridData->getScaleFactor(i, j).y;
//				}
//				else {
//					dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1).y*pn
//						- m_pGridData->getAuxiliaryVelocity(i, j).y*ps) / m_pGridData->getScaleFactor(i, j).y;
//				}
//
//			}
//			else {
//				dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j).x - m_pGridData->getAuxiliaryVelocity(i, j).x) / m_pGridData->getScaleFactor(i, j).x;
//				dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1).y - m_pGridData->getAuxiliaryVelocity(i, j).y) / m_pGridData->getScaleFactor(i, j).y;
//			}
//
//			divergent = dx + dy;
//
//			return divergent;
//
//		}
//		#pragma endregion SimulationFunctions
//
//		#pragma region InternalAuxiliaryFunctions
//		void LiquidSolver2D::updateConjugateGradientSolver(bool additionalCells) {
//			ConjugateGradient::solverParams_t *solverParams;
//			solverParams = (ConjugateGradient::solverParams_t *) m_params.getPressureSolverParams().getSpecificSolverParams();
//			solverParams->solveThinBoundaries = additionalCells;
//			if (additionalCells) {
//				solverParams->pSpecialDivergents = m_pCutCells2D->getDivergentsPtr();
//				solverParams->pSpecialPressures = m_pCutCells2D->getPressuresPtr();
//				solverParams->numberOfSpecialCells = m_pCutCells2D->getNumberOfCells();
//				((ConjugateGradient *)m_pPoissonSolver)->resizeScalarFields();
//				((ConjugateGradient *)m_pPoissonSolver)->reinitializePreconditioners();
//			} else {
//				solverParams->pSpecialDivergents = NULL;
//				solverParams->pSpecialPressures = NULL;
//				solverParams->numberOfSpecialCells = 0;
//			}
//		}
//	
//		Scalar LiquidSolver2D::calculateFinalDivergent(int i, int j) {
//			Scalar divergent = 0;
//
//			if (m_pGrid->isBoundaryCell(i, j) || m_pGrid->isSolidCell(i - 1, j) || m_pGrid->isSolidCell(i + 1, j)
//				|| m_pGrid->isSolidCell(i, j - 1) || m_pGrid->isSolidCell(i, j + 1))
//				return divergent;
//
//			Scalar dx = (m_pGridData->getVelocity(i + 1, j).x - m_pGridData->getVelocity(i, j).x) / m_pGridData->getScaleFactor(i, j).x;
//			Scalar dy = (m_pGridData->getVelocity(i, j + 1).y - m_pGridData->getVelocity(i, j).y) / m_pGridData->getScaleFactor(i, j).y;
//
//			divergent = dx + dy;
//			return divergent;
//		}
//		#pragma endregion InternalAuxiliaryFunctions
//
//	}
//	
//	
//}