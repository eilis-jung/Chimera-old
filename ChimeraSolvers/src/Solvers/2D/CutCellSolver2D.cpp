//  Copyright (c) 2013, Vinicius Costa Azevedo
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met: 
//
//1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer. 
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution. 
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//	The views and conclusions contained in the software and documentation are those
//	of the authors and should not be interpreted as representing official policies, 
//	either expressed or implied, of the FreeBSD Project.

#include "Solvers/2D/CutCellSolver2D.h"

namespace Chimera {

	namespace Solvers {
		#pragma region ConstructorsDestructors
		
		CutCellSolver2D::CutCellSolver2D(const params_t &params, StructuredGrid<Vector2> *pGrid,
												 const vector<BoundaryCondition<Vector2> *> &boundaryConditions, const vector<RigidObject2D<Vector2> *> &rigidObjects)
												: RegularGridSolver2D(params, pGrid, boundaryConditions, rigidObjects), m_nodalBasedVelocities(pGrid->getDimensions()), 
																														m_auxNodalBasedVelocities(pGrid->getDimensions()) {
				m_pAdvection = nullptr;

				m_pCutCells = initializeCutCells();

				/** Initialize cut-cells main and auxiliary velocities */
				m_pCutCellsVelocities2D = new CutCellsVelocities2D(m_pCutCells, m_params.solidBoundaryType);
				m_pAuxCutCellsVelocities2D = new CutCellsVelocities2D(m_pCutCells, m_params.solidBoundaryType);

				initializeInterpolants();

				m_pPoissonMatrix = createPoissonMatrix();
				if(m_pCutCells) {
					updatePoissonThinSolidWalls();
				}

				/**Poisson solver initialization depends on correct handleThinBoundaries parameters */
				initializePoissonSolver();

				/** Boundary conditions for initialization */
				enforceBoundaryConditions();

				m_pAdvection = initializeAdvectionClass();
				if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
					ParticleBasedAdvection<Vector2, Array2D> *pPBAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
					pPBAdv->getParticleBasedIntegrator()->setCutCells(m_pCutCells);
					ParticlesToNodalGrid2D *pParticlesToNodal = dynamic_cast<ParticlesToNodalGrid2D *>(pPBAdv->getParticlesToGrid());
					if (pParticlesToNodal == nullptr) {
						throw(exception("CutCellSolver2D: only particlesToNodalGrid is supported by cut-cells particle-based advection"));
					}
					pParticlesToNodal->setCutCellsVelocities(m_pCutCellsVelocities2D);

					pPBAdv->getParticleBasedIntegrator()->setCutCells(m_pCutCells);
					pPBAdv->getParticlesSampler()->setCutCells(m_pCutCells);
				}

				((ConjugateGradient *) m_pPoissonSolver)->reinitializePreconditioners();
		}
		#pragma endregion

		#pragma region UpdateFunctions
		//Updates the current flow solver with the given time step
		void CutCellSolver2D::update(Scalar dt) {
			MeanValueInterpolant2D<Vector2> *pMeanValueInterpolant = dynamic_cast<MeanValueInterpolant2D<Vector2> *>(m_pVelocityInterpolant);
			MeanValueInterpolant2D<Vector2> *pAuxMeanValueInterpolant = dynamic_cast<MeanValueInterpolant2D<Vector2> *>(m_pAuxVelocityInterpolant);
			ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
			if (pParticleBasedAdv == nullptr) {
				throw(exception("CutCellSolver2D: only particle based advection methods are supported now"));
			}

			m_numIterations++;
			m_totalSimulationTimer.start();

			if (m_numIterations == 2)
				updatePoissonThinSolidWalls();

			if (PhysicsCore<Vector2>::getInstance()->getElapsedTime() < dt) {
				//applyForces(dt);

				/** Update rigid objects velocities into cut-cells */
				for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
					if(m_rigidObjectsVec[i]->getLineMesh()->hasUpdated())
						m_rigidObjectsVec[i]->updateCutEdgesVelocities(0, m_pCutCells->getGridSpacing(), true);
				}

				enforceBoundaryConditions();
				if(m_pCutCells)
					updateCutCellsDivergence(dt);

				updateDivergents(dt);
				solvePressure();
				project(dt);
				enforceBoundaryConditions();

				/** Update rigid objects velocities into cut-cells */
				for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
					if (m_rigidObjectsVec[i]->getLineMesh()->hasUpdated())
						m_rigidObjectsVec[i]->updateCutEdgesVelocities(0, m_pCutCells->getGridSpacing());
				}

				if (pMeanValueInterpolant) {
					pMeanValueInterpolant->updateNodalVelocities(m_pGridData->getVelocityArray(), m_nodalBasedVelocities);
				}

				pParticleBasedAdv->getParticlesSampler()->interpolateVelocities(m_pVelocityInterpolant, pParticleBasedAdv->getParticlesData());
			}

			m_advectionTimer.start();
		
			/** Advection is not following traditional PBA pipeline since we need to update rigid bodies and cut-cells
				before splatting into the grid. */
			{
				//First update particle positions
				pParticleBasedAdv->updatePositions(dt);

				/** Then update rigid bodies*/
				for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
					m_rigidObjectsVec[i]->update(dt);
				}

				bool reinitializeSpecialCells = false;
				for (int i = 0; i < m_lineMeshes.size(); i++) {
					if (m_lineMeshes[i]->hasUpdated()) {
						reinitializeSpecialCells = true;
					}
				}

				/** If one of the objects moved (hasUpdated() == true), recreate all cut-cells */
				if (reinitializeSpecialCells) {
					reinitializeThinBounds();

					//Update velocities after reinitialization, since those are used on the update grid attributes
					for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
						m_rigidObjectsVec[i]->updateCutEdgesVelocities(1, m_pCutCells->getGridSpacing(), true);
					}
				}

				/** Transfer velocities to the newly created grid nodes */
				pParticleBasedAdv->updateGridAttributes();
				/** Update nodal velocities */
				pAuxMeanValueInterpolant->updateNodalVelocities(m_pGridData->getAuxVelocityArray(), m_auxNodalBasedVelocities, true);
			}
			m_advectionTimer.stop();
			m_advectionTime = m_advectionTimer.secondsElapsed();

			/** Solve pressure */
			enforceBoundaryConditions();
			m_solvePressureTimer.start();
			if (m_pCutCells)
				updateCutCellsDivergence(dt);
			updateDivergents(dt);
			solvePressure();
			m_solvePressureTimer.stop();
			m_solvePressureTime = m_solvePressureTimer.secondsElapsed();

			/** Project velocity */
			m_projectionTimer.start();
			project(dt);
			m_projectionTimer.stop();
			m_projectionTime = m_projectionTimer.secondsElapsed();
			enforceBoundaryConditions();

			/** Update rigid bodies velocities */
			for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
				if (m_rigidObjectsVec[i]->getLineMesh()->hasUpdated())
					m_rigidObjectsVec[i]->updateCutEdgesVelocities(1, m_pCutCells->getGridSpacing());
			}
			/** And nodal velocities with after advection grid-based velocities */
			pMeanValueInterpolant->updateNodalVelocities(m_pGridData->getVelocityArray(), m_nodalBasedVelocities);

			/** Update particles attributes and advect density*/
			m_pAdvection->postProjectionUpdate(dt);

			m_totalSimulationTimer.stop();
			m_totalSimulationTime = m_totalSimulationTimer.secondsElapsed();
		}
		#pragma endregion 

		#pragma region ObjectsInitialization
		void CutCellSolver2D::reinitializeThinBounds() {
			m_cutCellGenerationTimer.start();
			m_pCutCells->reinitialize(m_lineMeshes);
			m_cutCellGenerationTimer.stop();

			m_pPoissonMatrix->setNumberAdditionalCells(m_pCutCells->getNumberCutCells());
			m_cutCellsPressures.resize(m_pCutCells->getNumberCutCells());
			m_cutCellsDivergents.resize(m_pCutCells->getNumberCutCells());
			m_cutCellsDivergents.assign(m_cutCellsDivergents.size(), 0.f);
			m_cutCellsPressures.assign(m_cutCellsPressures.size(), 0.f);

			for(int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
				for(int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
					m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(i, j), -1, -1, 4, -1, -1);
				}
			}
			for(unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
				m_boundaryConditions[i]->updatePoissonMatrix(m_pPoissonMatrix);
			}
			//BoundaryCondition<Vector2>::updateSolidWalls(m_pPoissonMatrix, Array2D<bool>(m_pGrid->getSolidMarkers(), m_dimensions), m_params.getPressureSolverParams().getMethodCategory() == Krylov);
			m_pPoissonMatrix->applyCorrection(1e-4);

			updatePoissonThinSolidWalls();

			m_params.pPoissonSolverParams->solveThinBoundaries = true;
			m_params.pPoissonSolverParams->pSpecialDivergents = &m_cutCellsDivergents;
			m_params.pPoissonSolverParams->pSpecialPressures = &m_cutCellsPressures;
			m_params.pPoissonSolverParams->numberOfSpecialCells = m_pCutCells->getNumberCutCells();
			m_pPoissonSolver->setParams(*m_params.pPoissonSolverParams);
			((ConjugateGradient *) m_pPoissonSolver)->resizeScalarFields();
			((ConjugateGradient *) m_pPoissonSolver)->reinitializePreconditioners();
		}
		#pragma endregion

		#pragma region InitializationFunctions
		CutCells2D<Vector2> * CutCellSolver2D::initializeCutCells() {

			if (m_lineMeshes.size() > 0) {
				m_cutCellGenerationTimer.start();
				CutCells2D<Vector2> *pPlanarMesh = new CutCells2D<Vector2>(m_lineMeshes, m_pGridData->getGridSpacing(), m_pGrid->getDimensions());
				pPlanarMesh->initialize();
				m_cutCellsPressures.resize(pPlanarMesh->getNumberCutCells(), 0.0);
				m_cutCellsDivergents.resize(pPlanarMesh->getNumberCutCells(), 0.0);
				m_cutCellGenerationTimer.stop();
				
				m_params.pPoissonSolverParams->solveThinBoundaries = true;
				m_params.pPoissonSolverParams->pSpecialDivergents = &m_cutCellsDivergents;
				m_params.pPoissonSolverParams->pSpecialPressures = &m_cutCellsPressures;
				m_params.pPoissonSolverParams->numberOfSpecialCells = m_cutCellsPressures.size();
				
				return pPlanarMesh;
			}

			return nullptr;
		}
		
		PoissonMatrix * CutCellSolver2D::createPoissonMatrix() {
			dimensions_t poissonMatrixDim = m_pGridData->getDimensions();

			if(m_params.pPoissonSolverParams->solverCategory == Krylov && !m_pGrid->isPeriodic()) {
				poissonMatrixDim.x += -2; poissonMatrixDim.y += -2;
			} else if(m_params.pPoissonSolverParams->solverCategory == Krylov && m_pGrid->isPeriodic()) { //X peridiocity
				poissonMatrixDim.y += -2;
			}


			PoissonMatrix *pMatrix = NULL;
			bool initializeGPU = m_params.pPoissonSolverParams->platform == PlataformGPU;
			if(m_pCutCells) {
				pMatrix = new PoissonMatrix(poissonMatrixDim, m_pCutCells->getNumberCutCells(), initializeGPU);
			} else {
				pMatrix = new PoissonMatrix(poissonMatrixDim, initializeGPU);
			}

			for(int i = 0; i < poissonMatrixDim.x; i++) {
				for(int j = 0; j < poissonMatrixDim.y; j++) {
					pMatrix->setRow(pMatrix->getRowIndex(i, j), -1, -1, 4, -1, -1);
				}
			}
			if(m_pGrid->isPeriodic()) {
				for(int j = 0; j < poissonMatrixDim.y; j++) {
					pMatrix->setPeriodicWestValue(pMatrix->getRowIndex(0, j), -1);
					pMatrix->setPeriodicEastValue(pMatrix->getRowIndex(poissonMatrixDim.x - 1, j), -1);
					pMatrix->setWestValue(pMatrix->getRowIndex(0, j), 0);
					pMatrix->setEastValue(pMatrix->getRowIndex(poissonMatrixDim.x - 1, j), 0);
				}
			}

			for(unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
				m_boundaryConditions[i]->updatePoissonMatrix(pMatrix);
			}
			//BoundaryCondition<Vector2>::updateSolidWalls(pMatrix, Array2D<bool>(m_pGrid->getSolidMarkers(), m_dimensions), m_params.getPressureSolverParams().getMethodCategory() == Krylov);

			if(pMatrix->isSingular() && m_params.pPoissonSolverParams->solverCategory == Krylov)
				pMatrix->applyCorrection(1e-3);

			pMatrix->updateCudaData();

			return pMatrix;
		}

		void CutCellSolver2D::initializeInterpolants() {
			Scalar dx = m_pGridData->getGridSpacing();

			m_pVelocityInterpolant = new MeanValueInterpolant2D<Vector2>(m_nodalBasedVelocities, m_pCutCellsVelocities2D, dx);
			m_pAuxVelocityInterpolant = new MeanValueInterpolant2D<Vector2>(m_auxNodalBasedVelocities, m_pAuxCutCellsVelocities2D, dx, true);

			m_pVelocityInterpolant->setSiblingInterpolant(m_pAuxVelocityInterpolant);

			m_pDensityInterpolant = new BilinearStaggeredInterpolant2D<Scalar>(*m_pGridData->getDensityBuffer().getBufferArray1(), dx);
			m_pDensityInterpolant->setSiblingInterpolant(new BilinearStaggeredInterpolant2D<Scalar>(*m_pGridData->getDensityBuffer().getBufferArray2(), dx));
		}
		#pragma endregion 

		#pragma region PressureProjection
		void CutCellSolver2D::updatePoissonThinSolidWalls() {
			//Double check the initial offset number
			int initialOffset = 0;
			if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
				initialOffset = (m_pGrid->getDimensions().x - 2)*(m_pGrid->getDimensions().y - 2);
			}
			else {
				initialOffset = m_pGrid->getDimensions().x*m_pGrid->getDimensions().y;
			}

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

				for (uint j = 0; j < currCell.getHalfEdges().size(); j++) {
					Edge<Vector2> *pCurrEdge = currCell.getHalfEdges()[j]->getEdge();
					if (pCurrEdge->getConnectedHalfFaces().size() > 2) {
						throw(exception("Invalid number of faces connected to an edge"));
					}
					if (pCurrEdge->getType() != geometricEdge) {
						uint otherPressure;
						if (pCurrEdge->getConnectedHalfFaces().size() == 1) { //Neighbor to a regular grid face
							otherPressure = getRowIndex(currCell.getFace()->getGridCellLocation(), currCell.getHalfEdges()[j]->getLocation());
						}
						else {
							otherPressure = pCurrEdge->getConnectedHalfFaces()[0]->getID() == currFaceID ? pCurrEdge->getConnectedHalfFaces()[1]->getID() : pCurrEdge->getConnectedHalfFaces()[0]->getID();
							otherPressure += initialOffset;
						}

						DoubleScalar pressureCoefficient = pCurrEdge->getRelativeFraction();
						pc += pressureCoefficient;
						m_pPoissonMatrix->setValue(matrixInternalId++, row, otherPressure, -pressureCoefficient);
						if (pCurrEdge->getConnectedHalfFaces().size() == 1) //If we have a regular cell neighbor
							m_pPoissonMatrix->setValue(matrixInternalId++, otherPressure, row, -pressureCoefficient); //Guarantees matrix symmetry
					}
				}

				m_pPoissonMatrix->setValue(matrixInternalId++, row, row, pc);
			}


			if (m_pPoissonMatrix->isSingularCOO() && m_params.pPoissonSolverParams->solverCategory == Krylov)
				m_pPoissonMatrix->applyCorrection(1e-3);

			m_pPoissonMatrix->copyCOOtoHyb();
		}

		void CutCellSolver2D::updateCutCellsDivergence(Scalar dt) {
			if (m_numIterations == 20) {
				Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
				logVelocity("velocity_" + to_string(dx));
				logPressure("pressure_" + to_string(dx));
				logVorticity("vorticity_" + to_string(dx));
				//logVelocityForCutCells("velocity");
			}
			Scalar dx = m_pGridData->getGridSpacing();
			for (uint i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
				DoubleScalar divergent = 0;
				auto halfEdges = m_pCutCells->getCutCell(i).getHalfEdges();
				for (uint j = 0; j < halfEdges.size(); j++) {
					auto pEdge = halfEdges[j]->getEdge();
					divergent += halfEdges[j]->getNormal().dot(pEdge->getAuxiliaryVelocity())*pEdge->getRelativeFraction()*dx;
				}
				m_cutCellsDivergents[i] = -divergent / dt;
			}
		}

		Scalar CutCellSolver2D::calculateFluxDivergent(int i, int j) {
			Scalar divergent = 0;
	
			int row = 0;
			if(m_params.pPoissonSolverParams->solverCategory == Krylov) {
				row = m_pPoissonMatrix->getRowIndex(i - 1, j - 1);
			} else {
				row = m_pPoissonMatrix->getRowIndex(i, j );
			}

			Scalar dx, dy = 0;
			if(m_pCutCells && m_pCutCells->isCutCellAt(i, j)) {
				return 0;
			} else if(i > 1 && i < m_dimensions.x - 1 && j > 1 && j < m_dimensions.y - 1) {
				Scalar pn = abs(m_pPoissonMatrix->getNorthValue(row));
				Scalar pe = abs(m_pPoissonMatrix->getEastValue(row));
				Scalar ps = abs(m_pPoissonMatrix->getSouthValue(row));
				Scalar pw = abs(m_pPoissonMatrix->getWestValue(row));

				if(pn == 0.0f)
					pn = 1;
				if(pe == 0.0f)
					pe = 1;
				if(ps == 0.0f)
					ps = 1;
				if(pw == 0.0f)
					pw = 1;
			
				if(m_pCutCells && m_pCutCells->isCutCellAt(i + 1, j)) {
					Scalar specialCellVelocity = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), yAlignedEdge)[0]->getAuxiliaryVelocity().x 
													* m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), yAlignedEdge)[0]->getRelativeFraction();
					Scalar temp = m_pGridData->getAuxiliaryVelocity(i, j).x;
					dx = (specialCellVelocity - m_pGridData->getAuxiliaryVelocity(i, j).x*pw)/m_pGridData->getScaleFactor(i, j).x;
				} else if(m_pCutCells&& m_pCutCells->isCutCellAt(i - 1, j)) { //Use the last element on the face vector, since this cell will always be on "top"
					Scalar specialCellVelocity = m_pCutCells->getEdgeVector(dimensions_t(i, j), yAlignedEdge).back()->getAuxiliaryVelocity().x
													* m_pCutCells->getEdgeVector(dimensions_t(i, j), yAlignedEdge).back()->getRelativeFraction();
					Scalar temp = m_pGridData->getAuxiliaryVelocity(i + 1, j).x;
					dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j).x*pe - specialCellVelocity)/m_pGridData->getScaleFactor(i, j).x;
				} else {
					dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j).x*pe
						- m_pGridData->getAuxiliaryVelocity(i, j).x*pw)/m_pGridData->getScaleFactor(i, j).x;
				}

				if(m_pCutCells && m_pCutCells->isCutCellAt(i, j + 1)) {
					Scalar specialCellVelocity = m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), xAlignedEdge)[0]->getAuxiliaryVelocity().y
													* m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), xAlignedEdge)[0]->getRelativeFraction();
					Scalar temp = m_pGridData->getAuxiliaryVelocity(i, j).y;
					dy = (specialCellVelocity - m_pGridData->getAuxiliaryVelocity(i, j).y*ps)/m_pGridData->getScaleFactor(i, j).y;
				} else if (m_pCutCells && m_pCutCells->isCutCellAt(i, j - 1)) { //Use the last element on the face vector, since this cell will always be on "top"
					Scalar specialCellVelocity = m_pCutCells->getEdgeVector(dimensions_t(i , j), xAlignedEdge).back()->getAuxiliaryVelocity().y
													* m_pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge).back()->getRelativeFraction();
					Scalar temp = m_pGridData->getAuxiliaryVelocity(i, j + 1).y;
					dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1).y*pn - specialCellVelocity)/m_pGridData->getScaleFactor(i, j).y;
				} else {
					dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1).y*pn 
						- m_pGridData->getAuxiliaryVelocity(i, j).y*ps)/m_pGridData->getScaleFactor(i, j).y;
				}
			
			} else {
				dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j).x - m_pGridData->getAuxiliaryVelocity(i, j).x)/m_pGridData->getScaleFactor(i, j).x;
				dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1).y - m_pGridData->getAuxiliaryVelocity(i, j).y)/m_pGridData->getScaleFactor(i, j).y;
			}

			divergent = dx + dy;
		
			return divergent;

		}

		void CutCellSolver2D::divergenceFree(Scalar dt) {
			int i, j;
			Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
			Scalar dy = m_pGridData->getScaleFactor(0, 0).y;

			#pragma omp parallel for
			for (i = 1; i < m_dimensions.x - 1; i++) {
				for(j = 1; j < m_dimensions.y - 1; j++) {
					Vector2 velocity;
					if(m_pCutCells && m_pCutCells->getEdgeVector(dimensions_t(i, j), yAlignedEdge).size() > 0) {
						vector<Edge<Vector2> *> &currEdgeVec = m_pCutCells->getEdgeVector(dimensions_t(i, j), yAlignedEdge);
					
						for(int k = 0; k < currEdgeVec.size(); k++) {
							auto currEdge = currEdgeVec[k];

							if(currEdge->getType() == geometricEdge)
								continue;
							
							//Using ID to access the pressure vec
							Scalar p1 = m_cutCellsPressures[currEdge->getConnectedHalfFaces()[0]->getID()];
							Scalar p2;
							if(currEdge->getConnectedHalfFaces().size() == 2) {
								p2 = m_cutCellsPressures[currEdge->getConnectedHalfFaces()[1]->getID()];
								if(currEdge->getConnectedHalfFaces()[1]->getFace()->getGridCellLocation().x < currEdge->getConnectedHalfFaces()[0]->getFace()->getGridCellLocation().x) {
									swap(p1, p2);
								}
							} else {
								if(m_pCutCells->isCutCellAt(i, j)) {
									p2 = p1;
									p1 = m_pGridData->getPressure(i - 1, j);
								} else {
									p2 = m_pGridData->getPressure(i, j);
								}
							}

							Vector2 faceVelocity = currEdge->getAuxiliaryVelocity() - dt*(p2 - p1)/dx;
							faceVelocity.y = 0;
							velocity.x = faceVelocity.x;
							currEdge->setVelocity(faceVelocity);
						}
					} else {

						velocity.x = m_pGridData->getAuxiliaryVelocity(i, j).x - dt*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i - 1, j))/dx);
						if(m_pGrid->isSolidCell(i - 1, j))
							velocity.x = 0;
					}
					if(m_pCutCells && m_pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge).size() > 0) {
						vector<Edge<Vector2> *> &currEdgeVec =  m_pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge);
						for (int k = 0; k < currEdgeVec.size(); k++) {
							auto currEdge = currEdgeVec[k];
						
							if (currEdge->getType() == geometricEdge)
								continue;

							//Using ID to access the pressure vec
							Scalar p1 = m_cutCellsPressures[currEdge->getConnectedHalfFaces()[0]->getID()];
							Scalar p2;
							if (currEdge->getConnectedHalfFaces().size() == 2) {
								p2 = m_cutCellsPressures[currEdge->getConnectedHalfFaces()[1]->getID()];
								if (currEdge->getConnectedHalfFaces()[1]->getFace()->getGridCellLocation().y < currEdge->getConnectedHalfFaces()[0]->getFace()->getGridCellLocation().y) {
									swap(p1, p2);
								}
							}
							else {
								if (m_pCutCells->isCutCellAt(i, j)) {
									p2 = p1;
									p1 = m_pGridData->getPressure(i, j - 1);
								}
								else {
									p2 = m_pGridData->getPressure(i, j);
								}
							}

							Vector2 faceVelocity = currEdge->getAuxiliaryVelocity() - dt*(p2 - p1)/dx;
							faceVelocity.x = 0;
							velocity.y = faceVelocity.y;
							currEdge->setVelocity(faceVelocity);
						}
					} else {
						velocity.y = m_pGridData->getAuxiliaryVelocity(i, j).y - dt*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i, j - 1))/dy);
					
						if(m_pGrid->isSolidCell(i, j - 1))
							velocity.y = 0;
					} 

					m_pGridData->setVelocity(velocity, i, j);
				}
			}

			updateVorticity();

			if(m_pCutCells) {
				for (i = 1; i < m_dimensions.x - 1; i++) {
					for(j = 1; j < m_dimensions.y - 1; j++) {
						if(m_pCutCells->isCutCellAt(i, j))
							m_pGridData->setVelocity(Vector2(0, 0), i, j);
					}
				}
			}
		}
		#pragma endregion

	}
	
}