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

#include "Solvers/3D/CutVoxelSolver3D.h"

namespace Chimera {

	namespace Solvers {

		#pragma region Constructors
		CutVoxelSolver3D::CutVoxelSolver3D(const params_t &params, StructuredGrid<Vector3> *pGrid,
											const vector<BoundaryCondition<Vector3> *> &boundaryConditions /*= vector<BoundaryCondition<Vector3> *>()*/,
											const vector<PolygonalMesh<Vector3> *> &polygonMeshes /*= vector<PolygonalMesh<Vector3> *>()*/) 
											 : FlowSolver(params, pGrid), m_nodalBasedVelocities(pGrid->getDimensions()), m_auxNodalBasedVelocities(pGrid->getDimensions())  {
			m_pGrid = pGrid;
			m_dimensions = m_pGrid->getDimensions();
			m_boundaryConditions = boundaryConditions;
			m_pGridData = m_pGrid->getGridData3D();
			m_polyMeshesVec = polygonMeshes;
			m_pCutVoxels = nullptr;
			m_pAdvection = nullptr;
			m_pGridToParticlesTransfer = nullptr;
			m_pParticlesToGridTransfer = nullptr;
			
			m_pCutVoxels = initializeCutVoxels();

			/** Initialize cut-cells main and auxiliary velocities */
			m_pCutVoxelsVelocities3D = new CutVoxelsVelocities3D(m_pCutVoxels, m_params.solidBoundaryType);
			m_pAuxCutVoxelsVelocities3D = new CutVoxelsVelocities3D(m_pCutVoxels, m_params.solidBoundaryType);

			initializeInterpolants();

			m_pPoissonMatrix = createPoissonMatrix();
			if (m_pCutVoxels) {
				updatePoissonThinSolidWalls();
			}

			m_params.pPoissonSolverParams->numberOfSpecialCells = m_pCutVoxels->getNumberCutVoxels();

			/**Poisson solver initialization depends on correct handleThinBoundaries parameters */
			initializePoissonSolver();

			/** Boundary conditions for initialization */
			enforceBoundaryConditions();

			Logger::get() << "[dx dy dz] = " << m_pGridData->getScaleFactor(0, 0, 0).x << " " << m_pGridData->getScaleFactor(0, 0, 0).y << m_pGridData->getScaleFactor(0, 0, 0).z << endl;
			Logger::get() << "[dx/dy] = " << m_pGridData->getScaleFactor(0, 0, 0).x / m_pGridData->getScaleFactor(0, 0, 0).y << endl;

			m_pAdvection = initializeAdvectionClass();
			if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
				ParticleBasedAdvection<Vector3, Array3D> *pPBAdv = dynamic_cast<ParticleBasedAdvection<Vector3, Array3D> *>(m_pAdvection);
				pPBAdv->getParticleBasedIntegrator()->setCutVoxels(m_pCutVoxels);
				ParticlesToNodalGrid3D *pParticlesToNodal = dynamic_cast<ParticlesToNodalGrid3D *>(pPBAdv->getParticlesToGrid());

				pParticlesToNodal->setCutVoxelsVelocities(m_pCutVoxelsVelocities3D);
				pParticlesToNodal->setCutVoxels(m_pCutVoxels);
			}

			((ConjugateGradient *)m_pPoissonSolver)->reinitializePreconditioners();
		}

		#pragma endregion

		#pragma region UpdateFunctions
		//Updates the current flow solver with the given time step
		void CutVoxelSolver3D::update(Scalar dt) {
			MeanValueInterpolant3D<Vector3> *pMeanValueInterpolant = dynamic_cast<MeanValueInterpolant3D<Vector3> *>(m_pVelocityInterpolant);
			MeanValueInterpolant3D<Vector3> *pAuxMeanValueInterpolant = dynamic_cast<MeanValueInterpolant3D<Vector3> *>(m_pAuxVelocityInterpolant);
			m_numIterations++;

			m_totalSimulationTimer.start();

			/** Advection */
			enforceBoundaryConditions();

			if (PhysicsCore<Vector2>::getInstance()->getElapsedTime() < dt) {

				/** Update rigid objects velocities into cut-cells */
				//for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
				//	m_rigidObjectsVec[i]->updateCutEdgesVelocities(0, m_pPlanarMesh2D->getGridSpacing(), true);
				//}

				applyForces(dt);
				enforceBoundaryConditions();
				if (m_pCutVoxels)
					updateCutCellsDivergence(dt);
				updateDivergents(dt);
				enforceBoundaryConditions();
				solvePressure();
				enforceBoundaryConditions();
				project(dt);
				enforceBoundaryConditions();

				/** Update rigid objects velocities into cut-cells */
				//for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
				//	m_rigidObjectsVec[i]->updateCutEdgesVelocities(0, m_pPlanarMesh2D->getGridSpacing());
				//}

				if (pMeanValueInterpolant) {
					pMeanValueInterpolant->updateNodalVelocities(m_pGridData->getVelocityArray(), m_nodalBasedVelocities);
				}

				if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
					ParticleBasedAdvection<Vector3, Array3D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector3, Array3D> *>(m_pAdvection);
					pParticleBasedAdv->getParticlesSampler()->interpolateVelocities(m_pVelocityInterpolant, pParticleBasedAdv->getParticlesData());
				}
				
				//return;
			}
			
			enforceBoundaryConditions();
			if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
				//ParticleBasedAdvection<Vector3, Array3D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector3, Array3D> *>(m_pAdvection);
				/*CubicStreamfunctionInterpolant2D<Vector2> *pCInterpolant = dynamic_cast<CubicStreamfunctionInterpolant2D<Vector2> *>(pParticleBasedAdv->getParticleBasedIntegrator()->getInterpolant());
				if (pCInterpolant)
					pCInterpolant->computeStreamfunctions();
				else {
					BilinearStreamfunctionInterpolant2D<Vector2> *pSInterpolant = dynamic_cast<BilinearStreamfunctionInterpolant2D<Vector2> *>(pParticleBasedAdv->getParticleBasedIntegrator()->getInterpolant());
					if (pSInterpolant)
						pSInterpolant->computeStreamfunctions();
				}*/
			}


			m_advectionTimer.start();

			//for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
			//	m_rigidObjectsVec[i]->updateCutEdgesVelocities(0, m_pPlanarMesh2D->getGridSpacing());
			//}

			if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
				flipAdvection(dt);
				pAuxMeanValueInterpolant->updateNodalVelocities(m_pGridData->getAuxVelocityArray(), m_auxNodalBasedVelocities, true);
			}

			m_advectionTimer.stop();
			m_advectionTime = m_advectionTimer.secondsElapsed();
			enforceBoundaryConditions();

			/** Solve pressure */
			m_solvePressureTimer.start();
			if (m_pCutVoxels)
				updateCutCellsDivergence(dt);
			updateDivergents(dt);
			solvePressure();

			enforceBoundaryConditions();
			m_solvePressureTimer.stop();
			m_solvePressureTime = m_solvePressureTimer.secondsElapsed();

			/** Project velocity */
			m_projectionTimer.start();
			project(dt);
			m_projectionTimer.stop();
			m_projectionTime = m_projectionTimer.secondsElapsed();
			enforceBoundaryConditions();

			//for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
			//	m_rigidObjectsVec[i]->updateCutEdgesVelocities(1, m_pPlanarMesh2D->getGridSpacing());
			//}

			pMeanValueInterpolant->updateNodalVelocities(m_pGridData->getVelocityArray(), m_nodalBasedVelocities);

			//advectDensityField(dt);

			enforceBoundaryConditions();

			m_pAdvection->postProjectionUpdate(dt);
			m_totalSimulationTimer.stop();
			m_totalSimulationTime = m_totalSimulationTimer.secondsElapsed();

		}
		
		//An implementation of the cutCell pressure projection solve for static geometry
		void CutVoxelSolver3D::updatePoissonThinSolidWalls() {
			//Double check the initial offset number
			int initialOffset = 0;
			if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
				initialOffset = (m_pGrid->getDimensions().x - 2)*(m_pGrid->getDimensions().y - 2)*(m_pGrid->getDimensions().z - 2);
			}
			else {
				initialOffset = m_pGrid->getDimensions().x*m_pGrid->getDimensions().y*m_pGrid->getDimensions().z;
			}

			//Regular cells that are now special cells are treated as solid cells 
			if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
				for (int k = 2; k < m_pGrid->getDimensions().z - 2; k++) {
					for (int j = 2; j < m_pGrid->getDimensions().y - 2; ++j) {
						for (int i = 2; i < m_pGrid->getDimensions().x - 2; ++i) {
							if (m_pCutVoxels->isCutVoxel(i, j, k)) {
								m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(i - 1, j - 1, k - 1), 0, 0, 0, 1, 0, 0, 0);
							}
							else {
								int currRowIndex = m_pPoissonMatrix->getRowIndex(i - 1, j - 1, k - 1);

								if (m_pCutVoxels->isCutVoxel(i + 1, j, k)) {
									m_pPoissonMatrix->setWestValue(currRowIndex, 0);
								}
								if (m_pCutVoxels->isCutVoxel(i - 1, j, k)) {
									m_pPoissonMatrix->setEastValue(currRowIndex, 0);
								}
								if (m_pCutVoxels->isCutVoxel(i, j + 1, k)) {
									m_pPoissonMatrix->setNorthValue(currRowIndex, 0);
								}
								if (m_pCutVoxels->isCutVoxel(i, j - 1, k)) {
									m_pPoissonMatrix->setSouthValue(currRowIndex, 0);
								}
								if (m_pCutVoxels->isCutVoxel(i, j, k + 1)) {
									m_pPoissonMatrix->setFrontValue(currRowIndex, 0);
								}
								if (m_pCutVoxels->isCutVoxel(i, j, k - 1)) {
									m_pPoissonMatrix->setBackValue(currRowIndex, 0);
								}
							}
						}
					}
				}
			}

			m_pPoissonMatrix->copyDIAtoCOO();
			int numEntries = m_pPoissonMatrix->getNumberOfEntriesCOO();
			m_pPoissonMatrix->resizeToFitCOO();

			int matrixInternalId = numEntries;

			for (int i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {
				int row = initialOffset + i; //compute the matrix index. after the last regular cell.
				Scalar pc = 0;
				HalfVolume<Vector3> &currVoxel = m_pCutVoxels->getCutVoxel(i);
				uint cutVoxelID = currVoxel.getID();

				for (uint j = 0; j < currVoxel.getHalfFaces().size(); j++) {
					Face<Vector3> *pCurrFace = currVoxel.getHalfFaces()[j]->getFace();
					if (pCurrFace->getConnectedHalfVolumes().size() > 2) {
						throw(exception("Invalid number of faces connected to an edge"));
					}
					if (pCurrFace->getLocation() != geometricFace) {
						uint otherPressure;
						if (pCurrFace->getConnectedHalfVolumes().size() == 1) { //Neighbor to a regular grid face
							otherPressure = getRowIndex(currVoxel.getVolume()->getGridCellLocation(), currVoxel.getHalfFaces()[j]->getLocation());
							if (otherPressure > initialOffset)
								throw(exception("CutVoxelSolver3D updatePoissonThinSolidWalls invalid otherPressure connection to regular grid voxels"));
						}
						else {
							otherPressure = pCurrFace->getConnectedHalfVolumes()[0]->getID() == cutVoxelID ? 
											pCurrFace->getConnectedHalfVolumes()[1]->getID() : 
											pCurrFace->getConnectedHalfVolumes()[0]->getID();
							if (otherPressure > m_pCutVoxels->getNumberCutVoxels())
								throw(exception("CutVoxelSolver3D updatePoissonThinSolidWalls invalid otherPressure connection to cut-voxels"));
							otherPressure += initialOffset;
						}

						DoubleScalar pressureCoefficient = pCurrFace->getRelativeFraction();
						pc += pressureCoefficient;
						m_pPoissonMatrix->setValue(matrixInternalId++, row, otherPressure, -pressureCoefficient);
						if (pCurrFace->getConnectedHalfVolumes().size() == 1) //If we have a regular cell neighbor
							m_pPoissonMatrix->setValue(matrixInternalId++, otherPressure, row, -pressureCoefficient); //Guarantees matrix symmetry
					}
				}

				m_pPoissonMatrix->setValue(matrixInternalId++, row, row, pc);
			}


			if (m_pPoissonMatrix->isSingularCOO() && m_params.pPoissonSolverParams->solverCategory == Krylov)
				m_pPoissonMatrix->applyCorrection(1e-3);

			m_pPoissonMatrix->copyCOOtoHyb();
		}


		void CutVoxelSolver3D::updateCutCellsDivergence(Scalar dt) {
			Scalar dx = m_pGridData->getGridSpacing();
			for (uint i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {
				DoubleScalar divergent = 0;
				auto halfFaces = m_pCutVoxels->getCutVoxel(i).getHalfFaces();
				for (uint j = 0; j < halfFaces.size(); j++) {
					auto pFace = halfFaces[j]->getFace();
					divergent += halfFaces[j]->getNormal().dot(pFace->getAuxiliaryVelocity())*pFace->getRelativeFraction()*dx;
				}
				m_cutCellsDivergents[i] = -divergent / dt;
			}
		}
		#pragma endregion

		#pragma region InitializationFunctions
		CutVoxels3D<Vector3> * CutVoxelSolver3D::initializeCutVoxels() {
			if (m_polyMeshesVec.size() > 0) {
				m_cutCellGenerationTimer.start();
				CutVoxels3D<Vector3> *pCutVoxels = new CutVoxels3D<Vector3>(m_polyMeshesVec, m_pGridData->getGridSpacing(), m_pGrid->getDimensions());
				m_cutCellsPressures.resize(pCutVoxels->getNumberCutVoxels(), 0.0);
				m_cutCellsDivergents.resize(pCutVoxels->getNumberCutVoxels(), 0.0);
				m_cutCellGenerationTimer.stop();
				
				m_params.pPoissonSolverParams->solveThinBoundaries = true;
				m_params.pPoissonSolverParams->pSpecialDivergents = &m_cutCellsDivergents;
				m_params.pPoissonSolverParams->pSpecialPressures = &m_cutCellsPressures;
				m_params.pPoissonSolverParams->numberOfSpecialCells = m_cutCellsPressures.size();

				return pCutVoxels;
			}
			return nullptr;
		}

		PoissonMatrix * CutVoxelSolver3D::createPoissonMatrix() {
			dimensions_t poissonMatrixDim = m_pGridData->getDimensions();
			if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
				poissonMatrixDim.x += -2; poissonMatrixDim.y += -2; poissonMatrixDim.z += -2;
			}

			PoissonMatrix *pMatrix = NULL;
			if (m_params.pPoissonSolverParams->solverMethod == EigenCG ||
				m_params.pPoissonSolverParams->solverMethod == CPU_CG) {
				pMatrix = new PoissonMatrix(poissonMatrixDim, m_pCutVoxels->getNumberCutVoxels(), false);
			}
			else {
				pMatrix = new PoissonMatrix(poissonMatrixDim, m_pCutVoxels->getNumberCutVoxels(), true);
			}

			for (int i = 0; i < poissonMatrixDim.x; i++) {
				for (int j = 0; j < poissonMatrixDim.y; j++) {
					for (int k = 0; k < poissonMatrixDim.z; k++) {
						pMatrix->setRow(pMatrix->getRowIndex(i, j, k), -1, -1, -1, 6, -1, -1, -1);
					}
				}
			}

			for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
				m_boundaryConditions[i]->updatePoissonMatrix(pMatrix);
			}

			//BoundaryCondition<Vector3>::updateSolidWalls(pMatrix, Array3D<bool>(m_pGrid->getSolidMarkers(), m_dimensions), m_params.getPressureSolverParams().getMethodCategory() == Krylov);

			if (pMatrix->isSingular() && m_params.pPoissonSolverParams->solverCategory == Krylov)
				pMatrix->applyCorrection(1e-3);


			if (m_params.pPoissonSolverParams->solverMethod != EigenCG &&
				m_params.pPoissonSolverParams->solverMethod != CPU_CG)
				pMatrix->updateCudaData();

			return pMatrix;
		}

		void CutVoxelSolver3D::initializeInterpolants() {
			Scalar dx = m_pGridData->getGridSpacing();

			m_pVelocityInterpolant = new MeanValueInterpolant3D<Vector3>(m_nodalBasedVelocities, m_pCutVoxels, m_pCutVoxelsVelocities3D, dx);
			m_pAuxVelocityInterpolant = new MeanValueInterpolant3D<Vector3>(m_auxNodalBasedVelocities, m_pCutVoxels, m_pAuxCutVoxelsVelocities3D, dx, true);
			
			m_pVelocityInterpolant->setSiblingInterpolant(m_pAuxVelocityInterpolant);

			//m_pScalarInterpolant = new BilinearStaggeredInterpolant2D<Scalar>(*m_pGridData->getDensityBuffer().getBufferArray1(), dx);
		}

		GridToParticles<Vector3, Array3D> * CutVoxelSolver3D::createGridToParticles() {
			Scalar dx = m_pGridData->getGridSpacing();
	
			return new GridToParticlesFLIP3D(m_pVelocityInterpolant, 0.00f);
		}

		ParticlesToGrid<Vector3, Array3D> * CutVoxelSolver3D::createParticlesToGrid() {
			Scalar dx = m_pGridData->getGridSpacing();

			TransferKernel<Vector3> *pKernel = new SPHKernel<Vector3>(m_pGridData, dx * 2);

			ParticlesToNodalGrid3D *pParticlesToNodal = new ParticlesToNodalGrid3D(m_pGridData->getDimensions(), pKernel);
			pParticlesToNodal->setCutVoxelsVelocities(m_pCutVoxelsVelocities3D);
			pParticlesToNodal->setCutVoxels(m_pCutVoxels);

			return pParticlesToNodal;
		}
		#pragma endregion

		#pragma region PressureProjection
		Scalar CutVoxelSolver3D::calculateFluxDivergent(int i, int j, int k) {
			Scalar divergent = 0;

			int row = 0;
			if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
				row = m_pPoissonMatrix->getRowIndex(i - 1, j - 1, k - 1);
			}
			else {
				row = m_pPoissonMatrix->getRowIndex(i, j, k);
			}

			Scalar dx, dy, dz = 0;
			Scalar gridSpacing = m_pGridData->getGridSpacing();
			if (m_pCutVoxels && m_pCutVoxels->isCutVoxel(i, j, k)) {
				return 0;
			} else if (i > 1 && i < m_dimensions.x - 1 && j > 1 && j < m_dimensions.y - 1 && k > 1 && k < m_dimensions.z - 1) {
				/** Calculating X-aligned velocities and x-aligned derivative */
				Scalar prevVelX = m_pGridData->getAuxiliaryVelocity(i, j, k).x;
				Scalar nextVelX = m_pGridData->getAuxiliaryVelocity(i + 1, j, k).x;
				if (m_pCutVoxels && m_pCutVoxels->isCutVoxel(i + 1, j, k)) {
					nextVelX =  m_pCutVoxels->getFaceVector(dimensions_t(i + 1, j, k), YZFace)[0]->getAuxiliaryVelocity().x
								* m_pCutVoxels->getFaceVector(dimensions_t(i + 1, j, k), YZFace)[0]->getRelativeFraction();
				}
				if (m_pCutVoxels && m_pCutVoxels->isCutVoxel(i - 1, j, k)) { //Use the last element on the face vector, since this cell will always be on "top"
					prevVelX = m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), YZFace).back()->getAuxiliaryVelocity().x
							   * m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), YZFace).back()->getRelativeFraction();
				}
				dx = (nextVelX - prevVelX) / gridSpacing;
				
				/** Calculating Y-aligned velocities and y-aligned derivative */
				Scalar prevVelY = m_pGridData->getAuxiliaryVelocity(i, j, k).y;
				Scalar nextVelY = m_pGridData->getAuxiliaryVelocity(i, j + 1, k).y;
				if (m_pCutVoxels && m_pCutVoxels->isCutVoxel(i, j + 1, k)) {
					nextVelY = m_pCutVoxels->getFaceVector(dimensions_t(i, j + 1, k), XZFace)[0]->getAuxiliaryVelocity().y
							   * m_pCutVoxels->getFaceVector(dimensions_t(i, j + 1, k), XZFace)[0]->getRelativeFraction();
				}
				if (m_pCutVoxels && m_pCutVoxels->isCutVoxel(i, j - 1, k)) { //Use the last element on the face vector, since this cell will always be on "top"
					prevVelY = m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), XZFace).back()->getAuxiliaryVelocity().y
							   * m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), XZFace).back()->getRelativeFraction();
				}
				dy = (nextVelY - prevVelY) / gridSpacing;

				/** Calculating Z-aligned velocities and x-aligned derivative */
				Scalar prevVelZ = m_pGridData->getAuxiliaryVelocity(i, j, k).z;
				Scalar nextVelZ = m_pGridData->getAuxiliaryVelocity(i, j, k + 1).z;
				if (m_pCutVoxels && m_pCutVoxels->isCutVoxel(i, j, k + 1)) {
					nextVelZ = m_pCutVoxels->getFaceVector(dimensions_t(i, j, k + 1), XYFace)[0]->getAuxiliaryVelocity().z
							   * m_pCutVoxels->getFaceVector(dimensions_t(i, j, k + 1), XYFace)[0]->getRelativeFraction();
				}
				if (m_pCutVoxels && m_pCutVoxels->isCutVoxel(i, j, k - 1)) { //Use the last element on the face vector, since this cell will always be on "top"
					prevVelZ = m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), XYFace).back()->getAuxiliaryVelocity().z
							   * m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), XYFace).back()->getRelativeFraction();
				}
				dz = (nextVelZ - prevVelZ) / gridSpacing;

			}
			else {
				dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j, k).x - m_pGridData->getAuxiliaryVelocity(i, j, k).x) / gridSpacing;
				dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1, k).y - m_pGridData->getAuxiliaryVelocity(i, j, k).y) / gridSpacing;
				dz = (m_pGridData->getAuxiliaryVelocity(i, j, k + 1).z - m_pGridData->getAuxiliaryVelocity(i, j, k).z) / gridSpacing;
			}

			divergent = dx + dy + dz;

			return divergent;
		}


		void CutVoxelSolver3D::divergenceFree(Scalar dt) {
			int i, j, k;
			Scalar dx = m_pGridData->getGridSpacing();

			#pragma omp parallel for
			for (i = 1; i < m_dimensions.x - 1; i++) {
				for(j = 1; j < m_dimensions.y - 1; j++) {
					for (k = 1; k < m_dimensions.z - 1; k++) {
						Vector3 velocity;
						
						//X-component of the velocity first
						if (m_pCutVoxels && m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), YZFace).size() > 0) {
							vector<Face<Vector3> *> &faces = m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), YZFace);
							
							for (int l = 0; l < faces.size(); l++) {
								if (faces[l]->getLocation() == geometricFace)
									continue;

								projectCutCellVelocity(faces[l], dimensions_t(i, j, k), xComponent, dt);
							}
						}
						else {
							velocity.x = m_pGridData->getAuxiliaryVelocity(i, j, k).x - dt*((m_pGridData->getPressure(i, j, k) - m_pGridData->getPressure(i - 1, j, k)) / dx);
						}

						//Y-component of the velocity 
						if (m_pCutVoxels && m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), XZFace).size() > 0) {
							vector<Face<Vector3> *> &faces = m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), XZFace);

							for (int l = 0; l < faces.size(); l++) {
								if (faces[l]->getLocation() == geometricFace)
									continue;

								projectCutCellVelocity(faces[l], dimensions_t(i, j, k), yComponent, dt);
							}
						}
						else {
							velocity.y = m_pGridData->getAuxiliaryVelocity(i, j, k).y - dt*((m_pGridData->getPressure(i, j, k) - m_pGridData->getPressure(i, j - 1, k)) / dx);
						}

						//Z-component of the velocity 
						if (m_pCutVoxels && m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), XYFace).size() > 0) {
							vector<Face<Vector3> *> &faces = m_pCutVoxels->getFaceVector(dimensions_t(i, j, k), XYFace);

							for (int l = 0; l < faces.size(); l++) {
								if (faces[l]->getLocation() == geometricFace)
									continue;

								projectCutCellVelocity(faces[l], dimensions_t(i, j, k), zComponent, dt);
							}
						}
						else {
							velocity.z = m_pGridData->getAuxiliaryVelocity(i, j, k).z - dt*((m_pGridData->getPressure(i, j, k) - m_pGridData->getPressure(i, j, k - 1)) / dx);
						}

						m_pGridData->setVelocity(velocity, i, j, k);
					}
				}
			}

			updateVorticity();

			Vector3 zeroVector(0, 0, 0);
			if(m_pCutVoxels) {
				for (i = 1; i < m_dimensions.x - 1; i++) {
					for(j = 1; j < m_dimensions.y - 1; j++) {
						for (int k = 1; k < m_dimensions.z - 1; k++) {
							if (m_pCutVoxels->isCutVoxel(i, j, k)) {
								m_pGridData->setVelocity(zeroVector, i, j, k);
							}
						}
					}
				}
			}
		}

		Scalar CutVoxelSolver3D::projectCutCellVelocity(Face<Vector3> *pFace, const dimensions_t &voxelLocation, velocityComponent_t velocityComponent, Scalar dt) {
			if (pFace->getLocation() == geometricFace)
				return 0;

			//Using ID to access the pressure vec
			Scalar p1 = m_cutCellsPressures[pFace->getConnectedHalfVolumes()[0]->getID()];
			Scalar p2;
			if (pFace->getConnectedHalfVolumes().size() == 2) {
				p2 = m_cutCellsPressures[pFace->getConnectedHalfVolumes()[1]->getID()];
				if (pFace->getConnectedHalfVolumes()[1]->getVolume()->getGridCellLocation()[velocityComponent] < pFace->getConnectedHalfVolumes()[0]->getVolume()->getGridCellLocation()[velocityComponent]) {
					swap(p1, p2);
				}
			} else { //Size == 1
				if (m_pCutVoxels->isCutVoxel(voxelLocation)) {
					p2 = p1;
					if (velocityComponent == xComponent) {
						p1 = m_pGridData->getPressure(voxelLocation.x - 1, voxelLocation.y, voxelLocation.z);
					}
					else if (velocityComponent == yComponent) {
						p1 = m_pGridData->getPressure(voxelLocation.x, voxelLocation.y - 1, voxelLocation.z);
					}
					else if (velocityComponent == zComponent) {
						p1 = m_pGridData->getPressure(voxelLocation.x, voxelLocation.y, voxelLocation.z - 1);
					}
					else {
						throw(exception("CutVoxelSolver3D: projectCutCellVelocity invalid velocity component"));
					}
					
				}
				else {
					p2 = m_pGridData->getPressure(voxelLocation.x, voxelLocation.y, voxelLocation.z);
				}
			}

			Scalar dx = m_pGridData->getGridSpacing();
			Vector3 faceVelocity(0, 0, 0);
			faceVelocity[velocityComponent] = pFace->getAuxiliaryVelocity()[velocityComponent] - dt*(p2 - p1) / dx;
			pFace->setVelocity(faceVelocity);
		}
		#pragma endregion

		#pragma region Advection

		void CutVoxelSolver3D::flipAdvection(Scalar dt) {
			//m_pFLIP->setThinObjectAcceleration(-(nextThinObjectVelocity - currThinObjectVelocity)/(dt));
			
			ParticleBasedAdvection<Vector3, Array3D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector3, Array3D> *>(m_pAdvection);
			pParticleBasedAdv->updatePositions(dt);

			//for(int i = 0; i < m_rigidObjectsVec.size(); i++) {
			//	m_rigidObjectsVec[i]->update(dt);
			//}
		
			//bool reinitializeSpecialCells = false;
			//for (int i = 0; i < m_lineMeshes.size(); i++) {
			//	if (m_lineMeshes[i]->hasUpdated()) {
			//		reinitializeSpecialCells = true;
			//	}
			//}
			//if(reinitializeSpecialCells) {
			//	reinitializeThinBounds();

			//	//Update velocities after reinitialization, since those are used on the update grid attributes
			//	for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
			//		m_rigidObjectsVec[i]->updateCutEdgesVelocities(1, m_pPlanarMesh2D->getGridSpacing(), true);
			//	}
			//	for (int i = 0; i < m_lineMeshes.size(); i++) {
			//		m_lineMeshes[i]->setHasUpdated(false);
			//	}
			//}

			pParticleBasedAdv->updateGridAttributes();
			enforceBoundaryConditions();
		}
		#pragma endregion
	}
}