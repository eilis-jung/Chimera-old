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

#include "Physics/RaycastSolver2D.h"
#include "Physics/PhysicsCore.h"
#include <omp.h>

namespace Chimera {

	/************************************************************************/
	/* ctors                                                                */
	/************************************************************************/
	RaycastSolver2D::RaycastSolver2D(const FlowSolverParameters &params, StructuredGrid<Vector2> *pGrid, 
		const vector<Data::BoundaryCondition<Vector2> *> &boundaryConditions, const vector<RigidThinObject2D *> &thinObjectVec)
		: FlowSolver(params, pGrid), m_thinObjectVec(thinObjectVec), m_boundaryCells(pGrid->getDimensions()), 
		m_leftFacesVisibity(pGrid->getDimensions()), m_bottomFacesVisibility(pGrid->getDimensions()),
		m_leftCrossings(pGrid->getDimensions()), m_bottomCrossings(pGrid->getDimensions()) {
		m_pGrid = pGrid;
		m_pGridData = m_pGrid->getGridData2D();
		m_dimensions = m_pGrid->getDimensions();
		m_boundaryConditions = boundaryConditions;

		m_pPoissonMatrix = createPoissonMatrix();
		initializePoissonSolver();

		updateBoundaryCells();
		updatePoissonSolidWalls();

		/** Boundary conditions for initialization */
		enforceBoundaryConditions();

		initializeIntegrationParams();
		
		switch (m_params.getConvectionMethod()) {
			case CPU_S01SemiLagrangian:
				m_pSLIntegrator = new SemiLagrangianIntegrator<Array2D, Vector2>(m_pGridData->getVelocityArrayPtr(), m_pGridData->getAuxVelocityArrayPtr(), *m_pTrajectoryParams);
			break;
			case CPU_S02ModifiedMacCormack:
				m_pMCIntegrator = new MacCormackIntegrator<Array2D, Vector2>(m_pGridData->getVelocityArrayPtr(), m_pGridData->getAuxVelocityArrayPtr(), *m_pTrajectoryParams);
			break;
			case CPU_S01FLIP:
				m_pFLIP = new FLIPAdvection2D(m_params.getFlipParams(), m_pGridData, NULL, NULL);
			break;
		}

		Logger::get() << "[dx dy] = " << m_pGridData->getScaleFactor(0, 0).x << " " << m_pGridData->getScaleFactor(0, 0).y << endl;
		Logger::get() << "[dx/dy] = " << m_pGridData->getScaleFactor(0, 0).x / m_pGridData->getScaleFactor(0, 0).y << endl;
	}

	/************************************************************************/
	/* Boundary conditions	                                                */
	/************************************************************************/
	void RaycastSolver2D::enforceSolidWallsConditions() {
		switch (m_params.getSolidBoundaryType()) {
		case Solid_FreeSlip:
			//FreeSlipBC<Vector2>::enforceSolidWalls(dynamic_cast<QuadGrid *>(m_pGrid), m_solidVelocity);
			break;
		case Solid_NoSlip:
			//NoSlipBC<Vector2>::enforceSolidWalls(dynamic_cast<QuadGrid *>(m_pGrid), m_solidVelocity);
			break;

		case Solid_Interpolation:
			//BoundaryCondition<Vector2>::zeroSolidBoundaries(dynamic_cast<QuadGrid *>(m_pGrid));
			break;
		}
	}

	void RaycastSolver2D::applyForces(Scalar dt) {
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
		for (int k = 0; k < m_rotationalVelocities.size(); k++) {
			for (int i = 0; i < m_dimensions.x; i++) {
				for (int j = 0; j < m_dimensions.y; j++) {
					Vector2 velocity;
					Vector2 cellCenter(i*dx, (j + 0.5)*dx); //Staggered
					Vector2 radiusVec = cellCenter - m_rotationalVelocities[k].center;
					Scalar radius = radiusVec.length();
					if (radius > m_rotationalVelocities[k].minRadius && radius < m_rotationalVelocities[k].maxRadius) {
						if (m_rotationalVelocities[k].orientation) {
							velocity.x = -radiusVec.perpendicular().normalized().x*m_rotationalVelocities[k].strenght;
						}
						else {
							velocity.x = radiusVec.perpendicular().normalized().x*m_rotationalVelocities[k].strenght;
						}
						velocity.y = m_pGridData->getVelocity(i, j).y;
						m_pGridData->setVelocity(velocity, i, j);
						m_pGridData->setAuxiliaryVelocity(velocity, i, j);
					}
					cellCenter = Vector2((i + 0.5)*dx, j*dx);
					radiusVec = cellCenter - m_rotationalVelocities[k].center;
					radius = radiusVec.length();
					if (radius > m_rotationalVelocities[k].minRadius && radius < m_rotationalVelocities[k].maxRadius) {
						if (m_rotationalVelocities[k].orientation) {
							velocity.y = -radiusVec.perpendicular().normalized().y*m_rotationalVelocities[k].strenght;
						}
						else {
							velocity.y = radiusVec.perpendicular().normalized().y*m_rotationalVelocities[k].strenght;
						}
						velocity.x = m_pGridData->getVelocity(i, j).x;
						m_pGridData->setVelocity(velocity, i, j);
						m_pGridData->setAuxiliaryVelocity(velocity, i, j);
					}
				}
			}
		}
	}

	/************************************************************************/
	/* Initialization functions                                             */
	/************************************************************************/
	void RaycastSolver2D::initializeIntegrationParams() {
		m_pTrajectoryParams = new trajectoryIntegratorParams_t<Vector2>(&m_pGridData->getVelocityArray(),
																		&m_pGridData->getAuxVelocityArray(),
																		&m_pGridData->getTransformationMatrices(),
																		&m_pGridData->getScaleFactorsArray());

		if (m_params.getDiscretizationMethod() == finiteDifferenceMethod)
			m_pTrajectoryParams->transformVelocity = false;
		else
			m_pTrajectoryParams->transformVelocity = true;

		m_pTrajectoryParams->dt = Rendering::PhysicsCore::getInstance()->getParams()->timestep;
		m_pTrajectoryParams->integrationMethod = m_params.getIntegrationMethod();
		m_pTrajectoryParams->periodicDomain = m_pGrid->isPeriodic();
	}

	PoissonMatrix * RaycastSolver2D::createPoissonMatrix() {
		dimensions_t poissonMatrixDim = m_pGridData->getDimensions();
		if (m_params.getPressureSolverParams().getMethodCategory() == Krylov && !m_pGrid->isPeriodic()) {
			poissonMatrixDim.x += -2; poissonMatrixDim.y += -2;
		}
		else if (m_params.getPressureSolverParams().getMethodCategory() == Krylov && m_pGrid->isPeriodic()) { //X peridiocity
			poissonMatrixDim.y += -2;
		}

		PoissonMatrix *pMatrix = new PoissonMatrix(poissonMatrixDim);

		for (int i = 0; i < poissonMatrixDim.x; i++) {
			for (int j = 0; j < poissonMatrixDim.y; j++) {
				pMatrix->setRow(pMatrix->getRowIndex(i, j), -1, -1, 4, -1, -1);
			}
		}

		if (m_pGrid->isPeriodic()) {
			for (int j = 0; j < poissonMatrixDim.y; j++) {
				pMatrix->setPeriodicWestValue(pMatrix->getRowIndex(0, j), -1);
				pMatrix->setPeriodicEastValue(pMatrix->getRowIndex(poissonMatrixDim.x - 1, j), -1);
				pMatrix->setWestValue(pMatrix->getRowIndex(0, j), 0);
				pMatrix->setEastValue(pMatrix->getRowIndex(poissonMatrixDim.x - 1, j), 0);
			}
		}

		for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
			m_boundaryConditions[i]->updatePoissonMatrix(pMatrix);
		}
		BoundaryCondition<Vector2>::updateSolidWalls(pMatrix, Array2D<bool>(m_pGrid->getSolidMarkers(), m_dimensions), m_params.getPressureSolverParams().getMethodCategory() == Krylov);

		if (pMatrix->isSingular() && m_params.getPressureSolverParams().getMethodCategory() == Krylov)
			pMatrix->applyCorrection(1e-3);


		pMatrix->updateCudaData();

		return pMatrix;
	}

	/************************************************************************/
	/* Auxiliary Functions		                                            */
	/************************************************************************/
	/** Update */
	void RaycastSolver2D::updatePoissonBoundaryConditions() {
		/*RegularGridPMB::getInstance()->reinitializePoissonMatrix(m_pGridData, m_boundaryConditions, m_pPoissonMatrix);*/
	}

	void RaycastSolver2D::updatePoissonSolidWalls() {

		for (int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
			for (int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
				m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(i, j), -1, -1, 4, -1, -1);
			}
		}


		for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
			m_boundaryConditions[i]->updatePoissonMatrix(m_pPoissonMatrix);
		}

		
		BoundaryCondition<Vector2>::updateSolidWalls(m_pPoissonMatrix, Array2D<bool>(m_pGrid->getSolidMarkers(), m_dimensions), m_params.getPressureSolverParams().getMethodCategory() == Krylov);


		if (m_pPoissonMatrix->isSingular() && m_params.getPressureSolverParams().getMethodCategory() == Krylov)
			m_pPoissonMatrix->applyCorrection(1e-3);

		for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
			for (int j = 1; j < m_pGrid->getDimensions().y - 1; j++) {
				Scalar centralValue = 4.0f;
				int pressureI = i - 1, pressureJ = j - 1;
				bool updateMatrix = false;
				if (!m_leftFacesVisibity(i, j)) {
					centralValue -= 1;
					m_pPoissonMatrix->setWestValue(m_pPoissonMatrix->getRowIndex(pressureI, pressureJ), 0);
					//m_pPoissonMatrix->setEastValue(m_pPoissonMatrix->getRowIndex(pressureI - 1, pressureJ), 0);
					updateMatrix = true;
				}

				if (!m_leftFacesVisibity(i + 1, j)) {
					centralValue -= 1;
					m_pPoissonMatrix->setEastValue(m_pPoissonMatrix->getRowIndex(pressureI, pressureJ), 0);
					//m_pPoissonMatrix->setWestValue(m_pPoissonMatrix->getRowIndex(pressureI + 1, pressureJ), 0);
					updateMatrix = true;
				} 

				if (!m_bottomFacesVisibility(i, j)) {
					centralValue -= 1;
					m_pPoissonMatrix->setSouthValue(m_pPoissonMatrix->getRowIndex(pressureI, pressureJ), 0);
					//m_pPoissonMatrix->setNorthValue(m_pPoissonMatrix->getRowIndex(pressureI, pressureJ - 1), 0);
					updateMatrix = true;
				}

				if (!m_bottomFacesVisibility(i, j + 1)) {
					centralValue -= 1;
					m_pPoissonMatrix->setNorthValue(m_pPoissonMatrix->getRowIndex(pressureI, pressureJ), 0);
					//m_pPoissonMatrix->setSouthValue(m_pPoissonMatrix->getRowIndex(pressureI, pressureJ + 1), 0);
					updateMatrix = true;
				}

				if (centralValue == 0) { //All surrounded 
					centralValue = 1;
				}

				if (updateMatrix)
					m_pPoissonMatrix->setCentralValue(m_pPoissonMatrix->getRowIndex(pressureI, pressureJ), centralValue);
			}
		}

		m_pPoissonMatrix->updateCudaData();
	}

	/************************************************************************/
	/* Advection functions                                                  */
	/************************************************************************/

	void RaycastSolver2D::semiLagrangian(Scalar dt) {
		m_pSLIntegrator->integrateVelocityField();

		enforceSolidWallsConditions();
	}

	void RaycastSolver2D::modifiedMacCormack(Scalar dt) {
		m_pMCIntegrator->integrateVelocityField();
		enforceSolidWallsConditions();
	}

	/************************************************************************/
	/* Pseudo pressure step functions                                       */
	/************************************************************************/
	Scalar RaycastSolver2D::calculateFluxDivergent(int i, int j) {
		Scalar divergent = 0;

		Scalar x1, x0, y1, y0;
		if (m_boundaryCells(i, j)) {
			if (!m_leftFacesVisibity(i, j) && !m_leftFacesVisibity(i + 1, j) &&
				!m_bottomFacesVisibility(i, j) && !m_bottomFacesVisibility(i, j + 1)) {
				return 0;
			}
			if (!m_leftFacesVisibity(i, j)) {
				x0 = m_pGridData->getAuxiliaryVelocity(i, j).x;
			}
			else {
				x0 = m_pGridData->getAuxiliaryVelocity(i, j).x;
			}
			if (!m_leftFacesVisibity(i + 1, j)) {
				x1 = m_pGridData->getAuxiliaryVelocity(i + 1, j).x;
			}
			else {
				x1 = m_pGridData->getAuxiliaryVelocity(i + 1, j).x;
			}


			if (!m_bottomFacesVisibility(i, j)) {
				y0 = m_pGridData->getAuxiliaryVelocity(i, j).y;
			}
			else {
				y0 = m_pGridData->getAuxiliaryVelocity(i, j).y;
			}

			if (!m_bottomFacesVisibility(i, j + 1)) {
				y1 = m_pGridData->getAuxiliaryVelocity(i, j + 1).y;
			}
			else {
				y1 = m_pGridData->getAuxiliaryVelocity(i, j + 1).y;
			}
		}
		else {
			x0 = m_pGridData->getAuxiliaryVelocity(i, j).x;
			x1 = m_pGridData->getAuxiliaryVelocity(i + 1, j).x;
			y0 = m_pGridData->getAuxiliaryVelocity(i, j).y;
			y1 = m_pGridData->getAuxiliaryVelocity(i, j + 1).y;
		}

		Scalar dx = (x1 - x0) / m_pGridData->getScaleFactor(i, j).x;
		Scalar dy = (y1 - y0) / m_pGridData->getScaleFactor(i, j).y;
		
		divergent = dx + dy;

		return divergent;
	}

	Scalar RaycastSolver2D::calculateFinalDivergent(int i, int j) {
		Scalar divergent = 0;

		if (m_pGrid->isBoundaryCell(i, j) || m_pGrid->isSolidCell(i - 1, j) || m_pGrid->isSolidCell(i + 1, j)
			|| m_pGrid->isSolidCell(i, j - 1) || m_pGrid->isSolidCell(i, j + 1) || m_pGrid->isSolidCell(i, j))
			return divergent;

		Scalar dx = (m_pGridData->getVelocity(i + 1, j).x - m_pGridData->getVelocity(i, j).x);
		Scalar dy = (m_pGridData->getVelocity(i, j + 1).y - m_pGridData->getVelocity(i, j).y);

		divergent = dx + dy;
		return divergent;
	}

	/************************************************************************/
	/* Projection functions                                                 */
	/************************************************************************/
	void RaycastSolver2D::divergenceFree(Scalar dt) {
		int i, j;
	#pragma omp parallel for
		for (i = 1; i < m_dimensions.x - 1; i++) {
	#pragma omp parallel for private(j)
			for (j = 1; j < m_dimensions.y - 1; j++) {
				if (m_pGrid->isSolidCell(i, j) || m_pGrid->isBoundaryCell(i, j))
					continue;

				Vector2 velocity = m_pGridData->getVelocity(i, j);

				Scalar dx = m_pGridData->getScaleFactor(i, j).x;
				Scalar dy = m_pGridData->getScaleFactor(i, j).y;

				if (!m_pGrid->isSolidCell(i - 1, j) && m_leftFacesVisibity(i, j))
					velocity.x = m_pGridData->getAuxiliaryVelocity(i, j).x - dt*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i - 1, j)) / dx);

				if (!m_pGrid->isSolidCell(i, j - 1) && m_bottomFacesVisibility(i, j))
					velocity.y = m_pGridData->getAuxiliaryVelocity(i, j).y - dt*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i, j - 1)) / dy);

				m_pGridData->setVelocity(velocity, i, j);
			}
		}

		enforceSolidWallsConditions();

	}


	/************************************************************************/
	/* Functionalities                                                      */
	/************************************************************************/
	void RaycastSolver2D::updateBoundaryCells() {
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;

		/** Computing all crossings */
		for (unsigned int k = 0; k < m_thinObjectVec.size(); k++) {
			const vector<Vector2> &thinObjectPoints = m_thinObjectVec[k]->getLineMeshPtr()->getPoints();
			for (unsigned int i = 0; i < thinObjectPoints.size() - 1; i++) {
				Vector2 v0 = thinObjectPoints[i]; //Grid space 
				Vector2 v1 = thinObjectPoints[i + 1]; //Grid space 

				dimensions_t gridPoint;
				computeCrossing(v0, v1, i, verticalCrossing, dx, m_allCrossings);
				computeCrossing(v0, v1, i, horizontalCrossing, dx, m_allCrossings);
			}
		}
		/** Organize crossings on respective Arrays2D */
		for (int i = 0; i < m_allCrossings.size(); i++) {
			if (m_allCrossings[i].m_crossType == verticalCrossing) {
				m_leftCrossings(m_allCrossings[i].m_regularGridIndex) = m_allCrossings[i];
			} else if (m_allCrossings[i].m_crossType == horizontalCrossing) {
				m_bottomCrossings(m_allCrossings[i].m_regularGridIndex) = m_allCrossings[i];
			}
		}

		/** Resetting structures */
		for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
			for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
				m_leftFacesVisibity(i, j) = true;
				m_bottomFacesVisibility(i, j) = true;
				m_boundaryCells(i, j) = false;
			}
		}


	
		/*for (int i = 0; i < m_thinObjectVec.size(); i++) {
			LineMesh<Vector2> *pLineMesh = m_thinObjectVec[i]->getLineMeshPtr();
			for (int j = 0; j < pLineMesh->getPoints().size() - 1; j++) {
				Vector2 lineVec = pLineMesh->getPoints()[j + 1] - pLineMesh->getPoints()[j];
				lineVec.normalize();
				Scalar divisSize = 1 / float(m_numSubdivis);
				for (int k = 0; k < m_numSubdivis; k++) {
					Vector2 currVecPoint = pLineMesh->getPoints()[j] + lineVec*divisSize;
					m_boundaryCells(floor(currVecPoint.x / dx), floor(currVecPoint.y / dx)) = true;
				}
			}
		}*/

		/** Updating visibility*/
		for (int k = 0; k < m_thinObjectVec.size(); k++) {
			for (int i = 1; i < m_pGridData->getDimensions().x; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y; j++) {
					Vector2 centroidL((i - 0.5) * dx, (j + 0.5)*dx);
					Vector2 centroid((i + 0.5)*dx, (j + 0.5)*dx);
					Vector2 intersectionPoint;
					if (m_thinObjectVec[k]->getLineMeshPtr()->segmentIntersection(centroidL, centroid, intersectionPoint)) {
						Vector2 auxVelocity = m_pGridData->getAuxiliaryVelocity(i, j);
						Vector2 rotationAxis = intersectionPoint - m_thinObjectVec[k]->getLineMeshPtr()->getParams()->pointsCentroid;

						auxVelocity.x = -rotationAxis.y*m_thinObjectVec[k]->getLineMeshPtr()->getParams()->initialAngularVelocity;
						m_pGridData->setAuxiliaryVelocity(auxVelocity, i, j);

						auxVelocity = m_pGridData->getVelocity(i, j);
						auxVelocity.x = -rotationAxis.y*m_thinObjectVec[k]->getLineMeshPtr()->getParams()->initialAngularVelocity;
						m_pGridData->setVelocity(auxVelocity, i, j);
						m_leftFacesVisibity(i, j) = false;
					}
					Vector2 centroidB((i + 0.5)*dx, (j - 0.5)*dx);
					if (m_thinObjectVec[k]->getLineMeshPtr()->segmentIntersection(centroidB, centroid, intersectionPoint)) {
						Vector2 auxVelocity = m_pGridData->getAuxiliaryVelocity(i, j);
						Vector2 rotationAxis = intersectionPoint - m_thinObjectVec[k]->getLineMeshPtr()->getParams()->pointsCentroid;

						auxVelocity.y = rotationAxis.x*m_thinObjectVec[k]->getLineMeshPtr()->getParams()->initialAngularVelocity;
						m_pGridData->setAuxiliaryVelocity(auxVelocity, i, j);

						auxVelocity = m_pGridData->getVelocity(i, j);
						auxVelocity.y = rotationAxis.x*m_thinObjectVec[k]->getLineMeshPtr()->getParams()->initialAngularVelocity;
						m_pGridData->setVelocity(auxVelocity, i, j);
						m_bottomFacesVisibility(i, j) = false;
					}
				}
			}
		}

		/** Tagging boundary cells */
		for (int i = 1; i < m_pGridData->getDimensions().x; i++) {
			for (int j = 1; j < m_pGridData->getDimensions().y; j++) {
				if (!m_leftFacesVisibity(i, j)) {
					m_boundaryCells(i - 1, j) = true;
					m_boundaryCells(i, j) = true;
				}
				if (!m_bottomFacesVisibility(i, j)) {
					m_boundaryCells(i, j) = true;
					m_boundaryCells(i, j - 1) = true;
				}
			}
		}

	}

	/************************************************************************/
	/* Update functions                                                     */
	/************************************************************************/
	void RaycastSolver2D::update(Scalar dt) {
		m_numIterations++;
		m_params.m_totalSimulationTimer.start();

		/** Advection */
		enforceBoundaryConditions();

		if (PhysicsCore::getInstance()->getElapsedTime() < dt) {
			applyForces(dt);
			enforceBoundaryConditions();
			updateDivergents(dt);
			enforceBoundaryConditions();
			solvePressure();
			enforceBoundaryConditions();
			project(dt);
			if (m_params.getConvectionMethod() == CPU_S01FLIP) {
				m_pFLIP->backupVelocities();
				m_pFLIP->updateParticlesVelocities();
			}
		}

		

		m_params.m_advectionTimer.start();

		for (int i = 0; i < m_thinObjectVec.size(); i++) {
			m_thinObjectVec[i]->updateMesh(dt);
		}


		updateBoundaryCells();
		updatePoissonSolidWalls();

		if (!(m_params.getConvectionMethod() == CPU_S01FLIP)) {
			semiLagrangian(dt);
			//advect(dt);
		}
		else {
			m_pFLIP->updateParticlesPosition(dt);
			m_pFLIP->accumulateVelocitiesToGrid();
			enforceBoundaryConditions();
			enforceSolidWallsConditions();
			m_pFLIP->backupVelocities();
		}

		updateBoundaryCells();
		updatePoissonSolidWalls();


		m_params.m_advectionTimer.stop();
		m_advectionTime = m_params.m_advectionTimer.secondsElapsed();
		enforceBoundaryConditions();

		/** Solve pressure */
		m_params.m_solvePressureTimer.start();
		updateDivergents(dt);
		solvePressure();

		enforceBoundaryConditions();
		m_params.m_solvePressureTimer.stop();
		m_solvePressureTime = m_params.m_solvePressureTimer.secondsElapsed();

		/** Project velocity */
		m_params.m_projectionTimer.start();
		project(dt);
		m_params.m_projectionTimer.stop();
		m_projectionTime = m_params.m_projectionTimer.secondsElapsed();
		enforceBoundaryConditions();

		//advectDensityField(dt);

		//globalCoordinateSystemTransform(dt);

		//enforceBoundaryConditions();
		//enforceSolidWallsConditions();

		if ((m_params.getConvectionMethod() == CPU_S01FLIP)) {
			m_pFLIP->updateParticlesVelocities();
		}

		m_params.m_totalSimulationTimer.stop();
		m_totalSimulationTime = m_params.m_totalSimulationTimer.secondsElapsed();
	}
}