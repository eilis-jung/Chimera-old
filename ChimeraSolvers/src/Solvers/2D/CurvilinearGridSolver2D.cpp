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

#include "Physics/CurvilinearGridSolver2D.h"
#include "Physics/PhysicsCore.h"
#include <omp.h>

namespace Chimera {

	/************************************************************************/
	/* ctors                                                                */
	/************************************************************************/
	CurvilinearGridSolver2D::CurvilinearGridSolver2D(const FlowSolverParameters &params, StructuredGrid<Vector2> *pGrid, const vector<Data::BoundaryCondition<Vector2> *> &boundaryConditions)
		: FlowSolver(params, pGrid) {

		m_pGrid = pGrid;
		m_pGridData = m_pGrid->getGridData2D();
		m_dimensions = m_pGridData->getDimensions();
		m_boundaryConditions = boundaryConditions;

		m_pPoissonMatrix = createPoissonMatrix();
		initializePoissonSolver();

		/** Boundary conditions for initialization */
		enforceBoundaryConditions();

		initializeIntegrationParams();
		m_pSLIntegrator = new SemiLagrangianIntegrator<Array2D, Vector2>(m_pGridData->getVelocityArrayPtr(), m_pGridData->getAuxVelocityArrayPtr(), *m_pTrajectoryParams);
		m_pMCIntegrator = new MacCormackIntegrator<Array2D, Vector2>(m_pGridData->getVelocityArrayPtr(), m_pGridData->getAuxVelocityArrayPtr(), *m_pTrajectoryParams);
	}

	/************************************************************************/
	/* Initialization functions                                             */
	/************************************************************************/
	void CurvilinearGridSolver2D::initializeIntegrationParams() {
		m_pTrajectoryParams = new trajectoryIntegratorParams_t<Vector2>(	&m_pGridData->getVelocityArray(),
			&m_pGridData->getAuxVelocityArray(),
			&m_pGridData->getTransformationMatrices(),
			&m_pGridData->getScaleFactorsArray());
		if(m_params.getDiscretizationMethod() == finiteDifferenceMethod)
			m_pTrajectoryParams->transformVelocity = false;
		else
			m_pTrajectoryParams->transformVelocity = true;

		m_pTrajectoryParams->dt = Rendering::PhysicsCore::getInstance()->getParams()->timestep;
		m_pTrajectoryParams->integrationMethod = m_params.getIntegrationMethod();
		m_pTrajectoryParams->periodicDomain = m_pGrid->isPeriodic();
	}

	/************************************************************************/
	/* Poisson matrix creation                                              */
	/************************************************************************/
	PoissonMatrix * CurvilinearGridSolver2D::createPoissonMatrix() {
		dimensions_t poissonMatrixDim = m_pGridData->getDimensions();
		if(m_params.getPressureSolverParams().getMethodCategory() == Krylov) {
			if(!m_pGrid->isPeriodic()) {
				poissonMatrixDim.x += -2; poissonMatrixDim.y += -2;
			} else { //X peridiocity
				poissonMatrixDim.y += -2;
			}
		}

		PoissonMatrix *pMatrix = new PoissonMatrix(poissonMatrixDim, true, m_pGrid->isPeriodic());
		m_pPoissonMatrix = pMatrix;

		for(int i = 0; i < poissonMatrixDim.x; i++) {
			for(int j = 0; j < poissonMatrixDim.y; j++) {
				Scalar pn, pw, pc, pe, ps;
				if(!m_pGrid->isPeriodic() && i == 0) {
					pw = 0;
				} else {
					pw = calculateWestValue(i, j);
				}
				if(!m_pGrid->isPeriodic() && i == poissonMatrixDim.x - 1) {
					pe = 0;
				} else {
					pe = calculateEastValue(i, j);
				}
				if(j == 0)
					ps = 0;
				else
					ps = calculateSouthValue(i, j);

				if(j == poissonMatrixDim.y - 1)
					pn = 0;
				else
					pn = calculateNorthValue(i, j);
		
				pc = pw + pe + pn + ps;

				if(m_pGrid->isPeriodic()) {
					if(i == 0) {
						pMatrix->setPeriodicWestValue(pMatrix->getRowIndex(i, j), -pw);
						pw = 0;
					} else {
						pMatrix->setPeriodicWestValue(pMatrix->getRowIndex(i, j), 0);
					}

					if(i == poissonMatrixDim.x - 1) {
						pMatrix->setPeriodicEastValue(pMatrix->getRowIndex(i, j), -pe);
						pe = 0;
					} else {
						pMatrix->setPeriodicEastValue(pMatrix->getRowIndex(i, j), 0);
					}
				}

				pMatrix->setRow(pMatrix->getRowIndex(i, j), -pn, -pw, pc, -pe, -ps);
			}
		}

		for(unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
			m_boundaryConditions[i]->updatePoissonMatrix(pMatrix);
		}

		if(pMatrix->isSingular() && m_params.getPressureSolverParams().getMethodCategory() == Krylov)
			pMatrix->applyCorrection(1e-3);

		pMatrix->updateCudaData();

		return pMatrix;
	}

	void CurvilinearGridSolver2D::padValues(int &i, int &j) {
		if(m_params.getPressureSolverParams().getMethodCategory() == Krylov) {
			if(!m_pGrid->isPeriodic())
				i = i + 1;
			j = j + 1;
		}
	}

	Scalar CurvilinearGridSolver2D::calculateWestValue(int i, int j) {
		padValues(i, j);
		int prevI = roundClamp(i - 1, 0, m_pPoissonMatrix->getDimensions().x);

		Vector2 pDistVec = m_pGridData->getCenterPoint(i, j) - m_pGridData->getCenterPoint(prevI, j);
		Scalar pDist = pDistVec.length();
		pDistVec.normalize();

		Scalar angularContrib = abs(pDistVec.dot(-m_pGridData->getEtaBaseNormal(i, j)));
		Scalar wValue = m_pGridData->getScaleFactor(i, j).y*angularContrib/pDist;
		return wValue;
	}

	Scalar CurvilinearGridSolver2D::calculateEastValue(int i, int j) {
		padValues(i, j);

		int nextI = roundClamp(i + 1, 0, m_pPoissonMatrix->getDimensions().x);

		Vector2 pDistVec = m_pGridData->getCenterPoint(nextI, j) - m_pGridData->getCenterPoint(i, j);
		Scalar pDist = pDistVec.length();
		pDistVec.normalize();

		Scalar angularContrib = abs(pDistVec.dot(m_pGridData->getEtaBaseNormal(nextI, j)));
		Scalar eValue = m_pGridData->getScaleFactor(nextI, j).y*angularContrib/pDist;
		return eValue;
	}

	Scalar CurvilinearGridSolver2D::calculateNorthValue(int i, int j) {
		padValues(i, j);

		Vector2 pDistVec = m_pGridData->getCenterPoint(i, j + 1) - m_pGridData->getCenterPoint(i, j);
		Scalar pDist = pDistVec.length();
		pDistVec.normalize();

		Scalar angularContrib = abs(pDistVec.dot(m_pGridData->getXiBaseNormal(i, j + 1)));
		Scalar nValue = m_pGridData->getScaleFactor(i, j + 1).x*angularContrib/pDist;
		return nValue;
	}

	Scalar CurvilinearGridSolver2D::calculateSouthValue(int i, int j) {
		padValues(i, j);

		Vector2 pDistVec = m_pGridData->getCenterPoint(i, j) - m_pGridData->getCenterPoint(i, j - 1);
		Scalar pDist = pDistVec.length();
		pDistVec.normalize();

		Scalar angularContrib = abs(pDistVec.dot(m_pGridData->getXiBaseNormal(i, j)));
		Scalar sValue = m_pGridData->getScaleFactor(i, j).x*angularContrib/pDist;
		return sValue;
	}


	/************************************************************************/
	/* Auxiliary Functions		                                            */
	/************************************************************************/
	/** Update */
	void CurvilinearGridSolver2D::updatePoissonBoundaryConditions() {
		/*NonRegularGridPMB::getInstance()->reinitializePoissonMatrix(m_pGridData, m_boundaryConditions, m_pPoissonMatrix);*/
	}
	
	void CurvilinearGridSolver2D::updateVorticity() {
		/*QuadGrid *pQuadGrid = (QuadGrid *) m_pGrid;
		for(int i = 1; i < m_dimensions.x - 1; i++) {
			for(int j = 1; j < m_dimensions.y - 1; j++) {
				Scalar vorticity;
				Scalar dx = (m_pGridData->getScaleFactor(i, j).x + m_pGridData->getScaleFactor(i + 1, j).x)*0.5;
				Scalar dy = (m_pGridData->getScaleFactor(i, j).y + m_pGridData->getScaleFactor(i, j + 1).y)*0.5;

				vorticity = (m_pGridData->getVelocity(i + 1, j).y - m_pGridData->getVelocity(i, j).y)/dx
							- (m_pGridData->getVelocity(i, j + 1).x - m_pGridData->getVelocity(i, j).x)/dy;

				if(i == pQuadGrid->getConnectionPoint() - 1 && j > pQuadGrid->getLowerDimensions()[0].y  && j < pQuadGrid->getUpperDimensions()[0].y) {
					vorticity = 0;
				}

				m_pGridData->setVorticity(vorticity, i, j);
				m_pGridData->setDivergent(calculateFluxDivergent(i, j), i, j);
			}
		} */
	}

	void CurvilinearGridSolver2D::calculateBodyForce() {
	//	m_totalBodyForces = Vector2(0, 0);
	//	Scalar airDensity = 1.1839;
	//	for(int i = 1; i < m_dimensions.x - 1; i++) {
	//		Vector2 etaDirection = - m_pGridData->getEtaBase(i, 1);
	//		etaDirection.normalize();
	//		Scalar lift = etaDirection.y*m_pGridData->getPressure(i, 1)*m_pGridData->getScaleFactor(i, 1).x;
	//		Scalar v2 = 0.7*0.7;
	//		Scalar chordLenght = 1.0f;
	//		lift = lift/(airDensity*v2*chordLenght);
	//		m_totalBodyForces.y += lift;
	//		
	//	}

	//	//Log after stabilization a time
	//	if(PhysicsCore::getInstance()->getElapsedTime() > 0.6) {
	//		m_totalLiftForce += m_totalBodyForces.y;
	//		m_meanLiftForce = m_totalLiftForce/++m_numUpdates;
	//		if(abs(m_totalBodyForces.y) > abs(m_maxLiftForce))
	//			m_maxLiftForce = m_totalBodyForces.y;
	//	}
	//}

	///** Solid walls */
	//void CurvilinearGridSolver2D::enforceSolidWallsBC() {
	//	int connectionPoint = m_pGrid->getConnectionPoint();
	//	if(connectionPoint != -1) {
	//		for(int j = m_pGrid->getLowerDimensions()[0].y - 1; j < m_pGrid->getUpperDimensions()[0].y; j++) {
	//			if(m_params.getSolidBoundaryType() == Solid_NoSlip) {
	//				Vector2 localVelocity;
	//				localVelocity.x = 0;
	//				localVelocity.y = 0;
	//				m_pGridData->setAuxiliaryVelocity(localVelocity, connectionPoint - 1, j);
	//				m_pGridData->setAuxiliaryVelocity(localVelocity, connectionPoint, j);
	//				m_pGridData->setVelocity(localVelocity, connectionPoint - 1, j);
	//				m_pGridData->setVelocity(localVelocity, connectionPoint, j);
	//			} else if(m_params.getSolidBoundaryType() == Solid_FreeSlip) {
	//				Vector2 localVelocity;
	//				localVelocity = m_pGridData->getVelocity(connectionPoint - 1, j);
	//				localVelocity = transformToLocal(connectionPoint - 1, j, localVelocity, m_pGridData);
	//				localVelocity.x = 0;
	//				localVelocity = transformToGlobal(connectionPoint - 1, j, localVelocity, m_pGridData);
	//				m_pGridData->setVelocity(localVelocity, connectionPoint - 1, j);

	//				localVelocity = m_pGridData->getVelocity(connectionPoint, j);
	//				localVelocity = transformToLocal(connectionPoint, j, localVelocity, m_pGridData);
	//				localVelocity.x = 0;
	//				localVelocity = transformToGlobal(connectionPoint, j, localVelocity, m_pGridData);
	//				m_pGridData->setVelocity(localVelocity, connectionPoint, j);
	//			}
	//		}
	//	}
	}

	void CurvilinearGridSolver2D::applyForces(Scalar dt) {
		/*int i, j;

		int startingI, endingI;
		if(m_pGrid->isPeriodic()) {
			startingI = 0; endingI = m_dimensions.x;
		} else {
			startingI = 1; endingI = m_dimensions.x - 1;
		}*/
	}

	/************************************************************************/
	/* Local coordinate system transformations								*/
	/************************************************************************/

	void CurvilinearGridSolver2D::localCoordinateSystemTransform(Scalar dt) {
		/*for (int i = 0; i < m_dimensions.x; i++) {
		for(int j = 1; j < m_dimensions.y; j++) {
		Vector2 tempVelocity =  m_translationalVelocity;
		if(j == 1)
		tempVelocity.y = 0;
		m_pGridData->setVelocity(m_pGridData->getVelocity(i, j) + tempVelocity, i, j);
		}
		}*/
	}

	void CurvilinearGridSolver2D::globalCoordinateSystemTransform(Scalar dt) {
		/*for (int i = 0; i < m_dimensions.x; i++) {
			for(int j = 1; j < m_dimensions.y; j++) {
				Vector2 tempVelocity =  m_translationalVelocity;
				if(j == 1)
					tempVelocity.y = 0;
				m_pGridData->setVelocity(m_pGridData->getVelocity(i, j) - tempVelocity, i, j);
			}
		}
*/
	}

	/************************************************************************/
	/* Advection functions													*/
	/************************************************************************/
	void CurvilinearGridSolver2D::semiLagrangian(Scalar dt) {
		m_pSLIntegrator->integrateVelocityField();

		enforceSolidWallsBC();
	}

	void CurvilinearGridSolver2D::modifiedMacCormack(Scalar dt) {
		m_pMCIntegrator->integrateVelocityField();
		enforceSolidWallsBC();
	}

	void CurvilinearGridSolver2D::advectDensityField(Scalar dt) {
		if(m_params.getConvectionMethod()== convectionMethod_t::CPU_S02ModifiedMacCormack) {
			m_pMCIntegrator->integrateScalarField((DoubleBuffer<Array2D<Scalar>, Scalar> *) &m_pGridData->getDensityBuffer());
		} else {
			m_pSLIntegrator->integrateScalarField((DoubleBuffer<Array2D<Scalar>, Scalar> *) &m_pGridData->getDensityBuffer());
		}
	}


	/************************************************************************/
	/* Pseudo pressure step functions                                       */
	/************************************************************************/
	/** Discretization of the divergent operator in a finite volume fashion.
	 **	Caution: The interpolation of y values from neighboring cells,
	 **	can lead to instabilities */
	Scalar CurvilinearGridSolver2D::calculateFluxDivergent(int i, int j) {
		Vector2 cEast, cWest, cNorth, cSouth;

		int nextI = i + 1;
		if(m_pGrid->isPeriodic() && i == m_dimensions.x - 1) {
			nextI = 0;
		}
		int prevI = i - 1;
		if(m_pGrid->isPeriodic() && i == 0) {
			prevI = m_dimensions.x - 1;
		}

		/** Left flux*/
		cEast.x = m_pGridData->getAuxiliaryVelocity(i, j).x; 
		cEast.y = (m_pGridData->getAuxiliaryVelocity(i, j).y + m_pGridData->getAuxiliaryVelocity(prevI, j).y 
					+ m_pGridData->getAuxiliaryVelocity(i, j + 1).y + m_pGridData->getAuxiliaryVelocity(prevI, j + 1).y)*0.25;
		Scalar leftFlux = cEast.dot(m_pGridData->getEtaBaseNormal(i, j))*m_pGridData->getScaleFactor(i, j).y;

		/** Right flux*/
		cWest.x = m_pGridData->getAuxiliaryVelocity(nextI, j).x; 
		cWest.y = (m_pGridData->getAuxiliaryVelocity(nextI, j).y + m_pGridData->getAuxiliaryVelocity(i, j).y 
			+ m_pGridData->getAuxiliaryVelocity(nextI, j + 1).y + m_pGridData->getAuxiliaryVelocity(i, j + 1).y)*0.25;
		Scalar rightFlux = cWest.dot(-m_pGridData->getEtaBaseNormal(nextI, j))*m_pGridData->getScaleFactor(nextI, j).y;

		/** Bottom flux*/
		cSouth.y = m_pGridData->getAuxiliaryVelocity(i, j).y; 
		cSouth.x = (m_pGridData->getAuxiliaryVelocity(i, j).x + m_pGridData->getAuxiliaryVelocity(i, j - 1).x 
			+ m_pGridData->getAuxiliaryVelocity(nextI, j).x + m_pGridData->getAuxiliaryVelocity(nextI, j - 1).x)*0.25;
		Scalar bottomFlux = cSouth.dot(-m_pGridData->getXiBaseNormal(i, j))*m_pGridData->getScaleFactor(i, j).x;

		/** Top flux*/
		cNorth.y = m_pGridData->getAuxiliaryVelocity(i, j + 1).y;
		cNorth.x = (m_pGridData->getAuxiliaryVelocity(i, j + 1).x + m_pGridData->getAuxiliaryVelocity(i, j).x 
			+ m_pGridData->getAuxiliaryVelocity(nextI, j + 1).x + m_pGridData->getAuxiliaryVelocity(nextI, j).x)*0.25;
		Scalar topFlux = cNorth.dot(m_pGridData->getXiBaseNormal(i, j + 1))*m_pGridData->getScaleFactor(i, j + 1).x;

		Scalar divergent = (leftFlux + rightFlux + topFlux + bottomFlux)/m_pGridData->getVolume(i, j);
		return divergent;
	}

	Scalar CurvilinearGridSolver2D::calculateFinalDivergent(int i, int j) {
		Vector2 cEast, cWest, cNorth, cSouth;

		int nextI = i + 1;
		if(m_pGrid->isPeriodic() && i == m_dimensions.x - 1) {
			nextI = 0;
		}
		int prevI = i - 1;
		if(m_pGrid->isPeriodic() && i == 0) {
			prevI = m_dimensions.x - 1;
		}

		/** Left flux*/
		cEast.x = m_pGridData->getVelocity(i, j).x; 
		cEast.y = (m_pGridData->getVelocity(i, j).y + m_pGridData->getVelocity(prevI, j).y 
			+ m_pGridData->getVelocity(i, j + 1).y + m_pGridData->getVelocity(prevI, j + 1).y)*0.25;
		Scalar leftFlux = cEast.dot(m_pGridData->getEtaBaseNormal(i, j))*m_pGridData->getScaleFactor(i, j).y;

		/** Right flux*/
		cWest.x = m_pGridData->getVelocity(nextI, j).x; 
		cWest.y = (m_pGridData->getVelocity(nextI, j).y + m_pGridData->getVelocity(i, j).y 
			+ m_pGridData->getVelocity(nextI, j + 1).y + m_pGridData->getVelocity(i, j + 1).y)*0.25;
		Scalar rightFlux = cWest.dot(-m_pGridData->getEtaBaseNormal(nextI, j))*m_pGridData->getScaleFactor(nextI, j).y;

		/** Bottom flux*/
		cSouth.y = m_pGridData->getVelocity(i, j).y; 
		cSouth.x = (m_pGridData->getVelocity(i, j).x + m_pGridData->getVelocity(i, j - 1).x 
			+ m_pGridData->getVelocity(nextI, j).x + m_pGridData->getVelocity(nextI, j - 1).x)*0.25;
		Scalar bottomFlux = cSouth.dot(-m_pGridData->getXiBaseNormal(i, j))*m_pGridData->getScaleFactor(i, j).x;

		/** Top flux*/
		cNorth.y = m_pGridData->getVelocity(i, j + 1).y;
		cNorth.x = (m_pGridData->getVelocity(i, j + 1).x + m_pGridData->getVelocity(i, j).x 
			+ m_pGridData->getVelocity(nextI, j + 1).x + m_pGridData->getVelocity(nextI, j).x)*0.25;
		Scalar topFlux = cNorth.dot(m_pGridData->getXiBaseNormal(i, j + 1))*m_pGridData->getScaleFactor(i, j + 1).x;

		return (leftFlux + rightFlux + topFlux + bottomFlux);
	}



	/************************************************************************/
	/* Projection functions                                                 */
	/************************************************************************/
	
	void CurvilinearGridSolver2D::divergenceFree(Scalar dt) {
		int i, j;

		int startingI, endingI;
		if(m_pGrid->isPeriodic()) {
			startingI = 0; endingI = m_dimensions.x;
		} else {
			startingI = 1; endingI = m_dimensions.x - 1;
		}

#pragma omp parallel for 
		for(i = startingI; i < endingI; i++) {
#pragma omp parallel for private(j)
			for(j = 1; j < m_dimensions.y - 1; j++) {

				int nextI = i + 1;
				if(m_pGrid->isPeriodic() && i == m_dimensions.x - 1) {
					nextI = 0;
				}
				int prevI = i - 1;
				if(m_pGrid->isPeriodic() && i == 0) {
					prevI = m_dimensions.x - 1;
				}

				Vector2 velocity;
				Scalar pgXi, pgEta;
				Vector2 pgXiDir  = (m_pGridData->getCenterPoint(i, j) - m_pGridData->getCenterPoint(prevI, j));
				Vector2 pgEtaDir = (m_pGridData->getCenterPoint(i, j) - m_pGridData->getCenterPoint(i, j - 1));

				/** X component*/
				Scalar dx = pgXiDir.length();
				/** Y Component */
				Scalar dy = pgEtaDir.length();

				pgXiDir.normalize();
				pgEtaDir.normalize();

				pgXi = ((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(prevI, j))/dx)*pgXiDir.x;
				pgEta = (m_pGridData->getPressure(i, j + 1) +  m_pGridData->getPressure(prevI, j + 1))*0.5f -
					(m_pGridData->getPressure(i, j - 1) + m_pGridData->getPressure(prevI, j - 1))*0.5f;
				pgEta *= pgEtaDir.x/(dy*2);
				velocity.x = m_pGridData->getAuxiliaryVelocity(i, j).x - dt*(pgXi + pgEta);

				pgXi = (m_pGridData->getPressure(nextI, j) + m_pGridData->getPressure(nextI, j - 1))*0.5f -
					(m_pGridData->getPressure(prevI, j) + m_pGridData->getPressure(prevI, j - 1))*0.5f;
				pgXi *= pgXiDir.y/(dx*2);
				pgEta = ((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i, j - 1))/dy)*pgEtaDir.y;
				velocity.y = m_pGridData->getAuxiliaryVelocity(i, j).y - dt*(pgXi + pgEta);

				m_pGridData->setVelocity(velocity, i, j);
			}
		}


		enforceSolidWallsBC();
		calculateBodyForce();

	}

	


}