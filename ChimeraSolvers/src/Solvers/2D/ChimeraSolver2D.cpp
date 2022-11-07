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

#include "Physics/ChimeraSolver2D.h"
#include "Physics/PhysicsCore.h"
#include "Rendering/GLRenderer2D.h"

namespace Chimera {

	ChimeraSolver2D::ChimeraSolver2D(const FlowSolverParameters &params, SimulationConfig<Vector2> *pMainSimCfg) 
		: FlowSolver(params, pMainSimCfg->getGrid()) {
		m_simulationConfigs.push_back(pMainSimCfg);

		m_pGrid = pMainSimCfg->getGrid();
		m_dimensions = m_pGrid->getGridData2D()->getDimensions();
		//m_dimensions.x = m_dimensions.x - 1;
		m_boundaryConditions = *pMainSimCfg->getBoundaryConditions();
		m_pGridData = m_pGrid->getGridData2D();

		m_circularTranslation = false;

		m_initialIterations = 5;
	}

	void ChimeraSolver2D::globalCoordinateSystemTransform(Scalar dt) {}
	void ChimeraSolver2D::localCoordinateSystemTransform(Scalar dt){}
	/************************************************************************/
	/* Smoothing                                                            */
	/************************************************************************/
	void ChimeraSolver2D::smoothBoundaries(SimulationConfig<Vector2> *pBackSimCfg) {
		QuadGrid *pQuadGrid = dynamic_cast<QuadGrid *>(pBackSimCfg->getGrid());
		GridData2D *pGridData = pQuadGrid->getGridData2D();
		PoissonMatrix *pPoissonMatrix = pBackSimCfg->getFlowSolver()->getPoissonMatrix();
		for(int k = 0; k < m_boundarySmoothingIterations; k++) {
			for(int l = 0; l < m_boundaryIndices.size(); l++) {
				int i = m_boundaryIndices[l].x;
				int j = m_boundaryIndices[l].y;
				if(!pQuadGrid->isBoundaryCell(i, j)) {
					Scalar rhs = pGridData->getDivergent(i, j);
					Scalar h2 = pGridData->getScaleFactor(0, 0).x;
					h2 = h2*h2;	
					Scalar pressure =	rhs*h2 + (pGridData->getPressure(i + 1, j) + pGridData->getPressure(i - 1, j) +
						pGridData->getPressure(i, j + 1) + pGridData->getPressure(i, j - 1)); 
					pressure /= 4.0f;
					pGridData->setPressure(pressure, i, j);
				}
			}
		}
	}

	/************************************************************************/
	/* Interpolation functions	                                            */
	/************************************************************************/
	/** Velocity */
	void ChimeraSolver2D::backToFrontVelocityInterpolation(int ithFrontGrid) {
		SimulationConfig<Vector2> *pFrontSimCfg = m_simulationConfigs[ithFrontGrid];
		SimulationConfig<Vector2> *pBackSimCfg = m_simulationConfigs[0];

		GridData2D *pFrontGridData = pFrontSimCfg->getGrid()->getGridData2D();
		GridData2D *pBackGridData = pBackSimCfg->getGrid()->getGridData2D();
		
		Scalar dx = pBackGridData->getScaleFactor(0, 0).x;

		for(int i = 0; i < pFrontGridData->getDimensions().x; i++) {
			for(int j = pFrontGridData->getDimensions().y - 1; j < pFrontGridData->getDimensions().y; j++) {
				Vector2 frontGridCenter = pFrontGridData->getPoint(i, j) + pFrontSimCfg->getGrid()->getPosition();
				Vector2 velocityPosition = frontGridCenter - pBackSimCfg->getGrid()->getGridOrigin();

				Matrix2x2 cellBasis = m_pGridData->getInverseTransformationMatrix(i, j);
				/************************************************************************/
				/* U velocity position                                                  */
				/************************************************************************/
				velocityPosition += cellBasis.column[1]*pFrontGridData->getScaleFactor(i, j).y*0.5; //+ Eta base * 0.5
				velocityPosition = velocityPosition/dx;

				//Interpolation
				Vector2 interpVelocity = bilinearInterpolation(velocityPosition, pBackGridData->getVelocityArray());
				Vector2 interpAuxVel = bilinearInterpolation(velocityPosition, pBackGridData->getAuxVelocityArray());

				/************************************************************************/
				/* V velocity position                                                  */
				/************************************************************************/
				velocityPosition = frontGridCenter - pBackSimCfg->getGrid()->getGridOrigin();
				velocityPosition += cellBasis.column[0]*pFrontGridData->getScaleFactor(i, j).x;

				velocityPosition = velocityPosition/dx;
				//Interpolation
				interpVelocity.y = bilinearInterpolation(velocityPosition, pBackGridData->getVelocityArray()).y;
				interpAuxVel.y = bilinearInterpolation(velocityPosition, pBackGridData->getAuxVelocityArray()).y;

				pFrontGridData->setVelocity(interpVelocity - pFrontSimCfg->getGrid()->getVelocity(), i, j);
				pFrontGridData->setAuxiliaryVelocity(interpAuxVel - pFrontSimCfg->getGrid()->getVelocity(), i, j);
			}
		}
	}
	void ChimeraSolver2D::frontToBackVelocityInterpolation(int ithFrontGrid) {
		QuadGrid *pBackGrid = (QuadGrid* ) m_simulationConfigs[0]->getGrid();
		QuadGrid *pFrontGrid = (QuadGrid* ) m_simulationConfigs[ithFrontGrid]->getGrid();

		GridData2D *pBackGridData = pBackGrid->getGridData2D();
		GridData2D *pFrontGridData = pFrontGrid->getGridData2D();

		dimensions_t * pBoundaryMap = m_pBoundaryMaps[ithFrontGrid - 1];
		Array2D<dimensions_t> boundaryMap(pBoundaryMap, pBackGrid->getDimensions());

		Scalar dx = pBackGridData->getScaleFactor(0, 0).x;
	
		for(int i = 0; i < pBackGrid->getDimensions().x; i++) {
			for(int j = 0; j < pBackGrid->getDimensions().y; j++) {
				if(boundaryMap(i, j).x != -1) {

					dimensions_t frontGridIndex = boundaryMap(i, j);
					Matrix2x2 cellBasis = m_pGridData->getInverseTransformationMatrix(frontGridIndex.x, frontGridIndex.y);

					/************************************************************************/
					/* U position                                                           */
					/************************************************************************/
					Vector2 velocityPosition(i*dx, (j + 0.5)*dx);
					Vector2 velocityOrigin = pFrontGridData->getPoint(frontGridIndex.x, frontGridIndex.y);
					velocityOrigin += pFrontGrid->getPosition();
					velocityOrigin += cellBasis.column[1]*pFrontGridData->getScaleFactor(frontGridIndex.x, frontGridIndex.y).y*0.5; //+ Eta base * 0.5

					Vector2 transformedVec = velocityPosition - velocityOrigin;
					transformedVec = transformToCoordinateSystem(frontGridIndex.x, frontGridIndex.y, transformedVec, m_pGridData->getTransformationMatrices());
					transformedVec /= pFrontGridData->getScaleFactor(frontGridIndex.x, frontGridIndex.y);
					transformedVec += Vector2(frontGridIndex.x, frontGridIndex.y + 0.5);

					transformedVec.x = roundClamp<Scalar>(transformedVec.x, 0.0f, pFrontGridData->getDimensions().x);
					transformedVec.y = clamp<Scalar>(transformedVec.y, 0.0f, pFrontGridData->getDimensions().y - 0.51);

					Vector2 interpVelocity = bilinearInterpolation(transformedVec, pFrontGridData->getVelocityArray(), pFrontGrid->isPeriodic());
					Vector2 interpAuxVel = bilinearInterpolation(transformedVec,  pFrontGridData->getAuxVelocityArray(), pFrontGrid->isPeriodic());

					/************************************************************************/
					/* V position                                                           */
					/************************************************************************/
					velocityPosition= Vector2((i + 0.5)*dx, j*dx);
					velocityOrigin = pFrontGridData->getPoint(frontGridIndex.x, frontGridIndex.y);
					velocityOrigin += pFrontGrid->getPosition();
					velocityOrigin += cellBasis.column[0]*pFrontGridData->getScaleFactor(frontGridIndex.x, frontGridIndex.y).x*0.5; //+ Xi base * 0.5

					transformedVec = velocityPosition - velocityOrigin;
					transformedVec = transformToCoordinateSystem(frontGridIndex.x, frontGridIndex.y, transformedVec, m_pGridData->getTransformationMatrices());
					transformedVec /= pFrontGridData->getScaleFactor(frontGridIndex.x, frontGridIndex.y);
					transformedVec += Vector2(frontGridIndex.x + 0.5, frontGridIndex.y);

					transformedVec.x = roundClamp<Scalar>(transformedVec.x, 0.0f, pFrontGridData->getDimensions().x);
					transformedVec.y = clamp<Scalar>(transformedVec.y, 0.0f, pFrontGridData->getDimensions().y - 0.51);

					interpVelocity.y = bilinearInterpolation(transformedVec, pFrontGridData->getVelocityArray(), pFrontGrid->isPeriodic()).y;
					interpAuxVel.y = bilinearInterpolation(transformedVec, pFrontGridData->getAuxVelocityArray(), pFrontGrid->isPeriodic()).y;

					pBackGridData->setVelocity(interpVelocity + pFrontGrid->getVelocity(), i, j);
					pBackGridData->setAuxiliaryVelocity(interpAuxVel + pFrontGrid->getVelocity(), i, j);
				}
			}
		}
	}

	/** Pressure */
	void ChimeraSolver2D::backToFrontPressureInterpolation(int ithFrontGrid) {
		SimulationConfig<Vector2> *pFrontSimCfg = m_simulationConfigs[ithFrontGrid];
		SimulationConfig<Vector2> *pBackSimCfg = m_simulationConfigs[0];

		GridData2D *pFrontGridData = pFrontSimCfg->getGrid()->getGridData2D();
		GridData2D *pBackGridData = pBackSimCfg->getGrid()->getGridData2D();
		Scalar dx = pBackGridData->getScaleFactor(0, 0).x;

		for(int i = 0; i < pFrontGridData->getDimensions().x; i++) {
			for(int j = pFrontGridData->getDimensions().y - 1; j < pFrontGridData->getDimensions().y; j++) {
				Vector2 frontGridCenter = pFrontGridData->getCenterPoint(i, j);
				frontGridCenter += m_simulationConfigs[ithFrontGrid]->getGrid()->getPosition();
				Vector2 pressurePosition = (frontGridCenter - pBackSimCfg->getGrid()->getGridOrigin())/dx;

				Scalar pressure = interpolateScalar(pressurePosition, pBackGridData->getPressureArray());

				pFrontGridData->setPressure(pressure, i, j);
			}
		}
	}

	void ChimeraSolver2D::frontToBackPressureInterpolation(int ithFrontGrid) {
		QuadGrid *pBackGrid = (QuadGrid* ) m_simulationConfigs[0]->getGrid();
		QuadGrid *pFrontGrid = (QuadGrid* ) m_simulationConfigs[ithFrontGrid]->getGrid();

		GridData2D *pBackGridData = pBackGrid->getGridData2D();
		GridData2D *pFrontGridData = pFrontGrid->getGridData2D();

		dimensions_t * pBoundaryMap = m_pBoundaryMaps[ithFrontGrid - 1];
		Array2D<dimensions_t> boundaryMapArray(pBoundaryMap, pBackGridData->getDimensions());

		Scalar dx = pBackGridData->getScaleFactor(0, 0).x;

		for(int i = 0; i < pBackGrid->getDimensions().x; i++) {
			for(int j = 0; j < pBackGrid->getDimensions().y; j++) {
				if(boundaryMapArray(i, j).x != -1) {
					dimensions_t frontGridIndex = boundaryMapArray(i, j);

					Vector2 cellOrigin = pFrontGridData->getCenterPoint(frontGridIndex.x, frontGridIndex.y) + pFrontGrid->getPosition();
					Vector2 transformedVec = pBackGridData->getCenterPoint(i, j) - cellOrigin;
					transformedVec = transformToCoordinateSystem(frontGridIndex.x, frontGridIndex.y, transformedVec, m_pGridData->getTransformationMatrices());
					transformedVec /= pFrontGridData->getScaleFactor(frontGridIndex.x, frontGridIndex.y);
					transformedVec += Vector2(frontGridIndex.x + 0.5, frontGridIndex.y + 0.5);

					transformedVec.x = roundClamp<Scalar>(transformedVec.x, 0.0f, pFrontGridData->getDimensions().x);
					transformedVec.y = clamp<Scalar>(transformedVec.y, 0.0f, pFrontGridData->getDimensions().y - 0.51);

					Scalar pressure = interpolateScalar(transformedVec, pFrontGridData->getPressureArray(), pFrontGrid->isPeriodic());
					pBackGridData->setPressure(pressure, i, j);
				}
			}
		}
	}

	/** Density */
	void ChimeraSolver2D::backToFrontDensityInterpolation(int ithFrontGrid) {
		SimulationConfig<Vector2> *pFrontSimCfg = m_simulationConfigs[ithFrontGrid];
		SimulationConfig<Vector2> *pBackSimCfg = m_simulationConfigs[0];

		QuadGrid *pBackGrid = dynamic_cast<QuadGrid *>(pBackSimCfg->getGrid());
		QuadGrid *pFrontGrid = dynamic_cast<QuadGrid *>(pFrontSimCfg->getGrid());
		GridData2D *pFrontGridData = pFrontGrid->getGridData2D();
		GridData2D *pBackGridData = pBackGrid->getGridData2D();

		DoubleBuffer<Array2D<Scalar>, Scalar> *pDensityBuffer = &pFrontGridData->getDensityBuffer();

		Scalar dx = pBackGridData->getScaleFactor(0, 0).x;
		for(int i = 0; i < pFrontGrid->getGridData2D()->getDimensions().x; i++) {
			for(int j = pFrontGrid->getGridData2D()->getDimensions().y - 1; j < pFrontGrid->getGridData2D()->getDimensions().y; j++) {
				Vector2 frontGridCenter = pFrontGridData->getCenterPoint(i, j) + pFrontSimCfg->getGrid()->getPosition();
				Vector2 densityPosition = (frontGridCenter - pBackGrid->getGridOrigin())/dx;
				
				Scalar density = interpolateScalar(densityPosition, *pBackGridData->getDensityBuffer().getBufferArray1());
				
				pDensityBuffer->setValueBothBuffers(density, i, j);
			}
		}
	}
	void ChimeraSolver2D::frontToBackDensityInterpolation(int ithFrontGrid) {
		SimulationConfig<Vector2> *pFrontSimCfg = m_simulationConfigs[ithFrontGrid];
		SimulationConfig<Vector2> *pBackSimCfg = m_simulationConfigs[0];

		QuadGrid *pBackGrid = dynamic_cast<QuadGrid *>(pBackSimCfg->getGrid());
		QuadGrid *pFrontGrid = dynamic_cast<QuadGrid *>(pFrontSimCfg->getGrid());

		GridData2D *pFrontGridData = pFrontGrid->getGridData2D();
		GridData2D *pBackGridData = pBackGrid->getGridData2D();

		map<int, dimensions_t> *pBackMap = pBackGrid->getVelocityBoundaryMap();
		DoubleBuffer<Array2D<Scalar>, Scalar> *pDensityBuffer = &pFrontGridData->getDensityBuffer();

		dimensions_t * pBoundaryMap = m_pBoundaryMaps[ithFrontGrid - 1];
		Array2D<dimensions_t> boundaryMapArray(pBoundaryMap, pBackGridData->getDimensions());

		Scalar dx = pBackGridData->getScaleFactor(0, 0).x;

		for(int i = 0; i < pBackGrid->getDimensions().x; i++) {
			for(int j = 0; j < pBackGrid->getDimensions().y; j++) {
				if(boundaryMapArray(i, j).x != -1) {
					dimensions_t frontGridIndex = boundaryMapArray(i, j);

					Vector2 frontGridCenter = pFrontGridData->getCenterPoint(frontGridIndex.x, frontGridIndex.y) + pFrontSimCfg->getGrid()->getPosition();
					Vector2 transformedVec = pBackGridData->getCenterPoint(i, j) - frontGridCenter;
					transformedVec = transformToCoordinateSystem(frontGridIndex.x, frontGridIndex.y, transformedVec, m_pGridData->getTransformationMatrices());
					transformedVec += Vector2(frontGridIndex.x + 0.5, frontGridIndex.y + 0.5);

					transformedVec.x = clamp<Scalar>(transformedVec.x, 0.0f, pFrontGridData->getDimensions().x - 0.51);
					transformedVec.y = clamp<Scalar>(transformedVec.y, 0.0f, pFrontGridData->getDimensions().y - 0.51);

					Scalar density = interpolateScalar(transformedVec, *pFrontGridData->getDensityBuffer().getBufferArray1());
					pDensityBuffer->setValueBothBuffers(density, i, j);
				}
			}
		}
	}

	/************************************************************************/
	/* Auxiliary functions                                                  */
	/************************************************************************/
	void ChimeraSolver2D::applyForces(Scalar dt) {

	}
	
	bool ChimeraSolver2D::isClosestCell(const Vector2 &frontGridCenterPoint, int i, int j) {
		QuadGrid * pBackgrid = (QuadGrid*) m_simulationConfigs[0]->getGrid();
		Vector2 backGridCenterPoint = pBackgrid->getGridData2D()->getCenterPoint(i, j);
		Scalar centerDistance = (frontGridCenterPoint - backGridCenterPoint).length();
		int mapIndex = pBackgrid->getDimensions().x*j + i;
		map<int, Scalar>::iterator centerIter = m_minDistanceMap.find(mapIndex);
		if(centerIter != m_minDistanceMap.end()) {
			if(centerDistance < centerIter->second) {
				return true;
			}
		} else { //No frontgrid index was mapped to this background cell so far
			return true;
		}
		return false;
	}

	void ChimeraSolver2D::updateDivergents(Scalar dt) {
		for(int i = 0; i < m_simulationConfigs.size(); i++) {
			m_simulationConfigs[i]->getFlowSolver()->updateDivergents(dt);
		}
	}

 	/************************************************************************/
	/* Advection functions                                                  */
	/************************************************************************/
	void ChimeraSolver2D::semiLagrangian(Scalar dt) {
		/** Velocity */
		m_simulationConfigs[0]->getFlowSolver()->advect(dt);
		for(int i = 1; i < m_simulationConfigs.size(); i++) {
			m_simulationConfigs[i]->getFlowSolver()->localCoordinateSystemTransform(dt);
			backToFrontVelocityInterpolation(i);
			m_simulationConfigs[i]->getFlowSolver()->advect(dt);
			frontToBackVelocityInterpolation(i);
		}
	}

	void ChimeraSolver2D::modifiedMacCormack(Scalar dt) {
		semiLagrangian(dt);
	}

	void ChimeraSolver2D::advectDensityField(Scalar dt) {
		/** Density */
		m_simulationConfigs[0]->getFlowSolver()->advectDensityField(dt);
		for(int i = 1; i < m_simulationConfigs.size(); i++) {
			backToFrontDensityInterpolation(i);
			m_simulationConfigs[i]->getFlowSolver()->advectDensityField(dt);
			frontToBackDensityInterpolation(i);
		}
	}

	/************************************************************************/
	/* Pressure solving step                                                */
	/************************************************************************/
	void ChimeraSolver2D::solvePressure() {
		/************************************************************************/
		/* Pressure solving step                                                */
		/************************************************************************/
		Scalar resBackgrid = 0.0f, resFrontgrid = 0.0f;
		int outerIter;
		Scalar dt = PhysicsCore::getInstance()->getParams()->timestep;
		if(PhysicsCore::getInstance()->getElapsedTime()/dt < 10) {
			outerIter = m_initialIterations;	
		} else {
			outerIter = m_outerIterations;
		}


		for(int outer = 0; outer < outerIter; outer++) {
			//Solve pressure on the background grid
			m_simulationConfigs[0]->getFlowSolver()->solvePressure();
			
			//Dynamically adjusting the number of pressure solving iterations on the foreground grid
			if(resBackgrid < resFrontgrid) {
				m_innerIterations++;
			} else if(resBackgrid > resFrontgrid) {
				m_innerIterations--;
			}

			for(int i = 1; i < m_simulationConfigs.size(); i++) {
				//Setting up maximum and minimum number of inner iterations
				m_innerIterations = clamp<int>(m_innerIterations, 1, 10);	

				backToFrontPressureInterpolation(i);
				for(int k = 0; k < m_innerIterations; k++) {
					m_simulationConfigs[i]->getFlowSolver()->solvePressure();
				}
				frontToBackPressureInterpolation(i);
			}
			smoothBoundaries(m_simulationConfigs[0]);

			//Calculating residual
			resBackgrid = m_simulationConfigs[0]->getFlowSolver()->getParams().getPoissonSolver()->getResidual();
			resFrontgrid = m_simulationConfigs[1]->getFlowSolver()->getParams().getPoissonSolver()->getResidual();

		}
		
		GLRenderer2D::getInstance()->getSimulationStatsWindow()->setResidual(resBackgrid + resFrontgrid);
	}


	/************************************************************************/
	/* Projection functions                                                 */
	/************************************************************************/
	void ChimeraSolver2D::divergenceFree(Scalar dt) {
		m_simulationConfigs[0]->getFlowSolver()->enforceBoundaryConditions();
		m_simulationConfigs[0]->getFlowSolver()->project(dt);

		for(int i = 1; i < m_simulationConfigs.size(); i++) {
			m_simulationConfigs[i]->getFlowSolver()->enforceBoundaryConditions();
			m_simulationConfigs[i]->getFlowSolver()->project(dt);
			m_simulationConfigs[i]->getFlowSolver()->enforceBoundaryConditions();
		}
	}


	/************************************************************************/
	/* Grid holes and boundaries update                                     */
	/************************************************************************/
	void ChimeraSolver2D::updateGridHoles() {
		QuadGrid *pBackgrid = (QuadGrid *) m_simulationConfigs[0]->getGrid();
		for(int simIndex = 1; simIndex < m_simulationConfigs.size(); simIndex++) {
			QuadGrid *pFrontgrid = (QuadGrid *) m_simulationConfigs[simIndex]->getGrid();
			dimensions_t *pBoundaryMap = m_pBoundaryMaps[simIndex - 1];
			Array2D<dimensions_t> boundaryMapArray(pBoundaryMap, pBackgrid->getDimensions());

			/**Finding boundary cells first */
			int j = pFrontgrid->getDimensions().y - m_boundaryThreshold*simIndex;
			for(int i = 0; i < pFrontgrid->getDimensions().x; i++) {
				Vector2 centerPosition = pFrontgrid->getGridData2D()->getCenterPoint(i, j) + pFrontgrid->getPosition() - pBackgrid->getGridOrigin();
				Scalar dx = pBackgrid->getGridData2D()->getScaleFactor(0, 0).x;
				pBackgrid->setBoundaryCell(true, centerPosition.x/dx, centerPosition.y/dx);
				if(isClosestCell(centerPosition, centerPosition.x/dx, centerPosition.y/dx)) {
					boundaryMapArray(centerPosition.x/dx, centerPosition.y/dx) =  dimensions_t(i, j);
				} 
			}

			/************************************************************************/
			/* Filling in grid holes											    */
			/************************************************************************/
			int topBoundary = 0;
			for(int j = pBackgrid->getDimensions().y - 1; j > 0; j--) {
				int connectedBoundaryLines = 0;
				for(int i = 0; i < pBackgrid->getDimensions().x - 1; i++) {
					if(pBackgrid->isBoundaryCell(i, j) && boundaryMapArray(i, j).x != -1) {
						while(pBackgrid->isBoundaryCell(i + 1, j) && boundaryMapArray(i + 1, j).x != -1)  {
							++i;
						}
						connectedBoundaryLines++;
					}
				}
				if(connectedBoundaryLines > 1) { //Passed by two connected boundary lines
					topBoundary = j + 1; //The line above is the top boundary
					break;
				}
			}

			int bottomBoundary = 0;
			for(int j = 0; j < topBoundary; j++) {
				int connectedBoundaryLines = 0;
				for(int i = 0; i < pBackgrid->getDimensions().x - 1; i++) {
					if(pBackgrid->isBoundaryCell(i, j) && boundaryMapArray(i, j).x != -1) {
						while(pBackgrid->isBoundaryCell(i + 1, j) && boundaryMapArray(i + 1, j).x != -1)  {
							++i;
						}
						connectedBoundaryLines++;
					}
				}
				if(connectedBoundaryLines > 1) { //Passed by two connected boundary lines
					bottomBoundary = j - 1; //The line below is the bottom boundary
					break;
				}
			}

			for(int j = bottomBoundary + 1; j < topBoundary; j++) {
				for(int i = 0; i < pBackgrid->getDimensions().x - 1; i++) {
					if(pBackgrid->isBoundaryCell(i, j) && boundaryMapArray(i, j).x != -1) {
						while(!pBackgrid->isBoundaryCell(i + 1, j)  && !boundaryMapArray(i + 1, j).x != -1) {
							pBackgrid->setSolidCell(true, i + 1, j);
							++i;
							if(pBackgrid->isBoundaryCell(i + 1, j) && boundaryMapArray(i + 1, j).x  != -1) {
								i = pBackgrid->getDimensions().x - 1;
								break;
							}
						}
					}
				}

			}

			/************************************************************************/
			/* Removing unnecessary boundary cells                                  */
			/************************************************************************/
			for(int i = 0; i < pBackgrid->getDimensions().x; i++) {
				for(int j = 0; j < pBackgrid->getDimensions().y; j++) {
					bool adjacent = pBackgrid->isAdjacentToWall(i, j);
					if((!pBackgrid->isAdjacentToWall(i, j) && pBackgrid->isBoundaryCell(i, j))) {
						pBackgrid->setBoundaryCell(false, i, j);
					}
				}
			}
		}
		
	}

	void ChimeraSolver2D::updateGridHolesRobust() {
		QuadGrid *pBackGrid = (QuadGrid *) m_simulationConfigs[0]->getGrid();
		for(int simIndex = 1; simIndex < m_simulationConfigs.size(); simIndex++) {
			vector<Vector2> objectPoints;
			QuadGrid *pFrontGrid = (QuadGrid *) m_simulationConfigs[simIndex]->getGrid();
			for(int i = 0; i <= pFrontGrid->getDimensions().x; i++) {
				objectPoints.push_back(pFrontGrid->getGridData2D()->getPoint(i, 0));
			}

			dimensions_t *pBoundaryMap = m_pBoundaryMaps[simIndex - 1];
			Array2D<dimensions_t> boundaryMapArray(pBoundaryMap, pBackGrid->getDimensions());

			Scalar dx = pBackGrid->getGridData2D()->getScaleFactor(0, 0).x;

			//Setting all solid cells before hand.
			for(int i = 0; i < pBackGrid->getDimensions().x; i++) {
				for(int j = 0; j < pBackGrid->getDimensions().y; j++) {
					bool foundCell = false;

					Vector2 regularGridCellPos = pBackGrid->getGridData2D()->getCenterPoint(i, j);

					int k = 0, l = 0;
					for(k = 0; k < pFrontGrid->getDimensions().x && !foundCell; k++) {
						for(l = 0; l < pFrontGrid->getDimensions().y - m_boundaryThreshold; l++) {
							if(foundCell = pFrontGrid->isInsideCell(regularGridCellPos, k, l) ) {
								pBackGrid->setSolidCell(true, i, j);
								Vector2 fgCenterPos = pFrontGrid->getGridData2D()->getCenterPoint(i, j) + pFrontGrid->getPosition() - pBackGrid->getGridOrigin();
								if(isClosestCell(fgCenterPos, i, j)) {
									boundaryMapArray(i, j) = dimensions_t(k, l, 0);
								}
								break;
							}
						}
					}
					if(!foundCell) {
						if(isInsidePolygon(pBackGrid->getGridData2D()->getCenterPoint(i, j) - pFrontGrid->getPosition(), objectPoints) || 
							isInsidePolygon(pBackGrid->getGridData2D()->getPoint(i, j) - pFrontGrid->getPosition(), objectPoints)) {
								pBackGrid->setSolidCell(true, i, j);
						}
						boundaryMapArray(regularGridCellPos.x/dx, regularGridCellPos.y/dx) = dimensions_t(-1, -1, 0);
					}
				}
			}
			for(int ipass = 0; ipass < 2; ipass++) {
				for(int i = 1, iniJ = ipass + 1; i < pBackGrid->getDimensions().x - 1; i++, iniJ = 3 - iniJ) {
					for(int j = iniJ; j < pBackGrid->getDimensions().y - 1; j += 2) {
						if(pBackGrid->isSolidCell(i,j)) {
							if(!pBackGrid->isSolidCell(i + 1, j) || !pBackGrid->isSolidCell(i - 1, j) || 
								!pBackGrid->isSolidCell(i, j + 1) || !pBackGrid->isSolidCell(i, j - 1)) {
									pBackGrid->setBoundaryCell(true, i, j);
							}
						}
					}
				}
			}

			for(int i = 0; i < pBackGrid->getDimensions().x; i++) {
				for(int j = 0; j < pBackGrid->getDimensions().y; j++) {
					if(pBackGrid->isBoundaryCell(i, j))
						pBackGrid->setSolidCell(false, i, j);
				}
			}

			/************************************************************************/
			/* Removing unnecessary boundary cells                                  */
			/************************************************************************/
			for(int i = 0; i < pBackGrid->getDimensions().x; i++) {
				for(int j = 0; j < pBackGrid->getDimensions().y; j++) {
					bool adjacent = pBackGrid->isAdjacentToWall(i, j);
					if((!pBackGrid->isAdjacentToWall(i, j) && pBackGrid->isBoundaryCell(i, j))) {
						pBackGrid->setBoundaryCell(false, i, j);
					}
				}
			}
		}
	}

	void ChimeraSolver2D::updateGridBoundaries() {
		QuadGrid *pBackgrid = (QuadGrid *) m_simulationConfigs[0]->getGrid();
		bool *tempMarkers = new bool[pBackgrid->getDimensions().x*pBackgrid->getDimensions().y];
		Array2D<bool> tempMarkersArray(tempMarkers, pBackgrid->getDimensions());
		GridData2D *pGridData = pBackgrid->getGridData2D();

		/**Temporary boundary map is initialized with the original boundaries of the background grid. Additionally, each
		 ** cell in the local neighborhood is initialized as a temporary boundary map.*/
		for(int i = 0; i < pBackgrid->getDimensions().x; i++) {
			for(int j = 0; j < pBackgrid->getDimensions().y; j++) {
				tempMarkersArray(i, j) = 0;
				if(pBackgrid->isBoundaryCell(i, j)) {
					tempMarkersArray(i, j) = true;
					tempMarkersArray(i + 1, j) = true;
					tempMarkersArray(i - 1, j) = true;
					tempMarkersArray(i, j + 1) = true;
					tempMarkersArray(i, j - 1) = true;
				}
			}
		}

		for(int k = 0; k < m_boundarySmoothingLayers/2; k++) {
			for(int ipass = 0; ipass < 2; ipass++) {
				for(int i = 1, iniJ = ipass + 1; i < pGridData->getDimensions().x - 1; i++, iniJ = 3 - iniJ) {
					for(int j = iniJ; j < pGridData->getDimensions().y - 1; j += 2) {
						if(tempMarkersArray(i, j)) {
							tempMarkersArray(i + 1, j) = true;
							tempMarkersArray(i - 1, j) = true;
							tempMarkersArray(i, j + 1) = true;
							tempMarkersArray(i, j - 1) = true;
						} else {
							tempMarkersArray(i, j) = tempMarkersArray(i + 1, j) ||
														tempMarkersArray(i - 1, j) ||
														tempMarkersArray(i, j + 1) ||
														tempMarkersArray(i, j - 1);
						}
					}
				}
			}
		}

		/** Clearing boundary indices */
		m_boundaryIndices.clear();
		for(int i = 0; i < pBackgrid->getDimensions().x; i++) {
			for(int j = 0; j < pBackgrid->getDimensions().y; j++) {
				if(tempMarkersArray(i, j)) {
					m_boundaryIndices.push_back(dimensions_t(i, j));
				}
			}
		}

		delete tempMarkers;
	}

	/************************************************************************/
	/* Update function                                                      */
	/************************************************************************/
	void ChimeraSolver2D::update(Scalar dt) {
		/** Update all the grids positions and velocities */
		/**Updating grids position */
		for(int i = 1; i < m_simulationConfigs.size(); i++) {
			m_simulationConfigs[i]->getGrid()->setPosition(m_simulationConfigs[i]->getGrid()->getPosition() 
															+ m_simulationConfigs[i]->getGrid()->getVelocity()*dt);
			//m_simulationConfigs[i]->getGrid()->rotate(0.5*dt); //0.5 angular rotation per sec.

			if(m_circularTranslation) {
				Vector2 circularAxis = m_simulationConfigs[i]->getGrid()->getPosition() - m_simulationConfigs[0]->getGrid()->getGridCentroid();
				Vector2 circularVelocity = circularAxis.perpendicular().normalize();
				setOverlappingGridVelocity(i, circularVelocity);
			}
		}

		/** Update grid markers, grid holes and boundary indices */
		for(int i = 1; i < m_simulationConfigs.size(); i++) {
			QuadGrid *pFrontGrid = dynamic_cast<QuadGrid *>(m_simulationConfigs[i]->getGrid());
			if(pFrontGrid->getVelocity().length() > 0.0f) {
				QuadGrid *pBackgrid = dynamic_cast<QuadGrid *>(m_simulationConfigs[0]->getGrid());

				/**Resetting the temporary boundary map */
				dimensions_t *pBoundaryMap = m_pBoundaryMaps[i - 1];
				Array2D<dimensions_t> boundaryMapArray(pBoundaryMap, pBackgrid->getDimensions());
				for(int i = 0; i < pBackgrid->getDimensions().x; i++) {
					for(int j = 0; j < pBackgrid->getDimensions().y; j++) {
						boundaryMapArray(i, j) = dimensions_t(-1, -1, -1);
						pBackgrid->setSolidCell(false, i, j);
						pBackgrid->setBoundaryCell(false, i, j);
					}
				}

				updateGridHoles();
				updateGridBoundaries();
			}
		}

		//Updates the flow solver
		FlowSolver::update(dt);
	}

	/************************************************************************/
	/* Access functions                                                     */
	/************************************************************************/
	void ChimeraSolver2D::setOverlappingGridVelocity(int gridIndex, const Vector2 gridVelocity) {
		m_simulationConfigs[gridIndex]->getGrid()->setVelocity(gridVelocity);
		m_simulationConfigs[gridIndex]->getFlowSolver()->globalCoordinateSystemTransform(PhysicsCore::getInstance()->getParams()->timestep);
		//m_simulationConfigs[gridIndex]->getFlowSolver()->setTranslationalVelocity(-gridVelocity);
	}
}