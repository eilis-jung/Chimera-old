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

#include "Solvers/3D/RegularGridSolver3D.h"

namespace Chimera {

	namespace Solvers {

		#pragma region Constructors
		RegularGridSolver3D::RegularGridSolver3D(const params_t &params, StructuredGrid<Vector3> *pGrid, 
												const vector<BoundaryCondition<Vector3> *> &boundaryConditions)
			: FlowSolver(params, pGrid) {

			m_pGrid = pGrid;
			m_pGridData = m_pGrid->getGridData3D();
			m_dimensions = m_pGrid->getDimensions();
			m_boundaryConditions = boundaryConditions;
			m_pAdvection = nullptr;
			m_pGridToParticlesTransfer = nullptr;
			m_pParticlesToGridTransfer = nullptr;

			m_vorticityConfinementFactor = 6;
			m_buoyancyDensityCoefficient = 1.8;
			m_buoyancyTemperatureCoefficient = 3.6;
			m_buoyancyDensityCoefficient *= 8;
			m_buoyancyTemperatureCoefficient *= 8;

			m_pPoissonMatrix = createPoissonMatrix();
			initializePoissonSolver();

			/** Boundary conditions for initialization */
			enforceBoundaryConditions();

			Scalar dx = m_pGridData->getGridSpacing();

			/** Velocity and intermediary velocities interpolants */
			initializeInterpolants();

			m_pAdvection = initializeAdvectionClass();
			
			if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
				ParticleBasedAdvection<Vector3, Array3D> *pPBAdv = dynamic_cast<ParticleBasedAdvection<Vector3, Array3D> *>(m_pAdvection);
				pPBAdv->addScalarBasedAttribute("density", m_pDensityInterpolant);
				pPBAdv->addScalarBasedAttribute("temperature", m_pTemperatureInterpolant);
			}

			Logger::get() << "[dx/dy] = " << m_pGridData->getScaleFactor(0, 0, 0).x / m_pGridData->getScaleFactor(0, 0, 0).y << endl;
		}
		#pragma endregion 

		#pragma region Functionalities
		void RegularGridSolver3D::update(Scalar dt) {
			m_numIterations++;
			m_totalSimulationTimer.start();

			if (PhysicsCore<Vector3>::getInstance()->getElapsedTime() < dt) {
				applyForces(dt);
				addBuyoancy(dt);
				enforceBoundaryConditions();
				updateDivergents(dt);
				enforceBoundaryConditions();
				solvePressure();
				project(dt);
				enforceBoundaryConditions();
				
				if (m_pAdvection->getParams().advectionCategory == LagrangianAdvection) {
					ParticleBasedAdvection<Vector3, Array3D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector3, Array3D> *>(m_pAdvection);
					pParticleBasedAdv->getParticlesSampler()->interpolateVelocities(m_pVelocityInterpolant, pParticleBasedAdv->getParticlesData());
				}
			}
			else {
				addBuyoancy(dt);
				vorticityConfinement(dt);
				if (m_pAdvection->getParams().advectionCategory == LagrangianAdvection) {
					//Update particles velocities just before the time-step, after buoyancy
					m_pAdvection->postProjectionUpdate(dt);
				}
			}

			m_advectionTimer.start();
			m_pAdvection->advect(dt);
			m_advectionTimer.stop();
			m_advectionTime = m_advectionTimer.secondsElapsed();
			
			/** Solve pressure */
			m_solvePressureTimer.start();
			enforceBoundaryConditions();
			updateDivergents(dt);
			solvePressure();
			m_solvePressureTimer.stop();
			m_solvePressureTime = m_solvePressureTimer.secondsElapsed();

			/** Project velocity */
			m_projectionTimer.start();
			project(dt);
			m_projectionTimer.stop();
			m_projectionTime = m_projectionTimer.secondsElapsed();
			
			/** Post projection update for advection methods */
			if (m_pAdvection->getParams().advectionCategory == EulerianAdvection) {
				/**Advect densities on grid-based advection schemes */
				m_pAdvection->postProjectionUpdate(dt);
			}
			
			/** Enforce conditions on density and temperature */
			enforceScalarFieldMarkers();
			m_pGridData->getDensityBuffer().swapBuffers();
			m_pGridData->getTemperatureBuffer().swapBuffers();

			updateVorticity();
			enforceBoundaryConditions();

			m_totalSimulationTimer.stop();
			m_totalSimulationTime = m_totalSimulationTimer.secondsElapsed();
		}

		void RegularGridSolver3D::updatePoissonSolidWalls() {
			for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
				m_boundaryConditions[i]->updatePoissonMatrix(m_pPoissonMatrix);
			}

			//BoundaryCondition<Vector3>::updateSolidWalls(m_pPoissonMatrix, Array3D<bool>(m_pGrid->getSolidMarkers(), m_dimensions), m_params.getPressureSolverParams().getMethodCategory() == Krylov);

			if (m_pPoissonMatrix->isSingular() && m_params.pPoissonSolverParams->solverMethod == Krylov)
				m_pPoissonMatrix->applyCorrection(1e-3);

			m_pPoissonMatrix->updateCudaData();
		}


		void RegularGridSolver3D::addBuyoancy(Scalar dt) {
			Scalar dx = m_pGridData->getGridSpacing();

			Array3D<Vector3> buoyancyForces(m_pGridData->getDimensions());

			#pragma omp parallel for
			for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
					for (int k = 1; k < m_pGridData->getDimensions().z - 1; k++) {
						//Temperature is normalized
						Scalar temperature = m_pGridData->getTemperatureBuffer().getValue(i, j, k);
						Scalar density = m_pGridData->getDensityBuffer().getValue(i, j, k);
						Vector3 buoyancyForce;
						buoyancyForce.y = temperature*m_buoyancyTemperatureCoefficient - density*m_buoyancyDensityCoefficient;
						buoyancyForces(i, j, k) = buoyancyForce*dt;//dx;
					}
				}
			}

			#pragma omp parallel for
			for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
					for (int k = 1; k < m_pGridData->getDimensions().z - 1; k++) {
						Vector3 interpolatedBuoyancy;
						if(j == 1)
							interpolatedBuoyancy = buoyancyForces(i, j, k);
						else
							interpolatedBuoyancy = (buoyancyForces(i, j, k) + buoyancyForces(i - 1, j - 1, k - 1))*0.5;

						m_pGridData->setVelocity(m_pGridData->getVelocity(i, j, k) + interpolatedBuoyancy, i, j, k);
					}
				}
			}
			
		}

		void RegularGridSolver3D::vorticityConfinement(Scalar dt) {
			//Updates vorticity before computing vorticity confinement forces
			updateVorticity();
			
			Scalar dx = m_pGridData->getGridSpacing();
			Array3D<Vector3> vorticityConfinementForces(m_pGridData->getDimensions());

			//Apply only a bit further away from boundary conditions
			//Also, calculate forces for grid-centered, then interpolate it to velocity locations
			#pragma omp parallel for
			for (int i = 2; i < m_pGridData->getDimensions().x - 2; i++) {
				for (int j = 2; j < m_pGridData->getDimensions().y - 2; j++) {
					for (int k = 2; k < m_pGridData->getDimensions().z - 2; k++) {
						Vector3 vorticityGradient;

						vorticityGradient.x = (m_pGridData->getVorticity(i + 1, j, k) - m_pGridData->getVorticity(i, j, k)) / dx;
						vorticityGradient.y = (m_pGridData->getVorticity(i, j + 1, k) - m_pGridData->getVorticity(i, j, k)) / dx;
						vorticityGradient.z = (m_pGridData->getVorticity(i, j, k + 1) - m_pGridData->getVorticity(i, j, k)) / dx;

						vorticityGradient.normalize();

						Vector3 vorticity = calculateVorticity(i, j, k);

						Vector3 vConfinementForce = vorticityGradient.cross(vorticity)*dx*m_vorticityConfinementFactor*dt;
						vorticityConfinementForces(i, j, k) = vConfinementForce;
					}
				}
			}

			#pragma omp parallel for
			for (int i = 2; i < m_pGridData->getDimensions().x - 2; i++) {
				for (int j = 2; j < m_pGridData->getDimensions().y - 2; j++) {
					for (int k = 2; k < m_pGridData->getDimensions().z - 2; k++) {
						Vector3 interpolatedVorticityConfinement = vorticityConfinementForces(i, j, k) + vorticityConfinementForces(i - 1, j - 1, k - 1);
						interpolatedVorticityConfinement *= 0.5;
					
						m_pGridData->setVelocity(m_pGridData->getVelocity(i, j, k) + interpolatedVorticityConfinement, i, j, k);
					}
				}
			}


		}
		#pragma endregion 

		#pragma region InitializationFunctions
		PoissonMatrix * RegularGridSolver3D::createPoissonMatrix() {
			dimensions_t poissonMatrixDim = m_pGridData->getDimensions();
			if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
				poissonMatrixDim.x += -2; poissonMatrixDim.y += -2; poissonMatrixDim.z += -2;
			}

			PoissonMatrix *pMatrix = NULL;
			if (m_params.pPoissonSolverParams->solverMethod == EigenCG ||
				m_params.pPoissonSolverParams->solverMethod == CPU_CG) {
				pMatrix = new PoissonMatrix(poissonMatrixDim, false);
			}
			else {
				pMatrix = new PoissonMatrix(poissonMatrixDim);
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

			if (pMatrix->isSingular() && m_params.pPoissonSolverParams->solverMethod == Krylov)
				pMatrix->applyCorrection(1e-3);


			if (m_params.pPoissonSolverParams->solverMethod != EigenCG &&
				m_params.pPoissonSolverParams->solverMethod != CPU_CG)
				pMatrix->updateCudaData();

			return pMatrix;
		}

		void RegularGridSolver3D::initializeInterpolants() {
			Scalar dx = m_pGridData->getGridSpacing();

			/** Velocity and intermediary velocities interpolants */
			m_pVelocityInterpolant = new BilinearStaggeredInterpolant3D<Vector3>(m_pGridData->getVelocityArray(), dx);
			m_pAuxVelocityInterpolant = new BilinearStaggeredInterpolant3D<Vector3>(m_pGridData->getAuxVelocityArray(), dx);
			m_pVelocityInterpolant->setSiblingInterpolant(m_pAuxVelocityInterpolant);

			m_pDensityInterpolant = new BilinearStaggeredInterpolant3D<Scalar>(*m_pGridData->getDensityBuffer().getBufferArray1(), dx);
			m_pDensityInterpolant->setSiblingInterpolant(new BilinearStaggeredInterpolant3D<Scalar>(*m_pGridData->getDensityBuffer().getBufferArray2(), dx));

			m_pTemperatureInterpolant = new BilinearStaggeredInterpolant3D<Scalar>(*m_pGridData->getTemperatureBuffer().getBufferArray1(), dx);
			m_pTemperatureInterpolant->setSiblingInterpolant(new BilinearStaggeredInterpolant3D<Scalar>(*m_pGridData->getTemperatureBuffer().getBufferArray2(), dx));
		}
		#pragma endregion 

		#pragma region BoundaryConditions
		void RegularGridSolver3D::enforceSolidWallsConditions(const Vector3 &solidVelocity) {
			switch (m_params.solidBoundaryType) {
			case Solid_FreeSlip:
				FreeSlipBC<Vector3>::enforceSolidWalls(dynamic_cast<HexaGrid *>(m_pGrid), solidVelocity);
				break;
			case Solid_NoSlip:
				NoSlipBC<Vector3>::enforceSolidWalls(dynamic_cast<HexaGrid *>(m_pGrid), solidVelocity);
				break;

			case Solid_Interpolation:
				BoundaryCondition<Vector3>::zeroSolidBoundaries(m_pGrid->getGridData3D());
				break;
			}
		}

		void RegularGridSolver3D::enforceScalarFieldMarkers() {
			for (int k = 0; k < m_scalarFieldMarkers.size(); k++) {
				int lowerBoundX, lowerBoundY, lowerBoundZ, upperBoundX, upperBoundY, upperBoundZ;
				Scalar dx = m_pGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
				lowerBoundX = floor(m_scalarFieldMarkers[k].position.x / dx);
				lowerBoundY = floor(m_scalarFieldMarkers[k].position.y / dx);
				lowerBoundZ = floor(m_scalarFieldMarkers[k].position.z / dx);
				upperBoundX = lowerBoundX + floor(m_scalarFieldMarkers[k].size.x / dx);
				upperBoundY = lowerBoundY + floor(m_scalarFieldMarkers[k].size.y / dx);
				upperBoundZ = lowerBoundZ + floor(m_scalarFieldMarkers[k].size.z / dx);

				for (int i = lowerBoundX; i < upperBoundX; i++) {
					for (int j = lowerBoundY; j < upperBoundY; j++) {
						for (int k = lowerBoundZ; k < upperBoundZ; k++) {
							m_pGridData->getDensityBuffer().setValueBothBuffers(1, i, j,k);
						}
					}
				}
			}
		}
		#pragma endregion 

		#pragma region PressureProjection
		Scalar RegularGridSolver3D::calculateFluxDivergent(int i, int j, int k) {
			return (m_pGridData->getAuxiliaryVelocity(i + 1, j, k).x - m_pGridData->getAuxiliaryVelocity(i, j, k).x) / m_pGridData->getScaleFactor(i, j, k).x +
				(m_pGridData->getAuxiliaryVelocity(i, j + 1, k).y - m_pGridData->getAuxiliaryVelocity(i, j, k).y) / m_pGridData->getScaleFactor(i, j, k).y +
				(m_pGridData->getAuxiliaryVelocity(i, j, k + 1).z - m_pGridData->getAuxiliaryVelocity(i, j, k).z) / m_pGridData->getScaleFactor(i, j, k).z;
		}

		void RegularGridSolver3D::divergenceFree(Scalar dt) {
			int i, j;
			//#pragma omp parallel for
			for (i = 1; i < m_dimensions.x - 1; i++) {
				//#pragma omp parallel for private(j)
				for (j = 1; j < m_dimensions.y - 1; j++) {
					for (int k = 1; k < m_dimensions.z - 1; k++) {
						Vector3 velocity;

						if (m_pGrid->isSolidCell(i, j, k)) {
							m_pGridData->setVelocity(Vector3(0, 0, 0), i, j, k);
							continue;
						}


						Scalar dx = m_pGridData->getScaleFactor(i, j, k).x;
						Scalar dy = m_pGridData->getScaleFactor(i, j, k).y;
						Scalar dz = m_pGridData->getScaleFactor(i, j, k).z;
						Scalar pressG = (m_pGridData->getPressure(i, j, k) - m_pGridData->getPressure(i, j - 1, k)) / dy;
						velocity.x = m_pGridData->getAuxiliaryVelocity(i, j, k).x - dt*((m_pGridData->getPressure(i, j, k) - m_pGridData->getPressure(i - 1, j, k)) / dx);
						velocity.y = m_pGridData->getAuxiliaryVelocity(i, j, k).y - dt*((m_pGridData->getPressure(i, j, k) - m_pGridData->getPressure(i, j - 1, k)) / dy);
						velocity.z = m_pGridData->getAuxiliaryVelocity(i, j, k).z - dt*((m_pGridData->getPressure(i, j, k) - m_pGridData->getPressure(i, j, k - 1)) / dz);

						if (m_pGrid->isSolidCell(i - 1, j, k))
							velocity.x = 0;
						if (m_pGrid->isSolidCell(i, j - 1, k))
							velocity.y = 0;
						if (m_pGrid->isSolidCell(i, j, k - 1))
							velocity.z = 0;

						m_pGridData->setVelocity(velocity, i, j, k);
					}
				}
			}
		}
		#pragma endregion 

		#pragma region InternalUpdateFunctions
		void RegularGridSolver3D::updateVorticity() {
			for (int i = 1; i < m_dimensions.x - 1; i++) {
				for (int j = 1; j < m_dimensions.y - 1; j++) {
					for (int k = 1; k < m_dimensions.z - 1; k++) {
						Vector3 vorticityVec = calculateVorticity(i, j, k);
						m_pGridData->setVorticity(vorticityVec.length(), i, j, k);
					}
				}
			}
		}

		Vector3 RegularGridSolver3D::calculateVorticity(uint i, uint j, uint k) {
			Scalar dx = m_pGridData->getGridSpacing();
			Vector3 vorticityVec;
			vorticityVec.x = (m_pGridData->getVelocity(i, j + 1, k).z - m_pGridData->getVelocity(i, j, k).z) / dx;
			vorticityVec.x -= (m_pGridData->getVelocity(i, j, k + 1).y - m_pGridData->getVelocity(i, j, k).y) / dx;

			vorticityVec.y = (m_pGridData->getVelocity(i, j, k + 1).x - m_pGridData->getVelocity(i, j, k).x) / dx;
			vorticityVec.y -= (m_pGridData->getVelocity(i + 1, j, k).z - m_pGridData->getVelocity(i, j, k).z) / dx;

			vorticityVec.z = (m_pGridData->getVelocity(i + 1, j, k).y - m_pGridData->getVelocity(i, j, k).y) / dx;
			vorticityVec.z -= (m_pGridData->getVelocity(i, j + 1, k).x - m_pGridData->getVelocity(i, j, k).x) / dx;
			
			return vorticityVec;
		}
		#pragma endregion 
	}
	
}