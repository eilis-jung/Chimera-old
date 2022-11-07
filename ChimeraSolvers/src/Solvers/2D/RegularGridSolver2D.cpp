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

#include "Solvers/2D/RegularGridSolver2D.h"


namespace Chimera {

	namespace Solvers {

		#pragma region Constructors
		RegularGridSolver2D::RegularGridSolver2D(const params_t &params, StructuredGrid<Vector2> *pGrid, 
													const vector<BoundaryCondition<Vector2> *> &boundaryConditions,
													const vector<RigidObject2D<Vector2> *> &rigidObjects /*= vector<RigidObject2D<Vector2> *>()*/)
			: FlowSolver(params, pGrid), m_rigidObjectsVec(rigidObjects) {
			
			/** Initialize line meshes - potential objects */
			for (int i = 0; i < m_rigidObjectsVec.size(); i++) {
				m_lineMeshes.push_back(m_rigidObjectsVec[i]->getLineMesh());
			}

			m_pGrid = pGrid;
			m_pGridData = m_pGrid->getGridData2D();
			m_dimensions = m_pGrid->getDimensions();
			m_boundaryConditions = boundaryConditions;
			m_pAdvection = nullptr;

			/** Initializing Poisson Matrix and Solver */
			m_pPoissonMatrix = createPoissonMatrix();
			initializePoissonSolver();

			/** Boundary conditions for initialization */
			enforceBoundaryConditions();

			/** Velocity and intermediary velocities interpolants */
			initializeInterpolants();

			/* Initialize density and temperature fields simulate with 0 time-step */
			applyHotSmokeSources(0);

			/** Initializing Advection */
			m_pAdvection = initializeAdvectionClass();
			if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
				ParticleBasedAdvection<Vector2, Array2D> *pPBAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
				pPBAdv->addScalarBasedAttribute("density", m_pDensityInterpolant);
				pPBAdv->addScalarBasedAttribute("temperature", m_pTemperatureInterpolant);
			}
		}
		#pragma endregion 

		#pragma region Functionalities
		void RegularGridSolver2D::update(Scalar dt) {
			m_numIterations++;
			m_totalSimulationTimer.start();

			/** First time-step: Do an additional solve to get div-free velocities from boundary conditions everywhere */
			if (PhysicsCore<Vector2>::getInstance()->getElapsedTime() < dt) {
				applyForces(dt);
				enforceBoundaryConditions();
				updateDivergents(dt);
				solvePressure();
				project(dt);
				enforceBoundaryConditions();
				
				/** If using a particle-based advection, we need to initialize their initial velocities by fully interpolating 
					from the grid */
				if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
					ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
					pParticleBasedAdv->getParticlesSampler()->interpolateVelocities(m_pVelocityInterpolant, pParticleBasedAdv->getParticlesData());
				}
			} else {
				/** If not initial time-step, apply forcing functions into the grid */
				applyForces(dt);
				if (m_params.vorticityConfinementStrength) {
					vorticityConfinement(dt);
				}

				/** If particle-based velocities, update their velocities at the beginning of the time-step*/
				if (m_pAdvection->getParams().advectionCategory == LagrangianAdvection) {
					//Update particles velocities just before the time-step, after buoyancy
					m_pAdvection->postProjectionUpdate(dt);
				}
			}
			
			/** Advection */
			m_advectionTimer.start(); 
			m_pAdvection->advect(dt);
			m_advectionTimer.stop();
			m_advectionTime = m_advectionTimer.secondsElapsed();
			
			/** Solve pressure */
			enforceBoundaryConditions();
			m_solvePressureTimer.start();
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

			/** Post projection update: Usually means advection of density fields and update particles attributes */
			m_pAdvection->postProjectionUpdate(dt);

			/** Swapping buffers for densities and temperatures*/
			m_pGridData->getDensityBuffer().swapBuffers();
			m_pGridData->getTemperatureBuffer().swapBuffers();
		
			/** We also have to do this for advection methods' interpolants */
			if (m_pAdvection->getParams().advectionCategory == EulerianAdvection) {
				SemiLagrangianAdvection<Vector2, Array2D> *pSLAdv = dynamic_cast<SemiLagrangianAdvection<Vector2, Array2D> *>(m_pAdvection);
				//MCcormack should also work in this cast, since it is derived from SL;
				if(pSLAdv) {
					//Swapping interpolants buffers means that the main interpolant becomes its sibling
					//and its sibling its automatically the old interpolant
					pSLAdv->setDensityInterpolant(pSLAdv->getDensityInterpolant()->getSibilingInterpolant());
					pSLAdv->setTemperatureInterpolant(pSLAdv->getTemperatureInterpolant()->getSibilingInterpolant());
				}
			}
			else if (m_pAdvection->getParams().advectionCategory == LagrangianAdvection) {
				ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
				auto densityAttributeIter = pParticleBasedAdv->getGridToParticles()->getScalarBasedAttribute("density");
				densityAttributeIter->second = densityAttributeIter->second->getSibilingInterpolant();

				auto temperatureAttributeIter = pParticleBasedAdv->getGridToParticles()->getScalarBasedAttribute("temperature");
				temperatureAttributeIter->second = temperatureAttributeIter->second->getSibilingInterpolant();
			}
			
			/** Update tracking variables - not important to simulation, but to visualization only. */
			updateKineticEnergy();
			updateVorticity();

			m_totalSimulationTimer.stop();
			m_totalSimulationTime = m_totalSimulationTimer.secondsElapsed();
		}

		void RegularGridSolver2D::vorticityConfinement(Scalar dt) {
			Scalar dx = m_pGridData->getGridSpacing();
			updateVorticity();

			Array2D<Vector2> vorticityConfinementForces(m_pGridData->getDimensions());
			//Apply only a bit further away from boundary conditions
			//Also, calculate forces for grid-centered, then interpolate it to velocity locations
			#pragma omp parallel for
			for (int i = 2; i < m_pGridData->getDimensions().x - 2; i++) {
				for (int j = 2; j < m_pGridData->getDimensions().y - 2; j++) {
					Vector3 vorticityGradient;

					vorticityGradient.x = (m_pGridData->getVorticity(i + 1, j) - m_pGridData->getVorticity(i - 1, j)) /	(dx);
					vorticityGradient.y = (m_pGridData->getVorticity(i, j + 1) - m_pGridData->getVorticity(i, j - 1)) / (dx);
					vorticityGradient.z = 0;

					vorticityGradient.normalize();

					Vector3 vorticity(0, 0, m_pGridData->getVorticity(i, j));

					Vector3 vConfinementForce = vorticityGradient.cross(vorticity)*dx*m_params.vorticityConfinementStrength*dt;
					if (m_pGridData->getVorticity(i, j) < 0.0)
						vConfinementForce = -vConfinementForce;
					vorticityConfinementForces(i, j) = Vector2(vConfinementForce.x, vConfinementForce.y);
				}
			}

			#pragma omp parallel for
			for (int i = 2; i < m_pGridData->getDimensions().x - 2; i++) {
				for (int j = 2; j < m_pGridData->getDimensions().y - 2; j++) {
					Vector2 interpolatedVorticityConfinement = vorticityConfinementForces(i, j) + vorticityConfinementForces(i - 1, j - 1);
					interpolatedVorticityConfinement *= 0.5;

					m_pGridData->setVelocity(m_pGridData->getVelocity(i, j) + interpolatedVorticityConfinement, i, j);
				}
			}
		}

		void RegularGridSolver2D::applyHotSmokeSources(Scalar dt) {
			Scalar dx = m_pGridData->getGridSpacing();
			#pragma omp parallel for
			for (int currSource = 0; currSource < m_params.smokeSources.size(); currSource++) {
				for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
					for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
						Scalar distance = (Vector2(i + 0.5, j + 0.5)*dx - m_params.smokeSources[currSource]->position).length();
						if (distance < m_params.smokeSources[currSource]->size) {
							Scalar noise = rand() / ((float)RAND_MAX);
							noise *= m_params.smokeSources[currSource]->densityVariation;
							m_pGridData->getDensityBuffer().setValueBothBuffers(m_params.smokeSources[currSource]->densityValue - noise, i, j);

							noise = rand() / ((float)RAND_MAX);
							noise *= m_params.smokeSources[currSource]->temperatureVariation;
							m_pGridData->getTemperatureBuffer().setValueBothBuffers(m_params.smokeSources[currSource]->temperatureValue - noise, i, j);

							Vector2 currVelocity = m_pGridData->getVelocity(i, j);
							m_pGridData->setVelocity(currVelocity + m_params.smokeSources[currSource]->velocity*dt, i, j);
						}
					}
				}

				if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection && dt != 0) { //dt == 0 initialization step
					ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
					ParticlesData<Vector2> *pParticlesData = pParticleBasedAdv->getParticlesData();
					const vector<Vector2> &particlePositions = pParticlesData->getPositions();
					
					for (int i = 0; i < particlePositions.size(); i++) {
						Scalar distance = (particlePositions[i] - m_params.smokeSources[currSource]->position).length();
						if (distance < m_params.smokeSources[currSource]->size) {
							if (pParticlesData->hasScalarBasedAttribute("density")) {
								vector<Scalar> &densities = pParticlesData->getScalarBasedAttribute("density");
								densities[i] = m_pDensityInterpolant->interpolate(particlePositions[i]);
							}

							if (pParticlesData->hasScalarBasedAttribute("temperature")) {
								vector<Scalar> &temperatures = pParticlesData->getScalarBasedAttribute("temperature");
								temperatures[i] = m_pTemperatureInterpolant->interpolate(particlePositions[i]);
							}

							pParticlesData->getVelocities()[i] += m_params.smokeSources[currSource]->velocity*dt;
						}
					}
				}
			}
			
		}

		void RegularGridSolver2D::applyRotationalForces(Scalar dt) {
			Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
			for (int k = 0; k < m_params.rotationalVelocities.size(); k++) {
				for (int i = 0; i < m_dimensions.x; i++) {
					for (int j = 0; j < m_dimensions.y; j++) {
						Vector2 velocity;
						Vector2 cellCenter(i*dx, (j + 0.5)*dx); //Staggered
						Vector2 radiusVec = cellCenter - m_params.rotationalVelocities[k].center;
						Scalar radius = radiusVec.length();
						Scalar radiusS = radius / dx;
						Scalar scaleFactor = clamp(1.0f / (0.25f*radiusS), 0.0f, 1.0f);
						//scaleFactor = 1.0f;
						if (radius > m_params.rotationalVelocities[k].minRadius && radius < m_params.rotationalVelocities[k].maxRadius) {
							if (m_params.rotationalVelocities[k].orientation) {
								velocity.x = -radiusVec.perpendicular().normalized().x*m_params.rotationalVelocities[k].strenght*scaleFactor;
							}
							else {
								velocity.x = radiusVec.perpendicular().normalized().x*m_params.rotationalVelocities[k].strenght*scaleFactor;
							}
							velocity.y = m_pGridData->getVelocity(i, j).y;
							m_pGridData->setVelocity(velocity, i, j);
							m_pGridData->setAuxiliaryVelocity(velocity, i, j);
						}
						cellCenter = Vector2((i + 0.5)*dx, j*dx);
						radiusVec = cellCenter - m_params.rotationalVelocities[k].center;
						radius = radiusVec.length();
						if (radius > m_params.rotationalVelocities[k].minRadius && radius < m_params.rotationalVelocities[k].maxRadius) {
							if (m_params.rotationalVelocities[k].orientation) {
								velocity.y = -radiusVec.perpendicular().normalized().y*m_params.rotationalVelocities[k].strenght*scaleFactor;
							}
							else {
								velocity.y = radiusVec.perpendicular().normalized().y*m_params.rotationalVelocities[k].strenght*scaleFactor;
							}
							velocity.x = m_pGridData->getVelocity(i, j).x;
							m_pGridData->setVelocity(velocity, i, j);
							m_pGridData->setAuxiliaryVelocity(velocity, i, j);
						}
					}
				}
			}
		}

		void RegularGridSolver2D::addBuyoancy(Scalar dt) {
			Scalar dx = m_pGridData->getGridSpacing();

			Array2D<Vector2> buoyancyForces(m_pGridData->getDimensions());
			buoyancyForces.assign(Vector2(0, 0));

			#pragma omp parallel for
			if (m_params.smokeSources.size() > 0) {
				for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
					for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
						//Temperature is normalized
						Scalar temperature = m_pGridData->getTemperatureBuffer().getValue(i, j);
						Scalar density = m_pGridData->getDensityBuffer().getValue(i, j);
						Vector2 buoyancyForce;
						buoyancyForce.y = temperature*m_params.smokeSources.front()->temperatureBuoyancyCoefficient
							- density*m_params.smokeSources.front()->densityBuoyancyCoefficient;
						buoyancyForces(i, j) += buoyancyForce*dt;//dx;
					}
				}
			}

			#pragma omp parallel for
			for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
					Vector2 interpolatedBuoyancy;
					if (j == 1)
						interpolatedBuoyancy = buoyancyForces(i, j);
					else
						interpolatedBuoyancy = (buoyancyForces(i, j) + buoyancyForces(i - 1, j - 1))*0.5;

					m_pGridData->setVelocity(m_pGridData->getVelocity(i, j) + interpolatedBuoyancy, i, j);
				}
			}
			
		}

		#pragma endregion 

		#pragma region InitializationFunctions
		PoissonMatrix * RegularGridSolver2D::createPoissonMatrix() {
			dimensions_t poissonMatrixDim = m_pGridData->getDimensions();
			if (m_params.pPoissonSolverParams->solverCategory == Krylov && !m_pGrid->isPeriodic()) {
				poissonMatrixDim.x += -2; poissonMatrixDim.y += -2;
			}
			else if (m_params.pPoissonSolverParams->solverCategory && m_pGrid->isPeriodic()) { //X peridiocity
				poissonMatrixDim.y += -2;
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
			//BoundaryCondition<Vector2>::updateSolidWalls(pMatrix, Array2D<bool>(m_pGrid->getSolidMarkers(), m_dimensions), m_params.getPressureSolverParams().getMethodCategory() == Krylov);

			/*if (pMatrix->isSingular() && m_params.getPressureSolverParams().getMethodCategory() == Krylov)
				pMatrix->applyCorrection(1e-3);*/

			if (m_params.pPoissonSolverParams->solverMethod != EigenCG &&
				m_params.pPoissonSolverParams->solverMethod != CPU_CG)
				pMatrix->updateCudaData();

			return pMatrix;
		}

		void RegularGridSolver2D::initializeInterpolants() {
			Scalar dx = m_pGridData->getGridSpacing();

			if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
				/** Velocity and intermediary velocities interpolants */
				ParticleBasedAdvection<Vector2, Array2D>::params_t *pPBAdvParams = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D>::params_t *>(m_params.pAdvectionParams);
				switch (pPBAdvParams->gridToParticleTransferMethod) {
					case ParticleBasedAdvection<Vector2, Array2D>::params_t::gridToParticle_t::APIC:
					case ParticleBasedAdvection<Vector2, Array2D>::params_t::gridToParticle_t::RPIC:
						m_pVelocityInterpolant = new BilinearAPICStaggeredInterpolant2D(m_pGridData->getVelocityArray(), dx);
					break;

					default: // FLIP, PIC, etc.
						m_pVelocityInterpolant = new BilinearStaggeredInterpolant2D<Vector2>(m_pGridData->getVelocityArray(), dx);
					break;
				}
			}
			else {
				m_pVelocityInterpolant = new BilinearStaggeredInterpolant2D<Vector2>(m_pGridData->getVelocityArray(), dx);
			}
			
			m_pAuxVelocityInterpolant = new BilinearStaggeredInterpolant2D<Vector2>(m_pGridData->getAuxVelocityArray(), dx);
			m_pVelocityInterpolant->setSiblingInterpolant(m_pAuxVelocityInterpolant);
			
			m_pDensityInterpolant = new BilinearStaggeredInterpolant2D<Scalar>(*m_pGridData->getDensityBuffer().getBufferArray1(), dx);
			m_pDensityInterpolant->setSiblingInterpolant(new BilinearStaggeredInterpolant2D<Scalar>(*m_pGridData->getDensityBuffer().getBufferArray2(), dx));
			m_pDensityInterpolant->getSibilingInterpolant()->setSiblingInterpolant(m_pDensityInterpolant);

			m_pTemperatureInterpolant = new BilinearStaggeredInterpolant2D<Scalar>(*m_pGridData->getTemperatureBuffer().getBufferArray1(), dx);
			m_pTemperatureInterpolant->setSiblingInterpolant(new BilinearStaggeredInterpolant2D<Scalar>(*m_pGridData->getTemperatureBuffer().getBufferArray2(), dx));
			m_pTemperatureInterpolant->getSibilingInterpolant()->setSiblingInterpolant(m_pTemperatureInterpolant);
		}
		#pragma endregion 

		#pragma region BoundaryConditions
		void RegularGridSolver2D::enforceSolidWallsConditions(const Vector2 &solidVelocity) {
			switch (m_params.solidBoundaryType) {
			case Solid_FreeSlip:
				FreeSlipBC<Vector2>::enforceSolidWalls(dynamic_cast<QuadGrid *>(m_pGrid), solidVelocity);
				break;
			case Solid_NoSlip:
				NoSlipBC<Vector2>::enforceSolidWalls(dynamic_cast<QuadGrid *>(m_pGrid), solidVelocity);
				break;

			case Solid_Interpolation:
				BoundaryCondition<Vector2>::zeroSolidBoundaries(m_pGridData);
				break;
			}
		}
		#pragma endregion 

		#pragma region PressureProjection
		Scalar RegularGridSolver2D::calculateFluxDivergent(int i, int j) {
			Scalar divergent = 0;

			if (m_pGrid->isBoundaryCell(i, j))
				return divergent;

			Scalar dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j).x - m_pGridData->getAuxiliaryVelocity(i, j).x) / m_pGridData->getScaleFactor(i, j).x;
			Scalar dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1).y - m_pGridData->getAuxiliaryVelocity(i, j).y) / m_pGridData->getScaleFactor(i, j).y;
			divergent = dx + dy;

			return divergent;
		}

		Scalar RegularGridSolver2D::calculateFinalDivergent(int i, int j) {
			Scalar divergent = 0;

			if (m_pGrid->isBoundaryCell(i, j) || m_pGrid->isSolidCell(i - 1, j) || m_pGrid->isSolidCell(i + 1, j)
				|| m_pGrid->isSolidCell(i, j - 1) || m_pGrid->isSolidCell(i, j + 1) || m_pGrid->isSolidCell(i, j))
				return divergent;

			Scalar dx = (m_pGridData->getVelocity(i + 1, j).x - m_pGridData->getVelocity(i, j).x) / m_pGridData->getScaleFactor(i, j).x;
			Scalar dy = (m_pGridData->getVelocity(i, j + 1).y - m_pGridData->getVelocity(i, j).y) / m_pGridData->getScaleFactor(i, j).x;

			divergent = dx + dy;
			return divergent;
		}

		void RegularGridSolver2D::divergenceFree(Scalar dt) {
			int i, j;
			#pragma omp parallel for
			for (i = 1; i < m_dimensions.x - 1; i++) {
				#pragma omp parallel for private(j)
				for (j = 1; j < m_dimensions.y - 1; j++) {
					if (m_pGrid->isSolidCell(i, j) || m_pGrid->isBoundaryCell(i, j))
						continue;

					Vector2 velocity;

					Scalar dx = m_pGridData->getScaleFactor(i, j).x;
					Scalar dy = m_pGridData->getScaleFactor(i, j).y;

					if (!m_pGrid->isSolidCell(i - 1, j))
						velocity.x = m_pGridData->getAuxiliaryVelocity(i, j).x - dt*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i - 1, j)) / dx);

					if (!m_pGrid->isSolidCell(i, j - 1))
						velocity.y = m_pGridData->getAuxiliaryVelocity(i, j).y - dt*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i, j - 1)) / dy);

					m_pGridData->setVelocity(velocity, i, j);
				}
			}

			updateVorticity();
		}
		#pragma endregion 

		#pragma region Misc
		Scalar RegularGridSolver2D::calculateVorticity(uint i, uint j) {
			Scalar dx = m_pGridData->getGridSpacing();
			return (m_pGridData->getVelocity(i + 1, j).y - m_pGridData->getVelocity(i, j).y) / (dx)
					- (m_pGridData->getVelocity(i, j + 1).x - m_pGridData->getVelocity(i, j).x) / (dx);
		}

		void RegularGridSolver2D::updateVorticity() {
			for (int i = 1; i < m_dimensions.x - 2; i++) {
				for (int j = 1; j < m_dimensions.y - 2; j++) {
					m_pGridData->setVorticity(calculateVorticity(i, j), i, j);
				}
			}
			for (int i = 1; i < m_dimensions.x - 1; i++) {
				m_pGridData->setVorticity(m_pGridData->getVorticity(i, m_dimensions.y - 2), i, m_dimensions.y - 1);
			}
			for (int j = 1; j < m_dimensions.y - 1; j++) {
				m_pGridData->setVorticity(m_pGridData->getVorticity(m_dimensions.x - 2, j), m_dimensions.x - 1, j);
			}
		}

		void RegularGridSolver2D::updateKineticEnergy()
		{
			Scalar dx = m_pGridData->getGridSpacing();

			Scalar minChange = 1e10;
			Scalar maxChange = -1e10;

			for (int i = 0; i < m_dimensions.x; ++i) {
				for (int j = 0; j < m_dimensions.y; ++j) {
					Vector2 velocity = m_pVelocityInterpolant->interpolate(Vector2(i + 0.5, j + 0.5)*dx); //argument in world coordinates
					Scalar kineticEnergy = 0.5 * velocity.dot(velocity);

					Scalar change = kineticEnergy - m_pGridData->getKineticEnergyValue(i, j);
					if (change < minChange) {
						minChange = change;
					}
					if (change > maxChange) {
						maxChange = change;
					}

					m_pGridData->setKineticEnergyChangeValue(change, i, j);
					m_pGridData->setKineticEnergyValue(kineticEnergy, i, j);
				}
			}

			//normalize change in kinetic energy
			for (int i = 0; i < m_dimensions.x; ++i) {
				for (int j = 0; j < m_dimensions.y; ++j) {
					Scalar change = m_pGridData->getKineticEnergyChangeValue(i, j);
					Scalar normalizedChange;

					if (minChange < 0) {
						if (change < 0) {
							normalizedChange = -1 * change / minChange; //change and minChange both negative -> will be positive -> need to make negative again
						} else {
							normalizedChange = change / maxChange;
						}
					} else {
						normalizedChange = (change - minChange) / (maxChange - minChange);
					}

					m_pGridData->setKineticEnergyChangeValue(normalizedChange, i, j);
				}
			}
		}

		void RegularGridSolver2D::updateStreamfunctions() {
			Scalar dx = m_pGridData->getGridSpacing();
			m_pGridData->setStreamfunction(0, 0, 0);
			for (int i = 1; i < m_pGridData->getDimensions().x; i++) {
				m_pGridData->setStreamfunction((-m_pGridData->getVelocity(i - 1, 0).y*dx) + m_pGridData->getStreamfunction(i - 1, 0), i, 0);
			}
			for (int i = 1; i < m_pGridData->getDimensions().x; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y; j++) {
					m_pGridData->setStreamfunction(m_pGridData->getVelocity(i, j - 1).x*dx + m_pGridData->getStreamfunction(i, j - 1), i, j);
				}
			}

			for (int j = 1; j < m_pGridData->getDimensions().y; j++) {
				m_pGridData->setStreamfunction((-m_pGridData->getVelocity(0, j - 1).x*dx) + m_pGridData->getStreamfunction(0, j - 1), 0, j);
			}

		}

		Scalar RegularGridSolver2D::getTotalKineticEnergy() const
		{
			Scalar totalKineticEnergy = 0;

			for (int i = 0; i < m_dimensions.x; ++i) {
				for (int j = 0; j < m_dimensions.y; ++j) {
					totalKineticEnergy += m_pGridData->getKineticEnergyValue(i, j);
				}

			}
			return totalKineticEnergy;
		}
		#pragma endregion 
	}
	
}