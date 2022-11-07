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
//
#include "Solvers/2D/TurbulenceSolver2D.h"

namespace Chimera {
	namespace Solvers {

		#pragma region Constructors
		TurbulenceSolver2D::TurbulenceSolver2D(const params_t &params, unsigned int numSubdivisions, StructuredGrid<Vector2> *pGrid,
			const vector<BoundaryCondition<Vector2> *> &boundaryConditions) : RegularGridSolver2D(params, pGrid),
			m_streamfunctionGrid(dimensions_t(pGrid->getDimensions().x*pow(2, numSubdivisions), pGrid->getDimensions().y*pow(2, numSubdivisions))),
			m_auxStreamfunctionGrid(dimensions_t(pGrid->getDimensions().x*pow(2, numSubdivisions), pGrid->getDimensions().y*pow(2, numSubdivisions))),
			m_fineGridVelocities(dimensions_t(pGrid->getDimensions().x*pow(2, numSubdivisions), pGrid->getDimensions().y*pow(2, numSubdivisions))) {
			m_numDivisions = numSubdivisions;
			m_boundaryConditions = boundaryConditions;
			m_refinedDx = pGrid->getGridData2D()->getGridSpacing()/ (pow(2, m_numDivisions));
			m_pGrid = pGrid;
			m_pGridData = m_pGrid->getGridData2D();
			m_dimensions = m_pGrid->getDimensions();
			m_boundaryConditions = boundaryConditions;
			m_pAdvection = NULL;
			m_streamfunctionGrid.assign(0);
			m_pPoissonMatrix = createPoissonMatrix();
			initializePoissonSolver();

			/** Boundary conditions for initialization */
			enforceBoundaryConditions();

			m_streamfunctionGrid.assign(0.0f);
			m_auxStreamfunctionGrid.assign(0.0f);
			
			//Scalar scaleFactor = m_refinedDx / m_pGridData->getGridSpacing();
			//Scalar coarseGridDx = m_pGridData->getGridSpacing();
			//for (int i = 0; i < m_streamfunctionGrid.getDimensions().x; i++) {
			//	for (int j = 0; j < m_streamfunctionGrid.getDimensions().y; j++) {
			//		dimensions_t coarseGridIndex(i*scaleFactor, j*scaleFactor);
			//		if (coarseGridIndex.x == 7 && coarseGridIndex.y == 2) {
			//			Vector2 divergencePoint(7.5*coarseGridDx, 2.5*coarseGridDx);
			//			Vector2 currCellCenter(i*m_refinedDx, j*m_refinedDx);
			//			Scalar distance = (divergencePoint - currCellCenter).length()*15;
			//			//Scalar distance = (divergencePoint - currCellCenter).length();
			//			Scalar s = (1 - distance*distance);
			//			s = std::max(0.f, pow(s, 3));
			//			m_streamfunctionGrid(i, j) = s*m_refinedDx*0.001;
			//		}
			//	}
			//}
			Scalar dx = m_pGridData->getGridSpacing();
			m_pVelocityInterpolant = new BilinearStaggeredInterpolant2D<Vector2>(m_pGridData->getVelocityArray(), dx);
			m_pAuxVelocityInterpolant = new BilinearStaggeredInterpolant2D<Vector2>(m_pGridData->getAuxVelocityArray(), dx);
			m_pDensityInterpolant = new BilinearStaggeredInterpolant2D<Scalar>(*m_pGridData->getDensityBuffer().getBufferArray1(), dx);

			//if (m_params.getConvectionMethod() == CPU_ParticleBasedAdvection) {
			//	Scalar dx = m_pGridData->getGridSpacing();

			//	m_pTurbulenceInterpolant = new TurbulentInterpolant2D<Vector2>(m_pGridData->getVelocityArray(), &m_streamfunctionGrid, dx, m_refinedDx);
			//	
			//	m_pStreamfunctionsVecInterpolant = new CubicStreamfunctionInterpolant2D<Vector2>(&m_streamfunctionGrid, m_refinedDx);
			//	m_pStreamfunctionsScalarInterpolant = new BilinearNodalInterpolant2D<Scalar>(m_streamfunctionGrid, m_refinedDx);

			//	ParticlesSampler<Vector2, Array2D> *pSimpleParticleSampler = new ParticlesSampler<Vector2, Array2D>(m_pGridData, 64);

			//	PositionIntegrator<Vector2, Array2D> *pParticlesIntegrator;
			//	pParticlesIntegrator = new RungeKutta2Integrator<Vector2, Array2D>(pSimpleParticleSampler->getParticlesData(), m_pTurbulenceInterpolant, m_refinedDx);

			//	
			//	GridToParticles<Vector2, Array2D> *pGridToParticles = initializeGridToParticles();

			//	TransferKernel<Vector2> *pKernel = NULL;
			//	if (m_params.getFlipParams().velocityTransfer == FLIPParams_t::bilinearWeights) {
			//		pKernel = new BilinearKernel<Vector2>(m_pGridData, dx * 2);
			//	}
			//	else if (m_params.getFlipParams().velocityTransfer == FLIPParams_t::sphWeights) {
			//		pKernel = new SPHKernel<Vector2>(m_pGridData, dx * 2);
			//	}

			//	ParticlesToGrid<Vector2, Array2D> *pParticlesToGrid = new TurbulentParticlesGrid2D(m_pGridData->getDimensions(), pKernel);

			//	auto pParticleBasedAdv = new ParticleBasedAdvection<Vector2, Array2D>(m_pGridData, pSimpleParticleSampler,
			//																				pParticlesIntegrator, pGridToParticles,
			//																				pParticlesToGrid);
			//	//No need for interpolant, initially the velocities are 0
			//	pParticleBasedAdv->addScalarBasedAttribute("streamfunctionFine", m_fineGridVelocities.getDimensions(), m_pStreamfunctionsScalarInterpolant);
			//	typedef ParticlesToGrid<Vector2, Array2D>::accumulatedEntry<Scalar> accumulatedEntryType;
			//	Array2D<accumulatedEntryType> &streamfunctions = pParticlesToGrid->getScalarAttributeArray("streamfunctionFine");
			//	streamfunctions.assign(accumulatedEntryType(0.0f, 0.0f));

			//	pParticleBasedAdv->addScalarBasedAttribute("density", m_pDensityInterpolant);
			//}

			Logger::get() << "[dx dy] = " << m_pGridData->getScaleFactor(0, 0).x << " " << m_pGridData->getScaleFactor(0, 0).y << endl;
			Logger::get() << "[dx/dy] = " << m_pGridData->getScaleFactor(0, 0).x / m_pGridData->getScaleFactor(0, 0).y << endl;
		}

		#pragma endregion 

		#pragma region InitializationMethods
		GridToParticles<Vector2, Array2D>* TurbulenceSolver2D::initializeGridToParticles() {
			m_velocitiesInterpolants.first = m_pVelocityInterpolant;
			m_velocitiesInterpolants.second = m_pAuxVelocityInterpolant;

			static pair<ScalarInterpolant *, ScalarInterpolant *> scalarFieldInterpolants(m_pDensityInterpolant, NULL);

			m_fineVelocitiesInterpolants.first = m_pStreamfunctionsVecInterpolant;
			m_fineVelocitiesInterpolants.second = new CubicStreamfunctionInterpolant2D<Vector2>(&m_auxStreamfunctionGrid, m_refinedDx);
			m_fineVelocitiesInterpolants.first = NULL;
			m_fineVelocitiesInterpolants.second = NULL;

			m_streamfunctionInterpolants.first = m_pStreamfunctionsScalarInterpolant;
			m_streamfunctionInterpolants.second = new BilinearNodalInterpolant2D<Scalar>(m_auxStreamfunctionGrid, m_refinedDx);
			m_streamfunctionInterpolants.first = NULL;

			return new GridToParticlesTurbulent2D(m_velocitiesInterpolants, scalarFieldInterpolants, m_fineVelocitiesInterpolants, m_streamfunctionInterpolants, 0.00);

		}
		#pragma endregion 
		#pragma region Functionalities
		void TurbulenceSolver2D::update(Scalar dt) {
			m_numIterations++;
			//m_totalSimulationTimer.start();


			//if (PhysicsCore<Vector2>::getInstance()->getElapsedTime() < dt) {
			//	applyForces(dt);
			//	enforceBoundaryConditions();
			//	updateDivergents(dt);
			//	enforceBoundaryConditions();
			//	solvePressure();
			//	enforceBoundaryConditions();
			//	project(dt);
			//	enforceBoundaryConditions();
			//	
			//	generateTurbulence(dt);

			//	if (m_pAdvection->getType() == CPU_ParticleBasedAdvection) {
			//		ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
			//		pParticleBasedAdv->getParticlesSampler()->interpolateVelocities(m_pVelocityInterpolant, pParticleBasedAdv->getParticlesData());
			//	}
			//}

			///** Advection */
			//enforceBoundaryConditions();
			//applyForces(dt);
			//
			//m_params.m_advectionTimer.start();
			//
			//m_pAdvection->advect(dt);

			//if (m_pAdvection->getType() == CPU_ParticleBasedAdvection) {
			//	ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
			//	typedef ParticlesToGrid<Vector2, Array2D>::accumulatedEntry<Scalar> accumulatedEntryType;
			//	ParticlesToGrid<Vector2, Array2D> *pParticlesToGrid = pParticleBasedAdv->getParticlesToGrid();
			//	Array2D<accumulatedEntryType> &streamfunctions = pParticlesToGrid->getScalarAttributeArray("streamfunctionFine");
			//	for (int i = 0; i < streamfunctions.getDimensions().x; i++) {
			//		for (int j = 0; j < streamfunctions.getDimensions().y; j++) {
			//			m_auxStreamfunctionGrid(i, j) = streamfunctions(i, j).entry;
			//		}
			//	}
			//}
			//

			//m_params.m_advectionTimer.stop();
			//m_advectionTime = m_params.m_advectionTimer.secondsElapsed();
			//enforceBoundaryConditions();

			///** Solve pressure */
			///** Transfer fine velocities to coarse ones first */
			////streamfunctionsToVelocities();
			//m_params.m_solvePressureTimer.start();
			//updateDivergents(dt);
			//solvePressure();

			//enforceBoundaryConditions();
			//m_params.m_solvePressureTimer.stop();
			//m_solvePressureTime = m_params.m_solvePressureTimer.secondsElapsed();

			///** Project velocity */
			//m_params.m_projectionTimer.start();
			//project(dt);
			//m_params.m_projectionTimer.stop();
			//m_projectionTime = m_params.m_projectionTimer.secondsElapsed();
			//enforceBoundaryConditions();

			//advectDensityField(dt);
			//enforceScalarFieldMarkers();

			//enforceBoundaryConditions();

			//
			//m_pAdvection->postProjectionUpdate(dt);

			//m_pGridData->getDensityBuffer().swapBuffers();
			//m_pGridData->getTemperatureBuffer().swapBuffers();

			//for (int i = 0; i < m_auxStreamfunctionGrid.getDimensions().x; i++) {
			//	for (int j = 0; j < m_auxStreamfunctionGrid.getDimensions().y; j++) {
			//		m_streamfunctionGrid(i, j) = m_auxStreamfunctionGrid(i, j);
			//	}
			//}
			//generateTurbulence(dt);


			//
			//m_params.m_totalSimulationTimer.stop();
			//m_totalSimulationTime = m_params.m_totalSimulationTimer.secondsElapsed();
		}
		#pragma endregion 

		#pragma region TurbulenceStreamfunctions
		
		void TurbulenceSolver2D::generateTurbulence(Scalar dt) {
			Scalar scaleFactor = m_refinedDx/m_pGridData->getGridSpacing();
			Scalar coarseGridDx = m_pGridData->getGridSpacing();
			Scalar maxDivergent = -FLT_MAX;


			for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
					if (m_pGridData->getDivergent(i, j) > maxDivergent) {
						maxDivergent = m_pGridData->getDivergent(i, j);
					}
				}
			}
			for (int i = 1; i < m_streamfunctionGrid.getDimensions().x; i++) {
				for (int j = 1; j < m_streamfunctionGrid.getDimensions().y; j++) {
					Scalar divergent = m_pGridData->getDivergent(i*scaleFactor, j*scaleFactor);
					//streamfunctions(i, j).entry += 0.01*(rand()/RAND_MAX);
					Scalar streamfunction = m_streamfunctionGrid(i, j);
					streamfunction = m_streamfunctionGrid(i, j);
					Scalar randTemp = ((rand() / ((float) RAND_MAX)));
					m_streamfunctionGrid(i, j) += (divergent / maxDivergent)*randTemp*0.00025;
					streamfunction = m_streamfunctionGrid(i, j);
					streamfunction = m_streamfunctionGrid(i, j);
					//m_streamfunctionGrid(i, j) = s*m_refinedDx*0.001;
				}
			}

			
			int invScaleFactor = floor(1 / scaleFactor);
			for (int k = 0; k < 10; k++) {
				Array2D<Scalar> tempArray(m_streamfunctionGrid);
				for (int i = invScaleFactor; i < m_streamfunctionGrid.getDimensions().x - invScaleFactor; i++) {
					for (int j = invScaleFactor; j < m_streamfunctionGrid.getDimensions().y - invScaleFactor; j++) {
						Scalar tempArrayTemp;
						m_streamfunctionGrid(i, j) = tempArray(i, j) + tempArray(i - 1, j)*0.25f + 
														tempArray(i + 1, j)*0.25f + tempArray(i, j + 1)*0.25f + tempArray(i, j - 1)*0.25f;

						tempArrayTemp = m_streamfunctionGrid(i, j);
						tempArrayTemp = m_streamfunctionGrid(i, j);
					}
				}

			}
			

			

			//if (PhysicsCore<Vector2>::getInstance()->getElapsedTime() < dt) {
			//	
			//}
		}

		void TurbulenceSolver2D::updateCoarseStreamfunctionGrid() {
			
		}

		void TurbulenceSolver2D::projectFineGridStreamfunctions(Scalar dt) {

		}

		void TurbulenceSolver2D::updateFineGridVelocities() {
			Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
			Scalar fineGridDx = m_pGridData->getScaleFactor(0, 0).x / (pow(2, m_numDivisions));
			Scalar invNumSubdivis = fineGridDx / dx;
			int numSubdivis = floor(dx / fineGridDx);
			Interpolant<Vector2, Array2D, Vector2> *pInterpolant = new BilinearStaggeredInterpolant2D<Vector2>(m_pGridData->getVelocityArray(), dx);


			for (int i = 1; i < m_fineGridVelocities.getDimensions().x - 1; i++) {
				for (int j = 1; j < m_fineGridVelocities.getDimensions().y - 1; j++) {
					m_fineGridVelocities(i, j).x = pInterpolant->interpolate(Vector2(i, j + 0.5)*fineGridDx).x;
					m_fineGridVelocities(i, j).y = pInterpolant->interpolate(Vector2(i + 0.5, j )*fineGridDx).y;
				}
			}
		}

		void TurbulenceSolver2D::velocitiesToStreamfunctions() {
			//updateCoarseStreamfunctionGrid();
			//cubicInterpolateStreamfunctionVelocities();

			//Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
			//Scalar fineGridDx = m_pGridData->getScaleFactor(0, 0).x / (pow(2, m_numDivisions));
			//Scalar invNumSubdivis = fineGridDx / dx;
			//int numSubdivis = floor(dx / fineGridDx);
			//m_streamfunctionGrid(0, 0) = 0;
			//m_streamfunctionGrid.assign(0);

			///*for (int i = 0; i < m_streamfunctionGrid.getDimensions().x - 1; i++) {
			//for (int j = 0; j < m_streamfunctionGrid.getDimensions().y - 1; j++) {
			//Scalar streamfunctionValue;
			//if (i == 0)
			//m_streamfunctionGrid(i, j + 1) = (-bilinearInterpolation(Vector2(i, j + 0.5)*invNumSubdivis, m_pGridData->getVelocityArray()).x*fineGridDx - m_streamfunctionGrid(i, j));
			//if (j == 0)
			//m_streamfunctionGrid(i + 1, j) = -bilinearInterpolation(Vector2(i + 0.5, j)*invNumSubdivis, m_pGridData->getVelocityArray()).y*fineGridDx + m_streamfunctionGrid(i, j);

			//m_streamfunctionGrid(i + 1, j + 1) = bilinearInterpolation(Vector2(i + 1, j + 0.5)*invNumSubdivis, m_pGridData->getVelocityArray()).x*fineGridDx + m_streamfunctionGrid(i + 1, j);
			//Scalar st1, st2, st3, st4;
			//st1 = m_streamfunctionGrid(i, j + 1);
			//st2 = m_streamfunctionGrid(i + 1, j);
			//st3 = m_streamfunctionGrid(i + 1, j + 1);
			//st4 = 0;
			//}
			//}*/
			//Scalar sumOfDivergents = 0;
			//Scalar sumOfDivergents2 = 0;
			//for (int j = numSubdivis; j < m_streamfunctionGrid.getDimensions().y - numSubdivis; j++) {
			//	for (int i = numSubdivis; i < m_streamfunctionGrid.getDimensions().x - numSubdivis; i++) {

			//		Scalar streamfunctionValue;
			//		if (i == numSubdivis)
			//			m_streamfunctionGrid(i, j + 1) = -(-m_fineGridVelocities(i, j).x*fineGridDx + m_streamfunctionGrid(i, j));
			//		if (j == numSubdivis)
			//			m_streamfunctionGrid(i + 1, j) = -m_fineGridVelocities(i, j).y*fineGridDx + m_streamfunctionGrid(i, j);

			//		m_streamfunctionGrid(i + 1, j + 1) = m_fineGridVelocities(i + 1, j).x*fineGridDx + m_streamfunctionGrid(i + 1, j);

			//		Scalar v1 = m_fineGridVelocities(i + 1, j).x;
			//		Scalar v2 = m_fineGridVelocities(i, j).x;
			//		Scalar v3 = m_fineGridVelocities(i, j + 1).y;
			//		Scalar v4 = m_fineGridVelocities(i, j).y;

			//		Scalar divergent = (m_fineGridVelocities(i + 1, j).x - m_fineGridVelocities(i, j).x) / fineGridDx + (m_fineGridVelocities(i, j + 1).y - m_fineGridVelocities(i, j).y) / fineGridDx;
			//		sumOfDivergents += divergent;

			//		Scalar divergent2 = (bilinearInterpolation(Vector2(i + 1, j + 0.5), m_pGridData->getVelocityArray()).x - bilinearInterpolation(Vector2(i, j + 0.5), m_pGridData->getVelocityArray()).x) / fineGridDx +
			//			(bilinearInterpolation(Vector2(i + 0.5, j + 1.0), m_pGridData->getVelocityArray()).y - bilinearInterpolation(Vector2(i + 0.5, j), m_pGridData->getVelocityArray()).y) / fineGridDx;
			//		sumOfDivergents2 += divergent2;
			//		Scalar st1, st2, st3, st4;
			//		st1 = m_streamfunctionGrid(i, j + 1);
			//		st2 = m_streamfunctionGrid(i + 1, j);
			//		st3 = m_streamfunctionGrid(i + 1, j + 1);
			//		st4 = 0;
			//	}
			//}

		}

		void TurbulenceSolver2D::streamfunctionsToVelocities() {
		//	int subdivisScaleFactor = pow(2, m_numDivisions);
		//	Scalar velocityWeight = 1 / ((Scalar)subdivisScaleFactor);
		//	Scalar fineGridDx = m_pGridData->getScaleFactor(0, 0).x / subdivisScaleFactor;

		//	for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
		//		for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
		//			Vector2 auxVelocity;
		//			vector<Scalar> velocityFaceComponents(subdivisScaleFactor);
		//			Scalar avgVelocityComponent = 0;
		//			//Each cell calculates its left and bottom faces

		//			//Left face
		//			for (int k = 0; k < subdivisScaleFactor; k++) {
		//				velocityFaceComponents[k] = m_streamfunctionGrid(i*subdivisScaleFactor, j*subdivisScaleFactor + 1) - m_streamfunctionGrid(i*subdivisScaleFactor, j*subdivisScaleFactor);
		//				velocityFaceComponents[k] /= -fineGridDx;
		//				avgVelocityComponent += velocityFaceComponents[k] * velocityWeight;
		//			}

		//			auxVelocity.x = avgVelocityComponent;

		//			//Bottom face
		//			avgVelocityComponent = 0;
		//			for (int k = 0; k < subdivisScaleFactor; k++) {
		//				velocityFaceComponents[k] = m_streamfunctionGrid(i*subdivisScaleFactor + 1, j*subdivisScaleFactor) - m_streamfunctionGrid(i*subdivisScaleFactor, j*subdivisScaleFactor);
		//				velocityFaceComponents[k] /= -fineGridDx;
		//				avgVelocityComponent += velocityFaceComponents[k] * velocityWeight;
		//			}
		//			auxVelocity.y = avgVelocityComponent;

		//			Vector2 currAuxVel = m_pGridData->getAuxiliaryVelocity(i, j);
		//			m_pGridData->setAuxiliaryVelocity(auxVelocity, i, j);
		//		}
		//	}
		//}

		//void TurbulenceSolver2D::projectFineGridStreamfunctions(Scalar dt) {
		//	int subdivisScaleFactor = pow(2, m_numDivisions);
		//	Scalar fineGridDx = m_pGridData->getScaleFactor(0, 0).x / subdivisScaleFactor;
		//	Scalar invSubdivis = fineGridDx / m_pGridData->getScaleFactor(0, 0).x;

		//	Array2D<Vector2> streamfunctionFineVelocities(m_streamfunctionGrid.getDimensions());
		//	/**Apply pressure gradients to all streamfunction-based velocities*/
		//	for (int i = 1; i < m_streamfunctionGrid.getDimensions().x - 1; i++) {
		//		for (int j = 1; j < m_streamfunctionGrid.getDimensions().y - 1; j++) {
		//			Vector2 auxVelocity;
		//			auxVelocity.x = m_streamfunctionGrid(i, (j + 1)) - m_streamfunctionGrid(i, j);
		//			auxVelocity.x /= fineGridDx;

		//			auxVelocity.y = m_streamfunctionGrid((i + 1), j) - m_streamfunctionGrid(i, j);
		//			auxVelocity.y /= fineGridDx;

		//			Scalar pressureMinusX = interpolateScalar(Vector2(i - 0.5, j + 0.5)*invSubdivis, m_pGridData->getPressureArray());
		//			Scalar pressureMinusY = interpolateScalar(Vector2(i + 0.5, j - 0.5)*invSubdivis, m_pGridData->getPressureArray());
		//			Scalar pressure = interpolateScalar(Vector2(i + 0.5, j + 0.5)*invSubdivis, m_pGridData->getPressureArray());
		//			Scalar pressureGradX = (pressure - pressureMinusX) / fineGridDx;
		//			Scalar pressureGradY = (pressure - pressureMinusY) / fineGridDx;

		//			auxVelocity.x -= pressureGradX*dt;
		//			auxVelocity.y -= pressureGradY*dt;
		//			streamfunctionFineVelocities(i, j) = auxVelocity;
		//		}
		//	}

		//	/**Correct streamfunction values with the updated velocities*/
		//	m_streamfunctionGrid(0, 0) = 0;

		//	for (int i = 0; i < m_streamfunctionGrid.getDimensions().x - 1; i++) {
		//		for (int j = 0; j < m_streamfunctionGrid.getDimensions().y - 1; j++) {
		//			if (i == 0)
		//				m_streamfunctionGrid(i, j + 1) = -(-streamfunctionFineVelocities(i, j).x*fineGridDx + m_streamfunctionGrid(i, j));
		//			if (j == 0)
		//				m_streamfunctionGrid(i + 1, j) = -streamfunctionFineVelocities(i, j).y*fineGridDx + m_streamfunctionGrid(i, j);

		//			m_streamfunctionGrid(i + 1, j + 1) = streamfunctionFineVelocities(i + 1, j).x*fineGridDx + m_streamfunctionGrid(i + 1, j);
		//		}
		//	}
		}
		#pragma endregion 
	}
}