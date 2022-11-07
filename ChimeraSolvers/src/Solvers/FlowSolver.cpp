#include "Solvers/FlowSolver.h"

namespace Chimera {
	namespace Solvers {

		#pragma region SolvingFunctions
		template<>
		void FlowSolver<Vector2, Array2D>::enforceBoundaryConditions() {
			GridData2D *pGridData2D = m_pGrid->getGridData2D();
			for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
				m_boundaryConditions[i]->applyBoundaryCondition(pGridData2D, m_params.solverType);
			}
		}
		template<>
		void FlowSolver<Vector3, Array3D>::enforceBoundaryConditions() {
			GridData3D *pGridData3D = m_pGrid->getGridData3D();
			for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
				m_boundaryConditions[i]->applyBoundaryCondition(pGridData3D, m_params.solverType);
			}
		}


		#pragma region InitializationFunctions
		template<class VectorT, template <class> class ArrayType>
		void FlowSolver<VectorT, ArrayType>::initalizeCGSolver() {
			m_pPoissonSolver = new ConjugateGradient(*m_params.pPoissonSolverParams, m_pPoissonMatrix);
			m_pPoissonSolver->setPeriodic(m_pGrid->isPeriodic());
		}

		template<class VectorType, template <class> class ArrayType>
		void FlowSolver<VectorType, ArrayType>::initializeEigenCGSolver() {
			m_pPoissonSolver = new EigenConjugateGradient(*m_params.pPoissonSolverParams, m_pPoissonMatrix);
			m_pPoissonSolver->setPeriodic(m_pGrid->isPeriodic());
		}

		template<>
		void FlowSolver<Vector2, Array2D>::initializeMultigridSolver() {
			//Multigrid::solverParams_t *solverParams;
			//GridData2D *pGridData2D = m_pGrid->getGridData2D();
			//solverParams = (Multigrid::solverParams_t *) m_params.getPressureSolverParams().getSpecificSolverParams();
			//if (m_params.getDiscretizationMethod() == finiteDifferenceMethod) { //regular grid
			//	solverParams->gridRegularSpacing = pGridData2D->getScaleFactor(0, 0).x;
			//	//solverParams->pSolidCells = m_pGrid->getSolidMarkers();
			//	//solverParams->pBoundaryCells = m_pGrid->getBoundaryCells();
			//	for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
			//		if (m_boundaryConditions[i]->getType() == Outflow) {
			//			solverParams->m_boundaries[m_boundaryConditions[i]->getLocation()] = PoissonSolver::dirichlet;
			//		}
			//	}
			//}
			//else {
			//	solverParams->pCellsVolumes = (Scalar *)pGridData2D->getVolumeArray().getRawDataPointer();
			//	solverParams->pCellsAreas = pGridData2D->getScaleFactorsArray().getRawDataPointer();
			//	solverParams->pSolidCells = NULL;
			//	solverParams->pBoundaryCells = NULL;
			//	if (m_params.isChimeraGrid()) {
			//		if (m_pGrid->isPeriodic()) {
			//			solverParams->m_boundaries[West] = PoissonSolver::dirichlet;
			//			solverParams->m_boundaries[East] = PoissonSolver::dirichlet;
			//			solverParams->m_boundaries[South] = PoissonSolver::neumann;
			//			solverParams->m_boundaries[North] = PoissonSolver::dirichlet;
			//		}
			//		//solverParams->restrictInitialSolution = true;
			//	}
			//	else if (m_pGrid->isPeriodic()) {
			//		solverParams->m_boundaries[West] = PoissonSolver::dirichlet;
			//		solverParams->m_boundaries[East] = PoissonSolver::dirichlet;
			//		solverParams->m_boundaries[South] = PoissonSolver::neumann;
			//		solverParams->m_boundaries[North] = PoissonSolver::neumann;
			//	}
			//}
			//Multigrid *pMultigrid = NULL;
			//if (m_dimensions.z == 0) {
			//	pMultigrid = new Multigrid2D(m_pPoissonMatrix, *solverParams);
			//	pMultigrid->setPeriodic(m_pGrid->isPeriodic());
			//	pMultigrid->initializeMultigrid(*solverParams, (Scalar *)pGridData2D->getDivergentArray().getRawDataPointer(), (Scalar *)pGridData2D->getPressureArray().getRawDataPointer());
			//}
			//m_pPoissonSolver = pMultigrid;
		}

		template<>
		void FlowSolver<Vector3, Array3D>::initializeMultigridSolver() {
			//Multigrid::solverParams_t *solverParams;
			//GridData3D *pGridData3D = m_pGrid->getGridData3D();
			//solverParams = (Multigrid::solverParams_t *) m_params.getPressureSolverParams().getSpecificSolverParams();
			//if (m_params.getDiscretizationMethod() == finiteDifferenceMethod) { //regular grid
			//	solverParams->gridRegularSpacing = pGridData3D->getScaleFactor(0, 0, 0).x;
			//	//solverParams->pSolidCells = m_pGrid->getSolidMarkers();
			//	//solverParams->pBoundaryCells = m_pGrid->getBoundaryCells();
			//	for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
			//		if (m_boundaryConditions[i]->getType() == Outflow) {
			//			solverParams->m_boundaries[m_boundaryConditions[i]->getLocation()] = PoissonSolver::dirichlet;
			//		}
			//	}
			//}
			//else {
			//	solverParams->pCellsVolumes = (Scalar *)pGridData3D->getVolumeArray().getRawDataPointer();
			//	solverParams->pCellsAreas = pGridData3D->getScaleFactorsArray().getRawDataPointer();
			//	solverParams->pSolidCells = NULL;
			//	solverParams->pBoundaryCells = NULL;
			//	if (m_params.isChimeraGrid()) {
			//		if (m_pGrid->isPeriodic()) {
			//			solverParams->m_boundaries[West] = PoissonSolver::dirichlet;
			//			solverParams->m_boundaries[East] = PoissonSolver::dirichlet;
			//			solverParams->m_boundaries[South] = PoissonSolver::neumann;
			//			solverParams->m_boundaries[North] = PoissonSolver::dirichlet;
			//			solverParams->m_boundaries[Front] = PoissonSolver::dirichlet;
			//			solverParams->m_boundaries[Back] = PoissonSolver::dirichlet;
			//		}
			//		//solverParams->restrictInitialSolution = true;
			//	}
			//	else if (m_pGrid->isPeriodic()) {
			//		solverParams->m_boundaries[West] = PoissonSolver::dirichlet;
			//		solverParams->m_boundaries[East] = PoissonSolver::dirichlet;
			//		solverParams->m_boundaries[South] = PoissonSolver::neumann;
			//		solverParams->m_boundaries[North] = PoissonSolver::neumann;
			//		solverParams->m_boundaries[Front] = PoissonSolver::neumann;
			//		solverParams->m_boundaries[Back] = PoissonSolver::neumann;
			//	}
			//}
			//Multigrid *pMultigrid = NULL;

			//pMultigrid = new Multigrid3D(m_pPoissonMatrix, *solverParams);
			//pMultigrid->setPeriodic(m_pGrid->isPeriodic());
			//pMultigrid->initializeMultigrid(*solverParams, (Scalar *)pGridData3D->getDivergentArray().getRawDataPointer(), (Scalar *)pGridData3D->getPressureArray().getRawDataPointer());

			//m_pPoissonSolver = pMultigrid;
		}

		template<>
		void FlowSolver<Vector2, Array2D>::initializeGaussSeidelSolver() {
			GaussSeidel *pGaussSeidel = new GaussSeidel(*m_params.pPoissonSolverParams, m_pPoissonMatrix);
			//Set divergences and extra pressures for solvers:
			if (m_params.pPoissonSolverParams->pSpecialDivergents != nullptr && m_params.pPoissonSolverParams->pSpecialPressures != nullptr) {
				pGaussSeidel->setCutCellsDivergence(&m_params.pPoissonSolverParams->pSpecialDivergents->at(0),
					m_pGrid->getDimensions().y * m_pGrid->getDimensions().x, m_pPoissonMatrix->getNumberAdditionalCells());
				//if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
				//	
				//}
				pGaussSeidel->setCutCellsPressure(&m_params.pPoissonSolverParams->pSpecialPressures->at(0));
			}

			m_pPoissonSolver = pGaussSeidel;
		}

		template<>
		void FlowSolver<Vector3, Array3D>::initializeGaussSeidelSolver() {
			//GaussSeidel::solverParams_t *solverParams;
			//solverParams = (GaussSeidel::solverParams_t *) m_params.getPressureSolverParams().getSpecificSolverParams();
			//GridData3D *pGridData3D = m_pGrid->getGridData3D();
			////solverParams->pDisabledCells = m_pGrid->getSolidMarkers();
			////solverParams->pBoundaryCells = m_pGrid->getBoundaryCells();
			//if (m_params.getDiscretizationMethod() == finiteVolumeMethod) {
			//	solverParams->pCellsVolumes = (Scalar *)pGridData3D->getVolumeArray().getRawDataPointer();
			//	if (m_pGrid->isPeriodic()) {
			//		solverParams->northBoundary = PoissonSolver::dirichlet;
			//		solverParams->southBoundary = PoissonSolver::neumann;
			//		solverParams->eastBoundary = PoissonSolver::dirichlet;
			//		solverParams->westBoundary = PoissonSolver::dirichlet;
			//	}
			//	else {
			//		solverParams->northBoundary = PoissonSolver::dirichlet;
			//		solverParams->southBoundary = PoissonSolver::dirichlet;
			//		solverParams->eastBoundary = PoissonSolver::dirichlet;
			//		solverParams->westBoundary = PoissonSolver::dirichlet;
			//	}
			//}
			//else if (m_pGrid->isSubGrid()) {
			//	solverParams->northBoundary = PoissonSolver::dirichlet;
			//	solverParams->southBoundary = PoissonSolver::dirichlet;
			//	solverParams->eastBoundary = PoissonSolver::dirichlet;
			//	solverParams->westBoundary = PoissonSolver::dirichlet;
			//}
			//else {
			//	if (m_params.isChimeraGrid()) {
			//		solverParams->dx = pGridData3D->getScaleFactor(0, 0, 0).x;
			//		solverParams->northBoundary = PoissonSolver::neumann;
			//		solverParams->southBoundary = PoissonSolver::neumann;
			//		solverParams->eastBoundary = PoissonSolver::dirichlet;
			//		solverParams->westBoundary = PoissonSolver::neumann;
			//	}
			//	else {
			//		solverParams->dx = pGridData3D->getScaleFactor(0, 0, 0).x;
			//		solverParams->northBoundary = PoissonSolver::neumann;
			//		solverParams->southBoundary = PoissonSolver::neumann;
			//		solverParams->eastBoundary = PoissonSolver::dirichlet;
			//		solverParams->westBoundary = PoissonSolver::neumann;
			//	}
			//}
			//m_pPoissonSolver = new GaussSeidel(m_pPoissonMatrix, *solverParams);
			GaussSeidel *pGaussSeidel = new GaussSeidel(*m_params.pPoissonSolverParams, m_pPoissonMatrix);

			//Set divergences and extra pressures for solvers:
			if (m_params.pPoissonSolverParams->pSpecialDivergents != nullptr && m_params.pPoissonSolverParams->pSpecialPressures != nullptr) {
				pGaussSeidel->setCutCellsDivergence(&m_params.pPoissonSolverParams->pSpecialDivergents->at(0), m_pPoissonMatrix->getNumRows(), m_pPoissonMatrix->getNumberAdditionalCells());
				pGaussSeidel->setCutCellsPressure(&m_params.pPoissonSolverParams->pSpecialPressures->at(0));
			}

			m_pPoissonSolver = pGaussSeidel;

		}

		template<class VectorT, template <class> class ArrayType>
		AdvectionBase * FlowSolver<VectorT, ArrayType>::initializeAdvectionClass() {
			if (m_params.pAdvectionParams->advectionCategory == LagrangianAdvection) {
				ParticleBasedAdvection<VectorT, ArrayType>::params_t *pPbaParams = dynamic_cast<ParticleBasedAdvection<VectorT, ArrayType>::params_t *>(m_params.pAdvectionParams);
				if (pPbaParams == nullptr) {
					throw exception("Invalid custom parameters configuration for initializing advection class.");
				}
				ParticleBasedAdvection<VectorT, ArrayType> *pParticleBasedAdvection = new ParticleBasedAdvection<VectorT, ArrayType>(pPbaParams, m_pVelocityInterpolant, m_pGrid->getGridData());
				pParticleBasedAdvection->getParticlesSampler()->setBoundaryConditions(m_boundaryConditions);
				return pParticleBasedAdvection;
			}
			else if (m_params.pAdvectionParams->advectionCategory == EulerianAdvection) {
				PositionIntegrator<VectorT, ArrayType> *pPositionIntegrator = nullptr;
				Scalar dx = m_pGrid->getGridData()->getGridSpacing();
				switch (m_params.pAdvectionParams->integrationMethod)
				{
					case forwardEuler:
						pPositionIntegrator = new ForwardEulerIntegrator<VectorT, ArrayType>(nullptr, m_pVelocityInterpolant, dx);
					break;

					case RungeKutta_2:
						pPositionIntegrator = new RungeKutta2Integrator<VectorT, ArrayType>(nullptr, m_pVelocityInterpolant, dx);
					break;
				}
				
				if (m_params.pAdvectionParams->gridBasedAdvectionMethod == SemiLagrangian) {
					return new SemiLagrangianAdvection<VectorT, ArrayType>(*m_params.pAdvectionParams, m_pGrid->getGridData(), pPositionIntegrator,
																	m_pVelocityInterpolant, m_pDensityInterpolant, m_pTemperatureInterpolant);
				} else if (m_params.pAdvectionParams->gridBasedAdvectionMethod == MacCormack) {
					return new MacCormackAdvection<VectorT, ArrayType>(*m_params.pAdvectionParams, m_pGrid->getGridData(), pPositionIntegrator,
																	m_pVelocityInterpolant, m_pDensityInterpolant, m_pTemperatureInterpolant);
				}
			}
		}
		#pragma endregion
		#pragma region UpdateFunctions
		template<>
		void FlowSolver<Vector2, Array2D>::updateFineGridDivergence(Interpolant<Vector2, Array2D, Vector2> *pInterpolant) {
			Scalar dx = m_pGrid->getGridData2D()->getGridSpacing();
			Scalar fineDx = m_pGrid->getGridData2D()->getFineGridScalarFieldDx();
			int numSubCells = dx / fineDx;
			dimensions_t fineGridDimensions(m_pGrid->getGridData2D()->getDimensions().x*numSubCells, m_pGrid->getGridData2D()->getDimensions().y*numSubCells);

			for (int i = numSubCells * 3; i < fineGridDimensions.x - numSubCells * 3; i++) {
				for (int j = numSubCells * 3; j < fineGridDimensions.y - numSubCells * 3; j++) {
					Vector2 centroid((i + 0.5)*fineDx, (j + 0.5)*fineDx);
					Vector2 centroidCoarse(centroid.x / dx, centroid.y / dx);
					dimensions_t coarseGridIndex(floor(centroidCoarse.x), floor(centroidCoarse.y));

					Scalar deltaError = 1e-5;

					Scalar divDx = fineDx*0.5 - deltaError;

					Vector2 velTop = pInterpolant->interpolate(Vector2(centroid.x, centroid.y + divDx));
					Vector2 velBottom = pInterpolant->interpolate(Vector2(centroid.x, centroid.y - divDx));
					Vector2 velLeft = pInterpolant->interpolate(Vector2(centroid.x + divDx, centroid.y));
					Vector2 velRight = pInterpolant->interpolate(Vector2(centroid.x - divDx, centroid.y));

					Scalar currDiv = (velTop.y - velBottom.y) / (2 * divDx) + (velLeft.x - velRight.x) / (2 * divDx);
					m_pGrid->getGridData2D()->setFineGridScalarValue(abs(currDiv), i, j);
				}
			}
		}

		template<>
		void FlowSolver<Vector3, Array3D>::updateFineGridDivergence(Interpolant<Vector3, Array3D, Vector3> *pInterpolant) {
			
		}

		template<>
		void FlowSolver<Vector2, Array2D>::updateDivergents(Scalar dt) {
			Scalar dx = m_pGrid->getGridData2D()->getScaleFactor(0, 0).x;
			Scalar dx2 = dx*dx;
			for (int i = 1; i < m_dimensions.x - 1; i++) {
				for (int j = 1; j < m_dimensions.y - 1; j++) {
					Scalar divergent = 0;
					divergent = -calculateFluxDivergent(i, j)*dx2 / dt;
					m_pGrid->getGridData2D()->setDivergent(divergent, i, j);
				}
			}
		}

		template<>
		void FlowSolver<Vector3, Array3D>::updateDivergents(Scalar dt) {
			bool allNeumannBoundaries = true;
			for (int i = 0; i < m_boundaryConditions.size(); i++) {
				if (m_boundaryConditions[i]->getType() == Outflow) {
					allNeumannBoundaries = false;
				}
			}

			Scalar dx = m_pGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
			Scalar dx2 = dx*dx;
			for (int i = 1; i < m_dimensions.x - 1; i++) {
				for (int j = 1; j < m_dimensions.y - 1; j++) {
					for (int k = 1; k < m_dimensions.z - 1; k++) {
						Scalar divergent = -calculateFluxDivergent(i, j, k);
						m_pGrid->getGridData3D()->setDivergent(divergent*dx2 / dt, i, j, k);
					}
				}
			}
		}

		template<>
		void FlowSolver<Vector2, Array2D>::updatePostProjectionDivergence() {
			DoubleScalar meanGridDivergence = 0.0;
			DoubleScalar maxDivergence = 0;
			int totalCells = 0;
			GridData2D *pGridData2D = m_pGrid->getGridData2D();
			for (int i = 1; i < pGridData2D->getDimensions().x - 1; i++) {
				for (int j = 1; j < pGridData2D->getDimensions().y - 1; j++) {
					Scalar currDivergence = calculateFinalDivergent(i, j);
					if (abs(currDivergence) > maxDivergence) {
						maxDivergence = abs(currDivergence);
					}
					meanGridDivergence += currDivergence;
					++totalCells;
				}
			}
			m_totalDivergent = meanGridDivergence;
			m_meanDivergent = meanGridDivergence / totalCells;
		}
		template<>
		void FlowSolver<Vector3, Array3D>::updatePostProjectionDivergence() {

		}
		#pragma endregion

		template class FlowSolver<Vector2, Array2D>;
		template class FlowSolver<Vector3, Array3D>;
	}

 }