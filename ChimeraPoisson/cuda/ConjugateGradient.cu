#include "Poisson/ConjugateGradient.h"


namespace Chimera {

	namespace Poisson {

		/************************************************************************/
		/* Cuda scalar field                                                    */
		/************************************************************************/
		ConjugateGradient::CudaScalarField::CudaScalarField(const params_t &solverParams, const dimensions_t &gridDimensions)  {
			int totalSize;
			if(gridDimensions.z == 0) {
				totalSize = gridDimensions.x*gridDimensions.y;
			} else {
				totalSize = gridDimensions.x*gridDimensions.y*gridDimensions.z;
			}
			if(solverParams.solveThinBoundaries) {
				totalSize += solverParams.numberOfSpecialCells;
			}
			if (solverParams.platform == PlataformGPU) {
				pDevicePressure = new cusp::array1d<Scalar, cusp::device_memory>(totalSize);
				pDeviceFlux = new cusp::array1d<Scalar, cusp::device_memory>(totalSize);
			}
			else {
				pDeviceFlux = pDevicePressure = nullptr;
			}
			
			pPressure = new cusp::array1d<Scalar, cusp::host_memory>(totalSize);
			pFlux = new cusp::array1d<Scalar, cusp::host_memory>(totalSize);
		}
		
		/************************************************************************/
		/* Resizing scalar fields                                               */
		/************************************************************************/
		void ConjugateGradient::resizeScalarFields() {
			int totalSize;
			if(m_dimensions.z == 0) {
				totalSize = m_dimensions.x*m_dimensions.y;
			} else {
				totalSize = m_dimensions.x*m_dimensions.y*m_dimensions.z;
			}
			if(m_params.solveThinBoundaries) {
				totalSize += m_params.numberOfSpecialCells;
			}
			if(m_pCudaScalarField->pDevicePressure != NULL && m_params.platform == PlataformGPU)
				m_pCudaScalarField->pDevicePressure->resize(totalSize);
			else if(m_params.platform == PlataformGPU)
				m_pCudaScalarField->pDevicePressure = new cusp::array1d<Scalar, cusp::device_memory>(totalSize);
			if(m_pCudaScalarField->pDeviceFlux != NULL && m_params.platform == PlataformGPU)
				m_pCudaScalarField->pDeviceFlux->resize(totalSize);
			else if (m_params.platform == PlataformGPU)
				m_pCudaScalarField->pDeviceFlux = new cusp::array1d<Scalar, cusp::device_memory>(totalSize);

			if(m_pCudaScalarField->pPressure != NULL)
				m_pCudaScalarField->pPressure->resize(totalSize);
			else
				m_pCudaScalarField->pPressure = new cusp::array1d<Scalar, cusp::host_memory>(totalSize);
			
			if(m_pCudaScalarField->pFlux != NULL)
				m_pCudaScalarField->pFlux->resize(totalSize);
			else
				m_pCudaScalarField->pFlux = new cusp::array1d<Scalar, cusp::host_memory>(totalSize);
			
		}
		void ConjugateGradient::reinitializePreconditioners() {
			switch(m_preconditioner) {
				case Diagonal:
					delete m_pDiagonalPrecond; 
					if(m_pPoissonMatrix->supportGPU()) {
						delete m_pDeviceDiagonalPrecond;
					}
					break;
				case AINV:
					delete m_pAinvPrecond;
					if(m_pPoissonMatrix->supportGPU()) {
						delete m_pDeviceAinvPrecond;
					}
					break;
				case SmoothedAggregation:
					delete m_pSmoothedPrecond;
					if(m_pPoissonMatrix->supportGPU()) {
						delete m_pDeviceSmoothedPrecond;
					}
					break;
			}
			initializePreconditioners();
		}

		/************************************************************************/
		/* 2D copying functions                                                 */
		/************************************************************************/
		void ConjugateGradient::copyToCudaScalarField(const Array2D<Scalar> *pRhs, bool copyToGpu) {
			int linearIndex = 0;

			for(int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
				for(int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
					linearIndex = m_pPoissonMatrix->getRowIndex(i, j);
					int padI;
					if(m_pPoissonMatrix->isPeriodic())
						padI = i;
					else 
						padI = i + 1;

					(*m_pCudaScalarField->pFlux)[linearIndex] = (*pRhs)(padI, j + 1);
					(*m_pCudaScalarField->pPressure)[linearIndex] = 0.0f;
				}
			}

			if(m_params.solveThinBoundaries) {
				for(int i = 0; i < m_params.numberOfSpecialCells; i++) {
					linearIndex = m_pPoissonMatrix->getRowIndex(m_pPoissonMatrix->getDimensions().x - 1,  m_pPoissonMatrix->getDimensions().y - 1) + 1;
					linearIndex += i;

					(*m_pCudaScalarField->pFlux)[linearIndex] = m_params.pSpecialDivergents->at(i);
					(*m_pCudaScalarField->pPressure)[linearIndex] = 0.0f;
				}
			}

			if(m_params.platform == PlataformGPU) {
				*m_pCudaScalarField->pDeviceFlux = *m_pCudaScalarField->pFlux;
				*m_pCudaScalarField->pDevicePressure = *m_pCudaScalarField->pPressure;
			}
		}

		void ConjugateGradient::copyFromCudaScalarField(Array2D<Scalar> *pResult, bool copyFromGpu) {
			int linearIndex = 0;

			if (m_params.platform == PlataformGPU) 
				*m_pCudaScalarField->pPressure = *m_pCudaScalarField->pDevicePressure;
			
			for(int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
				for(int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
					linearIndex = m_pPoissonMatrix->getRowIndex(i, j);
					int padI;
					if(m_pPoissonMatrix->isPeriodic())
						padI = i;
					else 
						padI = i + 1;

					(*pResult)(padI, j + 1) = (*m_pCudaScalarField->pPressure)[linearIndex];
				}
			}

			if(m_params.solveThinBoundaries) {
				for(int i = 0; i < m_params.numberOfSpecialCells; i++) {
					linearIndex = m_pPoissonMatrix->getRowIndex(m_pPoissonMatrix->getDimensions().x - 1,  m_pPoissonMatrix->getDimensions().y - 1) + 1;
					linearIndex += i;
					(*m_params.pSpecialPressures)[i] = (*m_pCudaScalarField->pPressure)[linearIndex];
				}
			}

		}
		
		/************************************************************************/
		/* 3D copying functions                                                 */
		/************************************************************************/
		void ConjugateGradient::copyToCudaScalarField(const Array3D<Scalar> *pRhs, bool copyToGPU) {
			int linearIndex = 0;

			for(int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
				for(int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
					for(int k = 0; k < m_pPoissonMatrix->getDimensions().z; k++) {
						linearIndex = m_pPoissonMatrix->getRowIndex(i, j, k);
						(*m_pCudaScalarField->pFlux)[linearIndex] = (*pRhs)(i + 1, j + 1, k + 1);
						//cout << (*m_pCudaScalarField->pFlux)[linearIndex] << endl; 
						(*m_pCudaScalarField->pPressure)[linearIndex] = 0.0f;
					}
				}
			}
			if(m_params.solveThinBoundaries) {
				for(int i = 0; i < m_params.numberOfSpecialCells; i++) {
					linearIndex = m_pPoissonMatrix->getRowIndex(m_pPoissonMatrix->getDimensions().x - 1,  m_pPoissonMatrix->getDimensions().y - 1, 
																m_pPoissonMatrix->getDimensions().z - 1) + 1;
					linearIndex += i;
					(*m_pCudaScalarField->pFlux)[linearIndex] = m_params.pSpecialDivergents->at(i);
					//cout << (*m_pCudaScalarField->pFlux)[linearIndex] << endl;
					(*m_pCudaScalarField->pPressure)[linearIndex] = 0.0f;
				}
			}
			if (m_params.platform == PlataformGPU) {
				*m_pCudaScalarField->pDeviceFlux = *m_pCudaScalarField->pFlux;
				*m_pCudaScalarField->pDevicePressure = *m_pCudaScalarField->pPressure;
			}
		}

		void ConjugateGradient::copyFromCudaScalarField(Array3D<Scalar> *pResult, bool copyFromGPU) {
			int linearIndex = 0;
			
			if (m_params.platform == PlataformGPU) 
				*m_pCudaScalarField->pPressure = *m_pCudaScalarField->pDevicePressure;

			for(int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
				for(int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
					for(int k = 0; k < m_pPoissonMatrix->getDimensions().z; k++) {
						linearIndex = m_pPoissonMatrix->getRowIndex(i, j,  k);
						(*pResult)(i + 1, j + 1, k + 1) = (*m_pCudaScalarField->pPressure)[linearIndex];
					}
				}
			}

			if(m_params.solveThinBoundaries) {
				for(int i = 0; i < m_params.numberOfSpecialCells; i++) {
					linearIndex = m_pPoissonMatrix->getRowIndex(m_pPoissonMatrix->getDimensions().x - 1,  m_pPoissonMatrix->getDimensions().y - 1, 
																m_pPoissonMatrix->getDimensions().z - 1) + 1;
					linearIndex += i;
					
					(*m_params.pSpecialPressures)[i] = (*m_pCudaScalarField->pPressure)[linearIndex];
				}
			}
		}
		/************************************************************************/
		/* Conjugate gradient                                                   */
		/************************************************************************/
		void ConjugateGradient::initializePreconditioners() {
			switch(m_preconditioner) {
				case Diagonal:
					if(m_params.solveThinBoundaries)
						m_pDiagonalPrecond = new cusp::precond::diagonal<Scalar, cusp::host_memory>(*(m_pPoissonMatrix->getCPUDataHyb()));
					else
						m_pDiagonalPrecond = new cusp::precond::diagonal<Scalar, cusp::host_memory>(*(m_pPoissonMatrix->getCPUData()));

					if(m_pPoissonMatrix->supportGPU()) {
						if(m_params.solveThinBoundaries)
							m_pDeviceDiagonalPrecond = new cusp::precond::diagonal<Scalar, cusp::device_memory>(*(m_pPoissonMatrix->getGPUDataHyb()));
						else
							m_pDeviceDiagonalPrecond = new cusp::precond::diagonal<Scalar, cusp::device_memory>(*(m_pPoissonMatrix->getGPUData()));
					}
					break;

				case AINV:
					if(m_params.solveThinBoundaries)
						m_pAinvPrecond = new cusp::precond::bridson_ainv<Scalar, cusp::host_memory>(*(m_pPoissonMatrix->getCPUDataHyb()));
					else
						m_pAinvPrecond = new cusp::precond::bridson_ainv<Scalar, cusp::host_memory>(*(m_pPoissonMatrix->getCPUData()));
					

					if(m_pPoissonMatrix->supportGPU()) {
						if(m_params.solveThinBoundaries)
							m_pDeviceAinvPrecond = new cusp::precond::bridson_ainv<Scalar, cusp::device_memory>(*(m_pPoissonMatrix->getGPUDataHyb()));
						else
							m_pDeviceAinvPrecond = new cusp::precond::bridson_ainv<Scalar, cusp::device_memory>(*(m_pPoissonMatrix->getGPUData()));
					}
					break;

				case SmoothedAggregation:
					if(m_params.solveThinBoundaries)
						m_pSmoothedPrecond = new cusp::precond::aggregation::smoothed_aggregation<Integer, Scalar, cusp::host_memory>(*(m_pPoissonMatrix->getCPUDataHyb()));
					else
						m_pSmoothedPrecond = new cusp::precond::aggregation::smoothed_aggregation<Integer, Scalar, cusp::host_memory>(*(m_pPoissonMatrix->getCPUData()));
					if(m_pPoissonMatrix->supportGPU()) {
						if(m_params.solveThinBoundaries)
							m_pDeviceSmoothedPrecond = new cusp::precond::aggregation::smoothed_aggregation<Integer, Scalar, cusp::device_memory>(*(m_pPoissonMatrix->getGPUDataHyb()));
						else
							m_pDeviceSmoothedPrecond = new cusp::precond::aggregation::smoothed_aggregation<Integer, Scalar, cusp::device_memory>(*(m_pPoissonMatrix->getGPUData()));
					}
					break;
			}
		}
		


		bool ConjugateGradient::solveCPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) {
		/*	if(m_pMonitor != NULL)
				delete m_pMonitor;*/
		
			if(m_dimensions.z == 0)
				copyToCudaScalarField(dynamic_cast<const Array2D<Scalar> *>(pRhs), false);
			else
				copyToCudaScalarField(dynamic_cast<const Array3D<Scalar> *>(pRhs), false);

			m_pMonitor = new cusp::default_monitor<Scalar>(*m_pCudaScalarField->pFlux, m_params.maxIterations, m_params.maxResidual);

			switch(m_preconditioner) {
				case NoPreconditioner:
					if(m_params.solveThinBoundaries)
						cusp::krylov::cg(*(m_pPoissonMatrix->getCPUDataHyb()), *m_pCudaScalarField->pPressure, *m_pCudaScalarField->pFlux, *m_pMonitor);
					else
						cusp::krylov::cg(*(m_pPoissonMatrix->getCPUData()), *m_pCudaScalarField->pPressure, *m_pCudaScalarField->pFlux, *m_pMonitor);
				break;

				case Diagonal:
					if(m_params.solveThinBoundaries)
						cusp::krylov::cg(*(m_pPoissonMatrix->getCPUDataHyb()),*m_pCudaScalarField->pPressure, *m_pCudaScalarField->pFlux, *m_pMonitor, *m_pDiagonalPrecond);
					else
						cusp::krylov::cg(*(m_pPoissonMatrix->getCPUData()), *m_pCudaScalarField->pPressure, *m_pCudaScalarField->pFlux, *m_pMonitor, *m_pDiagonalPrecond);
					break;

				case AINV:
					if(m_params.solveThinBoundaries)
						cusp::krylov::cg(*(m_pPoissonMatrix->getCPUDataHyb()), *m_pCudaScalarField->pPressure, *m_pCudaScalarField->pFlux, *m_pMonitor, *m_pAinvPrecond);
					else
						cusp::krylov::cg(*(m_pPoissonMatrix->getCPUData()), *m_pCudaScalarField->pPressure, *m_pCudaScalarField->pFlux, *m_pMonitor, *m_pAinvPrecond);
					break;

				case SmoothedAggregation:
					if(m_params.solveThinBoundaries)
						cusp::krylov::cg(*(m_pPoissonMatrix->getCPUDataHyb()), *m_pCudaScalarField->pPressure, *m_pCudaScalarField->pFlux, *m_pMonitor, *m_pSmoothedPrecond);
					else
						cusp::krylov::cg(*(m_pPoissonMatrix->getCPUData()), *m_pCudaScalarField->pPressure, *m_pCudaScalarField->pFlux, *m_pMonitor, *m_pSmoothedPrecond);

					break;
			}

			m_lastResidual = m_pMonitor->residual_norm();
			m_numIterations = m_pMonitor->iteration_count();

			if(m_dimensions.z == 0)
				copyFromCudaScalarField(dynamic_cast<Array2D<Scalar> *>(pResult), false);
			else
				copyFromCudaScalarField(dynamic_cast<Array3D<Scalar> *>(pResult), false);

			if(m_pMonitor->converged())
				return true;
			else 
				return false;
		}


		bool ConjugateGradient::solveGPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) {
			if(m_dimensions.z == 0)
				copyToCudaScalarField(dynamic_cast<const Array2D<Scalar> *>(pRhs));
			else
				copyToCudaScalarField(dynamic_cast<const Array3D<Scalar> *>(pRhs));

			m_pMonitor = new cusp::default_monitor<Scalar>(*m_pCudaScalarField->pDeviceFlux, m_params.maxIterations, m_params.maxResidual);

			switch(m_preconditioner) {
				case NoPreconditioner:
					if(m_params.solveThinBoundaries)
						cusp::krylov::cg(*(m_pPoissonMatrix->getGPUDataHyb()), *m_pCudaScalarField->pDevicePressure, *m_pCudaScalarField->pDeviceFlux, *m_pMonitor);
					else
						cusp::krylov::cg(*(m_pPoissonMatrix->getGPUData()), *m_pCudaScalarField->pDevicePressure, *m_pCudaScalarField->pDeviceFlux, *m_pMonitor);
				break;
				case Diagonal:
					if(m_params.solveThinBoundaries)
						cusp::krylov::cg(*(m_pPoissonMatrix->getGPUDataHyb()), *m_pCudaScalarField->pDevicePressure, *m_pCudaScalarField->pDeviceFlux, *m_pMonitor, *m_pDeviceDiagonalPrecond);
					else
						cusp::krylov::cg(*(m_pPoissonMatrix->getGPUData()), *m_pCudaScalarField->pDevicePressure, *m_pCudaScalarField->pDeviceFlux, *m_pMonitor, *m_pDeviceDiagonalPrecond);
					break;

				case AINV:
					if(m_params.solveThinBoundaries)
						cusp::krylov::cg(*(m_pPoissonMatrix->getGPUDataHyb()), *m_pCudaScalarField->pDevicePressure, *m_pCudaScalarField->pDeviceFlux, *m_pMonitor, *m_pDeviceAinvPrecond);
					else
						cusp::krylov::cg(*(m_pPoissonMatrix->getGPUData()), *m_pCudaScalarField->pDevicePressure, *m_pCudaScalarField->pDeviceFlux, *m_pMonitor, *m_pDeviceAinvPrecond);
					break;

				case SmoothedAggregation:
					if(m_params.solveThinBoundaries)
						cusp::krylov::cg(*(m_pPoissonMatrix->getGPUDataHyb()), *m_pCudaScalarField->pDevicePressure, *m_pCudaScalarField->pDeviceFlux, *m_pMonitor, *m_pDeviceSmoothedPrecond);
					else
						cusp::krylov::cg(*(m_pPoissonMatrix->getGPUData()), *m_pCudaScalarField->pDevicePressure, *m_pCudaScalarField->pDeviceFlux, *m_pMonitor, *m_pDeviceSmoothedPrecond);
					break;
			}
			
			if(m_dimensions.z == 0)
				copyFromCudaScalarField(dynamic_cast<Array2D<Scalar> *>(pResult));
			else
				copyFromCudaScalarField(dynamic_cast<Array3D<Scalar> *>(pResult));

			m_lastResidual = m_pMonitor->residual_norm();
			m_numIterations = m_pMonitor->iteration_count();
			if(m_pMonitor->converged())
				return true;
			else 
				return false;
		}
		
	}
}