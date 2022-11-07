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

#include "Poisson/EigenConjugateGradient.h"

namespace Chimera {

	namespace Poisson {

		#pragma region Constructors
		EigenConjugateGradient::EigenConjugateGradient(const params_t &params, PoissonMatrix *A) :
			PoissonSolver(params, A),
				m_eigenA(A->getMatrixSize(), A->getMatrixSize()),
				m_eigenRhs(A->getMatrixSize()), m_eigenX(A->getMatrixSize()) {
				m_solverType = Krylov;
				if (A->getDimensions().z == 0)
					convertMatrixToEigen2D();
				else
					convertMatrixToEigen3D();
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		void EigenConjugateGradient::convertMatrixToEigen2D() {
			vector<Eigen::Triplet<Scalar>> coefficients;
			
			for (int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
				for (int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
					int row = m_pPoissonMatrix->getRowIndex(i, j);
					if (j > 0) {
						Eigen::Triplet<Scalar> southTriplet(row, row + m_pPoissonMatrix->getSouthOffset(), m_pPoissonMatrix->getSouthValue(row));
						coefficients.push_back(southTriplet);
					}
					if (i > 0) {
						Eigen::Triplet<Scalar> westTriplet(row, row + m_pPoissonMatrix->getWestOffset(), m_pPoissonMatrix->getWestValue(row));
						coefficients.push_back(westTriplet);
					}

					Eigen::Triplet<Scalar> centralTriplet(row, row, m_pPoissonMatrix->getCentralValue(row));
					coefficients.push_back(centralTriplet);
					if (i < m_pPoissonMatrix->getDimensions().x - 1) {
						Eigen::Triplet<Scalar> eastTriplet(row, row + m_pPoissonMatrix->getEastOffset(), m_pPoissonMatrix->getEastValue(row));
						coefficients.push_back(eastTriplet);
					}

					if (j < m_pPoissonMatrix->getDimensions().y - 1) {
						Eigen::Triplet<Scalar> northTriplet(row, row + m_pPoissonMatrix->getNorthOffset(), m_pPoissonMatrix->getNorthValue(row));
						coefficients.push_back(northTriplet);
					}	
				}
			}

			m_eigenA.setFromTriplets(coefficients.begin(), coefficients.end());
		}

		void EigenConjugateGradient::convertMatrixToEigen3D() {
			for (int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
				for (int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
					for (int k = 0; k < m_pPoissonMatrix->getDimensions().z; k++) {

					}
				}
			}
		}
		#pragma endregion

		#pragma region Functionalities
		bool EigenConjugateGradient::solveCPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) {
			const Array2D<Scalar> &rhsMatrix = dynamic_cast<const Array2D<Scalar> &>(*pRhs);
			for (int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
				for (int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
					m_eigenRhs(m_pPoissonMatrix->getRowIndex(i, j)) = rhsMatrix(i + 1, j + 1);
				}
			}

			Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>, Eigen::Lower | Eigen::Upper> cg;
			cg.compute(m_eigenA);
			cg.setMaxIterations(m_params.maxIterations);
			cg.setTolerance(m_params.maxResidual);
			m_eigenX = cg.solve(m_eigenRhs);
			
			m_numIterations = cg.iterations();
			m_lastResidual = cg.error();

			Array2D<Scalar> &resultMatrix = dynamic_cast<Array2D<Scalar> &>(*pResult);
			for (int i = 0; i < m_pPoissonMatrix->getDimensions().x; i++) {
				for (int j = 0; j < m_pPoissonMatrix->getDimensions().y; j++) {
					resultMatrix(i + 1, j + 1) = m_eigenX(m_pPoissonMatrix->getRowIndex(i, j));
				}
			}

			return true;
		}
		#pragma endregion
	}
}