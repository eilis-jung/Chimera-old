#include "Solver/LeastSquaresSolver.h"
#include <Eigen/Dense>

namespace Chimera {

	namespace EigenWrapper {
		template <class ScalarType>
		LeastSquaresSolver<ScalarType>::LeastSquaresSolver(MatrixNxN *A, ScalarType relativeTolerance /* = 1e-5 */, int maxIterations /* = 100 */) {
			m_pMatrix = A;
			m_relativeTolerance = relativeTolerance;
			m_maxIterations = maxIterations;
		}


		template <>
		vector<Scalar> LeastSquaresSolver<Scalar>::solve(const vector<Scalar> &rhs, solvingType_t solvingType /*= jacobiSVD*/) {
			Eigen::MatrixXf eigenMatrix(Eigen::MatrixXf::Zero(m_pMatrix->getNumRows(), m_pMatrix->getNumColumns()));
			Eigen::VectorXf b(rhs.size());

			//Probably not the best way to initialize a vector in Eigen, my Eigen skills are weak :O
			for (int i = 0; i < rhs.size(); i++) {
				b(i) = rhs[i];
			}

			Eigen::VectorXf x(m_pMatrix->getNumRows());

			//Again, my Eigen skillz are weak
			for (int j = 0; j < m_pMatrix->getNumColumns(); j++) {
				for (int i = 0; i < m_pMatrix->getNumRows(); i++) {
					eigenMatrix(i, j) = (*m_pMatrix)(i, j);
				}
			}
			switch (solvingType) {
				case jacobiSVD:
					x = eigenMatrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
				break;

				case matrixSquare:
					x = (eigenMatrix.transpose() * eigenMatrix).ldlt().solve(eigenMatrix.transpose() * b);
				break;

				case pivotQR:
					x = eigenMatrix.colPivHouseholderQr().solve(b);
				break;
			}
			
			vector<Scalar> result;
			result.reserve(x.size());
			for (int i = 0; i < x.size(); i++) {
				result.push_back(x(i));
			}
			return result;
		}

		template <>
		vector<DoubleScalar> LeastSquaresSolver<DoubleScalar>::solve(const vector<DoubleScalar> &rhs, solvingType_t solvingType /*= jacobiSVD*/) {
			Eigen::MatrixXd eigenMatrix(Eigen::MatrixXd::Zero(m_pMatrix->getNumRows(), m_pMatrix->getNumColumns()));
			Eigen::VectorXd b(rhs.size());

			//Probably not the best way to initialize a vector in Eigen, my Eigen skills are weak :O
			for (int i = 0; i < rhs.size(); i++) {
				b(i) = rhs[i];
			}
			Eigen::VectorXd x(m_pMatrix->getNumRows());
			
			//Again, my Eigen skillz are weak
			for (int j = 0; j < m_pMatrix->getNumColumns(); j++) {
				for (int i = 0; i < m_pMatrix->getNumRows(); i++) {
					DoubleScalar matrxA = (*m_pMatrix)(i, j);
					eigenMatrix(i, j) = (*m_pMatrix)(i, j);
				}
			}

			switch (solvingType) {
			case jacobiSVD:
				x = eigenMatrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
				break;

			case matrixSquare:
				x = (eigenMatrix.transpose() * eigenMatrix).ldlt().solve(eigenMatrix.transpose() * b);
				break;

			case pivotQR:
				x = eigenMatrix.colPivHouseholderQr().solve(b);
				break;
			}

			vector<DoubleScalar> result;
			result.reserve(x.size());
			for (int i = 0; i < x.size(); i++) {
				result.push_back(x(i));
			}
			return result;
		}

		template class LeastSquaresSolver <Scalar>;
		template class LeastSquaresSolver <DoubleScalar>;
	}
}