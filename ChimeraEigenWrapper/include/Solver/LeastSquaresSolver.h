#ifndef CHIMERA_EIGEN_WRAPPER_LEAST_SQUARES_SOLVER_H_ 
#define CHIMERA_EIGEN_WRAPPER_LEAST_SQUARES_SOLVER_H_ 
#pragma once


#include "ChimeraCore.h"

namespace Chimera {
	using namespace Core;

	namespace EigenWrapper {

		template <class ScalarType>
		class LeastSquaresSolver {
		
		public:

			#pragma region ExternalStructures
			typedef enum solvingType_t {
				jacobiSVD,
				matrixSquare,
				pivotQR
			} solvingType_t;
			#pragma endregion

			#pragma region Constructors
			LeastSquaresSolver(MatrixNxN *A, ScalarType relativeTolerance = 1e-5, int maxIterations = 100);
			#pragma endregion

			#pragma region Functionalities
			/** Solves the least squares system initialized in MatrixNxN A. One has to select a method for solving the least squares.
			  * Notice that each method pins down the null space in different ways, so results may vary according to that. Check the
			  * Eigen documentation for differences on solving methods. */
			vector<ScalarType> solve(const vector<ScalarType> &rhs, solvingType_t solvingType = jacobiSVD);
			#pragma endregion

		protected:
			#pragma region ClassMembers
			MatrixNxN *m_pMatrix;
			#pragma endregion
			ScalarType m_relativeTolerance;
			unsigned int m_maxIterations;
			ScalarType m_lastResidual;
			int m_numIterations;
			
			#pragma region PrivateFunctionalities
	
			#pragma endregion
			
		};

	}
}
#endif