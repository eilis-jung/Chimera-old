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