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

#ifndef _MATH_EIGEN_CONJUGATE_GRADIENT_
#define _MATH_EIGEN_CONJUGATE_GRADIENT_
#pragma  once

#include "ChimeraCore.h"

/************************************************************************/
/* Math API                                                             */
/************************************************************************/
#include "Poisson/PoissonMatrix.h"
#include "Poisson/PoissonSolver.h"

/*Eigen library */
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>

using namespace std;
namespace Chimera {
	namespace Poisson {

	class EigenConjugateGradient : public PoissonSolver {

	public:

		#pragma region Constructors
		EigenConjugateGradient(const params_t &params, PoissonMatrix *A);
		#pragma endregion

		#pragma region Functionalities
		bool solveCPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult);
		bool solveGPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) { return false; };
		#pragma endregion

		#pragma region AccessFunctions
		params_t * getParams() {
			return &m_params;
		}
		#pragma endregion

	private:

		#pragma region ClassMembers
		Eigen::VectorXf m_eigenX, m_eigenRhs;
		Eigen::SparseMatrix<Scalar> m_eigenA;

		#pragma endregion ClassMembers

		#pragma region PrivateFunctions
		void convertMatrixToEigen2D();
		void convertMatrixToEigen3D();
		#pragma endregion 

	};
}

}
#endif