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

#ifndef _MATH_GAUSS_SOLVER_H_ 
#define _MATH_GAUSS_SOLVER_H_ 
#pragma once

#include "ChimeraCore.h"

/************************************************************************/
/* Math API                                                             */
/************************************************************************/
#include "Poisson/PoissonMatrix.h"
#include "Poisson/PoissonSolver.h"

using namespace std;
namespace Chimera {
	namespace Poisson {

		class GaussSeidel : public PoissonSolver {

		public:

			typedef enum iterationType_t {
				serial, 
				redBlack
			} iterationType_t;


			typedef struct solverParams_t {
				iterationType_t iterationType;
				Scalar tolerance;
				int maxIterations;
				bool *pDisabledCells;
				bool *pBoundaryCells;
				Scalar *pCellsVolumes;
				Scalar dx;

				boundaryCondition_t northBoundary;
				boundaryCondition_t southBoundary;
				boundaryCondition_t westBoundary;
				boundaryCondition_t eastBoundary;

				solverParams_t() {
					iterationType = serial;
					tolerance = 1e-05f;
					maxIterations = 200;
					pDisabledCells = NULL;
					pBoundaryCells = NULL;
					pCellsVolumes = NULL;
					dx = 0;
					northBoundary = southBoundary = westBoundary = eastBoundary = neumann;
				}
			} solverParams_t;

			

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			GaussSeidel(const params_t & params, PoissonMatrix *A) : PoissonSolver(params, A) {
				m_totalResidual = 0;
				m_solverType = Relaxation;
				m_cutCellsDivergence = nullptr;
				m_cutCellsPressure = nullptr;

			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			virtual void updateResidual();

			/************************************************************************/
			/* Solving                                                              */
			/************************************************************************/
			bool solveCPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult);
			bool solveGPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			solverParams_t * getSolverParams() {
				return &m_params;
			}
			void setCutCellsDivergence(Scalar *pCutCellsDivergence, int regularCellNum, int cutCellNum) {
				m_cutCellsDivergence = pCutCellsDivergence;
				m_pPrevResult = new Scalar[regularCellNum];
				m_pPrevResultAdditional = new Scalar[cutCellNum];
			}

			void setCutCellsPressure(Scalar *pCutCellsPressure) {
				m_cutCellsPressure = pCutCellsPressure;
			}
			void serialIterationForCutCells(const Scalar *rhs, Scalar *result);

			~GaussSeidel() {
				delete[] m_pPrevResult;
				delete[] m_pPrevResultAdditional;
			}

		private:
			/************************************************************************/
			/* Private Functions                                                    */
			/************************************************************************/
			void dumpToFile() const;

			void redBlackIteration(int sweep, const Scalar *rhs, Scalar *result);
			void serialIteration(const Scalar *rhs, Scalar *result);
			uint getIndexForArray(uint index);
			void updateBoundaries(Scalar *result);
			/************************************************************************/
			/* Members                                                              */
			/************************************************************************/
			solverParams_t m_params;
			DoubleScalar m_totalResidual;
			Scalar *m_pResult;
			const Scalar *m_pRhs;
			Scalar *m_cutCellsDivergence;
			Scalar *m_cutCellsPressure;
			Scalar * m_pPrevResult;
			Scalar * m_pPrevResultAdditional;

		};
	}

}
#endif