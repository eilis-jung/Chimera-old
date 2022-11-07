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

#ifndef CHIMERA_MATH_POISSON_SOLVER_H_ 
#define CHIMERA_MATH_POISSON_SOLVER_H_
#pragma once


#include "ChimeraCore.h"
#include "Poisson/PoissonMatrix.h"

namespace Chimera {

	namespace Poisson {

		typedef enum SolverCategory_t {
			Krylov,
			Relaxation,
			NotSet
		} SolverCategory_t;

		class PoissonSolver {
		
		public:

			#pragma region InternalStructures
			typedef struct params_t {

				/** Residual reminiscent from the simulation */
				Scalar maxResidual;

				/** Number of iterations until algorithm end */
				int maxIterations;

				/** Cut-Cells variables*/
				bool solveThinBoundaries;
				int numberOfSpecialCells;
				vector<Scalar> *pSpecialDivergents;
				vector<Scalar> *pSpecialPressures;

				//CPU or GPU
				plataform_t platform;

				/** Solver type: Used in the PoissonMatrix building phase.
				It is categorized as Krylov solvers (doesn't need extra elements around
				matrix boundaries) or Relaxation solvers (need extra elements). It is
				a private element, since the direct access to it is not allowed. */
				SolverCategory_t solverCategory;

				/** Pressure sub-class */
				pressureMethod_t solverMethod;
				
				params_t() {
					maxResidual = 1e-7;
					maxIterations = 10000;
					solverMethod = GPU_CG;
					solverCategory = Poisson::Krylov;
					solveThinBoundaries = false;
					numberOfSpecialCells = 0;
					pSpecialPressures = nullptr;
					pSpecialDivergents = nullptr;
				}
			} params_t;
			
			typedef enum boundaryCondition_t {
				dirichlet,
				neumann
			};
			#pragma endregion
			
			#pragma region Constructors
			/** Default ctor*/
			PoissonSolver(const params_t & params, PoissonMatrix *A);
			#pragma endregion
			
			#pragma region Functionalities
			virtual bool solveCPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) = 0;
			virtual bool solveGPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) = 0;
			
			virtual void updateResidual() { }
			#pragma endregion

			#pragma region AccessFunctions
			void setParams(const params_t &params) {
				m_params = params;
			}

			params_t & getParams() {
				return m_params;
			}

			inline Scalar getResidual() const {
				return m_lastResidual;
			}

			inline int getNumberIterations() const {
				return m_numIterations;
			}

			FORCE_INLINE void setPeriodic(bool periodic) {
				m_isPeriodic = periodic;
			}

			FORCE_INLINE bool isPeriodic() const {
				return m_isPeriodic;
			}
			#pragma endregion

		protected:

			#pragma region ClassMembers
			params_t m_params;
			int m_matrixSize;
			PoissonMatrix *m_pPoissonMatrix;
			SolverCategory_t m_solverType;
			dimensions_t m_dimensions;
			bool m_isPeriodic;

			/*Convergence criteria*/
			Scalar m_lastResidual;
			int m_numIterations;
			
			/*Aux structures*/
			bool *m_pDisabledCells;
			bool *m_pBoundaryCells;
			#pragma endregion

			#pragma region PrivateFunctionalities
			/** Get index in a 2D linearized array */
			inline int getIndex(int i, int j) const {
				return j*m_dimensions.x + i;
			}

			/** Get index in a 3D linearized array */
			inline int getIndex(int i, int j, int k) const {
				return k*(m_dimensions.x + m_dimensions.y) + j*m_dimensions.x + i;
			}

			inline bool disabledCell(int i, int j) const {
				return m_pDisabledCells[getIndex(i, j)];
			}
			
			inline bool boundaryCell(int i, int j) const {
				return m_pBoundaryCells[getIndex(i, j)];
			}

			inline bool disabledCell(int i, int j, int k) const {
				return m_pDisabledCells[getIndex(i, j, k)];
			}
			#pragma endregion
			
		};

	}
}
#endif