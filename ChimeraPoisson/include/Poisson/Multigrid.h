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

#ifndef MATH_MULTIGRID_H_ 
#define MATH_MULTIGRID_H_

#include "ChimeraCore.h"

#include "Poisson/PoissonMatrix.h"
#include "Poisson/PoissonSolver.h"

namespace Chimera {
	
	namespace Poisson {
		

		class Multigrid : public PoissonSolver {

		public:
			/************************************************************************/
			/* Internal structures                                                  */
			/************************************************************************/
			typedef enum multigridType_t {
				FMG_SQUARE_GRID, // Numerical recipes: grid needs to be square
				FMG, // Numerical recipes adaptation to work with non-square grids
				STANDARD,
				YAVNEH_96 // Yavneh's and Cohen method
			} multigridType_t;
			

			typedef enum weightingScheme_t {
				fullWeighting,
				halfWeighting
			};
			
			typedef enum interpolationScheme_t {
				linearInterpolation
			};

			
			/** Different types of relaxation schemes. */
			typedef enum smoothingScheme_t {
				gaussSeidel,
				redBlackGaussSeidel,
				SOR,
				redBlackSOR,
				gaussJacobi
			};


			/** Defines how the operators are defined in different levels of the Multigrid algorithm. These parameters
			 ** only influence on non-regular grids discretizations, since the operators coarsening on regular grids 
			 ** remains invariant.
			 ** 
			 ** No coarsening: Direct injection of the regular operators carried out at the coarser levels.
			 ** 
			 ** Rediscretization: Manual points rediscretization are made in order to define the operators. That is,
			 ** redefining the points that are used on the PoissonMatrix building phase and recalculate the operators
			 ** values for the new grid points.
			 ** 
			 ** Garlekin: Uses the approach of "On Multigrid for Overlapping grids", that is based upon interpolation and
			 ** restriction of the operators to the coarser levels. This approach generates, for 2D, 9 points matrices, 
			 ** instead of the standard 5 point configuration.
			 ** 
			 ** Simple Garlekin: Same as Garlekin, but transforms the 9 points stencil matrices into 5 point stencil 
			 ** matrices. 
			 ** 
			 ** Geometric averaging: Perform an operator averaging method based on the paper: "Multigrid calculation of 
			 ** fluid flows in complex 3D geometries using curvilinear grids".
			 ** 
			 ** **/
			typedef enum operatorsCoarsening_t {
				directInjection,
				rediscretization,
				garlekin,
				garlekinSimple,
				geomtricAveraging
			};

			/** Solver used in the last level of the grid. Does not necessarily exactly solve the matrix, i.e. can be
			 ** a Gauss-Seidel relaxation method with a fixed step of iterations.
			 ** If the relaxation solver is selected, the smoothing scheme defined will be performed with a fixed number
			 ** of steps. */

			typedef enum exactSolve_t {
				relaxationSolver,
				conjugateGradientSolver
			};

			typedef struct solverParams_t : public PoissonSolver::params_t {

				multigridType_t multigridType;

				// Number of subgrids levels
				unsigned int numSubgrids; 

				// MuCycle: Number of recursive iterations that can be perform due a grid correction step. Known iteration
				// patterns:
				//	1 - Vcycle
				//	2 - Wcycle
				unsigned int muCycle;

				//Pre-smooths, post smooths and solutionSmooths
				unsigned int preSmooths, postSmooths, solutionSmooths;

				//If the grid has regular spacing, stores the dx between cells
				Scalar gridRegularSpacing;

				//Solid cell map, used for obstacles on regular grids
				bool *pSolidCells;

				//Boundary cell map, used for obstacles on regular grids
				bool *pBoundaryCells;

				//If the grid has non-regular grid spacing, stores the cell volumes.
				Scalar *pCellsVolumes;

				//Scale factors: for the 2D case they are used to calculate cell face areas
				void *pCellsAreas;

				// Post Full Multigrid cycles. If set to -1, perform nCycles until convergence
				int nCycles; 

				//Multigrid error stopping criteria
				Scalar tolerance; 

				//If performing nCycles until convergence, what is the maximum number of cycles permitted
				unsigned int maxCycles;

				//Weighting scheme
				weightingScheme_t weightingScheme;

				//Interpolation scheme
				interpolationScheme_t interpolationScheme;

				//Internal smoother solver
				smoothingScheme_t smoothingScheme;

				//Defines how the algorithm perform the coarsening of the operators
				operatorsCoarsening_t operatorsCoarseningType;

				//Defines the solving type in the last level of the coarsening
				exactSolve_t exactSolver;

				//Boundary conditions for the pressure solving scheme. They will use the same convention as 
				// boundaryLocation_t
				vector<boundaryCondition_t> m_boundaries;

				//Defines if the initial solution is restricted to the coarser grids - useful for Chimera Multigrid
				//implementations
				bool restrictInitialSolution;

				//Iteration weight used in Sucessive Over Relaxation algorithm
				Scalar wSor;

				solverParams_t() {
					preSmooths = postSmooths = 2;
					solutionSmooths = 10;
					numSubgrids = 3;
					muCycle = 1; //V Cycle
					nCycles = 2;
					gridRegularSpacing = 0.1f;
					tolerance = 1e-05f;
					maxCycles = 20;
					pCellsVolumes = NULL;
					pCellsAreas = NULL;
					pSolidCells = NULL;
					pBoundaryCells = NULL;

					multigridType = FMG;
					weightingScheme = fullWeighting;
					interpolationScheme = linearInterpolation;
					smoothingScheme = gaussSeidel;
					operatorsCoarseningType = directInjection;
					restrictInitialSolution = false;

					wSor = 1.0f;

					//Initialize all boundary conditions as Neumann
					for(int i = 0; i < 6; i++) {
						boundaryCondition_t neumannBD = neumann;
						m_boundaries.push_back(neumannBD);
					} 
				}
			} solverParams_t;

		protected:
			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			/** Multigrid parameters. */
			solverParams_t m_params;

			/** Different levels vectors. */
			vector<Scalar *> m_rhsVector;
			vector<Scalar *> m_resultVector;
			vector<Scalar *> m_residualVector;
			vector<dimensions_t> m_subGridDimensions;
			vector<PoissonMatrix *> m_pPoissonMatrices;
			//Non-regular multigrid solver auxiliary structs
			vector<Scalar *> m_cellsVolumes;
			//Regular multigrid solver auxiliary structs
			vector<bool *> m_solidCellMarkers;
		
		public:
			/************************************************************************/
			/* ctors and init                                                       */
			/************************************************************************/
			Multigrid(PoissonMatrix *A, solverParams_t solverParams) 
				: PoissonSolver(solverParams, A) { //Max iterations won't be used
				m_params = solverParams;
				m_solverType = Relaxation;
			} 

			void initializeMultigrid(solverParams_t params, Scalar *rhs, Scalar *result) {
				m_params = params;
				initializeGridLevels(rhs, result);
			} 

			virtual void initializeGridLevels(Scalar *rhs, Scalar *result) = 0;

			void initializeSubOperators();

			/************************************************************************/
			/* Solving functions													*/		
			/************************************************************************/
			bool solveCPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult);
			bool solveGPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult);

			/************************************************************************/
			/* Access functions	                                                    */
			/************************************************************************/
			/** In order to not mix different projects, the user has to handle the rediscretization operators coarsening
			 ** approach by manually loading grids and adding the correspondent Poisson Matrices to the solver.
			 ** It is unfortunate by now, but would save a lot of code re-writing or circular references 
			 ** using this approach. */
			void addPoissonMatrix(PoissonMatrix *pMatrix) {
				m_pPoissonMatrices.push_back(pMatrix);
			}

			Multigrid::solverParams_t & getParams() {
				return m_params;
			}

			protected:

			/************************************************************************/
			/* Operator coarsening                                                  */
			/************************************************************************/
			virtual PoissonMatrix * directInjectionCoarseningMatrix(int level) = 0;
			virtual PoissonMatrix * garlekinCoarseningMatrix(int level) = 0;
			virtual PoissonMatrix * geometricalAveragingCoarseningMatrix(int level) = 0;

			/************************************************************************/
			/* Multigrid functionalities                                            */
			/************************************************************************/
			/** Exact solve method only works for regular grid implementations and 3x3 grids */
			virtual void exactSolve(const Scalar *rhs, Scalar * result) = 0;

			/************************************************************************/
			/* Restriction functions                                                */
			/************************************************************************/
			virtual void restrictFullWeighting(int level, const Scalar *fineGrid, Scalar *coarseGrid) = 0;
			virtual void restrictHalfWeighting(int level, const Scalar *fineGrid, Scalar *coarseGrid) = 0;

			FORCE_INLINE void restrict(int level, const Scalar *fineGrid, Scalar *coarseGrid) {
				switch(m_params.weightingScheme) {
					case fullWeighting:
						restrictFullWeighting(level, fineGrid, coarseGrid);
					break;

					case halfWeighting:
						restrictHalfWeighting(level, fineGrid, coarseGrid);
					break;
				}
			}

			//Restrict the rhs to all levels
			FORCE_INLINE void restrictToAllLevels() {
				for(unsigned int i = 0; i < m_subGridDimensions.size() - 1; i++) {
					restrict(i, m_rhsVector[i], m_rhsVector[i + 1]);
				}
			}

			/************************************************************************/
			/* Prolongation functions                                               */
			/************************************************************************/
			virtual void prolongLinearInterpolation(int level, const Scalar *coarseGrid, Scalar *fineGrid) = 0;

			void prolong(int level, const Scalar *coarseGrid, Scalar *fineGrid) {
				switch(m_params.interpolationScheme) {
					case linearInterpolation:
						prolongLinearInterpolation(level, coarseGrid, fineGrid);
					break;
				}
			} 

			/************************************************************************/
			/* Auxiliary                                                            */
			/************************************************************************/
			virtual void copyBoundaries(int level) = 0;
			virtual void zeroResultVector(int level) = 0;

			/************************************************************************/
			/* Level functions                                                     */
			/************************************************************************/
			/** Residual calculation */
			virtual void updateResiduals(int level) = 0;
			
			/** Perform a vCycle on the ith level*/
			virtual void performVCycle(int level);

			/** Prolong coarse residuals to finer rhs, computing the defect */
			virtual void addNextLevelErrors(int level) = 0;

			/************************************************************************/
			/* Smoothers                                                            */
			/************************************************************************/
			void smoothSolution(int level);
			
			virtual void redBlackGaussSeidelRelaxation(int level) = 0;
			virtual void gaussSeidelRelaxation(int level) = 0;
			virtual void gaussSeidelRelaxationExtended(int level) = 0;
			virtual void sucessiveOverRelaxation(int level) = 0;
			virtual void redBlackSuccessiveOverRelaxation(int level) = 0;
			virtual void gaussJacobiRelaxation(int level) = 0;

			/************************************************************************/
			/* Multigrid types                                                      */
			/************************************************************************/
			void FullMultigrid();
			void StandardMultigrid();
		};

	}
}

#endif