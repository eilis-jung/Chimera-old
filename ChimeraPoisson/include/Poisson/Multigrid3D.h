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

#ifndef MATH_MULTIGRID_3D_H_ 
#define MATH_MULTIGRID_3D_H_ 

#include "ChimeraCore.h"
#include "Poisson/Multigrid.h"

namespace Chimera {
	namespace Poisson {


		/** 2D Multigrid solver */
		class Multigrid3D : public Multigrid {
		public:
			/************************************************************************/
			/* ctors and init                                                       */
			/************************************************************************/
			Multigrid3D(PoissonMatrix *A, solverParams_t solverParams) 
				: Multigrid(A, solverParams) { //Max iterations won't be used
					m_params = solverParams;
					m_solverType = Relaxation;
			}

			void initializeGridLevels(Scalar *rhs, Scalar *result);

		protected:

			/************************************************************************/
			/* Operator coarsening                                                  */
			/************************************************************************/
			PoissonMatrix * directInjectionCoarseningMatrix(int level);
			PoissonMatrix * garlekinCoarseningMatrix(int level);
			PoissonMatrix * geometricalAveragingCoarseningMatrix(int level);

			/************************************************************************/
			/* Multigrid functionalities                                            */
			/************************************************************************/
			/** Exact solve method only works for regular grid implementations and 3x3 grids */
			void exactSolve(const Scalar *rhs, Scalar * result);

			/************************************************************************/
			/* Restriction functions                                                */
			/************************************************************************/
			void restrictFullWeighting(int level, const Scalar *fineGrid, Scalar *coarseGrid);
			void restrictHalfWeighting(int level, const Scalar *fineGrid, Scalar *coarseGrid);

			/************************************************************************/
			/* Prolongation functions                                               */
			/************************************************************************/
			void prolongLinearInterpolation(int level, const Scalar *coarseGrid, Scalar *fineGrid);

			/************************************************************************/
			/* Auxiliary                                                            */
			/************************************************************************/
			void copyBoundaries(int level);
			FORCE_INLINE void zeroResultVector(int level) {
				dimensions_t gridDimensions = m_subGridDimensions[level + 1];
				for(int i =  0; i < gridDimensions.x; i++) {
					for(int j =  0; j < gridDimensions.y; j++) {
						for(int k = 0; k < gridDimensions.z; k++) {
							Scalar *result = m_resultVector[level + 1];
							result[getLevelIndex(level + 1, i, j, k)] = 0;
						}
					}
				}
			}

			FORCE_INLINE int getLevelIndex(int level, int i, int j, int k) {
				return k*m_subGridDimensions[level].y*m_subGridDimensions[level].x + 
						j*m_subGridDimensions[level].x + i;
			};

			FORCE_INLINE bool isSolidCell(int level, int i, int j, int k) {
				return m_solidCellMarkers[level][getLevelIndex(level, i, j, k)];
			}

			//Temporary function: Probably we should expand this too
			FORCE_INLINE bool isBoundaryCell(int i, int j, int k) { 
				return m_params.pBoundaryCells[getLevelIndex(0, i, j, k)];
			}

			/************************************************************************/
			/* Level functions                                                     */
			/************************************************************************/
			/** Residual calculation */
			void updateResiduals(int level);

			/** Prolong coarse residuals to finer rhs, computing the defect */
			void addNextLevelErrors(int level);

			/************************************************************************/
			/* Smoothers                                                            */
			/************************************************************************/
			void smoothSolution(int level);
			void redBlackGaussSeidelRelaxation(int level);
			void gaussSeidelRelaxation(int level);
			void gaussSeidelRelaxationExtended(int level);
			void sucessiveOverRelaxation(int level);
			void redBlackSuccessiveOverRelaxation(int level);
			void gaussJacobiRelaxation(int level);
		};
	}
}

#endif