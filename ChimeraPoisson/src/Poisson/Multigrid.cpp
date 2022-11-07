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

#include "Poisson/Multigrid.h"

namespace Chimera {
	namespace Poisson {

		/************************************************************************/
		/* Ctors and initialization												*/
		/************************************************************************/
		void Multigrid::initializeSubOperators() {
			//0 Level Poisson matrix is correspondent to the original fine grid operator discretization
			//if(m_params.operatorsCoarseningType != rediscretization)
				
			m_pPoissonMatrices.push_back(m_pPoissonMatrix);

			for(unsigned int level = 1; level < m_subGridDimensions.size(); level++) {
				if(m_params.pCellsVolumes == NULL) { //Regular grid
					m_pPoissonMatrices.push_back(directInjectionCoarseningMatrix(level));
				} else {
					switch(m_params.operatorsCoarseningType) {
					case directInjection:
						m_pPoissonMatrices.push_back(directInjectionCoarseningMatrix(level));
						break;

					case garlekin:
						m_pPoissonMatrices.push_back(garlekinCoarseningMatrix(level));
						break;

					case rediscretization:
						//Do nothing - matrices are externally loaded into the solver
						break;

					case geomtricAveraging:
						m_pPoissonMatrices.push_back(geometricalAveragingCoarseningMatrix(level));
						break;
					}
				}
			}
		}

		void Multigrid::performVCycle(int level) {
			if(level == m_rhsVector.size() - 1) {
				if(m_subGridDimensions[level].x == 3 && m_subGridDimensions[level].y == 3) {
					exactSolve(m_rhsVector[level], m_resultVector[level]);
				} else { //Solution on coarsest grid will have fixed pre-sweeps steps
					for (unsigned int sweep = 0; sweep < m_params.solutionSmooths; sweep++) {
						smoothSolution(level);
					}
				}

			} else {
				for (unsigned int sweep = 0; sweep < m_params.preSmooths; sweep++) { //Pre-sweeps
					smoothSolution(level);
				}

				//Update residuals for this level
				updateResiduals(level);

				//Restrict residuals as the next RHS
				restrict(level, m_residualVector[level], m_rhsVector[level + 1]);

				//Setting to zero next level, in order to Solve Ae = r
				zeroResultVector(level + 1);

				for(unsigned int muCycle = 0; muCycle < m_params.muCycle; muCycle++) {
					//Solve Ae = r on next level
					performVCycle(level + 1);
				}

				//Perform correction of the solved error on this level
				addNextLevelErrors(level);

				//Post sweeps
				for (unsigned int sweep = 0; sweep < m_params.postSmooths; sweep++) { //Pre-sweeps
					smoothSolution(level);
				}

			}
		}

		/************************************************************************/
		/* Smoothers                                                            */
		/************************************************************************/
		void Multigrid::smoothSolution(int level) {
			switch(m_params.smoothingScheme) {
				case gaussSeidel:
					if(m_params.operatorsCoarseningType == garlekin) {
						gaussSeidelRelaxationExtended(level);
					} else {
						gaussSeidelRelaxation(level);
					}
				break;

				case redBlackGaussSeidel:
					redBlackGaussSeidelRelaxation(level);
				break;

				case SOR:
					sucessiveOverRelaxation(level);
				break;
					
				case redBlackSOR:
					redBlackSuccessiveOverRelaxation(level);
				break;

				case gaussJacobi:
					gaussJacobiRelaxation(level);
				break;
			}
		}

		/************************************************************************/
		/* Multigrid types				                                        */
		/************************************************************************/
		void Multigrid::FullMultigrid() {
			for(unsigned int i = static_cast<unsigned int>(m_rhsVector.size() - 1); i >= 0; i--) {
				performVCycle(i);
				if(i != 0)
					prolong(i - 1, (const Scalar *) m_resultVector[i], m_resultVector[i - 1]); //Prolong solution from the 
																							   // coarser to the finer grid
			}

			//Post FMG V-cycles
			StandardMultigrid();
		}

		void Multigrid::StandardMultigrid() {
			int totalNumCycles = 0;
			if(m_params.nCycles == -1) { //Perform until convergence
				totalNumCycles = m_params.maxCycles;
			} else { //Fixed number of iterations
				totalNumCycles = m_params.nCycles;
			}

			//N-cycles
			for(int i = 0; i < totalNumCycles; i++) {
				performVCycle(0);
			}

			updateResiduals(0);
		}

		/************************************************************************/
		/* Solving functions                                                    */
		/************************************************************************/
		bool Multigrid::solveCPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) {
			//RHS-> Finer to coarse
			restrictToAllLevels();

			switch(m_params.multigridType) {
				case FMG:
					FullMultigrid();
				case FMG_SQUARE_GRID:
					FullMultigrid();
				break;
				
				case STANDARD:
					StandardMultigrid();
				break;
			}
			return true;
		}
		bool Multigrid::solveGPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) {
			return false;
		}

	}
}