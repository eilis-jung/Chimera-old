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

#include "Poisson/Multigrid3D.h"

namespace Chimera {
	namespace Poisson {
		/************************************************************************/
		/* Ctors and initialization												*/
		/************************************************************************/
		void Multigrid3D::initializeGridLevels(Scalar *rhs, Scalar *result) {
			int numGrids = 0;
			dimensions_t gridSize = m_pPoissonMatrix->getDimensions();
			dimensions_t tempGridSize;

			//Initializing root level: finest grid
			{
				m_subGridDimensions.push_back(gridSize);
				Scalar *pResidual = new Scalar[gridSize.x*gridSize.y*gridSize.y];
				for(int i = 0; i < gridSize.x; i++) {
					for(int j = 0; j < gridSize.y; j++) {
						for(int k = 0; k < gridSize.z; k++) {
							pResidual[getLevelIndex(0, i, j, k)] = 0;
						}
					}
				}
				m_rhsVector.push_back(rhs);
				m_resultVector.push_back(result);
				m_residualVector.push_back(pResidual);
				if(m_params.pCellsVolumes != NULL) {
					m_cellsVolumes.push_back(m_params.pCellsVolumes);
				}
			}
			//FMG types
			if(m_params.multigridType == FMG_SQUARE_GRID) {
				tempGridSize = gridSize;

				while (tempGridSize.x > 4*2 - 1) {
					tempGridSize.x = tempGridSize.x/2 + 1;
					tempGridSize.y = tempGridSize.y/2 + 1;
					tempGridSize.z = tempGridSize.z/2 + 1;
					numGrids++;
					m_subGridDimensions.push_back(tempGridSize);
				}
				tempGridSize.x = 3;
				tempGridSize.y = 3;
				tempGridSize.z = 3;
				numGrids++;
				m_subGridDimensions.push_back(tempGridSize);

			} else if(m_isPeriodic) { //Periodicity is on X
				numGrids = m_params.numSubgrids;
				tempGridSize = gridSize;
				for(int tempNumGrids = numGrids; tempNumGrids > 0; --tempNumGrids) {
					tempGridSize.x = tempGridSize.x/2;
					tempGridSize.y = tempGridSize.y/2 + 1;
					tempGridSize.z = tempGridSize.z/2 + 1;
					m_subGridDimensions.push_back(tempGridSize);
				}
			} else {
				numGrids = m_params.numSubgrids;
				tempGridSize = gridSize;
				for(int tempNumGrids = numGrids; tempNumGrids > 0; --tempNumGrids) {
					tempGridSize.x = tempGridSize.x/2 + 1;
					tempGridSize.y = tempGridSize.y/2 + 1;
					tempGridSize.z = tempGridSize.z/2 + 1;
					m_subGridDimensions.push_back(tempGridSize);
				}
			}

			/** Allocating sub-grids, sub-residuals, sub-rhs and sub-operators */
			for(int level = 1; level <= numGrids; level++) {
				tempGridSize = m_subGridDimensions[level];
				Scalar *pRhs, *pResult, *pResidual;
				pRhs	= new Scalar[tempGridSize.x*tempGridSize.y*tempGridSize.z];
				pResult = new Scalar[tempGridSize.x*tempGridSize.y*tempGridSize.z];
				pResidual = new Scalar[tempGridSize.x*tempGridSize.y*tempGridSize.z];

				for(int i = 0; i < tempGridSize.x; i++) {
					for(int j = 0; j < tempGridSize.y; j++) {
						for(int k = 0; k < tempGridSize.z; j++) {
							pRhs[getLevelIndex(level, i, j, k)] = 0;
							pResult[getLevelIndex(level, i, j, k)] = 0;
							pResidual[getLevelIndex(level, i, j, k)] = 0;
						}
					}
				}

				m_rhsVector.push_back(pRhs);
				m_resultVector.push_back(pResult);
				m_residualVector.push_back(pResidual);
			}

			if(m_params.pSolidCells != NULL) {
				m_solidCellMarkers.push_back(m_params.pSolidCells);
			}

			initializeSubOperators();
		}

		/************************************************************************/
		/* Multigrid functionalities                                            */
		/************************************************************************/
		void Multigrid3D::exactSolve(const Scalar *rhs, Scalar * result) {
			Scalar h = m_subGridDimensions.size()*m_params.gridRegularSpacing;
			dimensions_t gridDimensions = m_subGridDimensions[m_subGridDimensions.size() - 1];

			for (int i = 0;i < 3; i++)
				for (int j = 0; j < 3; j++)
					for(int k = 0; k < 3; k++)
						result[getLevelIndex(m_subGridDimensions.size() - 1, i, j, k)] = 0.0;

			result[getLevelIndex(m_subGridDimensions.size() - 1, 1, 1, 1)] = -h*h*h*rhs[getLevelIndex(m_subGridDimensions.size() - 1, 1, 1, 1)]/9.0f;
		}


		/************************************************************************/
		/* Auxiliary                                                            */
		/************************************************************************/
		void Multigrid3D::copyBoundaries(int level) {
				Scalar * result = m_resultVector[level];
				dimensions_t gridDimensions = m_subGridDimensions[level];

				if(m_params.m_boundaries[West] == neumann) { //	West
					for(int j = 0; j < gridDimensions.y; j++) {
						for(int k = 0; k < gridDimensions.z; k++) {
							result[getLevelIndex(level, 0, j, k)] = result[getLevelIndex(level, 1, j, k)];
						}
					}
				}

				if(m_params.m_boundaries[East] == neumann) { //	East
					for(int j = 0; j < gridDimensions.y; j++) {
						for(int k = 0; k < gridDimensions.z; k++) {
							result[getLevelIndex(level, gridDimensions.x - 1, j, k)] = result[getLevelIndex(level, gridDimensions.x - 2, j, k)];
						}
					}
				}

				if(m_params.m_boundaries[South] == neumann) { // South
					for(int i = 0; i < gridDimensions.x; i++) {
						for(int k = 0; k < gridDimensions.z; k++) {
							result[getLevelIndex(level, i, 0, k)] = result[getLevelIndex(level, i, 1, k)];
						}
					}
				}

				if(m_params.m_boundaries[North] == neumann) { // North
					for(int i = 0; i < gridDimensions.x; i++) {
						for(int k = 0; k < gridDimensions.z; k++) {
							result[getLevelIndex(level, i, gridDimensions.y - 1, k)] = result[getLevelIndex(level, i, gridDimensions.y - 2, k)];
						}
					}
				} 

				if(m_params.m_boundaries[Back] == neumann) { 
					for(int i = 0; i < gridDimensions.x; i++) {
						for(int j = 0; j < gridDimensions.y; j++) {
							result[getLevelIndex(level, i, j, 0)] = result[getLevelIndex(level, i, j, 1)];
						}
					}
				}

				if(m_params.m_boundaries[Back] == neumann) { 
					for(int i = 0; i < gridDimensions.x; i++) {
						for(int j = 0; j < gridDimensions.y; j++) {
							result[getLevelIndex(level, i, j, gridDimensions.z - 1)] = result[getLevelIndex(level, i, j, gridDimensions.z - 2)];
						}
					}
				}
			}

		/************************************************************************/
		/* Operator coarsening                                                  */
		/************************************************************************/
		PoissonMatrix * Multigrid3D::directInjectionCoarseningMatrix(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];

			//Initialize volumes used in relaxation phase
			if(m_params.pCellsVolumes != NULL) {
				Scalar *pFineCellsVolumes = m_cellsVolumes[0];
				Scalar *pCellsVolumes = new Scalar[gridDimensions.x*gridDimensions.y];	
				dimensions_t fineGridDimensions = m_subGridDimensions[0];

				for(int i = 0; i < gridDimensions.x; i++) {
					for(int j = 0; j < gridDimensions.y; j++) {
						for(int k = 0; k < gridDimensions.z; k++) {
							int fineI = static_cast<int> (i*pow(2.0f, level));
							int fineJ = static_cast<int> (j*pow(2.0f, level));
							int fineK = static_cast<int> (k*pow(2.0f, level));

							//If the grids size are twice the size in each dimension volume is multiplied by 8
							Scalar totalVolume = pow(8.0f, level)*pFineCellsVolumes[getLevelIndex(0, fineI, fineJ, fineK)];
							pCellsVolumes[getLevelIndex(level, i, j, k)] = totalVolume;
						}
					}
				}
				m_cellsVolumes.push_back(pCellsVolumes);
			}

			PoissonMatrix *pPoissonMatrix = new PoissonMatrix(gridDimensions, false, m_isPeriodic);

			int startingI, endingI, endingK;
			if(m_isPeriodic) {
				startingI = 0; endingI = gridDimensions.x; endingK = gridDimensions.z;
			} else {
				startingI = 1; endingI = gridDimensions.x - 1; endingK = gridDimensions.z - 1;
			}

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < gridDimensions.y - 1; j++) {
					for(int k = 1; k < gridDimensions.z - 1; k++) {
						int fineI = static_cast<int> (i*pow(2.0f, level));
						int fineJ = static_cast<int> (j*pow(2.0f, level));
						int fineK = static_cast<int> (k*pow(2.0f, level));

						Scalar pn = abs(m_pPoissonMatrix->getNorthValue(getLevelIndex(0, fineI, fineJ, fineK)));
						Scalar ps = abs(m_pPoissonMatrix->getSouthValue(getLevelIndex(0, fineI, fineJ, fineK)));
						Scalar pe = abs(m_pPoissonMatrix->getEastValue(getLevelIndex(0, fineI, fineJ, fineK)));
						Scalar pw = abs(m_pPoissonMatrix->getWestValue(getLevelIndex(0, fineI, fineJ, fineK)));
						Scalar pb = abs(m_pPoissonMatrix->getBackValue(getLevelIndex(0, fineI, fineJ, fineK)));
						Scalar pf = abs(m_pPoissonMatrix->getFrontValue(getLevelIndex(0, fineI, fineJ, fineK)));

						pPoissonMatrix->setEastValue(getLevelIndex(level,	i, j, k), m_pPoissonMatrix->getEastValue(getLevelIndex(0,	fineI, fineJ, fineK)));
						pPoissonMatrix->setWestValue(getLevelIndex(level,	i, j, k), m_pPoissonMatrix->getWestValue(getLevelIndex(0,	fineI, fineJ, fineK)));
						pPoissonMatrix->setNorthValue(getLevelIndex(level,	i, j, k), m_pPoissonMatrix->getNorthValue(getLevelIndex(0,	fineI, fineJ, fineK)));
						pPoissonMatrix->setSouthValue(getLevelIndex(level,	i, j, k), m_pPoissonMatrix->getSouthValue(getLevelIndex(0,	fineI, fineJ, fineK)));
						pPoissonMatrix->setBackValue(getLevelIndex(level,	i, j, k), m_pPoissonMatrix->getBackValue(getLevelIndex(0,	fineI, fineJ, fineK)));
						pPoissonMatrix->setFrontValue(getLevelIndex(level,	i, j, k), m_pPoissonMatrix->getFrontValue(getLevelIndex(0,	fineI, fineJ, fineK)));

						if(m_isPeriodic) {
							pw = abs(m_pPoissonMatrix->getPeriodicWestValue(getLevelIndex(0, fineI, fineJ, fineK)));
							pPoissonMatrix->setPeriodicWestValue(getLevelIndex(level, i, j, k), -pw);
							pe = abs(m_pPoissonMatrix->getPeriodicEastValue(getLevelIndex(0, static_cast<int> (fineI + pow(2.0f, level) - 1), fineJ, fineK)));
							pPoissonMatrix->setPeriodicEastValue(getLevelIndex(level, i, j, k), -pe);
							if(pe == 0)
								pe = abs(m_pPoissonMatrix->getEastValue(getLevelIndex(0, fineI, fineJ, fineK)));
							if(pw == 0)
								pw = abs(m_pPoissonMatrix->getWestValue(getLevelIndex(0, fineI, fineJ, fineK)));
						}

						Scalar pc = pn + ps + pw + pe + pb + pf;
						pPoissonMatrix->setCentralValue(getLevelIndex(level, i, j, k), pc);
					}
				}
			}

			return pPoissonMatrix;
		}
	
		PoissonMatrix * Multigrid3D::garlekinCoarseningMatrix(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];

			PoissonMatrix *pPoissonMatrix = new PoissonMatrix(gridDimensions, PoissonMatrix::ninePointLaplace, false, m_isPeriodic);


			return pPoissonMatrix;
		}

		PoissonMatrix * Multigrid3D::geometricalAveragingCoarseningMatrix(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			dimensions_t fineGridDimensions = m_subGridDimensions[0];
			
			////Initialize volumes used in relaxation phase
			//if(m_params.pCellsVolumes != NULL) {
			//	Scalar *pFineCellsVolumes = m_cellsVolumes[0];
			//	Scalar *pCellsVolumes = new Scalar[gridDimensions.x*gridDimensions.y];	
			//	dimensions_t fineGridDimensions = m_subGridDimensions[0];

			//	for(int i = 0; i < gridDimensions.x; i++) {
			//		for(int j = 0; j < gridDimensions.y - 1; j++) {
			//			for(int k = 0; k < gridDimensions.z - 1; k++) {
			//				int fineI = i*pow(2.0f, level);
			//				int fineJ = j*pow(2.0f, level);
			//				int fineK = k*pow(2.0f, level);

			//				//If the grids size are twice the size in each dimension, for 3D, volume is multiplied by 4
			//				Scalar totalVolume = 0;
			//				for(int i_temp = 0; i_temp < pow(2.0f, level); i_temp++) {
			//					for(int j_temp = 0; j_temp < pow(2.0f, level); j_temp++) {
			//						for(int k_temp = 0; k_temp < pow(2.0f, level); k_temp++) {
			//							totalVolume += pFineCellsVolumes[getLevelIndex(0, roundClamp(fineI + i_temp, 0, fineGridDimensions.x), fineJ + j_temp, fineK + k_temp)];
			//						}
			//					}
			//				}
			//				pCellsVolumes[getLevelIndex(level, i, j, k)] = totalVolume;
			//			}
			//		}
			//	}
			//	m_cellsVolumes.push_back(pCellsVolumes);
			//}

			//if(gridDimensions.z == 0) {

			//}
			//Vector2 *pFineCellsAreas = (Vector2 *) m_params.pCellsAreas;

			PoissonMatrix *pPoissonMatrix = new PoissonMatrix(gridDimensions, false, m_isPeriodic);

			//for(int i = 0; i < gridDimensions.x; i++) {
			//	for(int j = 0; j < gridDimensions.y - 1; j++) {
			//		int fineI = i*pow(2.0f, level);
			//		int fineJ = j*pow(2.0f, level);

			//		Scalar totalLeftFacesArea = 0, leftFaceArea = 0;
			//		for(int k = 0; k < pow(2.0f, level); k++) {
			//			totalLeftFacesArea += pFineCellsAreas[getLevelIndex(0, fineI, fineJ + k)].y;
			//		}
			//		Scalar pw = 0;
			//		for(int k = 0; k < pow(2.0f, level); k++) {
			//			leftFaceArea = pFineCellsAreas[getLevelIndex(0, fineI, fineJ + k)].y;
			//			pw += m_pPoissonMatrix->getWestValue(getLevelIndex(0, fineI, fineJ + k))*leftFaceArea/totalLeftFacesArea;
			//		}

			//		Scalar totalBottomFacesArea = 0, bottomFaceArea = 0;
			//		for(int k = 0; k < pow(2.0f, level); k++) {
			//			totalBottomFacesArea += pFineCellsAreas[getLevelIndex(0, roundClamp<int>(fineI + k, 0, fineGridDimensions.x), fineJ)].x;
			//		}
			//		Scalar ps = 0;
			//		for(int k = 0; k < pow(2.0f, level); k++) {
			//			bottomFaceArea = pFineCellsAreas[getLevelIndex(0, roundClamp<int>(fineI + k, 0, fineGridDimensions.x), fineJ)].x;
			//			ps += m_pPoissonMatrix->getSouthValue(getLevelIndex(0, roundClamp<int>(fineI + k, 0, fineGridDimensions.x), fineJ))*bottomFaceArea/totalBottomFacesArea;
			//		}

			//		pPoissonMatrix->setWestValue(getLevelIndex(level, i, j), pw);
			//		pPoissonMatrix->setSouthValue(getLevelIndex(level, i, j), ps);
			//	}
			//}

			//for(int i = 0; i < gridDimensions.x - 1; i++) {
			//	for(int j = 0; j < gridDimensions.y; j++) {
			//		pPoissonMatrix->setEastValue(getLevelIndex(level, i, j), pPoissonMatrix->getWestValue(getLevelIndex(level, i + 1, j)));
			//	}
			//}

			//for(int i = 0; i < gridDimensions.x; i++) {
			//	for(int j = 0; j < gridDimensions.y - 1; j++) {
			//		pPoissonMatrix->setNorthValue(getLevelIndex(level, i, j), pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j + 1)));
			//	}
			//}

			//for(int i = 0; i < gridDimensions.x; i++) {
			//	for(int j = 0; j < gridDimensions.y; j++) {
			//		Scalar centralValue = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j)) +
			//			pPoissonMatrix->getEastValue(getLevelIndex(level, i, j)) +
			//			pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j)) +
			//			pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j));
			//		pPoissonMatrix->setCentralValue(getLevelIndex(level, i, j), -centralValue);
			//	}
			//}


			//if(m_isPeriodic) {
			//	for(int j = 1; j < gridDimensions.y - 1; j++) {
			//		int fineJ = j*pow(2.0f, level);

			//		Scalar totalLeftFacesArea = 0, leftFaceArea = 0;
			//		for(int k = 0; k < pow(2.0f, level); k++) {
			//			totalLeftFacesArea += pFineCellsAreas[getLevelIndex(0, 0, fineJ + k)].y;
			//		}
			//		Scalar pw = 0;
			//		for(int k = 0; k < pow(2.0f, level); k++) {
			//			leftFaceArea = pFineCellsAreas[getLevelIndex(0, 0, fineJ + k)].y;
			//			pw += m_pPoissonMatrix->getPeriodicWestValue(getLevelIndex(0, 0, fineJ + k))*leftFaceArea/totalLeftFacesArea;
			//		}

			//		pPoissonMatrix->setPeriodicWestValue(getLevelIndex(level, 0, j), pw);
			//		pPoissonMatrix->setPeriodicEastValue(getLevelIndex(level, gridDimensions.x - 1, j), pw);

			//		Scalar centralValue = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, 0, j)) +
			//			pPoissonMatrix->getEastValue(getLevelIndex(level, 0, j)) +
			//			pPoissonMatrix->getNorthValue(getLevelIndex(level, 0, j)) +
			//			pPoissonMatrix->getSouthValue(getLevelIndex(level, 0, j));

			//		pPoissonMatrix->setCentralValue(getLevelIndex(level, 0, j), -centralValue);

			//		centralValue = pPoissonMatrix->getWestValue(getLevelIndex(level, gridDimensions.x - 1, j)) +
			//			pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, gridDimensions.x - 1, j)) +
			//			pPoissonMatrix->getNorthValue(getLevelIndex(level, gridDimensions.x - 1, j)) +
			//			pPoissonMatrix->getSouthValue(getLevelIndex(level, gridDimensions.x - 1, j));

			//		pPoissonMatrix->setCentralValue(getLevelIndex(level, gridDimensions.x - 1, j), -centralValue);
			//	}
			//}

			return pPoissonMatrix;
		}

		/************************************************************************/
		/* Restriction functions                                                */
		/************************************************************************/
		/** Full weighting in 3D becomes complex - only half-weighting is used */
		void Multigrid3D::restrictFullWeighting(int level, const Scalar *fineGrid, Scalar *coarseGrid) {
			restrictHalfWeighting(level, fineGrid, coarseGrid);
		}

		void Multigrid3D::restrictHalfWeighting(int level, const Scalar *fineGrid, Scalar *coarseGrid) {
			dimensions_t coarseGridSize = m_subGridDimensions[level + 1];

			int startingI = 1; int endingI = coarseGridSize.x - 1;
			if(m_isPeriodic) {
				startingI = 0; endingI = coarseGridSize.x;
			}

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < coarseGridSize.y - 1; j++) {
					for(int k = 1; k < coarseGridSize.z - 1; k++) {
						int fineI = i*2;
						int fineJ = j*2;
						int fineK = j*2;

						int nextI = fineI + 1;
						int prevI = fineI - 1;
						if(m_isPeriodic && i == 0) 
							prevI = coarseGridSize.x - 1;
						
						if(m_isPeriodic && i == coarseGridSize.x - 1)
							nextI = 0;

						Scalar restriction = 0.5f*fineGrid[getLevelIndex(level, fineI, fineJ, fineK)] 
														+ (0.5f/6.0f) * ( fineGrid[getLevelIndex(level, prevI, fineJ, fineK)]
																		+ fineGrid[getLevelIndex(level, nextI, fineJ, fineK)]
																		+ fineGrid[getLevelIndex(level, fineI, fineJ - 1, fineK)]
																		+ fineGrid[getLevelIndex(level, fineI, fineJ + 1, fineK)]
																		+ fineGrid[getLevelIndex(level, fineI, fineJ, fineK - 1)]
																		+ fineGrid[getLevelIndex(level, fineI, fineJ, fineK + 1)]);
						coarseGrid[getLevelIndex(level + 1, i, j, k)] = restriction;
					}
				}
			}

			//Top and bottom
			for (int i = 0; i < coarseGridSize.x; i++) {
				for(int k = 0; k < coarseGridSize.z; k++) {
					int fineI = i*2; int fineJ = (coarseGridSize.y - 1)*2; int fineK = k*2;
					coarseGrid[getLevelIndex(level + 1, i, 0, k)] = fineGrid[getLevelIndex(level, fineI, 0, fineK)];
					coarseGrid[getLevelIndex(level + 1, i, coarseGridSize.y - 1, k)] = fineGrid[getLevelIndex(level, fineI, fineJ, fineK)];
				}
			}

			//Back and front
			for (int i = 0; i < coarseGridSize.x; i++) {
				for (int j = 0; j < coarseGridSize.y; j++) {
					int fineI = i*2; int fineJ = j*2; int fineK = (coarseGridSize.z - 1)*2;
					coarseGrid[getLevelIndex(level + 1, i, j, 0)] = fineGrid[getLevelIndex(level, fineI, fineJ, 0)];
					coarseGrid[getLevelIndex(level + 1, i, j, coarseGridSize.z - 1)] = fineGrid[getLevelIndex(level, fineI, fineJ, fineK)];
				}
			}

			//East and west
			if(!m_isPeriodic) {
				for (int j = 0; j < coarseGridSize.y; j++) {
					for(int k = 0; k < coarseGridSize.z; k++) {
						int fineI = (coarseGridSize.x - 1)*2; int fineJ = j*2; int fineK = k*2;
						coarseGrid[getLevelIndex(level + 1, 0, j, k)] = fineGrid[getLevelIndex(level, 0, fineJ, fineK)];
						coarseGrid[getLevelIndex(level + 1, coarseGridSize.x - 1, j, k)] = fineGrid[getLevelIndex(level, fineI, fineJ, fineK)];
					}
				}
			}
		}

		/************************************************************************/
		/* Prolongation functions                                               */
		/************************************************************************/
		/**Review prolongation */
		void Multigrid3D::prolongLinearInterpolation(int level, const Scalar *coarseGrid, Scalar *fineGrid) {
			dimensions_t coarseGridSize = m_subGridDimensions[level + 1];

			//Bilinear interpolation
			for (int jc = 0; jc < m_subGridDimensions[level + 1].y; jc++) 
				for (int ic = 0; ic < m_subGridDimensions[level + 1].x; ic++)
					for(int kc = 0; kc < m_subGridDimensions[level + 1].z; kc++)
						fineGrid[getLevelIndex(level,2*ic, 2*jc, 2*kc)] = coarseGrid[getLevelIndex(level + 1, ic, jc, kc)];
			
			if(m_isPeriodic) {
				for (int jf = 0; jf < m_subGridDimensions[level].y; jf += 2) {
					for (int kf = 0; kf < m_subGridDimensions[level].z; kf += 2) {
						for (int iif = 1; iif < m_subGridDimensions[level].x - 1; iif += 2) {
							fineGrid[getLevelIndex(level, iif, jf, kf)] = 0.5f*(fineGrid[getLevelIndex(level, iif + 1, jf, kf)] + fineGrid[getLevelIndex(level, iif - 1, jf, kf)]);
						}
						fineGrid[getLevelIndex(level, m_subGridDimensions[level].x - 1, jf, kf)] = 0.5f*(	fineGrid[getLevelIndex(level, 0, jf, kf)] 
																										+	fineGrid[getLevelIndex(level, m_subGridDimensions[level].x - 2, jf, kf)]);
					}
				}
			} else {
				for (int jf = 0; jf < m_subGridDimensions[level].y; jf += 2)
					for (int kf = 0; kf < m_subGridDimensions[level].z; kf += 2)
						for (int iif = 1; iif < m_subGridDimensions[level].x - 1; iif += 2)
							fineGrid[getLevelIndex(level, iif, jf, kf)] = 0.5f*(fineGrid[getLevelIndex(level, iif + 1, jf, kf)] + fineGrid[getLevelIndex(level, iif - 1, jf, kf)]);
			}
			
			for (int jf = 1; jf < m_subGridDimensions[level].y - 1; jf += 2) 
				for (int iif = 0; iif < m_subGridDimensions[level].x; iif++) 
					for (int kkf = 0; kkf < m_subGridDimensions[level].z; kkf++) 
						fineGrid[getLevelIndex(level, iif, jf, kkf)] = 0.5f*(fineGrid[getLevelIndex(level, iif, jf + 1, kkf)] + fineGrid[getLevelIndex(level, iif, jf - 1, kkf)]);

		}

		/************************************************************************/
		/* Level functions                                                      */
		/************************************************************************/
		void Multigrid3D::updateResiduals(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result     = m_resultVector[level];
			Scalar *rhs	       = m_rhsVector[level];
			Scalar *pResiduals = m_residualVector[level];

			DoubleScalar totalError = 0.0l, maxRhs = 0.0l;

			Scalar h, h2 = 0.0f;
			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.gridRegularSpacing*(level + 1);
				h2 = 1.0f/(h*h*h);
			}

			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = gridDimensions.x;
			} else {
				startingI = 1; endingI = gridDimensions.x - 1;
			}

			/*copyBoundaries(level);*/

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < gridDimensions.y - 1; j++) {
					for(int k = 1; k < gridDimensions.z - 1; k++) {
						if(m_params.pSolidCells != NULL && level == 0 && (isSolidCell(level, i, j, k) || isBoundaryCell(i, j, k))) {
							pResiduals[getLevelIndex(level, i, j, k)] = 0;
							continue;
						}
						Scalar pe, pw = 0;

						PoissonMatrix *pPoissonMatrix = m_pPoissonMatrices[level];

						if(m_params.pCellsVolumes != NULL) { //Non-regular grid
							Scalar *pCellVolumes = m_cellsVolumes[level];
							h2 = 1.0f/(pCellVolumes[getLevelIndex(level, i, j, k)]);
						}

						int nextI = i + 1;
						pe = pPoissonMatrix->getEastValue(getLevelIndex(level, i, j, k));
						if(m_isPeriodic && i == gridDimensions.x - 1) {
							nextI = 0;
							pe = pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, i, j, k));
						}

						int prevI = i - 1;
						pw = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j, k));
						if(m_isPeriodic && i == 0) {
							prevI = gridDimensions.x - 1;
							pw = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, i, j, k));
						}

						Scalar residual = result[getLevelIndex(level, nextI, j, k)]*pe
										+  	result[getLevelIndex(level, prevI, j, k)]*pw
										+	result[getLevelIndex(level, i, j - 1, k)]*pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j, k))
										+	result[getLevelIndex(level, i, j + 1, k)]*pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j, k))
										+	result[getLevelIndex(level, i, j, k - 1)]*pPoissonMatrix->getBackValue(getLevelIndex(level, i, j, k))
										+	result[getLevelIndex(level, i, j, k + 1)]*pPoissonMatrix->getFrontValue(getLevelIndex(level, i, j, k))
										+	result[getLevelIndex(level, i, j, k)]*pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j, k));


						residual = -residual;
						residual *= h2;
						residual += rhs[getLevelIndex(level, i, j, k)];

						pResiduals[getLevelIndex(level, i, j, k)] = residual;

						if(level == 0) {
							totalError += residual*residual;
							maxRhs += rhs[getLevelIndex(level, i, j, k)]*rhs[getLevelIndex(level, i, j, k)];
							/*if(abs(rhs[getLevelIndex(level, i, j)]) > maxRhs) {
							maxRhs = abs(rhs[getLevelIndex(level, i, j)]);
							}*/
						}
					}
				}
			}

			if(level == 0) {
				totalError /= (gridDimensions.x)*(gridDimensions.y)*(gridDimensions.z);
				maxRhs /= (gridDimensions.x)*(gridDimensions.y)*(gridDimensions.z);
				if(maxRhs != 0) {
					m_lastResidual = static_cast<Scalar> (totalError/maxRhs);
				} else {
					m_lastResidual = static_cast<Scalar> (totalError);
				}
				
			}

			//for(int j = 0; j < gridDimensions.y; j++) {
			//	pResiduals[getLevelIndex(level, 0, j)] = 0;
			//	pResiduals[getLevelIndex(level, gridDimensions.x - 1, j)] = 0;
			//}
		}

		void Multigrid3D::addNextLevelErrors(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result = m_resultVector[level];

			Scalar *temp = new Scalar[gridDimensions.x*gridDimensions.y*gridDimensions.z];

			prolong(level, m_resultVector[level + 1], temp);
			
			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = gridDimensions.x;
			} else {
				startingI = 1; endingI = gridDimensions.x - 1;
			}

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < gridDimensions.y - 1; j++) {
					for(int k = 1; k < gridDimensions.z - 1; k++) {
						Scalar tempVal = temp[getLevelIndex(level, i, j, k)];
						if(m_params.pSolidCells != NULL && level == 0 && (isSolidCell(level, i, j, k) || isBoundaryCell(i, j, k))) {

						} else {
							result[getLevelIndex(level, i, j, k)] += temp[getLevelIndex(level, i, j, k)];
						}
					}
				}
			}

			delete temp;
		}

		/************************************************************************/
		/* Smoothers                                                            */
		/************************************************************************/
		void Multigrid3D::gaussSeidelRelaxation(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result = m_resultVector[level];
			Scalar *rhs	   = m_rhsVector[level];

			Scalar h2, h = 0;

			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.gridRegularSpacing*(level + 1);
				h2 = (h*h*h);
			}

			copyBoundaries(level);

			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = gridDimensions.x;
			} else {
				startingI = 1; endingI = gridDimensions.x - 1;
			}

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < gridDimensions.y - 1; j++) {
					for(int k = 1; k < gridDimensions.z - 1; k++) {
						if(m_params.pSolidCells != NULL && level == 0 && (isSolidCell(0, i, j, k) || isBoundaryCell(i, j, k))) {
							if(isBoundaryCell(i, j, k))
								continue;

							result[getLevelIndex(level, i, j, k)] = 0.0f;
						} else {
							Scalar pe, pw;
							PoissonMatrix *pPoissonMatrix = m_pPoissonMatrices[level];

							if(m_params.pCellsVolumes != NULL) {
								Scalar *pCellVolumes = m_cellsVolumes[level];
								h2 = pCellVolumes[getLevelIndex(level, i, j, k)];
							}

							int nextI = i + 1;
							pe = pPoissonMatrix->getEastValue(getLevelIndex(level, i, j, k));
							if(m_isPeriodic && i == gridDimensions.x - 1) {
								nextI = 0;
								pe = pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, i, j, k));
							}

							int prevI = i - 1;
							pw = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j, k));
							if(m_isPeriodic && i == 0) {
								prevI = gridDimensions.x - 1;
								pw = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, i, j, k));
							}

							Scalar relaxationStep =	 h2*rhs[getLevelIndex(level, i, j, k)] - 
								(	result[getLevelIndex(level, nextI, j, k)]*pe
								+  	result[getLevelIndex(level, prevI, j, k)]*pw
								+	result[getLevelIndex(level, i, j - 1, k)]*pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j, k))
								+	result[getLevelIndex(level, i, j + 1, k)]*pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j, k))
								+	result[getLevelIndex(level, i, j, k - 1)]*pPoissonMatrix->getBackValue(getLevelIndex(level, i, j, k))
								+	result[getLevelIndex(level, i, j, k + 1)]*pPoissonMatrix->getFrontValue(getLevelIndex(level, i, j, k))
								);

							result[getLevelIndex(level, i, j, k)] = (1/pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j, k))) * relaxationStep;
						}
					}
				}
			}

		}

		void Multigrid3D::gaussSeidelRelaxationExtended(int level) {
		
		}

		void Multigrid3D::redBlackGaussSeidelRelaxation(int level) {
			



		}

		void Multigrid3D::sucessiveOverRelaxation(int level) {
			
		}

		void Multigrid3D::redBlackSuccessiveOverRelaxation(int level) {
			
		}

		void Multigrid3D::gaussJacobiRelaxation(int level) {
		}
	}
}