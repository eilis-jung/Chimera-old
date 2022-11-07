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

#include "Poisson/Multigrid2D.h"

namespace Chimera {
	namespace Poisson {
		/************************************************************************/
		/* Ctors and initialization												*/
		/************************************************************************/
		void Multigrid2D::initializeGridLevels(Scalar *rhs, Scalar *result) {
			int numGrids = 0;
			dimensions_t gridSize = m_pPoissonMatrix->getDimensions();
			dimensions_t tempGridSize;


			//Initializing root level: finest grid
			{
				m_subGridDimensions.push_back(gridSize);
				Scalar *pResidual = new Scalar[gridSize.x*gridSize.y];
				for(int i = 0; i < gridSize.x; i++) {
					for(int j = 0; j < gridSize.y; j++) {
						pResidual[getLevelIndex(0, i, j)] = 0;
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
					numGrids++;
					m_subGridDimensions.push_back(tempGridSize);
				}
				tempGridSize.x = 3;
				tempGridSize.y = 3;
				numGrids++;
				m_subGridDimensions.push_back(tempGridSize);

			} else if(m_isPeriodic) {
				numGrids = m_params.numSubgrids;
				tempGridSize = gridSize;
				for(int tempNumGrids = numGrids; tempNumGrids > 0; --tempNumGrids) {
					tempGridSize.x = tempGridSize.x/2;
					tempGridSize.y = tempGridSize.y/2 + 1;
					m_subGridDimensions.push_back(tempGridSize);
				}
			} else {
				numGrids = m_params.numSubgrids;
				tempGridSize = gridSize;
				for(int tempNumGrids = numGrids; tempNumGrids > 0; --tempNumGrids) {
					tempGridSize.x = tempGridSize.x/2 + 1;
					tempGridSize.y = tempGridSize.y/2 + 1;
					m_subGridDimensions.push_back(tempGridSize);
				}
			}

			/** Allocating sub-grids, sub-residuals, sub-rhs and sub-operators */
			for(int level = 1; level <= numGrids; level++) {
				tempGridSize = m_subGridDimensions[level];
				Scalar *pRhs, *pResult, *pResidual;
				pRhs	= new Scalar[tempGridSize.x*tempGridSize.y];
				pResult = new Scalar[tempGridSize.x*tempGridSize.y];
				pResidual = new Scalar[tempGridSize.x*tempGridSize.y];

				for(int i = 0; i < tempGridSize.x; i++) {
					for(int j = 0; j < tempGridSize.y; j++) {
						pRhs[getLevelIndex(level, i, j)] = 0;
						pResult[getLevelIndex(level, i, j)] = 0;
						pResidual[getLevelIndex(level, i, j)] = 0;
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


			for(int i = 0; i < gridSize.x; i++) {
				for(int j = 0; j < gridSize.y; j++) {
					Scalar volume;
					if(m_params.pCellsVolumes != NULL) {
						Scalar *pCellVolumes = m_cellsVolumes[0];
						volume = pCellVolumes[getLevelIndex(0, i, j)];
					} else {
						volume = m_params.gridRegularSpacing*m_params.gridRegularSpacing;
					}
					Scalar *rhs	= m_rhsVector[0];
					rhs[getLevelIndex(0, i, j)] = rhs[getLevelIndex(0, i, j)]/volume;
				}
			}
		}

		/************************************************************************/
		/* Multigrid functionalities                                            */
		/************************************************************************/
		void Multigrid2D::exactSolve(const Scalar *rhs, Scalar * result) {
			Scalar h = m_subGridDimensions.size()*m_params.gridRegularSpacing;
			dimensions_t gridDimensions = m_subGridDimensions[m_subGridDimensions.size() - 1];

			for (int i = 0;i < 3; i++)
				for (int j = 0; j < 3; j++)
					result[getLevelIndex(m_subGridDimensions.size() - 1, i, j)]=0.0;

			result[getLevelIndex(m_subGridDimensions.size() - 1, 1, 1)] = -h*h*rhs[getLevelIndex(m_subGridDimensions.size() - 1, 1, 1)]/4.0f;
		}


		/************************************************************************/
		/* Auxiliary                                                            */
		/************************************************************************/
		void Multigrid2D::copyBoundaries(int level) {
				Scalar * result = m_resultVector[level];
				dimensions_t gridDimensions = m_subGridDimensions[level];

				if(m_params.m_boundaries[West] == neumann) { //	West
					for(int j = 0; j < gridDimensions.y; j++) {
						result[getLevelIndex(level, 0, j)] = result[getLevelIndex(level, 1, j)];
					}
				}

				if(m_params.m_boundaries[East] == neumann) { //	East
					for(int j = 0; j < gridDimensions.y; j++) {
						result[getLevelIndex(level, gridDimensions.x - 1, j)] = result[getLevelIndex(level, gridDimensions.x - 2, j)];
					}
				}

				if(m_params.m_boundaries[South] == neumann) { // South
					for(int i = 0; i < gridDimensions.x; i++) {
						result[getLevelIndex(level, i, 0)] = result[getLevelIndex(level, i, 1)];
					}
				}

				if(m_params.m_boundaries[North] == neumann) { // North
					for(int i = 0; i < gridDimensions.x; i++) {
						result[getLevelIndex(level, i, gridDimensions.y - 1)] = result[getLevelIndex(level, i, gridDimensions.y - 2)];
					}
				} /*else if(m_params.pCellsVolumes != NULL && level > 0) {
					for(int i = 0; i < gridDimensions.x; i++) {
						result[getLevelIndex(level, i, gridDimensions.y - 1)] = result[getLevelIndex(level, i, gridDimensions.y - 2)];
					}
				}*/
			}

		/************************************************************************/
		/* Operator coarsening                                                  */
		/************************************************************************/
		PoissonMatrix * Multigrid2D::directInjectionCoarseningMatrix(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];

			//Initialize volumes used in relaxation phase
			if(m_params.pCellsVolumes != NULL) {
				Scalar *pFineCellsVolumes = m_cellsVolumes[0];
				Scalar *pCellsVolumes = new Scalar[gridDimensions.x*gridDimensions.y];	
				dimensions_t fineGridDimensions = m_subGridDimensions[0];

				for(int i = 0; i < gridDimensions.x; i++) {
					for(int j = 0; j < gridDimensions.y - 1; j++) {
						int fineI = static_cast<int> (i*pow(2.0f, level));
						int fineJ = static_cast<int> (j*pow(2.0f, level));

						//If the grids size are twice the size in each dimension, for 2D, volume is multiplied by 4
						Scalar totalVolume = pow(4.0f, level)*pFineCellsVolumes[getLevelIndex(0, fineI, fineJ)];
						pCellsVolumes[getLevelIndex(level, i, j)] = totalVolume;
					}
				}
				m_cellsVolumes.push_back(pCellsVolumes);
			}

			PoissonMatrix *pPoissonMatrix = new PoissonMatrix(gridDimensions, false, m_isPeriodic);

			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = gridDimensions.x;
			} else {
				startingI = 1; endingI = gridDimensions.x - 1;
			}

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < gridDimensions.y - 1; j++) {
					int fineI = static_cast<int> (i*pow(2.0f, level));
					int fineJ = static_cast<int> (j*pow(2.0f, level));

					Scalar pn = abs(m_pPoissonMatrix->getNorthValue(getLevelIndex(0, fineI, fineJ)));
					Scalar ps = abs(m_pPoissonMatrix->getSouthValue(getLevelIndex(0, fineI, fineJ)));
					Scalar pe = abs(m_pPoissonMatrix->getEastValue(getLevelIndex(0, fineI, fineJ)));
					Scalar pw = abs(m_pPoissonMatrix->getWestValue(getLevelIndex(0, fineI, fineJ)));

					pPoissonMatrix->setEastValue(getLevelIndex(level, i, j), m_pPoissonMatrix->getEastValue(getLevelIndex(0, fineI, fineJ)));
					pPoissonMatrix->setWestValue(getLevelIndex(level, i, j), m_pPoissonMatrix->getWestValue(getLevelIndex(0, fineI, fineJ)));
					pPoissonMatrix->setNorthValue(getLevelIndex(level, i, j), m_pPoissonMatrix->getNorthValue(getLevelIndex(0, fineI, fineJ)));
					pPoissonMatrix->setSouthValue(getLevelIndex(level, i, j), m_pPoissonMatrix->getSouthValue(getLevelIndex(0, fineI, fineJ)));
					
					
					if(m_isPeriodic) {
						pw = abs(m_pPoissonMatrix->getPeriodicWestValue(getLevelIndex(0, fineI, fineJ)));
						pPoissonMatrix->setPeriodicWestValue(getLevelIndex(level, i, j), -pw);
						pe = abs(m_pPoissonMatrix->getPeriodicEastValue(getLevelIndex(0, static_cast<int> (fineI + pow(2.0f, level) - 1), fineJ)));
						pPoissonMatrix->setPeriodicEastValue(getLevelIndex(level, i, j), -pe);
						if(pe == 0)
							pe = abs(m_pPoissonMatrix->getEastValue(getLevelIndex(0, fineI, fineJ)));
						if(pw == 0)
							pw = abs(m_pPoissonMatrix->getWestValue(getLevelIndex(0, fineI, fineJ)));
					}
					Scalar pc = pn + ps + pw + pe;
					pPoissonMatrix->setCentralValue(getLevelIndex(level, i, j), pn + ps + pw + pe);
				}
			}

			return pPoissonMatrix;
		}
	
		PoissonMatrix * Multigrid2D::garlekinCoarseningMatrix(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];

			PoissonMatrix *pPoissonMatrix = new PoissonMatrix(gridDimensions, PoissonMatrix::ninePointLaplace, false, m_isPeriodic);

			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = gridDimensions.x;
			} else {
				startingI = 1; endingI = gridDimensions.x - 1;
			}

			//Scale factors
			Scalar cornerSF = 1/8.0f;
			Scalar primSF = 3/4.0f;
			Scalar secSF = 1 - primSF;
			Scalar centralSF = 3/2.0f;
			
			//Finer level Poisson matrix
			PoissonMatrix *pFinePoissonMatrix = m_pPoissonMatrices[level - 1];

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < gridDimensions.y - 1; j++) {

					int fineI = i*2;
					int fineJ = j*2;

					Scalar pe = abs(pFinePoissonMatrix->getEastValue(getLevelIndex(level - 1, fineI, fineJ)));
					if(m_isPeriodic && i == gridDimensions.x - 1) {
						pe = abs(pFinePoissonMatrix->getPeriodicEastValue(getLevelIndex(level - 1, fineI, fineJ)));
					}
					Scalar pw = abs(pFinePoissonMatrix->getWestValue(getLevelIndex(level - 1, fineI, fineJ)));
					if(m_isPeriodic && i == 0) {
						pw = abs(pFinePoissonMatrix->getPeriodicWestValue(getLevelIndex(level - 1, fineI, fineJ)));
					}
					Scalar pn = abs(pFinePoissonMatrix->getNorthValue(getLevelIndex(level - 1, fineI, fineJ)));
					Scalar ps = abs(pFinePoissonMatrix->getSouthValue(getLevelIndex(level - 1, fineI, fineJ)));

					//Corners
					pPoissonMatrix->setNorthWestValue(getLevelIndex(level, i, j), -cornerSF*(pw + pn));
					pPoissonMatrix->setNorthEastValue(getLevelIndex(level, i, j), -cornerSF*(pe + pn));
					pPoissonMatrix->setSouthWestValue(getLevelIndex(level, i, j), -cornerSF*(pw + ps));
					pPoissonMatrix->setSouthEastValue(getLevelIndex(level, i, j), -cornerSF*(pe + ps));

					//X axis
					Scalar newPw = primSF*pw - secSF*0.5f*(pn + ps);
					newPw = -newPw;
					pPoissonMatrix->setWestValue(getLevelIndex(level, i, j), newPw);
					Scalar newPe = primSF*pe - secSF*0.5f*(pn + ps);
					newPe = -newPe;
					pPoissonMatrix->setEastValue(getLevelIndex(level, i, j), newPe);

					//Y axis
					Scalar newPn = primSF*pn - secSF*0.5f*(pe + pw);
					newPn = -newPn;
					pPoissonMatrix->setNorthValue(getLevelIndex(level, i, j), newPn);
					Scalar newPs = primSF*ps - secSF*0.5f*(pe + pw);
					newPs = -newPs;
					pPoissonMatrix->setEastValue(getLevelIndex(level, i, j), newPs);

					//Central
					pPoissonMatrix->setCentralValue(getLevelIndex(level, i, j), centralSF*(0.5f*(pe + pw) + 0.5f*(ps + pn)));


					/*if(m_isPeriodic) {
						pPoissonMatrix->setPeriodicWestValue(getLevelIndex(level, i, j), m_pPoissonMatrix->getPeriodicWestValue(getLevelIndex(0, fineI, fineJ)));
						pPoissonMatrix->setPeriodicEastValue(getLevelIndex(level, i, j), m_pPoissonMatrix->getPeriodicEastValue(getLevelIndex(0, fineI, fineJ)));
					}*/
				}
			}

			return pPoissonMatrix;
		}

		PoissonMatrix * Multigrid2D::geometricalAveragingCoarseningMatrix(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			dimensions_t fineGridDimensions = m_subGridDimensions[0];
			
			//Initialize volumes used in relaxation phase
			if(m_params.pCellsVolumes != NULL) {
				Scalar *pFineCellsVolumes = m_cellsVolumes[0];
				Scalar *pCellsVolumes = new Scalar[gridDimensions.x*gridDimensions.y];	
				dimensions_t fineGridDimensions = m_subGridDimensions[0];

				for(int i = 0; i < gridDimensions.x; i++) {
					for(int j = 0; j < gridDimensions.y - 1; j++) {
						int fineI = static_cast<int> (i*pow(2.0f, level));
						int fineJ = static_cast<int> (j*pow(2.0f, level));

						//If the grids size are twice the size in each dimension, for 2D, volume is multiplied by 4
						Scalar totalVolume = 0;
						for(int k = 0; k < pow(2.0f, level); k++) {
							for(int l = 0; l < pow(2.0f, level); l++) {
								totalVolume += pFineCellsVolumes[getLevelIndex(0, roundClamp(fineI + k, 0, fineGridDimensions.x), fineJ + l)];
							}
						}
						pCellsVolumes[getLevelIndex(level, i, j)] = totalVolume;
					}
				}
				m_cellsVolumes.push_back(pCellsVolumes);
			}

			if(gridDimensions.z == 0) {

			}
			Vector2 *pFineCellsAreas = (Vector2 *) m_params.pCellsAreas;

			PoissonMatrix *pPoissonMatrix = new PoissonMatrix(gridDimensions, false, m_isPeriodic);

			for(int i = 0; i < gridDimensions.x; i++) {
				for(int j = 0; j < gridDimensions.y - 1; j++) {
					int fineI = static_cast<int> (i*pow(2.0f, level));
					int fineJ = static_cast<int> (j*pow(2.0f, level));

					Scalar totalLeftFacesArea = 0, leftFaceArea = 0;
					for(int k = 0; k < pow(2.0f, level); k++) {
						totalLeftFacesArea += pFineCellsAreas[getLevelIndex(0, fineI, fineJ + k)].y;
					}
					Scalar pw = 0;
					for(int k = 0; k < pow(2.0f, level); k++) {
						leftFaceArea = pFineCellsAreas[getLevelIndex(0, fineI, fineJ + k)].y;
						pw += m_pPoissonMatrix->getWestValue(getLevelIndex(0, fineI, fineJ + k))*leftFaceArea/totalLeftFacesArea;
					}

					Scalar totalBottomFacesArea = 0, bottomFaceArea = 0;
					for(int k = 0; k < pow(2.0f, level); k++) {
						totalBottomFacesArea += pFineCellsAreas[getLevelIndex(0, roundClamp<int>(fineI + k, 0, fineGridDimensions.x), fineJ)].x;
					}
					Scalar ps = 0;
					for(int k = 0; k < pow(2.0f, level); k++) {
						bottomFaceArea = pFineCellsAreas[getLevelIndex(0, roundClamp<int>(fineI + k, 0, fineGridDimensions.x), fineJ)].x;
						ps += m_pPoissonMatrix->getSouthValue(getLevelIndex(0, roundClamp<int>(fineI + k, 0, fineGridDimensions.x), fineJ))*bottomFaceArea/totalBottomFacesArea;
					}

					pPoissonMatrix->setWestValue(getLevelIndex(level, i, j), pw);
					pPoissonMatrix->setSouthValue(getLevelIndex(level, i, j), ps);
				}
			}

			for(int i = 0; i < gridDimensions.x - 1; i++) {
				for(int j = 0; j < gridDimensions.y; j++) {
					pPoissonMatrix->setEastValue(getLevelIndex(level, i, j), pPoissonMatrix->getWestValue(getLevelIndex(level, i + 1, j)));
				}
			}

			for(int i = 0; i < gridDimensions.x; i++) {
				for(int j = 0; j < gridDimensions.y - 1; j++) {
					pPoissonMatrix->setNorthValue(getLevelIndex(level, i, j), pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j + 1)));
				}
			}

			for(int i = 0; i < gridDimensions.x; i++) {
				for(int j = 0; j < gridDimensions.y; j++) {
					Scalar centralValue = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j)) +
						pPoissonMatrix->getEastValue(getLevelIndex(level, i, j)) +
						pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j)) +
						pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j));
					pPoissonMatrix->setCentralValue(getLevelIndex(level, i, j), -centralValue);
				}
			}


			if(m_isPeriodic) {
				for(int j = 1; j < gridDimensions.y - 1; j++) {
					int fineJ = static_cast<int> (j*pow(2.0f, level));

					Scalar totalLeftFacesArea = 0, leftFaceArea = 0;
					for(int k = 0; k < pow(2.0f, level); k++) {
						totalLeftFacesArea += pFineCellsAreas[getLevelIndex(0, 0, fineJ + k)].y;
					}
					Scalar pw = 0;
					for(int k = 0; k < pow(2.0f, level); k++) {
						leftFaceArea = pFineCellsAreas[getLevelIndex(0, 0, fineJ + k)].y;
						pw += m_pPoissonMatrix->getPeriodicWestValue(getLevelIndex(0, 0, fineJ + k))*leftFaceArea/totalLeftFacesArea;
					}

					pPoissonMatrix->setPeriodicWestValue(getLevelIndex(level, 0, j), pw);
					pPoissonMatrix->setPeriodicEastValue(getLevelIndex(level, gridDimensions.x - 1, j), pw);

					Scalar centralValue = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, 0, j)) +
						pPoissonMatrix->getEastValue(getLevelIndex(level, 0, j)) +
						pPoissonMatrix->getNorthValue(getLevelIndex(level, 0, j)) +
						pPoissonMatrix->getSouthValue(getLevelIndex(level, 0, j));

					pPoissonMatrix->setCentralValue(getLevelIndex(level, 0, j), -centralValue);

					centralValue = pPoissonMatrix->getWestValue(getLevelIndex(level, gridDimensions.x - 1, j)) +
						pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, gridDimensions.x - 1, j)) +
						pPoissonMatrix->getNorthValue(getLevelIndex(level, gridDimensions.x - 1, j)) +
						pPoissonMatrix->getSouthValue(getLevelIndex(level, gridDimensions.x - 1, j));

					pPoissonMatrix->setCentralValue(getLevelIndex(level, gridDimensions.x - 1, j), -centralValue);
				}
			}

			return pPoissonMatrix;
		}

		/************************************************************************/
		/* Restriction functions                                                */
		/************************************************************************/
		void Multigrid2D::restrictFullWeighting(int level, const Scalar *fineGrid, Scalar *coarseGrid) {
			dimensions_t coarseGridSize = m_subGridDimensions[level + 1];
			dimensions_t fineGridSize = m_subGridDimensions[level];
			if(level ==  m_subGridDimensions.size() - 2 && coarseGridSize.x == 3 && coarseGridSize.y == 3) {
				coarseGrid[getLevelIndex(level + 1, 1, 1)] = 0.25f*(fineGrid[getLevelIndex(level, 1, 1)] + fineGrid[getLevelIndex(level, 1, 2)] +
					fineGrid[getLevelIndex(level, 2, 1)] + fineGrid[getLevelIndex(level, 2, 2)]); 
				return;
			}

			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = coarseGridSize.x;
			} else {
				startingI = 1; endingI = coarseGridSize.x - 1;
			}

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < coarseGridSize.y - 1; j++) {
					int fineI = i*2;
					int fineJ = j*2;

					int nextI = fineI + 1;
					int prevI = fineI - 1;
					if(m_isPeriodic && i == 0) {
						prevI = fineGridSize.x - 1;
					}

					Scalar restriction = 0.0625f*(4*fineGrid[getLevelIndex(level, fineI, fineJ)] +
												 2*(fineGrid[getLevelIndex(level, fineI, fineJ - 1)] + 
													fineGrid[getLevelIndex(level, fineI, fineJ + 1)] + 
													fineGrid[getLevelIndex(level, prevI, fineJ)] + 
													fineGrid[getLevelIndex(level, nextI, fineJ)]) +
													fineGrid[getLevelIndex(level, prevI, fineJ - 1)] + 
													fineGrid[getLevelIndex(level, prevI, fineJ + 1)] +
													fineGrid[getLevelIndex(level, nextI, fineJ - 1)] + 
													fineGrid[getLevelIndex(level, nextI,fineJ + 1)]
												);

					coarseGrid[getLevelIndex(level + 1, i, j)] = restriction;

				}
			}

			///************************************************************************/
			///* Boundaries                                                           */
			///************************************************************************/
			//Bottom left
			//coarseGrid[getLevelIndex(level + 1, 0, 0)] = 0.0625*(11*fineGrid[getLevelIndex(level, 0, 0)] +
			//													 2*(fineGrid[getLevelIndex(level, 1, 0)] + 
			//														fineGrid[getLevelIndex(level, 0, 1)]) +
			//														fineGrid[getLevelIndex(level, 1, 1)]);
			////Top left
			//coarseGrid[getLevelIndex(level + 1, 0, coarseGridSize.y - 1)] = 0.0625*(11*fineGrid[getLevelIndex(level, 0, (coarseGridSize.y - 1)*2)] +
			//																		2*(fineGrid[getLevelIndex(level, 1, (coarseGridSize.y - 1)*2)] + 
			//																			fineGrid[getLevelIndex(level, 0, (coarseGridSize.y - 2)*2)]) +
			//																			fineGrid[getLevelIndex(level, 1, (coarseGridSize.y - 2)*2)]);
			////Bottom right
			//coarseGrid[getLevelIndex(level + 1, coarseGridSize.x - 1, 0)] = 0.0625*(11*fineGrid[getLevelIndex(level, (coarseGridSize.x - 1)*2, 0)] +
			//																		 2*(fineGrid[getLevelIndex(level, (coarseGridSize.x - 1)*2, 1)] + 
			//																			fineGrid[getLevelIndex(level, (coarseGridSize.x - 2)*2, 0)]) +
			//																			fineGrid[getLevelIndex(level, (coarseGridSize.x - 2)*2, 1)]);
			////Top right
			//coarseGrid[getLevelIndex(level + 1, 0, coarseGridSize.y - 1)] = 0.0625*(11*fineGrid[getLevelIndex(level, (coarseGridSize.x - 1)*2, (coarseGridSize.y - 1)*2)] +
			//													 2*(fineGrid[getLevelIndex(level, (coarseGridSize.x - 2)*2, (coarseGridSize.y - 1)*2)] + 
			//														fineGrid[getLevelIndex(level, (coarseGridSize.x - 1)*2, (coarseGridSize.y - 2)*2)]) +
			//														fineGrid[getLevelIndex(level, (coarseGridSize.x - 2)*2, (coarseGridSize.y - 2)*2)]);

			///*for(int j = 1; j < coarseGridSize.y - 1; j++) {
			//	int fineJ = j*2;
			//	coarseGrid[getLevelIndex(level + 1, 0, j)] = 0.0625*(8*fineGrid[getLevelIndex(level, 0, fineJ)] +
			//		2*(fineGrid[getLevelIndex(level, 0, fineJ + 1)] + 
			//		fineGrid[getLevelIndex(level, 1, fineJ)] + 
			//		fineGrid[getLevelIndex(level, 0, fineJ - 1)]) +
			//		fineGrid[getLevelIndex(level, 1, fineJ + 1)] + 
			//		fineGrid[getLevelIndex(level, 1, fineJ - 1)]);

			//	coarseGrid[getLevelIndex(level + 1, coarseGridSize.x - 1, j)] = 0.0625*(8*fineGrid[getLevelIndex(level, (coarseGridSize.x - 1)*2, fineJ)] +
			//		2*(fineGrid[getLevelIndex(level, (coarseGridSize.x - 2)*2, fineJ)] + 
			//		fineGrid[getLevelIndex(level, (coarseGridSize.x - 1)*2, fineJ + 1)]  + 
			//		fineGrid[getLevelIndex(level, (coarseGridSize.x - 1)*2, fineJ - 1)]) +
			//		fineGrid[getLevelIndex(level, (coarseGridSize.x - 2)*2, fineJ + 1)] + 
			//		fineGrid[getLevelIndex(level, (coarseGridSize.x - 2)*2, fineJ - 1)]);
			//}*/

			for(int i = 1; i < coarseGridSize.x - 1; i++) {
				int fineI = i*2;
				coarseGrid[getLevelIndex(level + 1, i, 0)] = 0.0625f*(8*fineGrid[getLevelIndex(level, fineI, 0)] +
					2*(fineGrid[getLevelIndex(level, fineI, 1)] + 
					fineGrid[getLevelIndex(level, fineI + 1, 0)] + 
					fineGrid[getLevelIndex(level, fineI - 1, 0)]) +
					fineGrid[getLevelIndex(level, fineI + 1, 1)] + 
					fineGrid[getLevelIndex(level, fineI - 1, 1)]);

				coarseGrid[getLevelIndex(level + 1, i, coarseGridSize.y - 1)] = 0.0625f*(8*fineGrid[getLevelIndex(level, fineI, (coarseGridSize.y - 1)*2)] +
					2*(fineGrid[getLevelIndex(level, fineI, (coarseGridSize.y - 2)*2)] + 
					fineGrid[getLevelIndex(level, fineI + 1, (coarseGridSize.y - 1)*2)] + 
					fineGrid[getLevelIndex(level, fineI - 1, (coarseGridSize.y - 1)*2)]) +
					fineGrid[getLevelIndex(level, fineI + 1, (coarseGridSize.y - 2)*2)] + 
					fineGrid[getLevelIndex(level, fineI - 1, (coarseGridSize.y - 2)*2)]);
			}

		}

		void Multigrid2D::restrictHalfWeighting(int level, const Scalar *fineGrid, Scalar *coarseGrid) {
			dimensions_t coarseGridSize = m_subGridDimensions[level + 1];

			for(int i = 1; i < coarseGridSize.x - 1; i++) {
				for(int j = 1; j < coarseGridSize.y - 1; j++) {
					int fineI = i*2;
					int fineJ = j*2;
					Scalar restriction = 0.5f*fineGrid[getLevelIndex(level, fineI, fineJ)] 
										+ 0.125f * ( fineGrid[getLevelIndex(level, fineI - 1, fineJ)]
													+ fineGrid[getLevelIndex(level, fineI + 1, fineJ)]
													+ fineGrid[getLevelIndex(level, fineI, fineJ - 1)]
													+ fineGrid[getLevelIndex(level, fineI, fineJ + 1)]);
					coarseGrid[getLevelIndex(level + 1, i, j)] = restriction;
				}
			}

			for (int i = 0; i < coarseGridSize.x; i++) {
				int fineI = i*2;
				int cc = (coarseGridSize.y - 1)*2;
				coarseGrid[getLevelIndex(level + 1, i, 0)] = fineGrid[getLevelIndex(level, fineI, 0)];
				coarseGrid[getLevelIndex(level + 1, i, coarseGridSize.y - 1)] = fineGrid[getLevelIndex(level, fineI, cc)];
			}

			for (int j = 0; j < coarseGridSize.y; j++) {
				int fineJ = j*2;
				int cc = (coarseGridSize.x - 1)*2;
				coarseGrid[getLevelIndex(level + 1, 0, j)] = fineGrid[getLevelIndex(level, 0, fineJ)];
				coarseGrid[getLevelIndex(level + 1, coarseGridSize.x - 1, j)] = fineGrid[getLevelIndex(level, cc, fineJ)];
			}
		}

		/************************************************************************/
		/* Prolongation functions                                               */
		/************************************************************************/
		void Multigrid2D::prolongLinearInterpolation(int level, const Scalar *coarseGrid, Scalar *fineGrid) {
			dimensions_t coarseGridSize = m_subGridDimensions[level + 1];

			//Bilinear interpolation
			for (int jc = 0; jc < m_subGridDimensions[level + 1].y; jc++) 
				for (int ic = 0; ic < m_subGridDimensions[level + 1].x; ic++) 
					fineGrid[getLevelIndex(level,2*ic, 2*jc)] = coarseGrid[getLevelIndex(level + 1, ic, jc)];
			
			if(m_isPeriodic) {
				for (int jf = 0; jf < m_subGridDimensions[level].y; jf += 2) {
					for (int iif = 1; iif < m_subGridDimensions[level].x - 1; iif += 2) {
						fineGrid[getLevelIndex(level, iif, jf)] = 0.5f*(fineGrid[getLevelIndex(level, iif + 1, jf)] + fineGrid[getLevelIndex(level, iif - 1, jf)]);
					}
					fineGrid[getLevelIndex(level, m_subGridDimensions[level].x - 1, jf)] = 0.5f*(	fineGrid[getLevelIndex(level, 0, jf)] 
																								+	fineGrid[getLevelIndex(level, m_subGridDimensions[level].x - 2, jf)]);
				}
			} else {
				for (int jf = 0; jf < m_subGridDimensions[level].y; jf += 2)
					for (int iif = 1; iif < m_subGridDimensions[level].x - 1; iif += 2)
						fineGrid[getLevelIndex(level, iif, jf)] = 0.5f*(fineGrid[getLevelIndex(level, iif + 1, jf)] + fineGrid[getLevelIndex(level, iif - 1, jf)]);
			}
			
	
			for (int jf = 1; jf < m_subGridDimensions[level].y - 1; jf += 2) 
				for (int iif = 0; iif < m_subGridDimensions[level].x; iif++) 
					fineGrid[getLevelIndex(level, iif, jf)] = 0.5f*(fineGrid[getLevelIndex(level, iif, jf + 1)] + fineGrid[getLevelIndex(level, iif, jf - 1)]);

		}

		/************************************************************************/
		/* Level functions                                                      */
		/************************************************************************/
		void Multigrid2D::updateResiduals(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result     = m_resultVector[level];
			Scalar *rhs	       = m_rhsVector[level];
			Scalar *pResiduals = m_residualVector[level];

			DoubleScalar totalError = 0.0l, maxRhs = 0.0l;

			Scalar h, h2 = 0.0f;
			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.gridRegularSpacing*(level + 1);
				h2 = 1.0f/(h*h);
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
					if(m_params.pSolidCells != NULL && level == 0 && (isSolidCell(level, i, j) || isBoundaryCell(i, j))) {
						pResiduals[getLevelIndex(level, i, j)] = 0;
						continue;
					}
					Scalar pe, pw = 0;

					PoissonMatrix *pPoissonMatrix = m_pPoissonMatrices[level];

					if(m_params.pCellsVolumes != NULL) { //Non-regular grid
						Scalar *pCellVolumes = m_cellsVolumes[level];
						h2 = 1.0f/(pCellVolumes[getLevelIndex(level, i, j)]);
					}

					int nextI = i + 1;
					pe = pPoissonMatrix->getEastValue(getLevelIndex(level, i, j));
					if(m_isPeriodic && i == gridDimensions.x - 1) {
						nextI = 0;
						pe = pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, i, j));
					}

					int prevI = i - 1;
					pw = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j));
					if(m_isPeriodic && i == 0) {
						prevI = gridDimensions.x - 1;
						pw = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, i, j));
					}

					Scalar residual = result[getLevelIndex(level, nextI, j)]*pe
						+  	result[getLevelIndex(level, prevI, j)]*pw
						+	result[getLevelIndex(level, i, j - 1)]*pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j))
						+	result[getLevelIndex(level, i, j + 1)]*pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j))
						+	result[getLevelIndex(level, i, j)]*pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j));


					residual = -residual;
					residual *= h2;
					residual += rhs[getLevelIndex(level, i, j)];

					pResiduals[getLevelIndex(level, i, j)] = residual;

					Scalar pn = pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j));
					Scalar ps = pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j));
					Scalar pc = pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j));

					Scalar centralBalance = pc + ps + pn + pe + pw;

					if(level == 0) {
						totalError += residual*residual;
						maxRhs += rhs[getLevelIndex(level, i, j)]*rhs[getLevelIndex(level, i, j)];
						/*if(abs(rhs[getLevelIndex(level, i, j)]) > maxRhs) {
							maxRhs = abs(rhs[getLevelIndex(level, i, j)]);
						}*/
					}

				}
			}

			if(level == 0) {
				totalError /= (gridDimensions.x)*(gridDimensions.y);
				maxRhs /= (gridDimensions.x)*(gridDimensions.y);
				if(maxRhs != 0) {
					m_lastResidual = static_cast<Scalar> (totalError/maxRhs);
				} else {
					m_lastResidual = static_cast<Scalar> (totalError);
				}
				
			}

			for(int j = 0; j < gridDimensions.y; j++) {
				pResiduals[getLevelIndex(level, 0, j)] = 0;
				pResiduals[getLevelIndex(level, gridDimensions.x - 1, j)] = 0;
			}
		}

		void Multigrid2D::addNextLevelErrors(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result = m_resultVector[level];

			Scalar *temp = new Scalar[gridDimensions.x*gridDimensions.y];

			prolong(level, m_resultVector[level + 1], temp);
			
			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = gridDimensions.x;
			} else {
				startingI = 1; endingI = gridDimensions.x - 1;
			}

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < gridDimensions.y - 1; j++) {
					Scalar tempVal = temp[getLevelIndex(level, i, j)];
					if(m_params.pSolidCells != NULL && level == 0 && (isSolidCell(level, i, j) || isBoundaryCell(i, j))) {

					} else {
						result[getLevelIndex(level, i, j)] += temp[getLevelIndex(level, i, j)];
					}
				}
			}

			delete temp;
		}

		/************************************************************************/
		/* Smoothers                                                            */
		/************************************************************************/
		void Multigrid2D::gaussSeidelRelaxation(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result = m_resultVector[level];
			Scalar *rhs	   = m_rhsVector[level];

			Scalar h2, h = 0;

			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.gridRegularSpacing*(level + 1);
				h2 = (h*h);
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
					if(m_params.pSolidCells != NULL && level == 0 && (isSolidCell(0, i, j) || isBoundaryCell(i, j))) {

						if(isBoundaryCell(i, j))
							continue;

						result[getLevelIndex(level, i, j)] = 0.0f;
					} else {
						Scalar pe, pw;

						PoissonMatrix *pPoissonMatrix = m_pPoissonMatrices[level];

						if(m_params.pCellsVolumes != NULL) {
							Scalar *pCellVolumes = m_cellsVolumes[level];
							h2 = pCellVolumes[getLevelIndex(level, i, j)];
						}



						int nextI = i + 1;
						pe = pPoissonMatrix->getEastValue(getLevelIndex(level, i, j));
						if(m_isPeriodic && i == gridDimensions.x - 1) {
							nextI = 0;
							pe = pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, i, j));
						}

						int prevI = i - 1;
						pw = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j));
						if(m_isPeriodic && i == 0) {
							prevI = gridDimensions.x - 1;
							pw = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, i, j));
						}

						Scalar pn = pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j));
						Scalar ps = pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j));
						Scalar relaxationStep =	 h2*rhs[getLevelIndex(level, i, j)] - 
							(	result[getLevelIndex(level, nextI, j)]*pe
							+  	result[getLevelIndex(level, prevI, j)]*pw
							+	result[getLevelIndex(level, i, j - 1)]*pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j))
							+	result[getLevelIndex(level, i, j + 1)]*pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j))
							);

						result[getLevelIndex(level, i, j)] = (1/pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j))) * relaxationStep;
					}
				}
			}

		}

		void Multigrid2D::gaussSeidelRelaxationExtended(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result = m_resultVector[level];
			Scalar *rhs	   = m_rhsVector[level];

			Scalar h2, h = 0;
			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.gridRegularSpacing*(level + 1);
				h2 = (h*h);
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
					Scalar pe, pw;

					PoissonMatrix *pPoissonMatrix = m_pPoissonMatrices[level];

					if(m_params.pCellsVolumes != NULL) {
						Scalar *pCellVolumes = m_cellsVolumes[level];
						h2 = pCellVolumes[getLevelIndex(level, i, j)];
					}

					int nextI = i + 1;
					pe = pPoissonMatrix->getEastValue(getLevelIndex(level, i, j));
					if(m_isPeriodic && i == gridDimensions.x - 1) {
						nextI = 0;
						pe = pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, i, j));
					}

					int prevI = i - 1;
					pw = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j));
					if(m_isPeriodic && i == 0) {
						prevI = gridDimensions.x - 1;
						pw = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, i, j));
					}

					Scalar relaxationStep =	 h2*rhs[getLevelIndex(level, i, j)] - 
						(	result[getLevelIndex(level, nextI, j)]*pe
						+  	result[getLevelIndex(level, prevI, j)]*pw
						+	result[getLevelIndex(level, i, j - 1)]*pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j))
						+	result[getLevelIndex(level, i, j + 1)]*pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j))
						//  Additional points
						+	result[getLevelIndex(level, nextI, j + 1)]*pPoissonMatrix->getNorthWestValue(getLevelIndex(level, i, j))
						+	result[getLevelIndex(level, prevI, j + 1)]*pPoissonMatrix->getNorthEastValue(getLevelIndex(level, i, j))
						+	result[getLevelIndex(level, nextI, j - 1)]*pPoissonMatrix->getSouthWestValue(getLevelIndex(level, i, j))
						+	result[getLevelIndex(level, prevI, j - 1)]*pPoissonMatrix->getSouthEastValue(getLevelIndex(level, i, j))
						);

					result[getLevelIndex(level, i, j)] = (1/pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j))) * relaxationStep;
				}
			}
		}

		void Multigrid2D::redBlackGaussSeidelRelaxation(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result = m_resultVector[level];
			Scalar *rhs	   = m_rhsVector[level];
			Scalar *pResiduals = m_residualVector[level];

			Scalar h2, h = 0;
			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.gridRegularSpacing*(level + 1);
				h2 = (h*h);
			}

			copyBoundaries(level);

			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = gridDimensions.x;
			} else {
				startingI = 1; endingI = gridDimensions.x - 1;
			}


			for(int ipass = 0; ipass < 2; ipass++) {
				for(int i = startingI, iniJ = ipass + 1; i < endingI; i++, iniJ = 3 - iniJ) {
#pragma omp parallel for
					for(int j = iniJ; j < gridDimensions.y - 1; j += 2) {

						PoissonMatrix *pPoissonMatrix = m_pPoissonMatrices[level];

						if(m_params.pCellsVolumes != NULL) {
							Scalar *pCellVolumes = m_cellsVolumes[level];
							h2 = pCellVolumes[getLevelIndex(level, i, j)];
						}

						int nextI = i + 1;
						Scalar pe = pPoissonMatrix->getEastValue(getLevelIndex(level, i, j));
						if(m_isPeriodic && i == gridDimensions.x - 1) {
							nextI = 0;
							pe = pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, i, j));
						}

						int prevI = i - 1;
						Scalar pw = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j));
						if(m_isPeriodic && i == 0) {
							prevI = gridDimensions.x - 1;
							pw = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, i, j));
						}

						Scalar pn = pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j));
						Scalar ps = pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j));
						Scalar relaxationStep =	 h2*rhs[getLevelIndex(level, i, j)] - 
							(	result[getLevelIndex(level, nextI, j)]*pe
							+  	result[getLevelIndex(level, prevI, j)]*pw
							+	result[getLevelIndex(level, i, j - 1)]*pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j))
							+	result[getLevelIndex(level, i, j + 1)]*pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j))
							);

						result[getLevelIndex(level, i, j)] = (1/pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j))) * relaxationStep;
					}
				}
			}



		}

		void Multigrid2D::sucessiveOverRelaxation(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result = m_resultVector[level];
			Scalar *rhs	   = m_rhsVector[level];

			Scalar h2, h = 0;
			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.gridRegularSpacing*(level + 1);
				h2 = (h*h);
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
					if(m_params.pSolidCells != NULL && level == 0 && (isSolidCell(0, i, j) || isBoundaryCell(i, j))) {

						if(isBoundaryCell(i, j))
							continue;

						result[getLevelIndex(level, i, j)] = 0.0f;
					} else {
						Scalar pe, pw;

						PoissonMatrix *pPoissonMatrix = m_pPoissonMatrices[level];

						if(m_params.pCellsVolumes != NULL) {
							Scalar *pCellVolumes = m_cellsVolumes[level];
							h2 = pCellVolumes[getLevelIndex(level, i, j)];
						}



						int nextI = i + 1;
						pe = pPoissonMatrix->getEastValue(getLevelIndex(level, i, j));
						if(m_isPeriodic && i == gridDimensions.x - 1) {
							nextI = 0;
							pe = pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, i, j));
						}

						int prevI = i - 1;
						pw = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j));
						if(m_isPeriodic && i == 0) {
							prevI = gridDimensions.x - 1;
							pw = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, i, j));
						}

						Scalar pn = pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j));
						Scalar ps = pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j));
						Scalar relaxationStep =	 h2*rhs[getLevelIndex(level, i, j)] - 
							(	result[getLevelIndex(level, nextI, j)]*pe
							+  	result[getLevelIndex(level, prevI, j)]*pw
							+	result[getLevelIndex(level, i, j - 1)]*pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j))
							+	result[getLevelIndex(level, i, j + 1)]*pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j))
							);

						result[getLevelIndex(level, i, j)] =  result[getLevelIndex(level, i, j)]*(1 - m_params.wSor)
							+ (m_params.wSor/pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j))) * relaxationStep;
					}
				}
			}
		}

		void Multigrid2D::redBlackSuccessiveOverRelaxation(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result = m_resultVector[level];
			Scalar *rhs	   = m_rhsVector[level];
			Scalar *pResiduals = m_residualVector[level];


			Scalar h2, h = 0;
			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.gridRegularSpacing*(level + 1);
				h2 = (h*h);
			}

			copyBoundaries(level);

			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = gridDimensions.x;
			} else {
				startingI = 1; endingI = gridDimensions.x - 1;
			}


			for(int ipass = 0; ipass < 2; ipass++) {
				for(int i = startingI, iniJ = ipass + 1; i < endingI; i++, iniJ = 3 - iniJ) {
					for(int j = iniJ; j < gridDimensions.y - 1; j += 2) {
						if(m_params.pSolidCells != NULL && level == 0 && (isSolidCell(0, i, j) || isBoundaryCell(i, j))) {

							if(isBoundaryCell(i, j))
								continue;

							result[getLevelIndex(level, i, j)] = 0.0f;
						} else {
							Scalar pe, pw;

							PoissonMatrix *pPoissonMatrix = m_pPoissonMatrices[level];

							if(m_params.pCellsVolumes != NULL) {
								Scalar *pCellVolumes = m_cellsVolumes[level];
								h2 = pCellVolumes[getLevelIndex(level, i, j)];
							}



							int nextI = i + 1;
							pe = pPoissonMatrix->getEastValue(getLevelIndex(level, i, j));
							if(m_isPeriodic && i == gridDimensions.x - 1) {
								nextI = 0;
								pe = pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, i, j));
							}

							int prevI = i - 1;
							pw = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j));
							if(m_isPeriodic && i == 0) {
								prevI = gridDimensions.x - 1;
								pw = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, i, j));
							}

							Scalar pn = pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j));
							Scalar ps = pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j));
							Scalar relaxationStep =	 h2*rhs[getLevelIndex(level, i, j)] - 
								(	result[getLevelIndex(level, nextI, j)]*pe
								+  	result[getLevelIndex(level, prevI, j)]*pw
								+	result[getLevelIndex(level, i, j - 1)]*pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j))
								+	result[getLevelIndex(level, i, j + 1)]*pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j))
								);

							result[getLevelIndex(level, i, j)] = result[getLevelIndex(level, i, j)]*(1 - m_params.wSor)
								+ (m_params.wSor/pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j))) * relaxationStep;
						}
					}
				}
			}
		}

		void Multigrid2D::gaussJacobiRelaxation(int level) {
			dimensions_t gridDimensions = m_subGridDimensions[level];
			Scalar *result = m_resultVector[level];
			Scalar *rhs	   = m_rhsVector[level];

			Scalar *oldResult = new Scalar[gridDimensions.x*gridDimensions.y];
			for(int i = 0; i < gridDimensions.x; i++) {
				for(int j = 0; j < gridDimensions.y; j++) {
					oldResult[getLevelIndex(level, i, j)] = result[getLevelIndex(level, i, j)];
				}
			}

			Scalar h2, h = 0;

			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.gridRegularSpacing*(level + 1);
				h2 = (h*h);
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
					Scalar pe, pw;

					PoissonMatrix *pPoissonMatrix = m_pPoissonMatrices[level];

					if(m_params.pCellsVolumes != NULL) {
						Scalar *pCellVolumes = m_cellsVolumes[level];
						h2 = pCellVolumes[getLevelIndex(level, i, j)];
					}

					int nextI = i + 1;
					pe = pPoissonMatrix->getEastValue(getLevelIndex(level, i, j));
					if(m_isPeriodic && i == gridDimensions.x - 1) {
						nextI = 0;
						pe = pPoissonMatrix->getPeriodicEastValue(getLevelIndex(level, i, j));
					}

					int prevI = i - 1;
					pw = pPoissonMatrix->getWestValue(getLevelIndex(level, i, j));
					if(m_isPeriodic && i == 0) {
						prevI = gridDimensions.x - 1;
						pw = pPoissonMatrix->getPeriodicWestValue(getLevelIndex(level, i, j));
					}

					Scalar pn = pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j));
					Scalar ps = pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j));
					Scalar relaxationStep =	 h2*rhs[getLevelIndex(level, i, j)] - 
						(	oldResult[getLevelIndex(level, nextI, j)]*pe
						+  	oldResult[getLevelIndex(level, prevI, j)]*pw
						+	oldResult[getLevelIndex(level, i, j - 1)]*pPoissonMatrix->getSouthValue(getLevelIndex(level, i, j))
						+	oldResult[getLevelIndex(level, i, j + 1)]*pPoissonMatrix->getNorthValue(getLevelIndex(level, i, j))
						);

					result[getLevelIndex(level, i, j)] = (1/pPoissonMatrix->getCentralValue(getLevelIndex(level, i, j))) * relaxationStep;
				}
			}

			delete oldResult;

		}
	}
}