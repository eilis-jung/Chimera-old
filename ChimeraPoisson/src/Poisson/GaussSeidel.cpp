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

#include "Poisson/GaussSeidel.h"
#include <omp.h>

namespace Chimera {

	namespace Poisson {


		/************************************************************************/
		/* OpenMP riddle                                                        */
		/************************************************************************/

		/**int tSweep = 0;
		if(t2Sweep == 0)
		if(tSweep % 2 == 0)
		tSweep = 1;
		else tSweep = 2;
		else 
		if(tSweep % 2 == 0)
		tSweep = 2;
		else tSweep = 1;

		int j;
		 **/

		/************************************************************************/
		/* Solving                                                              */
		/************************************************************************/
		void GaussSeidel::redBlackIteration(int sweep, const Scalar *rhs, Scalar *result) {
			Scalar h = 1/((float)m_dimensions.x);
			Scalar hh = h*h;
			

			////#pragma omp parallel for
			//for(int i = 2; i < m_dimensions.x - 1; i += 2) {
			//	//#pragma omp parallel for private(j)
			//	for(int j = 1; j < m_dimensions.y - 1; j += 2) {
			//		Scalar relaxationStep = hh*rhs[getIndex(i, j)] - (m_pPoissonMatrix->getEastValue(getIndex(i, j))*result[getIndex(i + 1, j)]
			//								+ m_pPoissonMatrix->getNorthValue(getIndex(i, j))*result[getIndex(i, j + 1)]
			//								+ m_pPoissonMatrix->getWestValue(getIndex(i, j))*result[getIndex(i - 1, j)]
			//								+ m_pPoissonMatrix->getSouthValue(getIndex(i, j))*result[getIndex(i, j - 1)]);

			//		result[getIndex(i, j)] = (1/m_pPoissonMatrix->getCentralValue(getIndex(i, j))) * relaxationStep;
			//	}	
			//}
			////#pragma omp parallel for
			//for(int i = 1; i < m_dimensions.x - 1; i += 2) {
			//	//#pragma omp parallel for private(j)
			//	for(int j = 2; j < m_dimensions.y - 1; j += 2) {
			//		Scalar relaxationStep = hh*rhs[getIndex(i, j)] - (m_pPoissonMatrix->getEastValue(getIndex(i, j))*result[getIndex(i + 1, j)]
			//								+ m_pPoissonMatrix->getNorthValue(getIndex(i, j))*result[getIndex(i, j + 1)]
			//								+ m_pPoissonMatrix->getWestValue(getIndex(i, j))*result[getIndex(i - 1, j)]
			//								+ m_pPoissonMatrix->getSouthValue(getIndex(i, j))*result[getIndex(i, j - 1)]);

			//		result[getIndex(i, j)] = (1/m_pPoissonMatrix->getCentralValue(getIndex(i, j))) * relaxationStep;
			//	}	
			//}
			
			//#pragma omp parallel for
			for(int i = 1; i < m_dimensions.x - 1; i++) {
				//#pragma omp parallel for private(j)
				for(int j = sweep; j < m_dimensions.y - 1; j+=2) {
					Scalar relaxationStep = hh*rhs[getIndex(i, j)] - (m_pPoissonMatrix->getEastValue(getIndex(i, j))*result[getIndex(i + 1, j)]
																	+ m_pPoissonMatrix->getNorthValue(getIndex(i, j))*result[getIndex(i, j + 1)]
																	+ m_pPoissonMatrix->getWestValue(getIndex(i, j))*result[getIndex(i - 1, j)]
																	+ m_pPoissonMatrix->getSouthValue(getIndex(i, j))*result[getIndex(i, j - 1)]);

					result[getIndex(i, j)] = (1/m_pPoissonMatrix->getCentralValue(getIndex(i, j))) * relaxationStep;
				}
				sweep = 3 - sweep;
			}
			//#pragma omp parallel for
			//for(int i = 1; i < m_dimensions.x - 1; i++) {
			//	int j;
			//	#pragma omp parallel for private(j)
			//	for(j = 1; j < m_dimensions.y - 1; j++) {
			//		/*Scalar residual = rhs[getIndex(i, j)] -  ( m_pPoissonMatrix->getEastValue(getIndex(i, j))*result[getIndex(i + 1, j)]
			//													+ m_pPoissonMatrix->getNorthValue(getIndex(i, j))*result[getIndex(i, j + 1)]
			//													+ m_pPoissonMatrix->getWestValue(getIndex(i, j))*result[getIndex(i - 1, j)]
			//													+ m_pPoissonMatrix->getSouthValue(getIndex(i, j))*result[getIndex(i, j - 1)]
			//													+ m_pPoissonMatrix->getCentralValue(getIndex(i, j))*result[getIndex(i, j)]);*/
			//		
			//		Scalar residual = i*i;																		
			//		//#pragma omp atomic
			//		m_totalResidual += sqrt(residual*residual);
			//	}
			//}
		}
		void GaussSeidel::serialIterationForCutCells(const Scalar *rhs, Scalar *result) {
			// Array result is for grid with borders. Different from PoissonMatrix, its dimensions are with +2.
			// Also for array rhs.

			// Initialize results from previous step
			int totalLength = m_pPoissonMatrix->getNumRows() + m_pPoissonMatrix->getNumberAdditionalCells();

			Scalar dx = 0.0625;
			Scalar hh = 1;

			for (uint i = 0; i < (m_dimensions.x + 2) * (m_dimensions.y + 2); i++) {
				m_pPrevResult[i] = result[i];
			}
			for (uint i = 0; i < m_pPoissonMatrix->getNumberAdditionalCells(); i++) {
				m_pPrevResultAdditional[i] = m_cutCellsPressure[i];
			}
			for (uint index = 0; index < totalLength; index++) {
				Scalar coeff = (1 / m_pPoissonMatrix->getValue(index, index));

				Scalar dk = 0;
				if (index >= m_pPoissonMatrix->getNumRows()) {
					dk = m_cutCellsDivergence[getIndexForArray(index)];
				}
				else {
					dk = rhs[getIndexForArray(index)];
				}
				vector <pair <uint, Scalar>> row = m_pPoissonMatrix->getRowCOOMatrix(index);
				Scalar relaxationStep(0);
				vector <pair <uint, Scalar>>::iterator it;
				for (it = row.begin(); it != row.end(); it++) {
					uint otherIndex = it->first;
					uint test = getIndexForArray(otherIndex);
					if (otherIndex != index) {
						Scalar p = it->second;
						Scalar x = 0;
						//if (otherIndex > index) {
						//	if (otherIndex >= m_pPoissonMatrix->getNumRows()) {
						//		x = m_pPrevResultAdditional[getIndexForArray(otherIndex)];
						//	}
						//	else {
						//		x = m_pPrevResult[getIndexForArray(otherIndex)];
						//	}
						//}
						//else {
						if (otherIndex >= m_pPoissonMatrix->getNumRows())
							x = m_cutCellsPressure[getIndexForArray(otherIndex)];
						else {
							x = result[getIndexForArray(otherIndex)];
						}
						//}
						relaxationStep += p*x;
					}
				}
				if (index >= m_pPoissonMatrix->getNumRows()) {
					m_cutCellsPressure[getIndexForArray(index)] = coeff * (hh*dk - relaxationStep);
				}
				else {
					result[getIndexForArray(index)] = coeff * (hh*dk - relaxationStep);
				}
			}
		}

		uint GaussSeidel::getIndexForArray(uint index) {
			int j = index / (m_dimensions.x);
			int i = index % (m_dimensions.x);
			uint actualIndForRhs = index;
			if (index < m_pPoissonMatrix->getNumRows())
				// regular cells
				actualIndForRhs = (j + 1) * (m_dimensions.x + 2) + i + 1;
			else
				actualIndForRhs = (index - m_pPoissonMatrix->getNumRows());
			return actualIndForRhs;
		}

		void GaussSeidel::serialIteration(const Scalar *rhs, Scalar *result) {
			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = m_dimensions.x;
			} else {
				startingI = 1; endingI = m_dimensions.x - 1;
			}
			if (m_pPoissonMatrix->getNumberAdditionalCells() > 0)
				return serialIterationForCutCells(rhs, result);

			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < m_dimensions.y - 1; j++) {
					Scalar relaxationStep, pw, pe;

					if(disabledCell(i, j) || boundaryCell(i, j)) {
					//	//Scalar pressure = result[getIndex(i, j)];
						continue; 
					}

					int nextI = i + 1;
					pe = m_pPoissonMatrix->getEastValue(getIndex(i, j));
					if(m_isPeriodic && i == m_dimensions.x - 1) {
						nextI = 0;
						pe = m_pPoissonMatrix->getPeriodicEastValue(getIndex(i, j));
					}

					int prevI = i - 1;
					pw = m_pPoissonMatrix->getWestValue(getIndex(i, j));
					if(m_isPeriodic && i == 0) {
						prevI = m_dimensions.x - 1;
						pw = m_pPoissonMatrix->getPeriodicWestValue(getIndex(i, j));
					}

					Scalar hh = 0;
					if(m_params.pCellsVolumes == NULL) {
						Scalar h = m_params.dx;
						hh = h*h;
					} else {
						Scalar clVolume = m_params.pCellsVolumes[getIndex(i, j)];
						hh = m_params.pCellsVolumes[getIndex(i, j)];
					}
					/*Scalar h = 1/((float)m_dimensions.x);
					hh = h*h;*/
					Scalar pn = m_pPoissonMatrix->getNorthValue(getIndex(i, j));
					Scalar ps = m_pPoissonMatrix->getSouthValue(getIndex(i, j));
					relaxationStep		= hh*rhs[getIndex(i, j)] - 
										(	pe*result[getIndex(nextI, j)]
										+	pn*result[getIndex(i, j + 1)]
										+	pw*result[getIndex(prevI, j)]
										+	ps*result[getIndex(i, j - 1)]);
					
					pe = pe*result[getIndex(nextI, j)];
					pw = pw*result[getIndex(prevI, j)];
					pn = pn*result[getIndex(i, j + 1)];
					ps = ps*result[getIndex(i, j - 1)];
					Scalar pc = (1/m_pPoissonMatrix->getCentralValue(getIndex(i, j)));

					result[getIndex(i, j)] = (1/m_pPoissonMatrix->getCentralValue(getIndex(i, j))) * relaxationStep;
				}
			}
			//#pragma omp parallel for
			//for(int i = 1; i < m_dimensions.x - 1; i++) {
			//	for(int j = 1; j < m_dimensions.y - 1; j++) {
			//		/*	Scalar residual = rhs[getIndex(i, j)] -  ( m_pPoissonMatrix->getEastValue(getIndex(i, j))*result[getIndex(i + 1, j)]
			//														+ m_pPoissonMatrix->getNorthValue(getIndex(i, j))*result[getIndex(i, j + 1)]
			//		+ m_pPoissonMatrix->getWestValue(getIndex(i, j))*result[getIndex(i - 1, j)]
			//		+ m_pPoissonMatrix->getSouthValue(getIndex(i, j))*result[getIndex(i, j - 1)]
			//		+ m_pPoissonMatrix->getCentralValue(getIndex(i, j))*result[getIndex(i, j)]);*/
			//		//residual += rhs[getIndex(i, j)];

			//		Scalar residual = i*i;
			//		m_totalResidual += sqrt(residual*residual);
			//	}
			//}

			
		}
		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void GaussSeidel::updateBoundaries(Scalar *result) {
			if(m_params.northBoundary == neumann) {
				for(int i = 0; i  < m_dimensions.x; i++) {
					result[getIndex(i, m_dimensions.y - 1)] = result[getIndex(i, m_dimensions.y - 2)];
				}
			} 
			if(m_params.southBoundary == neumann) {
				for(int i = 0; i  < m_dimensions.x; i++) {
					result[getIndex(i, 0)] = result[getIndex(i, 1)];
				}
			}
			if(m_params.westBoundary == neumann) {
				for(int j = 0; j < m_dimensions.y; j++) {
					result[getIndex(0, j)] = result[getIndex(1, j)];

				}
			}
			if(m_params.eastBoundary == neumann) {
				for(int j = 0; j < m_dimensions.y; j++) {
					result[getIndex(m_dimensions.x - 1, j)] = result[getIndex(m_dimensions.x - 2, j)];
				}
			}

		}

		/************************************************************************/
		/* Solving                                                              */
		/************************************************************************/
		bool GaussSeidel::solveCPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) {
			unsigned int iter;
			m_pResult = (Scalar *)pResult->getRawDataPointer();
			m_pRhs = (Scalar *)pRhs->getRawDataPointer();
			//Zero out pressures 
			for (int i = 0; i < pResult->size(); i++) {
				m_pResult[i] = 0;
			}
			for (int i = 0; i < m_pPoissonMatrix->getNumberAdditionalCells(); i++) {
				m_cutCellsPressure[i] = 0;
			}

			for(iter = 0; iter < 1000; iter++) {
				m_totalResidual = 0;
				//updateBoundaries((Scalar *)pResult->getRawDataPointer());
				switch(m_params.iterationType) {
					case redBlack:
						redBlackIteration(iter % 2 + 1, (Scalar *)pRhs->getRawDataPointer(), (Scalar *)pResult->getRawDataPointer());
						//redBlackIteration(2, rhs, result);
					break;
					case serial:
						serialIteration((Scalar *)pRhs->getRawDataPointer(), (Scalar *)pResult->getRawDataPointer());
					break;
				}
			}

			m_totalResidual = sqrt(m_totalResidual);
			if(m_dimensions.z == 0)
				m_totalResidual /= (Scalar) m_dimensions.x*m_dimensions.y;
			else 
				m_totalResidual /= m_dimensions.x*m_dimensions.y*m_dimensions.z;
			
			m_lastResidual = static_cast <Scalar>(m_totalResidual);
			m_numIterations = iter;

			
			return true;	
		}

		bool GaussSeidel::solveGPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult) {
			return false;
		}

		void GaussSeidel::updateResidual() {
			DoubleScalar totalError = 0.0l;

			Scalar h, h2 = 0.0f;
			if(m_params.pCellsVolumes == NULL) { //Regular grid 
				h = m_params.dx;;
				h2 = 1.0f/(h*h);
			}

			int startingI, endingI;
			if(m_isPeriodic) {
				startingI = 0; endingI = m_dimensions.x;
			} else {
				startingI = 1; endingI = m_dimensions.x - 1;
			}

			Scalar maxRhs = 0.0f;


			for(int i = startingI; i < endingI; i++) {
				for(int j = 1; j < m_dimensions.y - 1; j++) {
					Scalar pe, pw = 0;

					if(m_params.pCellsVolumes != NULL) { //Non-regular grid
						h2 = 1.0f/(m_params.pCellsVolumes[getIndex(i, j)]);
					}

					int nextI = i + 1;
					pe = m_pPoissonMatrix->getEastValue(getIndex(i, j));
					if(m_isPeriodic && i == m_dimensions.x - 1) {
						nextI = 0;
						pe = m_pPoissonMatrix->getPeriodicEastValue(getIndex(i, j));
					}

					int prevI = i - 1;
					pw = m_pPoissonMatrix->getWestValue(getIndex(i, j));
					if(m_isPeriodic && i == 0) {
						prevI = m_dimensions.x - 1;
						pw = m_pPoissonMatrix->getPeriodicWestValue(getIndex(i, j));
					}

					Scalar residual = m_pResult[getIndex(nextI, j)]*pe
												+  	m_pResult[getIndex(prevI, j)]*pw
												+	m_pResult[getIndex(i, j - 1)]*m_pPoissonMatrix->getSouthValue(getIndex(i, j))
												+	m_pResult[getIndex(i, j + 1)]*m_pPoissonMatrix->getNorthValue(getIndex(i, j))
												+	m_pResult[getIndex(i, j)]*m_pPoissonMatrix->getCentralValue(getIndex(i, j));


					residual = -residual;
					residual *= h2;
					residual += m_pRhs[getIndex(i, j)];

					Scalar pn = m_pPoissonMatrix->getNorthValue(getIndex(i, j));
					Scalar ps = m_pPoissonMatrix->getSouthValue(getIndex(i, j));
					Scalar pc = m_pPoissonMatrix->getCentralValue(getIndex(i, j));


					totalError += residual*residual;
					if(abs(m_pRhs[getIndex(i, j)]) > maxRhs) {
						maxRhs = abs(m_pRhs[getIndex(i, j)]);
					}


				}
			}


			totalError /= (m_dimensions.x)*(m_dimensions.y);
			if(maxRhs != 0) {
				m_lastResidual = static_cast <Scalar>(totalError/maxRhs);
			} else {
				m_lastResidual = static_cast <Scalar>(totalError);
			}
		}
		
	}
}