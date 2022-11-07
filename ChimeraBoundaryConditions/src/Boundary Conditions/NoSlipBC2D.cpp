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


#include "Boundary Conditions/NoSlipBC.h"

namespace Chimera {
	namespace BoundaryConditions {

		template<>
		void NoSlipBC<Vector2>::applyBoundaryConditionNorth(GridData<Vector2> *gridData, solverType_t solverType) {
			Vector2 tempVelocity;
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(gridData);
			dimensions_t gridDimensions = gridData->getDimensions();
			for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) {
				tempVelocity = pGridData2D->getVelocity(i, gridDimensions.y - 2);
				tempVelocity.y = m_solidVelocity.y;

				pGridData2D->setVelocity(-tempVelocity, i, gridDimensions.y - 1);
				pGridData2D->setAuxiliaryVelocity(m_solidVelocity, i, gridDimensions.y - 1);
				pGridData2D->setPressure(pGridData2D->getPressure(i, gridDimensions.y - 2), i, gridDimensions.y - 1);
			}
		}

		template<>
		void NoSlipBC<Vector2>::applyBoundaryConditionSouth(GridData<Vector2> *gridData, solverType_t solverType){
			Vector2 tempVelocity;
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(gridData);
			for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) {
				tempVelocity = pGridData2D->getVelocity(i, 1);
				tempVelocity.y = m_solidVelocity.y;

				pGridData2D->setVelocity(tempVelocity, i, 1);
				tempVelocity.x = 0;
				pGridData2D->setVelocity(-tempVelocity, i, 0);

				tempVelocity = pGridData2D->getAuxiliaryVelocity(i, 1);
				tempVelocity.y = m_solidVelocity.y;

				pGridData2D->setAuxiliaryVelocity(tempVelocity, i, 1);
				tempVelocity.x = 0;
				pGridData2D->setAuxiliaryVelocity(-tempVelocity, i, 0);
				
				pGridData2D->setPressure(pGridData2D->getPressure(i, 1), i, 0);
			}
		}

		template<>
		void NoSlipBC<Vector2>::applyBoundaryConditionWest(GridData<Vector2> *gridData, solverType_t solverType) {
			Vector2 tempVelocity;
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(gridData);
			for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) { 
				tempVelocity = pGridData2D->getVelocity(1, i);
				tempVelocity.x = m_solidVelocity.x;
				
				pGridData2D->setVelocity(tempVelocity, 1, i);
				pGridData2D->setVelocity(-tempVelocity, 0, i);

				tempVelocity = pGridData2D->getAuxiliaryVelocity(1, i);
				tempVelocity.x = m_solidVelocity.x;

				pGridData2D->setAuxiliaryVelocity(tempVelocity, 1, i);
				pGridData2D->setAuxiliaryVelocity(-tempVelocity, 0, i);

				pGridData2D->setPressure(pGridData2D->getPressure(1, i), 0, i);
			}
		}

		template<>
		void NoSlipBC<Vector2>::applyBoundaryConditionEast(GridData<Vector2> *gridData, solverType_t solverType) {
			Vector2 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(gridData);
			for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) { 
				tempVelocity = pGridData2D->getVelocity(gridDimensions.x - 2, i);
				tempVelocity.x = m_solidVelocity.x;

				pGridData2D->setVelocity(-tempVelocity, gridDimensions.x - 1, i);
				pGridData2D->setAuxiliaryVelocity(m_solidVelocity, gridDimensions.x - 1, i);

				pGridData2D->setPressure(pGridData2D->getPressure(gridDimensions.x - 2, i), gridDimensions.x - 1, i);
			}
		}

		template<>
		void NoSlipBC<Vector2>::applyBoundaryConditionFront(GridData<Vector2> *gridData, solverType_t solverType) {
			gridData; solverType;
		}

		template<>
		void NoSlipBC<Vector2>::applyBoundaryConditionBack(GridData<Vector2> *gridData, solverType_t solverType) {
			gridData; solverType;
		}

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		template <>
		static void NoSlipBC<Vector2>::enforceSolidWalls(QuadGrid *pQuadGrid, Vector2 solidVelocity /* = Vector2 */) {
			Vector2 tempVelocity;
			Vector2 zeroVelocity;
			GridData2D *pGridData = pQuadGrid->getGridData2D();
			for(int i = 0; i < pGridData->getDimensions().x; i++) {
				for(int j = 0; j < pGridData->getDimensions().y; j++) {
					if(!pQuadGrid->isSolidCell(i, j)) {
						if(pQuadGrid->isSolidCell(i - 1, j)) {
							tempVelocity = pGridData->getVelocity(i, j);
							tempVelocity.x = solidVelocity.x;
							pGridData->setVelocity(tempVelocity, i, j);

							tempVelocity = -tempVelocity;	
							if(!pQuadGrid->isSolidCell(i - 1, j - 1)) {
								tempVelocity.y = solidVelocity.y;
							}
							pGridData->setVelocity(tempVelocity, i - 1, j);

							tempVelocity = pGridData->getAuxiliaryVelocity(i,j);
							tempVelocity.x = solidVelocity.x;
							pGridData->setAuxiliaryVelocity(tempVelocity, i, j);
						}

						if(pQuadGrid->isSolidCell(i, j - 1)) {
							tempVelocity = pGridData->getVelocity(i,j);
							tempVelocity.y = solidVelocity.y;
							pGridData->setVelocity(tempVelocity, i, j);

							tempVelocity = -tempVelocity;	
							if(!pQuadGrid->isSolidCell(i - 1, j - 1)) {
								tempVelocity.x = solidVelocity.x;
							}
							pGridData->setVelocity(tempVelocity, i, j - 1);

							tempVelocity = pGridData->getAuxiliaryVelocity(i, j);
							tempVelocity.y = solidVelocity.y;
							pGridData->setAuxiliaryVelocity(tempVelocity, i, j);
						}

						if(pQuadGrid->isSolidCell(i + 1, j)) {
							tempVelocity = -pGridData->getVelocity(i, j);
							tempVelocity.x = 0;
							if(!pQuadGrid->isSolidCell(i + 1, j - 1)) {
								tempVelocity.y = solidVelocity.y;
							}
							pGridData->setVelocity(tempVelocity, i + 1, j);
							pGridData->setAuxiliaryVelocity(zeroVelocity, i + 1, j);
						}

						if(pQuadGrid->isSolidCell(i, j + 1)) {
							tempVelocity = -pGridData->getVelocity(i, j);
							tempVelocity.y = solidVelocity.y;
							if(!pQuadGrid->isSolidCell(i - 1, j + 1)) {
								tempVelocity.x = solidVelocity.x;
							}
							pGridData->setVelocity(tempVelocity, i, j + 1);
							pGridData->setAuxiliaryVelocity(zeroVelocity, i, j + 1);
						}
					}
				}
			}
		}


		
	}
}
