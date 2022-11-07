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
		void NoSlipBC<Vector3>::applyBoundaryConditionNorth(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					tempVelocity = pGridData3D->getVelocity(x, gridDimensions.y - 2, z);
					tempVelocity.y = m_solidVelocity.y;

					pGridData3D->setVelocity(-tempVelocity, x, gridDimensions.y - 1, z);
					pGridData3D->setAuxiliaryVelocity(m_solidVelocity, x, gridDimensions.y - 1, z);
					pGridData3D->setPressure(pGridData3D->getPressure(x, gridDimensions.y - 2, z), x, gridDimensions.y - 1, z);
				}	
			}
		}

		template<>
		void NoSlipBC<Vector3>::applyBoundaryConditionSouth(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					tempVelocity = pGridData3D->getVelocity(x, 1, z);
					tempVelocity.y = m_solidVelocity.y;
					pGridData3D->setVelocity(tempVelocity, x, 1, z);
					pGridData3D->setVelocity(-tempVelocity, x, 0, z);

					tempVelocity = pGridData3D->getAuxiliaryVelocity(x, 1, z);
					tempVelocity.y = m_solidVelocity.y;
					pGridData3D->setAuxiliaryVelocity(tempVelocity, x, 1, z);
					pGridData3D->setAuxiliaryVelocity(-tempVelocity, x, 0, z);

					pGridData3D->setPressure(pGridData3D->getPressure(x, 1, z), x, 0, z);
				}	
			}
		}


		template<>
		void NoSlipBC<Vector3>::applyBoundaryConditionEast(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int y = 0; y < gridDimensions.y; y++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					tempVelocity = pGridData3D->getVelocity(1, y, z);
					tempVelocity.x = m_solidVelocity.x;
					pGridData3D->setVelocity(tempVelocity, 1, y, z);
					pGridData3D->setVelocity(-tempVelocity, 0, y, z);

					tempVelocity = pGridData3D->getAuxiliaryVelocity(1, y, z);
					tempVelocity.x = m_solidVelocity.x;
					pGridData3D->setAuxiliaryVelocity(tempVelocity, 1, y, z);
					pGridData3D->setAuxiliaryVelocity(-tempVelocity, 0, y, z);

					pGridData3D->setPressure(pGridData3D->getPressure(1, y, z), 0, y, z);
				}
			}

		}
		template<>
		void NoSlipBC<Vector3>::applyBoundaryConditionWest(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int y = 0; y < gridDimensions.y; y++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					tempVelocity = pGridData3D->getVelocity(gridDimensions.x - 2, y, z);
					tempVelocity.x = m_solidVelocity.x;
					pGridData3D->setVelocity(-tempVelocity, gridDimensions.x - 1, y, z);
					pGridData3D->setAuxiliaryVelocity(m_solidVelocity, gridDimensions.x - 1, y, z);

					pGridData3D->setPressure(pGridData3D->getPressure(gridDimensions.x - 2, y, z), gridDimensions.x - 1, y, z);
				}
			}

		}

		template<>
		void NoSlipBC<Vector3>::applyBoundaryConditionBack(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int y = 0; y < gridDimensions.y; y++) {
					tempVelocity = pGridData3D->getVelocity(x, y, 1);
					tempVelocity.z = m_solidVelocity.z;
					pGridData3D->setVelocity(tempVelocity, x, y, 1);
					pGridData3D->setVelocity(-tempVelocity, x, y, 0);

					tempVelocity = pGridData3D->getAuxiliaryVelocity(x, y, 1);
					tempVelocity.z = m_solidVelocity.z;
					pGridData3D->setAuxiliaryVelocity(tempVelocity, x, y, 1);
					pGridData3D->setAuxiliaryVelocity(-tempVelocity, x, y, 0);

					pGridData3D->setPressure(pGridData3D->getPressure(x, y, 1), x, y, 0);
				}
			}
		}

		template<>
		void NoSlipBC<Vector3>::applyBoundaryConditionFront(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int y = 0; y < gridDimensions.y; y++) {
					tempVelocity = pGridData3D->getVelocity(x, y, gridDimensions.z - 1);
					tempVelocity.z = m_solidVelocity.z;
					pGridData3D->setVelocity(tempVelocity, x, y, gridDimensions.z - 1);
					pGridData3D->setAuxiliaryVelocity(m_solidVelocity, x, y, gridDimensions.z - 1);

					pGridData3D->setPressure(pGridData3D->getPressure(x, y, gridDimensions.z - 2), x, y, gridDimensions.z - 1);
				}
			}
		}


		template <>
		static void NoSlipBC<Vector3>::enforceSolidWalls(HexaGrid *pHexaGrid, Vector3 solidVelocity) {
			Vector3 tempVelocity;
			Vector3 zeroVelocity;
			GridData3D *pGridData = pHexaGrid->getGridData3D();
			for(int i = 1; i < pGridData->getDimensions().x - 1; i++) {
				for(int j = 1; j < pGridData->getDimensions().y - 1; j++) {
					for(int k = 1; k < pGridData->getDimensions().z - 1; k++) {
						/*if(!pHexaGrid->isSolidCell(i, j, k)) {
						if(pHexaGrid->isSolidCell(i - 1, j, k)) {
						tempVelocity = pGridData->getAuxiliaryVelocity(i, j, k);
						tempVelocity.x = solidVelocity.x;
						pGridData->setAuxiliaryVelocity(tempVelocity, i, j, k);

						tempVelocity = pGridData->getVelocity(i, j, k);
						tempVelocity.x = solidVelocity.x;
						pGridData->setVelocity(tempVelocity, i, j, k);

						tempVelocity = -tempVelocity;	
						if(!pHexaGrid->isSolidCell(i - 1, j - 1, k)) {
						tempVelocity.y = solidVelocity.y;
						} 
						if(!pHexaGrid->isSolidCell(i - 1, j, k - 1)) {
						tempVelocity.z = solidVelocity.z;
						}
						pGridData->setVelocity(tempVelocity, i - 1, j, k);
						}

						if(pHexaGrid->isSolidCell(i, j - 1, k)) {
						tempVelocity = pGridData->getAuxiliaryVelocity(i, j, k);
						tempVelocity.y = solidVelocity.y;
						pGridData->setAuxiliaryVelocity(tempVelocity, i, j, k);

						tempVelocity = pGridData->getVelocity(i, j, k);
						tempVelocity.y = solidVelocity.y;
						pGridData->setVelocity(tempVelocity, i, j, k);

						tempVelocity = -tempVelocity;	
						if(!pHexaGrid->isSolidCell(i - 1, j - 1, k)) {
						tempVelocity.x = solidVelocity.x;
						} 
						if(!pHexaGrid->isSolidCell(i, j - 1, k - 1)) {
						tempVelocity.z = solidVelocity.z;
						}
						pGridData->setVelocity(tempVelocity, i, j - 1, k);
						}

						if(pHexaGrid->isSolidCell(i, j, k - 1)) {
						tempVelocity = pGridData->getAuxiliaryVelocity(i, j, k);
						tempVelocity.z = solidVelocity.z;
						pGridData->setAuxiliaryVelocity(tempVelocity, i, j, k);

						tempVelocity = pGridData->getVelocity(i, j, k);
						tempVelocity.z = solidVelocity.z;
						pGridData->setVelocity(tempVelocity, i, j, k);

						tempVelocity = -tempVelocity;	
						if(!pHexaGrid->isSolidCell(i, j - 1, k - 1)) {
						tempVelocity.y = solidVelocity.y;
						} 
						if(!pHexaGrid->isSolidCell(i - 1, j, k - 1)) {
						tempVelocity.x = solidVelocity.x;
						}
						pGridData->setVelocity(tempVelocity, i, j, k - 1);
						}

						if(pHexaGrid->isSolidCell(i + 1, j, k)) {
						tempVelocity = pGridData->getVelocity(i + 1, j, k);
						tempVelocity.x = solidVelocity.x;
						pGridData->setVelocity(tempVelocity, i + 1, j, k);
						pGridData->setAuxiliaryVelocity(solidVelocity, i + 1, j, k);
						}
						if(pHexaGrid->isSolidCell(i, j + 1, k)) {
						tempVelocity = pGridData->getVelocity(i, j + 1, k);
						tempVelocity.y = solidVelocity.y;
						pGridData->setVelocity(tempVelocity, i, j + 1, k);
						pGridData->setAuxiliaryVelocity(solidVelocity, i, j + 1, k);
						}
						if(pHexaGrid->isSolidCell(i, j, k + 1)) {
						tempVelocity = pGridData->getVelocity(i, j, k + 1);
						tempVelocity.z = solidVelocity.z;
						pGridData->setVelocity(tempVelocity, i, j, k + 1);
						pGridData->setAuxiliaryVelocity(solidVelocity, i, j, k + 1);
						}
						} else */
						if(pHexaGrid->isSolidCell(i, j, k)) {
							pGridData->setVelocity(Vector3(0, 0, 0), i, j, k);
							pGridData->setAuxiliaryVelocity(Vector3(0, 0, 0), i, j, k );
						}
					}
				}
			}
		}
	}
}
