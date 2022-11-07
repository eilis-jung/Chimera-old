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

#include "Boundary Conditions/InflowBC.h"

namespace Chimera {
	namespace BoundaryConditions {

		template<>
		void InflowBC<Vector3>::applyBoundaryConditionNorth(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					pGridData3D->setVelocity(m_velocity, x, gridDimensions.y - 1, z);
					pGridData3D->setAuxiliaryVelocity(m_velocity, x, gridDimensions.y - 1, z);

					pGridData3D->setPressure(pGridData3D->getPressure(x, gridDimensions.y - 2, z), x, gridDimensions.y - 1, z);
				}	
			}
		}

		template<>
		void InflowBC<Vector3>::applyBoundaryConditionSouth(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					pGridData3D->setVelocity(m_velocity, x, 0, z);
					pGridData3D->setAuxiliaryVelocity(m_velocity, x, 0, z);

					tempVelocity = pGridData3D->getVelocity(x, 1, z);
					tempVelocity.y = m_velocity.y;
					pGridData3D->setVelocity(tempVelocity, x, 1, z);

					tempVelocity = pGridData3D->getAuxiliaryVelocity(x, 1, z);
					tempVelocity.y = m_velocity.y;
					pGridData3D->setAuxiliaryVelocity(tempVelocity, x, 1, z);

					pGridData3D->setPressure(pGridData3D->getPressure(x, 1, z), x, 0, z);
				}	
			}
		}


		template<>
		void InflowBC<Vector3>::applyBoundaryConditionEast(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int y = 0; y < gridDimensions.y; y++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					pGridData3D->setVelocity(m_velocity, 0, y, z);
					pGridData3D->setAuxiliaryVelocity(m_velocity, 0, y, z);

					tempVelocity = pGridData3D->getVelocity(1, y, z);
					tempVelocity.x = m_velocity.x;
					pGridData3D->setVelocity(tempVelocity, 1, y, z);

					tempVelocity = pGridData3D->getAuxiliaryVelocity(1, y, z);
					tempVelocity.x = m_velocity.x;
					pGridData3D->setAuxiliaryVelocity(tempVelocity, 1, y, z);
					
					pGridData3D->setPressure(pGridData3D->getPressure(1, y, z), 0, y, z);
				}
			}

		}
		template<>
		void InflowBC<Vector3>::applyBoundaryConditionWest(GridData<Vector3> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int y = 0; y < gridDimensions.y; y++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					pGridData3D->setVelocity(m_velocity, gridDimensions.x - 1, y, z);
					pGridData3D->setAuxiliaryVelocity(m_velocity, gridDimensions.x - 1, y, z);

					pGridData3D->setPressure(pGridData3D->getPressure(gridDimensions.x - 2, y, z), gridDimensions.x - 1, y, z);
				}
			}

		}

		template<>
		void InflowBC<Vector3>::applyBoundaryConditionBack(GridData<Vector3> *gridData, solverType_t solverType) {
			Vector3 tempVelocity;
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int y = 0; y < gridDimensions.y; y++) {
					pGridData3D->setVelocity(m_velocity, x, y, 0);
					pGridData3D->setAuxiliaryVelocity(m_velocity, x, y, 0);

					tempVelocity = pGridData3D->getVelocity(x, y, 1);
					tempVelocity.z = m_velocity.z;
					pGridData3D->setVelocity(tempVelocity, x, y, 1);

					tempVelocity = pGridData3D->getAuxiliaryVelocity(x, y, 1);
					tempVelocity.z = m_velocity.z;
					pGridData3D->setAuxiliaryVelocity(tempVelocity, x, y, 1);

					pGridData3D->setPressure(pGridData3D->getPressure(x, y, 1), x, y, 0);
				}
			}
		}

		template<>
		void InflowBC<Vector3>::applyBoundaryConditionFront(GridData<Vector3> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int y = 0; y < gridDimensions.y; y++) {
					pGridData3D->setVelocity(m_velocity, x, y, gridDimensions.z - 1);
					pGridData3D->setAuxiliaryVelocity(m_velocity, x, y, gridDimensions.z - 1);

					pGridData3D->setPressure(pGridData3D->getPressure(x, y, gridDimensions.z - 2), x, y, gridDimensions.z - 1);
				}
			}
		}

	}
}
