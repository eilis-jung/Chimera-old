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

#include "Boundary Conditions/JetBC.h"

namespace Chimera {
	namespace BoundaryConditions {

		template<>
		void JetBC<Vector3>::setScalarValues(int x, int y, int z, GridData3D *pGridData3D) {
			Scalar density = (rand()/(Scalar)RAND_MAX)*m_densityVariation + m_minDensity;
			density = clamp(density, 0.0f, 1.0f);
			pGridData3D->getDensityBuffer().setValueBothBuffers(density, x, y, z);

			Scalar temperature = (rand()/(Scalar)RAND_MAX)*m_temperatureVariation + m_minTemperature;
			density = clamp(density, 0.0f, 1.0f);
			pGridData3D->getTemperatureBuffer().setValueBothBuffers(temperature, x, y, z);
		}

		template<>
		void JetBC<Vector3>::applyBoundaryConditionNorth(GridData<Vector3> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			Vector3 centralPoint = pGridData3D->getCenterPoint(gridDimensions.x/2, gridDimensions.y - 1, gridDimensions.z/2);
			for(int y = gridDimensions.y - 1; y >= 0; gridDimensions.y--) {
				for(int x = 0; x < gridDimensions.x; x++) {
					for(int z = 0; z < gridDimensions.z; z++) {
						if((pGridData3D->getCenterPoint(x, y, z) - centralPoint).length() < m_size) {
							setScalarValues(x, y, z, pGridData3D);
						}
					}	
				}
				if((pGridData3D->getCenterPoint(gridDimensions.x/2, y, gridDimensions.z/2) - centralPoint).length() < m_size) {
					break;
				}
			}

			for(int x = 0; x < gridDimensions.x; x++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					pGridData3D->setVelocity(pGridData3D->getVelocity(x, m_dimensions.y - 2, z), x, m_dimensions.y - 1, z);
					pGridData3D->setPressure(pGridData3D->getPressure(x, m_dimensions.y - 2, z), x, m_dimensions.y - 1, z);
				}	
			}
		}

		template<>
		void JetBC<Vector3>::applyBoundaryConditionSouth(GridData<Vector3> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			Vector3 zeroVector;
			Scalar dx = gridData->getGridSpacing();
			Vector3 centralPoint = Vector3((gridDimensions.x + 1)/2, 0, (gridDimensions.z + 1)/2)*dx;
			for (int x = 0; x < gridDimensions.x; x++) {
				for (int z = 0; z < gridDimensions.z; z++) {
					pGridData3D->setPressure(pGridData3D->getPressure(x, 1, z), x, 0, z);

					pGridData3D->setVelocity(zeroVector, x, 0, z);
					pGridData3D->setAuxiliaryVelocity(zeroVector, x, 0, z);
					pGridData3D->setVelocity(zeroVector, x, 1, z);
					pGridData3D->setAuxiliaryVelocity(zeroVector, x, 1, z);

				}
			}
			for(int y = 0; y < gridDimensions.y; y++) {
				for(int x = 0; x < gridDimensions.x; x++) {
					for(int z = 0; z < gridDimensions.z; z++) {
						if((pGridData3D->getCenterPoint(x, y, z) - centralPoint).length() < m_size) {
							setScalarValues(x, y, z, pGridData3D);
							if (y <= 1) {
								pGridData3D->setVelocity(m_velocity, x, y, z);
								pGridData3D->setAuxiliaryVelocity(m_velocity, x, y, z);
							}
						}
					}	
				}
				/*if((pGridData3D->getCenterPoint(gridDimensions.x/2, y, gridDimensions.z/2) - centralPoint).length() < m_size) {
					break;
				}*/
			}

			/*for(int x = 0; x < gridDimensions.x; x++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					pGridData3D->setVelocity(pGridData3D->getVelocity(x, 1, z), x, 0, z);
					pGridData3D->setPressure(pGridData3D->getPressure(x, 1, z), x, 0, z);
				}	
			}*/
		}


		template<>
		void JetBC<Vector3>::applyBoundaryConditionEast(GridData<Vector3> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			Vector3 centralPoint = pGridData3D->getCenterPoint(0, gridDimensions.y/2, gridDimensions.z/2);
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int y = 0; y < gridDimensions.y; y++) {
					for(int z = 0; z < gridDimensions.z; z++) {
						if((pGridData3D->getCenterPoint(x, y, z) - centralPoint).length() < m_size) {
							setScalarValues(x, y, z, pGridData3D);
						}
					}
				}
				if((pGridData3D->getCenterPoint(x, gridDimensions.y/2, gridDimensions.z/2) - centralPoint).length() < m_size) {
					break;
				}
			}
			for(int y = 0; y < gridDimensions.y; y++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					pGridData3D->setVelocity(pGridData3D->getVelocity(1, y, z), 0, y, z);
					pGridData3D->setPressure(pGridData3D->getPressure(1, y, z), 0, y, z);
				}
			}
		}
		template<>
		void JetBC<Vector3>::applyBoundaryConditionWest(GridData<Vector3> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			Vector3 centralPoint = pGridData3D->getCenterPoint(gridDimensions.x - 1, gridDimensions.y/2, gridDimensions.z/2);
			for(int x = gridDimensions.x - 1; x >= 0; x--) {
				for(int y = 0; y < gridDimensions.y; y++) {
					for(int z = 0; z < gridDimensions.z; z++) {
						if((pGridData3D->getCenterPoint(x, y, z) - centralPoint).length() < m_size) {
							setScalarValues(x, y, z, pGridData3D);
						}
					}
				}
				if((pGridData3D->getCenterPoint(x, gridDimensions.y/2, gridDimensions.z/2) - centralPoint).length() < m_size) {
					break;
				}
			}
			for(int y = 0; y < gridDimensions.y; y++) {
				for(int z = 0; z < gridDimensions.z; z++) {
					pGridData3D->setVelocity(pGridData3D->getVelocity(gridDimensions.x - 2, y, z), gridDimensions.x - 1, y, z);
					pGridData3D->setPressure(pGridData3D->getPressure(gridDimensions.x - 2, y, z), gridDimensions.x - 1, y, z);
				}
			}
		}

		template<>
		void JetBC<Vector3>::applyBoundaryConditionBack(GridData<Vector3> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			Vector3 centralPoint = pGridData3D->getCenterPoint(gridDimensions.x/2, gridDimensions.y/2, 0);
			for(int z = 0; z < gridDimensions.z; z++) {
				for(int x = 0; x < gridDimensions.x; x++) {
					for(int y = 0; y < gridDimensions.y; y++) {
						if((pGridData3D->getCenterPoint(x, y, z) - centralPoint).length() < m_size) {
							setScalarValues(x, y, z, pGridData3D);
						}
					}
				}
				if((pGridData3D->getCenterPoint(gridDimensions.x/2, gridDimensions.y/2, z) - centralPoint).length() < m_size) {
					break;
				}
			}
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int y = 0; y < gridDimensions.y; y++) {
					pGridData3D->setVelocity(pGridData3D->getVelocity(x, y, 1), x, y, 0);
					pGridData3D->setPressure(pGridData3D->getPressure(x, y, 1), x, y, 0);
				}
			}
		}

		template<>
		void JetBC<Vector3>::applyBoundaryConditionFront(GridData<Vector3> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(gridData);
			Vector3 centralPoint = pGridData3D->getCenterPoint(gridDimensions.x/2, gridDimensions.y/2, gridDimensions.z - 1);
			for(int z = gridDimensions.z; z >= 0; z--) {
				for(int x = 0; x < gridDimensions.x; x++) {
					for(int y = 0; y < gridDimensions.y; y++) {
						if((pGridData3D->getCenterPoint(x, y, z) - centralPoint).length() < m_size) {
							setScalarValues(x, y, z, pGridData3D);
						}
					}
				}
				if((pGridData3D->getCenterPoint(gridDimensions.x/2, gridDimensions.y/2, z) - centralPoint).length() < m_size) {
					break;
				}
			}
			for(int x = 0; x < gridDimensions.x; x++) {
				for(int y = 0; y < gridDimensions.y; y++) {
					pGridData3D->setVelocity(pGridData3D->getVelocity(x, y, gridDimensions.z - 2), x, y, gridDimensions.z - 1);
					pGridData3D->setPressure(pGridData3D->getPressure(x, y, gridDimensions.z - 2), x, y, gridDimensions.z - 1);
				}
			}
		}

	}
}
