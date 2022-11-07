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
		void JetBC<Vector2>::setScalarValues(int x, int y, GridData2D *pGridData2D) {
			Scalar density = (rand()/(Scalar)RAND_MAX)*m_densityVariation + m_minDensity;
			density = clamp(density, 0.0f, 1.0f);
			pGridData2D->getDensityBuffer().setValueBothBuffers(density, x, y);

			Scalar temperature = (rand()/(Scalar)RAND_MAX)*m_temperatureVariation + m_minTemperature;
			density = clamp(density, 0.0f, 1.0f);
			pGridData2D->getTemperatureBuffer().setValueBothBuffers(temperature, x, y);
		}

		template<>
		void JetBC<Vector2>::applyBoundaryConditionNorth(GridData<Vector2> *gridData, solverType_t solverType) {
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(gridData);
			dimensions_t gridDimensions = gridData->getDimensions();
			Vector2 centralPoint = pGridData2D->getCenterPoint(gridDimensions.x/2, m_dimensions.y - 1);
			for(int y = gridDimensions.y - 1; y >= 0; y--) {
				for(int x = 0; x < gridDimensions.x; x++) {
					if((pGridData2D->getCenterPoint(x, y) - centralPoint).length() < m_size) {
						setScalarValues(x, y, pGridData2D);
					}
				}
				if((pGridData2D->getCenterPoint(gridDimensions.x/2, y) - centralPoint).length() < m_size) {
					break;
				}
			}
			for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) {
				pGridData2D->setVelocity(pGridData2D->getVelocity(i, m_dimensions.y - 2), i, m_dimensions.y - 1);
				pGridData2D->setPressure(pGridData2D->getPressure(i, m_dimensions.y - 2), i, m_dimensions.y - 1);
			}
		}

		template<>
		void JetBC<Vector2>::applyBoundaryConditionSouth(GridData<Vector2> *gridData, solverType_t solverType){
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(gridData);
			Vector2 centralPoint = pGridData2D->getCenterPoint(gridDimensions.x/2, 0);
			Vector2 zeroVelocity;
			for(int y = 0; y < gridDimensions.y; y++) {
				for(int x = 0; x < gridDimensions.x; x++) {
					if((pGridData2D->getCenterPoint(x, y) - centralPoint).length() < m_size) {
						setScalarValues(x, y, pGridData2D);
					}
				}
			}
			for(int x = 0; x < gridDimensions.x; x++) {
				if ((pGridData2D->getCenterPoint(x, 0) - centralPoint).length() < m_size) {
					pGridData2D->setVelocity(m_velocity, x, 0);
					pGridData2D->setAuxiliaryVelocity(m_velocity, x, 0);
				}
				else {
					pGridData2D->setVelocity(zeroVelocity, x, 0);
				}

				if ((pGridData2D->getCenterPoint(x, 1) - centralPoint).length() < m_size) {
					pGridData2D->setAuxiliaryVelocity(m_velocity, x, 1);
					pGridData2D->setVelocity(m_velocity, x, 1);
				}
				else {
					pGridData2D->setVelocity(zeroVelocity, x, 1);
				}
				
				pGridData2D->setPressure(pGridData2D->getPressure(x, 1), x, 0);
			}

		}

		template<>
		void JetBC<Vector2>::applyBoundaryConditionWest(GridData<Vector2> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(gridData);
			Vector2 centralPoint = pGridData2D->getCenterPoint(0, gridDimensions.y/2);
			for(int x = 0; x < gridDimensions.x; x++) {
					for(int y = 0; y < gridDimensions.y; y++) {
					if((pGridData2D->getCenterPoint(x, y) - centralPoint).length() < m_size) {
						setScalarValues(x, y, pGridData2D);
					}
				}
				if((pGridData2D->getCenterPoint(x, gridDimensions.y/2) - centralPoint).length() < m_size) {
					break;
				}
			}
			for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) { 
				pGridData2D->setVelocity(pGridData2D->getVelocity(1, i), 0, i);
				pGridData2D->setPressure(pGridData2D->getPressure(1, i), 0, i);
			}
		}

		template<>
		void JetBC<Vector2>::applyBoundaryConditionEast(GridData<Vector2> *gridData, solverType_t solverType) {
			dimensions_t gridDimensions = gridData->getDimensions();
			solverType; //Un-referenced trick - Outflow bds are not dependent on solverType
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(gridData);
			Vector2 centralPoint = pGridData2D->getCenterPoint(m_dimensions.x - 1, gridDimensions.y/2);
			for(int x = m_dimensions.x - 1; x >= 0; x--) {
				for(int y = 0; y < gridDimensions.y; y++) {
					if((pGridData2D->getCenterPoint(x, y) - centralPoint).length() < m_size) {
						setScalarValues(x, y, pGridData2D);
					}
				}
				if((pGridData2D->getCenterPoint(x, gridDimensions.y/2) - centralPoint).length() < m_size) {
					break;
				}
			}

			for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) { 
				pGridData2D->setVelocity(pGridData2D->getVelocity(m_dimensions.x - 2, i), m_dimensions.x - 1, i);
				pGridData2D->setPressure(pGridData2D->getPressure(m_dimensions.x - 2, i), m_dimensions.x - 1, i);
			}
		}

		template<>
		void JetBC<Vector2>::applyBoundaryConditionFront(GridData<Vector2> *gridData, solverType_t solverType) {
			gridData; solverType;
		}

		template<>
		void JetBC<Vector2>::applyBoundaryConditionBack(GridData<Vector2> *gridData, solverType_t solverType) {
			gridData; solverType;
		}

	}
}
