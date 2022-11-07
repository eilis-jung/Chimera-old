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

#include "Boundary Conditions/BoundaryCondition.h"
#include "Boundary Conditions/FarFieldBC.h"

/** This cpp implements only the 2D boundary functions of BoundaryCondition.
 ** !!!! There is no BoundaryCondition2D.h or BoundaryCondition2D class !!!!!
 ** */

namespace Chimera {
	namespace BoundaryConditions {

		/************************************************************************/
		/* Functionalities				                                        */
		/************************************************************************/
		template<>
		void BoundaryCondition<Vector2>::zeroVelocity(GridData<Vector2> *pGridData) {
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(pGridData);
			Vector2 zeroVelocity(0, 0), tempVelocity;
			switch(m_boundaryLocation) {
				case North:
					for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) {
						pGridData2D->setVelocity(zeroVelocity, i, m_dimensions.y - 1);
					}
					break;
				case South:
					for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) {
						pGridData2D->setVelocity(zeroVelocity, i, 0);
						tempVelocity = pGridData2D->getVelocity(i, 1);
						tempVelocity.y = 0;
						pGridData2D->setVelocity(tempVelocity, i, 1);
					}
					break;
				case West:
					for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) { 
						pGridData2D->setVelocity(zeroVelocity, 0, i);
						tempVelocity = pGridData2D->getVelocity(1, i);
						tempVelocity.x = 0;
						pGridData2D->setVelocity(tempVelocity, 1, i);
					}
					break;

				case East:
					for(int i = m_boundaryRange.initialRange; i < m_boundaryRange.finalRange; i++) {
						pGridData2D->setVelocity(zeroVelocity, m_dimensions.x - 1, i);
					}
					break;
			}
		}

		/************************************************************************/
		/* Poisson Matrix update                                                */
		/************************************************************************/
		template<> 
		void BoundaryCondition<Vector2>::updateWestBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, pw;
			switch (getType()) {
				case Inflow:
				case NoSlip:
				case FreeSlip:
				case Jet:
					for(int i = 0; i < dimensions.y; i++) {
						pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(0, i));
						pw = pPoissonMatrix->getWestValue(pPoissonMatrix->getRowIndex(0, i));
						pPoissonMatrix->setWestValue(pPoissonMatrix->getRowIndex(0, i), 0);
						pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(0, i), pc + pw);
					}
				break;

				case Outflow:
					for(int i = 0; i < dimensions.y; i++) {
						pPoissonMatrix->setWestValue(pPoissonMatrix->getRowIndex(0, i), 0);
					}
				break;
				
				case Periodic:
					
				break;

				case Farfield:
					FarFieldBC<Vector2> *pFarfield = dynamic_cast<FarFieldBC<Vector2>*>(this);
					if(pFarfield->getVelocity().x >= 0) { //Inflow
						for(int i = 0; i < dimensions.y; i++) {
							pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(0, i));
							pw = pPoissonMatrix->getWestValue(pPoissonMatrix->getRowIndex(0, i));
							pPoissonMatrix->setWestValue(pPoissonMatrix->getRowIndex(0, i), 0);
							pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(0, i), pc + pw);
						}
					} else { //Outflow
						for(int i = 0; i < dimensions.y; i++) {
							pPoissonMatrix->setWestValue(pPoissonMatrix->getRowIndex(0, i), 0);
						}
					}	
				break;
			}
		}

		template<> 
		void BoundaryCondition<Vector2>::updateEastBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, pe;

			switch (getType()) {
				case Inflow:
				case NoSlip:
				case FreeSlip:
				case Jet:
					for(int i = 0; i < dimensions.y; i++) {
						pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i));
						pe = pPoissonMatrix->getEastValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i));
						pPoissonMatrix->setEastValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i), 0);
						pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i), pc + pe);
					}
				break;

				case Outflow:
					for(int i = 0; i < dimensions.y; i++) {
						pPoissonMatrix->setEastValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i), 0);
					}
				break;
				case Periodic:
					
				break;

				case Farfield:
					FarFieldBC<Vector2> *pFarfield = dynamic_cast<FarFieldBC<Vector2>*>(this);
					if(pFarfield->getVelocity().x < 0) { //Inflow
						for(int i = 0; i < dimensions.y; i++) {
							pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i));
							pe = pPoissonMatrix->getEastValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i));
							pPoissonMatrix->setEastValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i), 0);
							pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i), pc + pe);
						}
					} else { //Outflow
						for(int i = 0; i < dimensions.y; i++) {
							pPoissonMatrix->setEastValue(pPoissonMatrix->getRowIndex(dimensions.x - 1,  i), 0);
						}
					}
				break;
			}
		}

		template<> 
		void BoundaryCondition<Vector2>::updateNorthBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, pn;

			switch (getType()) {
				case Inflow:
				case NoSlip:
				case FreeSlip:
				case Jet:
					for(int i = 0; i < dimensions.x; i++) {
						pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1));
						pn = pPoissonMatrix->getNorthValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1));
						pPoissonMatrix->setNorthValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1), 0);
						pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1), pc + pn);
					}
				break;

				case Outflow:
					for(int i = 0; i < dimensions.x; i++) {
						pPoissonMatrix->setNorthValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1), 0);
					}
				break;
				case Periodic:

				break;

				case Farfield:
					FarFieldBC<Vector2> *pFarfield = dynamic_cast<FarFieldBC<Vector2>*>(this);
					if(pFarfield->getVelocity().y < 0) { //Inflow
						for(int i = 0; i < dimensions.x; i++) {
							pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1));
							pn = pPoissonMatrix->getNorthValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1));
							pPoissonMatrix->setNorthValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1), 0);
							pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1), pc + pn);
						}
					} else { //Outflow
						for(int i = 0; i < dimensions.x; i++) {
							pPoissonMatrix->setNorthValue(pPoissonMatrix->getRowIndex(i, dimensions.y - 1), 0);
						}
					}	
				break;
			}
			
		}

		template<> 
		void BoundaryCondition<Vector2>::updateSouthBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, ps;

			switch (getType()) {
				case Inflow:
				case NoSlip:
				case FreeSlip:
				case Jet:
					for(int i = 0; i < dimensions.x; i++) {
						pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(i, 0));
						ps = pPoissonMatrix->getSouthValue(pPoissonMatrix->getRowIndex(i, 0));
						pPoissonMatrix->setSouthValue(pPoissonMatrix->getRowIndex(i, 0), 0);
						pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(i, 0), pc + ps);
					}
				break;

				case Outflow:
					for(int i = 0; i < dimensions.x; i++) {
						pPoissonMatrix->setSouthValue(pPoissonMatrix->getRowIndex(i, 0), 0);
					}
				break;
				case Periodic:

				break;

				case Farfield:
					FarFieldBC<Vector2> *pFarfield = dynamic_cast<FarFieldBC<Vector2>*>(this);
					if(pFarfield->getVelocity().x < 0) { //Inflow
						for(int i = 0; i < dimensions.x; i++) {
							pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(i, 0));
							ps = pPoissonMatrix->getSouthValue(pPoissonMatrix->getRowIndex(i, 0));
							pPoissonMatrix->setSouthValue(pPoissonMatrix->getRowIndex(i, 0), 0);
							pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(i, 0), pc + ps);
						}
					} else { //Outflow
						for(int i = 0; i < dimensions.x; i++) {
							pPoissonMatrix->setSouthValue(pPoissonMatrix->getRowIndex(i, 0), 0);
						}
					}	
				break;
			}
		}
		template <>
		static void BoundaryCondition<Vector2>::updateSolidWalls(PoissonMatrix *pPoissonMatrix, Array<char> solidWalls, bool padPoissonMatrix /* = true */) {
			bool updateMatrix = false;
			Array2D<char> *solidWalls2D = (Array2D<char> *)(&solidWalls);
			for(int i = 1; i < pPoissonMatrix->getDimensions().x - 1; i++) {
				for(int j = 1; j < pPoissonMatrix->getDimensions().y - 1; j++) {
					int tempI,tempJ;
					if(padPoissonMatrix) {
						tempI = i + 1; tempJ = j + 1;
					} else {
						tempI = i; tempJ = j;
					}

					if((*solidWalls2D)(tempI, tempJ) ) {
						pPoissonMatrix->setRow(pPoissonMatrix->getRowIndex(i, j), 0, 0, 1, 0, 0);
					} else {
						Scalar pw, pe, ps, pn;
						pw = pPoissonMatrix->getWestValue(pPoissonMatrix->getRowIndex(i, j));
						pe = pPoissonMatrix->getEastValue(pPoissonMatrix->getRowIndex(i, j));
						ps = pPoissonMatrix->getSouthValue(pPoissonMatrix->getRowIndex(i, j));
						pn = pPoissonMatrix->getNorthValue(pPoissonMatrix->getRowIndex(i, j));

						if((*solidWalls2D)(tempI + 1, tempJ)) {
							pe = 0;
							updateMatrix = true;
						}
						if((*solidWalls2D)(tempI - 1, tempJ)) {
							pw = 0;
							updateMatrix = true;
						}
						if((*solidWalls2D)(tempI, tempJ + 1)) {
							pn = 0;
							updateMatrix = true;
						}
						if((*solidWalls2D)(tempI, tempJ - 1)) {
							ps = 0;
							updateMatrix = true;
						}

						if(updateMatrix) {
							Scalar pc = pw + pe + ps + pn;
							pPoissonMatrix->setRow(pPoissonMatrix->getRowIndex(i, j), pn, pw, -pc, pe, ps);
							updateMatrix = false;
						}
					}
				}
			}
		}

		template <>
		void BoundaryCondition<Vector2>::updatePoissonMatrix(PoissonMatrix *pPoissonMatrix) {
			switch(getLocation()) {
				case East:
					updateEastBoundary(pPoissonMatrix);
				break;

				case West:
					updateWestBoundary(pPoissonMatrix);
				break;

				case North:
					updateNorthBoundary(pPoissonMatrix);
				break;

				case South:
					updateSouthBoundary(pPoissonMatrix);
				break;
			}
		}

		template<>
		void BoundaryCondition<Vector2>::applyBoundaryCondition(GridData<Vector2> *gridData, solverType_t solverType) {
			switch(m_boundaryLocation) {
			case North:
				applyBoundaryConditionNorth(gridData, solverType);
				break;
			case South:
				applyBoundaryConditionSouth(gridData, solverType);
				break;
			case West:
				applyBoundaryConditionWest(gridData, solverType);	
				break;
			case East:
				applyBoundaryConditionEast(gridData, solverType);
				break;
			}
		}

		template<>
		void BoundaryCondition<Vector2>::zeroSolidBoundaries(GridData<Vector2> * pGridData) {
			Vector2 zeroVelocity;
			GridData2D *pGridData2D = dynamic_cast<GridData2D *>(pGridData);
			for(int i = 0; i < pGridData2D->getDimensions().x; i++) {
				for(int j = 0; j < pGridData2D->getDimensions().y; j++) {
					/*if(pGridData2D-> -> ->isSolidCell(i, j)) {
						pGridData->setVelocity(zeroVelocity, i, j);
						pGridData->setAuxiliaryVelocity(zeroVelocity, i, j);
					}*/
				}
			}
		}


		template<>
		void BoundaryCondition<Vector2>::updateBackBoundary(PoissonMatrix *pPoissonMatrix) {
			pPoissonMatrix;
		}
		template<>
		void BoundaryCondition<Vector2>::updateFrontBoundary(PoissonMatrix *pPoissonMatrix) {
			pPoissonMatrix;
		}
	}
}