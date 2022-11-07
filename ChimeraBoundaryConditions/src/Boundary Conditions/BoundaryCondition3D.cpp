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

/** This cpp implements only the 2D boundary functions of BoundaryCondition.
 ** !!!! There is no BoundaryCondition2D.h or BoundaryCondition2D class !!!!!
 ** */

namespace Chimera {
	namespace BoundaryConditions {

		/************************************************************************/
		/* Functionalities				                                        */
		/************************************************************************/
		template<>
		void BoundaryCondition<Vector3>::zeroVelocity(GridData<Vector3> *pGridData) {
			Vector3 zeroVelocity(0, 0, 0), tempVelocity;
			dimensions_t dimensions = pGridData->getDimensions();
			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(pGridData);
			switch(m_boundaryLocation) {
				case North:
					for(int x = 0; x < dimensions.x; x++) {
						for(int z = 0; z < dimensions.z; z++) {
							pGridData3D->setVelocity(zeroVelocity, x, m_dimensions.y - 1, z);
						}
					}
				break;
				case South:
					for(int x = 0; x < dimensions.x; x++) {
						for(int z = 0; z < dimensions.z; z++) {
							pGridData3D->setVelocity(zeroVelocity, x, 0, z);
							tempVelocity = pGridData3D->getVelocity(x, 1, z);
							tempVelocity.y = 0; 
							pGridData3D->setVelocity(zeroVelocity, x, 1, z);
						}
					}
				break;
				case West:
					for(int z = 0; z < dimensions.z; z++) {
						for(int y = 0; y < dimensions.y; y++) {
							pGridData3D->setVelocity(zeroVelocity, dimensions.x - 1, y, z);
						}
					}
				break;

				case East:
					for(int z = 0; z < dimensions.z; z++) {
						for(int y = 0; y < dimensions.y; y++) {
							pGridData3D->setVelocity(zeroVelocity, 0, y, z);
							tempVelocity = pGridData3D->getVelocity(1, y, z);
							tempVelocity.x = 0; 
							pGridData3D->setVelocity(tempVelocity, 1, y, z);
						}
					}
				break;

				case Front:
					for(int y = 0; y < dimensions.y; y++) {
						for(int x = 0; x < dimensions.x; x++) {
							pGridData3D->setVelocity(zeroVelocity, x, y, dimensions.z - 1);
						}
					}
				break;

				case Back:
					for(int y = 0; y < dimensions.y; y++) {
						for(int x = 0; x < dimensions.x; x++) {
							pGridData3D->setVelocity(zeroVelocity, x, y, 0);
							tempVelocity = pGridData3D->getVelocity(x, y, 1);
							tempVelocity.z = 0;
							pGridData3D->setVelocity(tempVelocity, x, y, 1);
						}
					}

				break;
			}
		}


		/************************************************************************/
		/* Poisson Matrix update                                                */
		/************************************************************************/
		template<> 
		void BoundaryCondition<Vector3>::updateWestBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, pw;
			switch (getType()) {
				case Inflow:
				case NoSlip:
				case FreeSlip:
				case Jet:
					for(int y = 0; y < dimensions.y; y++) {
						for(int z = 0; z < dimensions.z; z++) {
							pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(dimensions.x - 1, y, z));
							pw = pPoissonMatrix->getWestValue(pPoissonMatrix->getRowIndex(dimensions.x - 1, y, z));
							pPoissonMatrix->setWestValue(pPoissonMatrix->getRowIndex(dimensions.x - 1, y, z), 0);
							pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(dimensions.x - 1, y, z), pc + pw);
						}
					}
				break;

				case Outflow:
					for(int y = 0; y < dimensions.y; y++) {
						for(int z = 0; z < dimensions.z; z++) {
							pPoissonMatrix->setWestValue(pPoissonMatrix->getRowIndex(dimensions.x - 1, y, z), 0);
						}
					}
				break;
				
				case Periodic:

				break;
			}
		}

		template<> 
		void BoundaryCondition<Vector3>::updateEastBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, pe;

			switch (getType()) {
				case Inflow:
				case NoSlip:
				case FreeSlip:
				case Jet:
					for(int y = 0; y < dimensions.y; y++) {
						for(int z = 0; z < dimensions.z; z++) {
							pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(0, y, z));
							pe = pPoissonMatrix->getEastValue(pPoissonMatrix->getRowIndex(0, y, z));
							pPoissonMatrix->setEastValue(pPoissonMatrix->getRowIndex(0, y, z), 0);
							pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(0, y, z), pc + pe);
						}
					}
				break;

				case Outflow:
					for(int y = 0; y < dimensions.y; y++) {
						for(int z = 0; z < dimensions.z; z++) {
							pPoissonMatrix->setEastValue(pPoissonMatrix->getRowIndex(0, y, z), 0);
						}
					}
				break;
				case Periodic:

				break;
			}
		}

		template<> 
		void BoundaryCondition<Vector3>::updateNorthBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, pn;

			switch (getType()) {
				case Inflow:
				case NoSlip:
				case FreeSlip:
				case Jet:
					for(int x = 0; x < dimensions.x; x++) {
						for(int z = 0; z < dimensions.z; z++) {
							pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(x, dimensions.y - 1, z));
							pn = pPoissonMatrix->getNorthValue(pPoissonMatrix->getRowIndex(x, dimensions.y - 1, z));
							pPoissonMatrix->setNorthValue(pPoissonMatrix->getRowIndex(x, dimensions.y - 1, z), 0);
							pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(x, dimensions.y - 1, z), pc + pn);
						}
					}
				break;

				case Outflow:
					for(int x = 0; x < dimensions.x; x++) {
						for(int z = 0; z < dimensions.z; z++) {
							pPoissonMatrix->setNorthValue(pPoissonMatrix->getRowIndex(x, dimensions.y - 1, z), 0);
						}
					}
				break;
				case Periodic:

				break;
			}
			
		}

		template<> 
		void BoundaryCondition<Vector3>::updateSouthBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, ps;

			switch (getType()) {
				case Inflow:
				case NoSlip:
				case FreeSlip:
				case Jet:
					for(int x = 0; x < dimensions.x; x++) {
						for(int z = 0; z < dimensions.z; z++) {
							pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(x, 0, z));
							ps = pPoissonMatrix->getSouthValue(pPoissonMatrix->getRowIndex(x, 0, z));
							pPoissonMatrix->setSouthValue(pPoissonMatrix->getRowIndex(x, 0, z), 0);
							pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(x, 0, z), pc + ps);
						}
					}
				break;

				case Outflow:
					for(int x = 0; x < dimensions.x; x++) {
						for(int z = 0; z < dimensions.z; z++) {
							pPoissonMatrix->setSouthValue(pPoissonMatrix->getRowIndex(x, 0, z), 0);
						}
					}
				break;
				case Periodic:

				break;
			}
		}

		void BoundaryCondition<Vector3>::updateBackBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, pb;

			switch (getType()) {
			case Inflow:
			case NoSlip:
			case FreeSlip:
			case Jet:
				for(int y = 0; y < dimensions.y; y++) {
					for(int x = 0; x < dimensions.x; x++) {
						pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(x, y, 0));
						pb = pPoissonMatrix->getBackValue(pPoissonMatrix->getRowIndex(x, y, 0));
						pPoissonMatrix->setBackValue(pPoissonMatrix->getRowIndex(x, y, 0), 0);
						pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(x, y, 0), pc + pb);
					}
				}
				break;

			case Outflow:
				for(int y = 0; y < dimensions.y; y++) {
					for(int x = 0; x < dimensions.x; x++) {
						pPoissonMatrix->setBackValue(pPoissonMatrix->getRowIndex(x, y, 0), 0);
					}
				}
				break;
			case Periodic:

				break;
			}
		}

		void BoundaryCondition<Vector3>::updateFrontBoundary(PoissonMatrix *pPoissonMatrix) {
			dimensions_t dimensions = pPoissonMatrix->getDimensions();
			Scalar pc, pf;

			switch (getType()) {
			case Inflow:
			case NoSlip:
			case FreeSlip:
			case Jet:
				for(int y = 0; y < dimensions.y; y++) {
					for(int x = 0; x < dimensions.x; x++) {
						pc = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(x, y, dimensions.z - 1));
						pf = pPoissonMatrix->getFrontValue(pPoissonMatrix->getRowIndex(x, y, dimensions.z - 1));
						pPoissonMatrix->setFrontValue(pPoissonMatrix->getRowIndex(x, y, dimensions.z - 1), 0);
						pPoissonMatrix->setCentralValue(pPoissonMatrix->getRowIndex(x, y, dimensions.z - 1), pc + pf);
					}
				}
				break;

			case Outflow:
				for(int y = 0; y < dimensions.y; y++) {
					for(int x = 0; x < dimensions.x; x++) {
						pPoissonMatrix->setFrontValue(pPoissonMatrix->getRowIndex(x, y, dimensions.z - 1), 0);
					}
				}
				break;
			case Periodic:

				break;
			}
		}

		template <>
		static void BoundaryCondition<Vector3>::updateSolidWalls(PoissonMatrix *pPoissonMatrix, Array<char> solidWalls /* = Array<bool> */, bool padPoissonMatrix /* = true */) {
			bool updateMatrix = false;
			Array3D<char> *solidWalls3D = (Array3D<char> *)(&solidWalls);
			for(int i = 0; i < pPoissonMatrix->getDimensions().x; i++) {
				for(int j = 0; j < pPoissonMatrix->getDimensions().y; j++) {
					for(int k = 0; k < pPoissonMatrix->getDimensions().z; k++) {
						int tempI,tempJ, tempK;
						if(padPoissonMatrix) {
							tempI = i + 1; tempJ = j + 1; tempK = k + 1;
						} else {
							tempI = i; tempJ = j; tempK = k;
						}

						if((*solidWalls3D)(tempI, tempJ, tempK) ) {
							pPoissonMatrix->setRow(pPoissonMatrix->getRowIndex(i, j, k), 0, 0, 0, 1, 0, 0, 0);
						} else {
							Scalar pw, pe, ps, pn, pf, pb;

							pw = pPoissonMatrix->getWestValue(pPoissonMatrix->getRowIndex(i, j, k));
							pe = pPoissonMatrix->getEastValue(pPoissonMatrix->getRowIndex(i, j, k));
							ps = pPoissonMatrix->getSouthValue(pPoissonMatrix->getRowIndex(i, j, k));
							pn = pPoissonMatrix->getNorthValue(pPoissonMatrix->getRowIndex(i, j, k));
							pb = pPoissonMatrix->getBackValue(pPoissonMatrix->getRowIndex(i, j, k));
							pf = pPoissonMatrix->getFrontValue(pPoissonMatrix->getRowIndex(i, j, k));

							if((*solidWalls3D)(tempI - 1, tempJ, tempK)) {
								pe = 0;
								updateMatrix = true;
							}
							if((*solidWalls3D)(tempI + 1, tempJ, tempK)) {
								pw = 0;
								updateMatrix = true;
							}
							if((*solidWalls3D)(tempI, tempJ + 1, tempK)) {
								pn = 0;
								updateMatrix = true;
							}
							if((*solidWalls3D)(tempI, tempJ - 1, tempK)) {
								ps = 0;
								updateMatrix = true;
							}
							if((*solidWalls3D)(tempI, tempJ, tempK + 1)) {
								pf = 0;
								updateMatrix = true;
							}
							if((*solidWalls3D)(tempI, tempJ, tempK - 1)) {
								pb = 0;
								updateMatrix = true;
							}

							if(updateMatrix) {
								Scalar pc = pw + pe + ps + pn + pb + pf;
								pPoissonMatrix->setRow(pPoissonMatrix->getRowIndex(i, j, k), pn, pw, pb, -pc, pe, ps, pf);
								updateMatrix = false;
							}
						}
					}
				}
			}
		}

		template <>
		void BoundaryCondition<Vector3>::updatePoissonMatrix(PoissonMatrix *pPoissonMatrix)  {
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

				case Back:
					updateBackBoundary(pPoissonMatrix);
				break;

				case Front:
					updateFrontBoundary(pPoissonMatrix);
				break;
			}
		}

		template<>
		void BoundaryCondition<Vector3>::applyBoundaryCondition(GridData<Vector3>*gridData, solverType_t solverType) {
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
			case Back:
				applyBoundaryConditionBack(gridData, solverType);
				break;
			case Front:
				applyBoundaryConditionFront(gridData, solverType);
				break;
			}
		}

		template<>
		void BoundaryCondition<Vector3>::zeroSolidBoundaries(GridData<Vector3> * pGridData) {
			//pHexaGrid;
		}

		template<>
		void BoundaryCondition<Vector3>::fixAllNeumannConditions(PoissonMatrix *pPoissonMatrix, vector<BoundaryCondition<Vector3>*> boundaryConditions) {
			bool allNeumannBoundaries = true;
			for (int i = 0; i < boundaryConditions.size(); i++) {
				if (boundaryConditions[i]->getType() == Outflow) {
					allNeumannBoundaries = false;
				}
			}

			if (allNeumannBoundaries) {
				Logger::getInstance()->get() << "Fixing neumann boundary conditions" << endl;
				pPoissonMatrix->setRow(pPoissonMatrix->getRowIndex(0, 0, 0), 0, 0, 0, 1, 0, 0, 0);
				pPoissonMatrix->setEastValue(pPoissonMatrix->getRowIndex(1, 0, 0), 0);
				pPoissonMatrix->setSouthValue(pPoissonMatrix->getRowIndex(0, 1, 0), 0);
				pPoissonMatrix->setBackValue(pPoissonMatrix->getRowIndex(0, 0, 1), 0);

			}
		}

	}
}