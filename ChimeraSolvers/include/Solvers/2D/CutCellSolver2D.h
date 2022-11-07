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

#ifndef __CHIMERA_VARIATIONAL_SOLVER_2D__
#define __CHIMERA_VARIATIONAL_SOLVER_2D__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraAdvection.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraCutCells.h"
#include "ChimeraMesh.h"
#include "ChimeraSolids.h"
#include "Solvers/2D/RegularGridSolver2D.h"

namespace Chimera {

	using namespace Core;
	using namespace Advection;
	using namespace Meshes;
	using namespace Solids;

	/** Implementation of the classic Navier-Stokes solver, for unsteady incompressible flows It uses a CutCell 
	  * formulation for rigid body objects and a new approach for treatment of thin objects. 
			Following configurations:
			Dependent variables: Pressure, Cartesian velocity components;
			Variable arrangement: Nodal (only one supported);
			Pressure Coupling: Fractional Step; */

	namespace Solvers {
			class CutCellSolver2D : public RegularGridSolver2D {

			public:
		
			#pragma region Constructors
			CutCellSolver2D(const params_t &params, StructuredGrid<Vector2> *pGrid,
							const vector<BoundaryCondition<Vector2> *> &boundaryConditions = vector<BoundaryCondition<Vector2> *>(), 
							const vector<RigidObject2D<Vector2> *> &rigidObjects = vector<RigidObject2D<Vector2> *>());
			#pragma endregion 

			#pragma region UpdateFunctions
			void update(Scalar dt) override;
			#pragma endregion 

			#pragma region ObjectsInitialization
			/** Reinitializes cut-cell structures after the update of any object present on the simulation domain */
			void reinitializeThinBounds();
			#pragma endregion 

			#pragma region AccessFunctions
			CutCells2D<Vector2> * getCutCells() {
				return m_pCutCells;
			}
			
			FORCE_INLINE Scalar getCutCellDivergence(uint cellID) {
				return m_cutCellsDivergents[cellID];
			}
			FORCE_INLINE vector<Scalar> * getDivergentsVectorPtr() {
				return &m_cutCellsDivergents;
			}

			FORCE_INLINE Scalar getCutCellPressure(uint cellID) {
				return m_cutCellsPressures[cellID];
			}
			FORCE_INLINE vector<Scalar> * getPressuresVectorPtr() {
				return &m_cutCellsPressures;
			}
			

			const Array2D<Vector2> & getNodalVelocityField() const {
				return m_nodalBasedVelocities;
			}
			#pragma endregion

			protected:

			#pragma region ClassMembers
			/** Planar mesh data */
			CutCells2D<Vector2> *m_pCutCells;
			vector<Scalar> m_cutCellsPressures;
			vector<Scalar> m_cutCellsDivergents;

			/** Velocity data for cut-cells */
			CutCellsVelocities2D *m_pCutCellsVelocities2D;

			/** Storing intermediary velocity data for cut-cells */
			CutCellsVelocities2D *m_pAuxCutCellsVelocities2D;

			/** Nodal based velocities */
			Array2D<Vector2> m_nodalBasedVelocities;
			Array2D<Vector2> m_auxNodalBasedVelocities;
			#pragma endregion

			#pragma region InitializationFunctions
			/**Initializes cut-cell structure */
			CutCells2D<Vector2>* initializeCutCells();

			/** Initializes the Poisson matrix accordingly with Finite Differences formulation */
			PoissonMatrix * createPoissonMatrix();

			/** Initializes several interpolants used for the simulation and debugging */
			void initializeInterpolants() override;
			#pragma endregion

			#pragma region PressureProjection
			/** Updates thin objects Poisson Matrix */
			virtual void updatePoissonThinSolidWalls();

			/** Updates internal cut-cells divergence */
			virtual void updateCutCellsDivergence(Scalar dt);

			/** Divergent calculation, based on the finite difference stencils.*/
			virtual Scalar calculateFluxDivergent(int i, int j) override;

			FORCE_INLINE virtual void logVelocity(string filename) {
				string fullFileNameX = filename + "_x.csv";
				string fullFileNameY = filename + "_y.csv";
				std::ofstream ofsX(fullFileNameX, std::ofstream::out);
				std::ofstream ofsY(fullFileNameY, std::ofstream::out);
				for (int i = 0; i < (m_pGrid->getDimensions().x - 2); i++) {
					for (int j = 0; j < (m_pGrid->getDimensions().y - 2); j++) {
						ofsX << m_pGridData->getVelocity(i, j).x << "," << m_pGridData->getCenterPoint(i, j).x << "," << m_pGridData->getCenterPoint(i, j).y << endl;
						ofsY << m_pGridData->getVelocity(i, j).y << "," << m_pGridData->getCenterPoint(i, j).x << "," << m_pGridData->getCenterPoint(i, j).y << endl;
					}
				}

				ofsX.close();
				ofsY.close();
			}

			FORCE_INLINE virtual void logPressure(string filename) {
				string fullFileName = filename + ".csv";
				std::ofstream ofs(fullFileName, std::ofstream::out);
				for (int i = 0; i < (m_pGrid->getDimensions().x - 2); i++) {
					for (int j = 0; j < (m_pGrid->getDimensions().y - 2); j++) {
						ofs << m_pGridData->getPressure(i, j) << "," << m_pGridData->getCenterPoint(i, j).x << "," << m_pGridData->getCenterPoint(i, j).y << "," << "0" << endl;
					}
				}
				for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
					ofs << m_cutCellsPressures[i] << "," << m_pCutCells->getCutCell(i).getCentroid().x << "," << m_pCutCells->getCutCell(i).getCentroid().y << "," << "1" << endl;
				}
				ofs.close();
			}


			FORCE_INLINE virtual void logVorticity(string filename) {
				string fullFileName = filename + ".csv";
				std::ofstream ofs(fullFileName, std::ofstream::out);
				for (int i = 0; i < (m_pGrid->getDimensions().x - 2); i++) {
					for (int j = 0; j < (m_pGrid->getDimensions().y - 2); j++) {
						ofs << m_pGridData->getVorticity(i, j) << "," << m_pGridData->getCenterPoint(i, j).x << "," << m_pGridData->getCenterPoint(i, j).y << endl;
					}
				}

				ofs.close();
			}

			FORCE_INLINE virtual void logVelocityForCutCells(string filename) {
				string fullFileNameX = filename + "_x_cutcells.csv";
				string fullFileNameY = filename + "_y_cutcells.csv";
				std::ofstream ofsX(fullFileNameX, std::ofstream::out);
				std::ofstream ofsY(fullFileNameY, std::ofstream::out);
				for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
					ofsX << m_params.cutFaceVelocities[i].velocity.x << "," ;
					ofsY << m_params.cutFaceVelocities[i].velocity.y << ",";
				}

				ofsX.close();
				ofsY.close();
			}

			FORCE_INLINE virtual void solvePressure() {
				const Array<Scalar> *pRhs;
				Array<Scalar> *pPressures;

				if (m_dimensions.z == 0) {
					pRhs = &m_pGrid->getGridData2D()->getDivergentArray();
					pPressures = (Array<Scalar> *)(&m_pGrid->getGridData2D()->getPressureArray());
				}
				else {
					pRhs = &m_pGrid->getGridData3D()->getDivergentArray();
					pPressures = (Array<Scalar> *)(&m_pGrid->getGridData3D()->getPressureArray());
				}
				if (m_params.pPoissonSolverParams->platform == PlataformGPU)
					m_pPoissonSolver->solveGPU(pRhs, pPressures);
				else {
					if (m_params.pPoissonSolverParams->solverMethod == GaussSeidelMethod) {
						GaussSeidel *pGS = dynamic_cast<GaussSeidel *>(m_pPoissonSolver);
						uint numIter = pGS->getParams().maxIterations;
						Scalar dt = PhysicsCore<Vector2>::getInstance()->getParams()->timestep;
						for (int i = 0; i < numIter; i++) {
							pGS->serialIterationForCutCells((Scalar *)pRhs->getRawDataPointer(), (Scalar *)pPressures->getRawDataPointer());
							//continue;
							updateDivergents(dt);
							updateCutCellsDivergence(dt);
						}
					}
					else {
						m_pPoissonSolver->solveCPU(pRhs, pPressures);
					}
				}


				m_linearSolverIterations = m_pPoissonSolver->getNumberIterations();
			}

			/** Projects the velocity into its divergence-free state*/
			virtual void divergenceFree(Scalar dt);
			#pragma endregion

			#pragma region Misc
			/** Overrides parent's vorticity calculation to accommodate cut-cells */
			//Scalar calculateVorticity(uint i, uint j) override;

			/** Helps create Poisson matrix method by computing Poisson matrix indices */
			int getRowIndex(const dimensions_t &currDim, halfEdgeLocation_t edgeLocation) {
				switch (edgeLocation) {
				case rightHalfEdge:
					return m_pPoissonMatrix->getRowIndex(currDim.x, currDim.y - 1);
					break;
				case bottomHalfEdge:
					return m_pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y - 2);
					break;
				case leftHalfEdge:
					return m_pPoissonMatrix->getRowIndex(currDim.x - 2, currDim.y - 1);
					break;
				case topHalfEdge:
					return m_pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y);
					break;
				default:
					return -1;
					break;
				}
			}
			#pragma endregion
		};
	}
	

}

#endif