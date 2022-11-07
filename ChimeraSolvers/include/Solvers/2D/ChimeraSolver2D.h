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

#ifndef _CHIMERA2D_CHIMERA_SOLVER_2D_
#define _CHIMERA2D_CHIMERA_SOLVER_2D_
#pragma once

#include "ChimeraCore.h"

#include "Solvers/FlowSolver.h"

#include "Solvers/2D/RegularGridSolver2D.h"
#include "Solvers/2D/CurvilinearGridSolver2D.h"


namespace Chimera {

	/** ChimeraSolver2D: Integrates different solvers into a single one. With this approach, we can update the flow simulation
	 ** by simply calling the update function of the ChimeraSolver2D class.
	 ** 
	 ** The update function consists in:
	 **		- Update grids position and velocity
	 **		- Update relative velocities: rotational and translational
	 **		- Advection on all grids
	 **		- Interpolate intermediary velocities between grids
	 **		- Pressure Solve:
	 **			- 1 step for the background grid
	 **			- interpolate pressures from the background to all the foreground grids
	 **			- 1 step for each foreground grid
	 **			- interpolate back the pressures from all foreground grids to the background grid
	 **		- Projection on all grids
	*/

	namespace Solvers {
		class ChimeraSolver2D : public FlowSolver<Vector2> {

			/************************************************************************/
			/* Class members	                                                    */
			/************************************************************************/
			/** Multigrid simulation can have various configurations, which are stored
			/** in a config map. */
			vector<SimulationConfig<Vector2> *> m_simulationConfigs;
			/** Each overlapped grid will have a temporary boundary map attached to it. It dictates the cell correspondence
			** between the background the overlapped grid. */
			vector<dimensions_t *> m_pBoundaryMaps;

			/** Pressure solving iterations */
			int m_initialIterations;
			int m_innerIterations;
			int m_outerIterations;
			int m_boundarySmoothingIterations;

			/** Number of outer layers on the boundary smoothing step*/
			int m_boundarySmoothingLayers;
			/** Boundary indices: used to track which cells we must perform the smoothing in the smooth boundaries phase */
			vector<dimensions_t> m_boundaryIndices;
			/** Number of front grid cells that will be considered when calculating background grid boundaries */
			int m_boundaryThreshold;

			map<int, Scalar> m_minDistanceMap;

			/** 2D Grid Data facilitator*/
			GridData2D *m_pGridData;

			/** Debug circular translation */
			bool m_circularTranslation;
			/************************************************************************/
			/* Auxiliary Functions		                                            */
			/************************************************************************/
			void applyForces(Scalar dt);
			/** Calculates if the frontGridCenterPoint is the closest cell to the ith and jth cell of the background grid.
			** This is used in case of various front grid cells being mapped to same background grid cells - if the front
			** grid cells are smaller, this may happen. Therefore, this algorithm checks if the front grid center point
			** is the closest cell (so far) to the background grid cell by maintaining a map of the minimum distance found
			** so far for that cell.*/
			bool isClosestCell(const Vector2 &frontGridCenterPoint, int i, int j);

			/**Overrides the default update divergents function */
			void updateDivergents(Scalar dt);

			/************************************************************************/
			/* Advection functions                                                  */
			/************************************************************************/
			void semiLagrangian(Scalar dt);
			void modifiedMacCormack(Scalar dt);

			/************************************************************************/
			/* Pseudo pressure step functions                                       */
			/************************************************************************/
			/** Solvers for the Poisson matrix in that have to be solved each time step.*/
			void solvePressure();

			/************************************************************************/
			/* Projection functions                                                 */
			/************************************************************************/
			/** Projects the velocity matrix into its divergence free part.*/
			void divergenceFree(Scalar dt);

			/************************************************************************/
			/* Animation                                                            */
			/************************************************************************/
			void advectDensityField(Scalar dt);

			/************************************************************************/
			/* Interpolation functions                                              */
			/************************************************************************/
			/** Velocity interpolation */
			void backToFrontVelocityInterpolation(int ithFrontGrid);
			void frontToBackVelocityInterpolation(int ithFrontGrid);

			/** Pressure interpolation */
			void backToFrontPressureInterpolation(int ithFrontGrid);
			void frontToBackPressureInterpolation(int ithFrontGrid);

			/** Density interpolation */
			void backToFrontDensityInterpolation(int ithFrontGrid);
			void frontToBackDensityInterpolation(int ithFrontGrid);


			/************************************************************************/
			/* Smoothing                                                            */
			/************************************************************************/
			/** Smooth background boundaries - improves convergence */
			void smoothBoundaries(SimulationConfig<Vector2> *pBackSimCfg);


		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			ChimeraSolver2D(const FlowSolverParameters &params, SimulationConfig<Vector2> *pMainSimCfg);

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			void addSimulationConfig(SimulationConfig<Vector2> *pSimCfg) {
				m_simulationConfigs.push_back(pSimCfg);

				QuadGrid *pBackgrid = dynamic_cast<QuadGrid *>(m_simulationConfigs[0]->getGrid());
				QuadGrid *pFrontGrid = dynamic_cast<QuadGrid *>(pSimCfg->getGrid());

				dimensions_t *pTempBoundaryMap = new dimensions_t[pBackgrid->getDimensions().x*pBackgrid->getDimensions().y];
				Array2D<dimensions_t> boundaryMapArray(pTempBoundaryMap, pBackgrid->getDimensions());
				for (int i = 0; i < pBackgrid->getDimensions().x; i++) {
					for (int j = 0; j < pBackgrid->getDimensions().y; j++) {
						boundaryMapArray(i, j) = dimensions_t(-1, -1, -1);
					}
				}
				m_pBoundaryMaps.push_back(pTempBoundaryMap);

			}

			FORCE_INLINE void enforceBoundaryConditions() {
				/*for(unsigned int k = 0; k < m_simulationConfigs.size(); k++) {
				vector<BoundaryCondition<Vector2> *> *pBoundaryConditions = m_simulationConfigs[k]->getBoundaryConditions();
				for(unsigned int i = 0; i < pBoundaryConditions->size(); i++) {
				(*pBoundaryConditions)[i]->applyBoundaryCondition(m_simulationConfigs[k]->getGrid()->getGridData(), m_simulationConfigs[k]->getFlowSolver()->getParams().getDiscretizationMethod());
				}
				}*/
			}

			/************************************************************************/
			/* Update function                                                      */
			/************************************************************************/
			void update(Scalar dt);
			/** Updates the grid holes with a fast algorithm. The algorithm generates an outer solid cell boundary through an
			** iteration on the outer boundaries of the front grid, performing a direct mapping to the background grid. Then
			** the interior cells of the outer boundary on the background are filled by an linear iteration on the cell columns.
			** This approach can leave background grid cells unmarked if the front grid cells sizes are too big. */
			void updateGridHoles();
			// Updates the cell map which stores the cells that are going to be extra-smoothed on multigrid phase
			void updateGridBoundaries();


			/************************************************************************/
			/* Grid holes and boundaries update                                     */
			/************************************************************************/
			/** Updates the grid holes with a robust algorithm. The algorithm compares all the background points with all
			** the front grid points with a point in triangle test to check if there is an overlap. Thus, the complexity
			** is O(nxm), where n = number of background grid cells and m = number of front grid cells. */
			void updateGridHolesRobust();

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			void setBoundaryThreshold(int boundaryThreshold) {
				m_boundaryThreshold = boundaryThreshold;
			}

			int getBoundaryThreshold() const {
				return m_boundaryThreshold;
			}

			void setBoundarySmoothingLayers(int boundarySmoothingLayers) {
				m_boundarySmoothingLayers = boundarySmoothingLayers;
			}

			void setBoundarySmoothingIterations(int numBoundarySmoothing) {
				m_boundarySmoothingIterations = numBoundarySmoothing;
			}

			void setInnerIterations(int innerIterations) {
				m_innerIterations = innerIterations;
			}

			void setInitIterations(int initIterations) {
				m_initialIterations = initIterations;
			}

			void setOuterIterations(int outerIterations) {
				m_outerIterations = outerIterations;
			}

			void setCircularTranslation(bool circularTranslation) {
				m_circularTranslation = circularTranslation;
			}

			dimensions_t * getBoundaryMap(int bdIndex) const {
				return m_pBoundaryMaps[bdIndex];
			}

			/************************************************************************/
			/* Translational and rotational overlapping grids			            */
			/************************************************************************/
			void setOverlappingGridVelocity(int gridIndex, const Vector2 gridVelocity);
			/** Sets the angular velocity of the ith overlapping grid present in the simulation */
			void setRotationSpeed(int gridIndex, Scalar rotationSpeed);

			/************************************************************************/
			/* Local coordinate system transformations								*/
			/************************************************************************/
			void localCoordinateSystemTransform(Scalar dt);
			void globalCoordinateSystemTransform(Scalar dt);

		};
	}
}

#endif