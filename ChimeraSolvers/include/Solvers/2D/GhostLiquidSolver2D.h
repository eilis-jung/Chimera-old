#pragma once
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

#ifndef __CHIMERA_GHOST_LIQUIDS_SOLVER_2D__
#define __CHIMERA_GHOST_LIQUIDS_SOLVER_2D__
#pragma once


#include "ChimeraCore.h"
#include "ChimeraAdvection.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"

#include "Solvers/2D/RegularGridSolver2D.h"

#include "ChimeraLevelSets.h"

namespace Chimera {

	using namespace LevelSets;
	/** Implementation of the ghost-pressures method for liquids */
	class GhostLiquidSolver : public RegularGridSolver2D {
	public:

		#pragma region ConstructorsDestructors
		/** Standard constructor. Receives params that will configure solver's several characteristics, the underlying
		** structured grid *pGrid, boundary conditions and the liquids original mesh representation. */
		GhostLiquidSolver(const params_t &params, StructuredGrid<Vector2> *pGrid,
			const vector<BoundaryCondition<Vector2> *> &boundaryConditions,
			LiquidRepresentation2D<Vector2> *pLiquidRepresentation) 
			: RegularGridSolver2D(params, pGrid, boundaryConditions), m_cellTypes(pGrid->getDimensions()), m_boundaryCells(pGrid->getDimensions()) {
			
			m_liquidDensityCoeff = 10;
			m_airDensityCoeff = 1.201;
			m_surfaceTensionCoeff = 0.073;

			/**Poisson solver initialization is after the poisson matrix basic setup is done */
			initializePoissonSolver();
		};
		#pragma endregion ConstructorsDestructors

		#pragma region UpdateFunctions
		/** Updates thin objects Poisson Matrix. This function is implementation-specific. Returns if the method needs
		* additional entries on the Poisson matrix. */
		bool updatePoissonMatrix();
		#pragma endregion UpdateFunctions

		#pragma region SimulationFunctions
		/** Divergence free pressure projection. This function is implementation specific */
		void divergenceFree(Scalar dt);

		/** Calculates flux divergent */
		Scalar calculateFluxDivergent(int i, int j);

		/** Force applied */
		void applyForces(Scalar dt) override;

		/** Updates ghost liquid solver */
		void update(Scalar dt) override;
		#pragma endregion SimulationFunctions		

		#pragma region AccessFunctions
		bool isBoundaryCell(int i, int j) {
			return m_boundaryCells(i, j);
		}
		void setLiquidRepresentation(LiquidRepresentation2D<Vector2> *pLiquidRepresentation) {
			m_pLiquidRepresentation = pLiquidRepresentation;
		}
		#pragma endregion AccessFunctions

	protected:

		#pragma region InternalFunctionalities
		/** Updates the pressure matrix entry for a given boundary cell*/
		void updatePressureMatrixBoundaryCell(const dimensions_t &cellIndex);

		/** Calculates the pressure matrix coefficient considering a given cellType and faceLocation. This function 
		  * considers possible combinations for cell neighbors to calculate the coefficients. */
		Scalar calculatePoissonMatrixCoefficient(const dimensions_t &cellIndex, LiquidRepresentation2D<Vector2>::levelSetCellType_t cellType, halfEdgeLocation_t heLocation);

		/** Helper function: gets the 2-D offset given a cell location, e.g the offset for a left location is (-1, 0) */
		dimensions_t getFaceOffset(halfEdgeLocation_t heLocation);

		/** Given the face location, calculates the face fraction of the cut-cell that has the centroid of cellIndex */ 
		Scalar calculateCellFraction(const dimensions_t &cellIndex, halfEdgeLocation_t faceLocation);  

		/** Given a ray traced from cellIndex to nextCell, calculates the distance between the intersection of this ray
		  * with the liquids geometry */
		Scalar distanceToLiquid(const dimensions_t &cellIndex, const dimensions_t &nextcellIndex);

		/** Updates cell verifying if the centroid is inside or outside liquids surface */
		void updateCellTypes();

		/** Liquid curvature functions. They calculate the liquid's curvature at a boundary cell.*/ 
		
		/** This first implementation uses the cut-cell data structure to extract normal information from the mesh. 
		  *	The curvature is simply the normal's average.  */
		Scalar calculateLiquidCurvature(const dimensions_t &cellIndex);
		/** This implementation is based on Osher-Fedkiw 2003 level-set book. It takes the divergent of the normalized
		  * level set field. */
		Scalar calculateLiquidCurvatureLS(const dimensions_t &cellIndex);
		#pragma endregion InternalFunctionalities


		#pragma region 
		Scalar m_liquidDensityCoeff;
		Scalar m_airDensityCoeff;
		Scalar m_surfaceTensionCoeff;
		LiquidRepresentation2D<Vector2> *m_pLiquidRepresentation;

		Array2D<LiquidRepresentation2D<Vector2>::levelSetCellType_t> m_cellTypes;
		Array2D<char> m_boundaryCells;
		#pragma endregion
	};

}

#endif