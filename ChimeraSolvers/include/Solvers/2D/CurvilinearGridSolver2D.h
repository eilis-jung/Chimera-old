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

#ifndef __CHIMERA_CurvilinearGrid_SOLVER_2D__
#define __CHIMERA_CurvilinearGrid_SOLVER_2D__
#pragma once

/************************************************************************/
/* Chimera Math                                                         */
/************************************************************************/
#include "ChimeraMath.h"

/************************************************************************/
/* Chimera Data			                                                */
/************************************************************************/
#include "ChimeraData.h"


namespace Chimera {

	/** Finite Volume Solver: Solver Navier-Stokes equations for arbitrary structured 2D grids.
		
		For the advection phase, implements some classical algorithms of computer graphics: 
			- transformed domain semi-lagrangian;
		
		The Poisson equation is solved by approximating the divergence operator as the classical finite volume solvers: 
		approximating fluxes on cell areas and dividing the total net flux by the cell volume.
		
		The projection step can be done by several ways, since now the derivatives uses pressure values that may not lie 
		exactly in the cell's center. This code uses a novel method that interpolate nearby pressure values in order to 
		find the correct pressure gradient.  
	*/
	class CurvilinearGridSolver2D : public FlowSolver<Vector2> {
	

		/************************************************************************/
		/* Class members	                                                    */
		/************************************************************************/
		/** Semi Lagrangian integrator */
		SemiLagrangianIntegrator<Array2D, Vector2> *m_pSLIntegrator;
		MacCormackIntegrator<Array2D, Vector2> *m_pMCIntegrator;
		/** GridData shortcut*/
		GridData2D *m_pGridData;
		/** Integration parameters*/
		trajectoryIntegratorParams_t<Vector2> *m_pTrajectoryParams;


		/************************************************************************/
		/* Initialization functions                                             */
		/************************************************************************/
		/** CUDA space function */
		void initializeIntegrationParams();

		/************************************************************************/
		/* Poisson matrix creation                                              */
		/************************************************************************/
		/** Initializes the Poisson matrix accordingly with Finite Differences formulation */
		PoissonMatrix * createPoissonMatrix();

		//Pad values accordingly with the solver type and boundary conditions
		void padValues(int &i, int &j);
		/**Poisson matrix Values calculation*/
		Scalar calculateWestValue(int i, int j);
		Scalar calculateEastValue(int i, int j);
		Scalar calculateSouthValue(int i, int j);
		Scalar calculateNorthValue(int i, int j);

		/************************************************************************/
		/* Auxiliary Functions		                                            */
		/************************************************************************/
		/** Update */
		void updatePoissonBoundaryConditions();

		void updateVorticity();

		void applyForces(Scalar dt);

		
		Scalar calculateFinalDivergent(int i, int j);

		/************************************************************************/
		/* Boundary conditions	                                                */
		/************************************************************************/
		/** Solid walls */
		void enforceSolidWallsBC() { };

		///** Boundary check */
		//FORCE_INLINE bool isOnBoundary(int i, int j) {
		//	return isOnLeftBoundary(i, j) || isOnRightBoundary(i, j);
		//}

		//FORCE_INLINE bool isOnRightBoundary(int i, int j) {
		//	int connectionPoint = m_pGrid->getConnectionPoint();
		//	if(i == connectionPoint) {
		//		if(j >= m_pGrid->getLowerDimensions()[0].y - 1 && j < m_pGrid->getUpperDimensions()[0].y) {
		//			return true;
		//		}
		//	}
		//}

		//FORCE_INLINE bool isOnLeftBoundary(int i, int j) {
		//	int connectionPoint = m_pGrid->getConnectionPoint();
		//	if(i == connectionPoint - 1) {
		//		if(j >= m_pGrid->getLowerDimensions()[0].y - 1 && j < m_pGrid->getUpperDimensions()[0].y) {
		//			return true;
		//		}
		//	}
		//}

		/************************************************************************/
		/* Advection functions                                                  */
		/************************************************************************/
		/** Semi Lagrangian - uses the transformation vector to search the correct index to interpolate the velocity.
			The interpolation is done in the transformed space.*/
		void semiLagrangian(Scalar dt);
		void modifiedMacCormack(Scalar dt);
		void robustSemiLagrangian(Scalar dt);

		/************************************************************************/
		/* Pseudo pressure step functions                                       */
		/************************************************************************/
		/** Divergent calculation, based on the finite volume algorithm.*/
		Scalar calculateFluxDivergent(int i, int j);
		
		/************************************************************************/
		/* Projection functions                                                 */
		/************************************************************************/
		/** Projects the velocity matrix into its divergence free part.*/
		void divergenceFree(Scalar dt);

		/************************************************************************/
		/* Animation                                                            */
		/************************************************************************/
		void advectDensityField(Scalar dt);
	public:

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		CurvilinearGridSolver2D(const FlowSolverParameters &params, StructuredGrid<Vector2> *pGrid, 
					const vector<Data::BoundaryCondition<Vector2> *> &boundaryConditions = vector<Data::BoundaryCondition<Vector2> *>());

		/************************************************************************/
		/* Local coordinate system transformations								*/
		/************************************************************************/
		void localCoordinateSystemTransform(Scalar dt);
		void globalCoordinateSystemTransform(Scalar dt);

		void calculateBodyForce();
	};

}

#endif