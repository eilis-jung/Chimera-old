////  Copyright (c) 2013, Vinicius Costa Azevedo
////	All rights reserved.
////
////	Redistribution and use in source and binary forms, with or without
////	modification, are permitted provided that the following conditions are met: 
////
////1. Redistributions of source code must retain the above copyright notice, this
////	list of conditions and the following disclaimer. 
////	2. Redistributions in binary form must reproduce the above copyright notice,
////	this list of conditions and the following disclaimer in the documentation
////	and/or other materials provided with the distribution. 
////
////	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
////	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
////	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
////	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
////	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
////	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
////LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
////	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
////	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
////	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////
////	The views and conclusions contained in the software and documentation are those
////	of the authors and should not be interpreted as representing official policies, 
////	either expressed or implied, of the FreeBSD Project.
//
//#ifndef __CHIMERA_RAYCAST_SOLVER_2D__
//#define __CHIMERA_RAYCAST_SOLVER_2D__
//#pragma once
//
//#include "ChimeraCore.h"
//#include "ChimeraAdvection.h"
//
//#include "Physics/RigidThinObject2D.h"
//#include "Physics/FLIPAdvection2D.h"
//
//namespace Chimera {
//
//	using namespace Core;
//	using namespace Advection;
//
//	/** Implementation of the classic Navier-Stokes solver, for unsteady incompressible flows. 
//	 ** It uses Christopher's CutCell formulation for rigid body objects and a new approach for treatment of thin
//	 ** objects. 
//		Following configurations:
//			Dependent variables: Pressure, Cartesian velocity components;
//			Variable arrangement: Staggered;
//			Pressure Coupling: Fractional Step;
//	*/
//
//	class RaycastSolver2D : public FlowSolver<Vector2> {
//	public:
//
//	private:
//
//		#pragma region ClassMembers
//		/************************************************************************/
//		/*		- Thin objects variables	                                    */
//		/************************************************************************/
//		/** ThinObject class */
//		vector<RigidThinObject2D *> m_thinObjectVec;
//		/** ThinObject points */
//		vector<LineMesh<Vector2> *>  m_lineMeshes;
//
//		/************************************************************************/
//		/*		- General Variables                                             */
//		/************************************************************************/
//		/** Integrators */
//		SemiLagrangianIntegrator<Array2D, Vector2> *m_pSLIntegrator;
//		MacCormackIntegrator<Array2D, Vector2> *m_pMCIntegrator;
//
//		/** GridData shortcut*/
//		GridData2D *m_pGridData;
//
//		/** Integration parameters*/
//		trajectoryIntegratorParams_t<Vector2> *m_pTrajectoryParams;		
//
//		Array2D<bool> m_boundaryCells;
//		Array2D<bool> m_leftFacesVisibity;
//		Array2D<bool> m_bottomFacesVisibility;
//		vector<Crossing<Vector2>> m_allCrossings;
//		Array2D<Crossing<Vector2>> m_bottomCrossings;
//		Array2D<Crossing<Vector2>> m_leftCrossings;
//
//		/** FLIP parameters*/
//		FLIPAdvection2D *m_pFLIP;
//		#pragma endregion ClassMembers
//
//		#pragma region InitializationFunctions
//		/************************************************************************/
//		/* Initialization functions                                             */
//		/************************************************************************/
//		/** Initialize trajectory integrator parameters */
//		void initializeIntegrationParams();
//
//		/** Initializes the Poisson matrix accordingly with Finite Differences formulation */
//		PoissonMatrix * createPoissonMatrix();
//		
//		#pragma endregion InitializationFunctions
//	
//		#pragma region InternalAuxiliaryFunctions
//		
//		/************************************************************************/
//		/* Auxiliary Functions		                                            */
//		/************************************************************************/
//
//		/** Update */
//		void updatePoissonBoundaryConditions();
//		Scalar calculateFinalDivergent(int i, int j);
//		
//		#pragma endregion InternalAuxiliaryFunctions
//
//		#pragma region SimulationFunctions
//		/************************************************************************/
//		/* Advection functions                                                  */
//		/************************************************************************/
//		/** Semi Lagrangian algorithm: Based in the method of characteristics, it solves the intermediary 
//			velocity required in the fractional step. Doesn't account for viscosity, because it is overly dissipative.*/
//		void semiLagrangian(Scalar dt);
//		void modifiedMacCormack(Scalar dt);
//
//		void flipAdvection(Scalar dt);
//
//		/************************************************************************/
//		/* Pseudo pressure step functions                                       */
//		/************************************************************************/
//		/** Divergent calculation, based on the finite difference stencils.*/
//		Scalar calculateFluxDivergent(int i, int j);
//		
//		/************************************************************************/
//		/* Projection functions                                                 */
//		/************************************************************************/
//		void divergenceFree(Scalar dt);
//
//		#pragma endregion SimulationFunctions
//
//	public:
//		
//		#pragma region ConstructorsDestructors
//		/************************************************************************/
//		/* ctors                                                                */
//		/************************************************************************/
//		RaycastSolver2D(const FlowSolverParameters &params, StructuredGrid<Vector2> *pGrid, 
//			const vector<Data::BoundaryCondition<Vector2> *> &boundaryConditions = vector<Data::BoundaryCondition<Vector2> *>(), 
//					const vector<RigidThinObject2D *> &thinObjectVec = vector<RigidThinObject2D *>());
//		#pragma endregion ConstructorsDestructors
//
//		#pragma region UpdateFunctions
//		/************************************************************************/
//		/* Update                                                               */
//		/************************************************************************/
//		/**Overrides FlowSolver's update function in order to perform advection & thinObject movement correctly */
//		virtual void update(Scalar dt);
//
//		/** Updates Solid objects boundaries */
//		void updatePoissonSolidWalls();
//		
//		void updateBoundaryCells();
//
//		void enforceSolidWallsConditions();
//
//		void applyForces(Scalar dt);
//		void advectDensityField(Scalar dt) { };
//		#pragma endregion UpdateFunctions
//
//
//		#pragma region AccesFunctions
//		/************************************************************************/
//		/* Access functions														*/
//		/************************************************************************/
//		FORCE_INLINE FLIPAdvection2D * getFlipAdvection() {
//			return m_pFLIP;
//		}
//
//		FORCE_INLINE const vector<RigidThinObject2D*> getThinObjectVec() const {
//			return m_thinObjectVec;
//		}
//
//		FORCE_INLINE Array2D<Crossing<Vector2>> * getLeftCrossingsPtr() {
//			return &m_leftCrossings;
//		}
//		FORCE_INLINE Array2D<Crossing<Vector2>> * getBottomCrossingsPtr() {
//			return &m_bottomCrossings;
//		}
//
//		FORCE_INLINE Array2D<bool> * getLeftFacesVisibilityPtr() {
//			return &m_leftFacesVisibity;
//		}
//		FORCE_INLINE Array2D<bool> * getBottomFacesVisibilityPtr() {
//			return &m_bottomFacesVisibility;
//		}
//
//		FORCE_INLINE Array2D<bool> * getBoundaryCellsPtr() {
//			return &m_boundaryCells;
//		}
//		#pragma endregion AccessFunctions
//
//	};
//
//}
//
//#endif