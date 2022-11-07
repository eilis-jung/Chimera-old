//  Copyright (c) 2013, Vinicius Costa zevedo
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

#ifndef __CHIMERA_FDM_SOLVER_2D__
#define __CHIMERA_FDM_SOLVER_2D__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraAdvection.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraLevelSets.h"
#include "ChimeraSolids.h"
#include "Solvers/FlowSolver.h"

namespace Chimera {

	using namespace Core;
	using namespace Advection;
	using namespace Solvers;
	using namespace Particles;
	using namespace Solids;

	/** Implementation of the classic Navier-Stokes solver, for unsteady incompressible flows. 
		It uses the following configurations:
			Dependent variables: Pressure, Cartesian velocity components;
			Variable arrangement: Staggered;
			Pressure Coupling: Fractional Step;
	*/

	namespace Solvers {
		class RegularGridSolver2D : public FlowSolver<Vector2, Array2D> {

		public:

			#pragma region Constructors
			//Default constructor for derived classes
			RegularGridSolver2D(const params_t &params, StructuredGrid<Vector2> *pGrid) : FlowSolver(params, pGrid) {
				m_pGridData = pGrid->getGridData2D();
				m_pAdvection = nullptr;
			}

			//Default constructor for using this class
			RegularGridSolver2D(const params_t &params, StructuredGrid<Vector2> *pGrid,
									const vector<BoundaryCondition<Vector2> *> &boundaryConditions, const vector<RigidObject2D<Vector2> *> &rigidObjects = vector<RigidObject2D<Vector2> *>());
			
			#pragma endregion 

			#pragma region Functionalities
			/**Overrides FlowSolver's update function in order to perform advection & cloth movement correctly */
			virtual void update(Scalar dt) override;

			virtual void vorticityConfinement(Scalar dt);
			virtual void applyForces(Scalar dt) {
				applyHotSmokeSources(dt);
				applyRotationalForces(dt);
				addBuyoancy(dt);
			}
			virtual void applyHotSmokeSources(Scalar dt);
			virtual void applyRotationalForces(Scalar dt);
			virtual void addBuyoancy(Scalar dt);
			#pragma endregion

			#pragma region AccessFunctions
			FORCE_INLINE const vector<RigidObject2D<Vector2> *> getThinObjectVec() const {
				return m_rigidObjectsVec;
			}

			FORCE_INLINE const vector<LineMesh<Vector2> *> getLineMeshes() const {
				return m_lineMeshes;
			}
			#pragma endregion
		protected:
			typedef Interpolant <Vector2, Array2D, Vector2> VelocityInterpolant;
			typedef Interpolant <Scalar, Array2D, Vector2> ScalarInterpolant;

			#pragma region ClassMembers
			/** Rigid objects */
			vector<RigidObject2D<Vector2> *> m_rigidObjectsVec;

			/** Line meshes used to initialize planar mesh*/
			vector<LineMesh<Vector2> *>  m_lineMeshes;

			/** GridData shortcut*/
			GridData2D *m_pGridData;
			#pragma endregion

			#pragma region InitializationFunctions
			/** Initializes the Poisson matrix accordingly with Finite Differences formulation */
			PoissonMatrix * createPoissonMatrix();

			/** Initializes several interpolants used for the simulation and debugging */
			virtual void initializeInterpolants() override;
			#pragma endregion	

			#pragma region BoundaryConditions
			/** Enforces solid walls boundary conditions */
			virtual void enforceSolidWallsConditions(const Vector2 &solidVelocity);
			#pragma endregion

			#pragma region PressureProjection
			/** Calculates the final divergent (after projection) */
			virtual Scalar calculateFinalDivergent(int i, int j) override;
			/** Calculates the intermediary divergent (before projection) */
			virtual Scalar calculateFluxDivergent(int i, int j) override;

			/** Given the pressure solved by the Linear system, projects the velocity in its divergence-free part. */
			void divergenceFree(Scalar dt) override;
			#pragma endregion

			#pragma region Misc
			//Calculates cell centered voriticity
			virtual Scalar calculateVorticity(uint i, uint j);

			/** Updates vorticity at each grid node by calculating partial derivatives from sttagered velocity locations.*/
			virtual void updateVorticity() override;
			/** Updates kinetic energy in each grid cell and updates the difference to the last time step*/
			virtual void updateKineticEnergy() override;
			/** Updates streamfunctions with divergence-free velocities fields on the grid */
			virtual void updateStreamfunctions();

			/** Gets total kinetic energy in system */
			Scalar getTotalKineticEnergy() const override;
			#pragma endregion
		};
	}
}

#endif