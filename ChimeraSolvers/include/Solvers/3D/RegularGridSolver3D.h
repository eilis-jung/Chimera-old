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

#ifndef __CHIMERA_REGULAR_GRID_SOLVER_3D__
#define __CHIMERA_REGULAR_GRID_SOLVER_3D__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraAdvection.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraLevelSets.h"

#include "Solvers/FlowSolver.h"

namespace Chimera {

	using namespace Core;
	using namespace Advection;
	using namespace Solvers;
	using namespace Particles;
	using namespace LevelSets;

	/** Implementation of the classic Navier-Stokes solver, for unsteady incompressible flows.
	It uses the following configurations:
	Dependent variables: Pressure, Cartesian velocity components;
	Variable arrangement: Staggered;
	Pressure Coupling: Fractional Step;
	*/

	namespace Solvers {
		class RegularGridSolver3D : public FlowSolver<Vector3, Array3D> {

		
			public:
		
				#pragma region Constructors
				//Default constructor for derived classes
				RegularGridSolver3D(const params_t &params, StructuredGrid<Vector3> *pGrid) : FlowSolver(params, pGrid) {
					m_pGridData = pGrid->getGridData3D();
					m_pAdvection = nullptr;
					m_pGridToParticlesTransfer = nullptr;  m_pParticlesToGridTransfer = nullptr;
				}

				//Default constructor for using this class
				RegularGridSolver3D(const params_t &params, StructuredGrid<Vector3> *pGrid, const vector<BoundaryCondition<Vector3> *> &boundaryConditions);

				#pragma endregion 

				#pragma region Functionalities
				/**Overrides FlowSolver's update function in order to perform advection & cloth movement correctly */
				void update(Scalar dt) override;

				/** Updates Poisson Solid walls after some change is done on the regular grid */
				void updatePoissonSolidWalls();

				virtual void applyForces(Scalar dt) { };

				virtual void addBuyoancy(Scalar dt);

				void vorticityConfinement(Scalar dt);
				#pragma endregion
	
			protected:
				typedef Interpolant <Vector3, Array3D, Vector3> VelocityInterpolant;
				typedef Interpolant <Scalar, Array3D, Vector3> ScalarInterpolant;

				#pragma region ClassMembers
				/** GridData shortcut*/
				GridData3D *m_pGridData;


				Scalar m_vorticityConfinementFactor;
				Scalar m_buoyancyDensityCoefficient;
				Scalar m_buoyancyTemperatureCoefficient;

				/** Particle-based advection */
				GridToParticles<Vector3, Array3D> *m_pGridToParticlesTransfer;
				ParticlesToGrid<Vector3, Array3D> *m_pParticlesToGridTransfer;
				#pragma endregion


				#pragma region InitializationFunctions
				/** Initializes the Poisson matrix accordingly with Finite Differences formulation */
				PoissonMatrix * createPoissonMatrix();

				/** Initializes several interpolants used for the simulation and debugging */
				void initializeInterpolants() override;
				#pragma endregion


				#pragma region PressureProjection
				/** Calculates the intermediary divergent (before projection) */
				Scalar calculateFluxDivergent(int i, int j, int k) override;

				/** Given the pressure solved by the Linear system, projects the velocity in its divergence-free part. */
				void divergenceFree(Scalar dt) override;
				#pragma endregion

				#pragma region BoundaryConditionsx
				/** Enforces solid walls boundary conditions */
				void enforceSolidWallsConditions(const Vector3 &solidVelocity);
		
				/** Enforces configuration-based scalar fields at each time-step*/
				void enforceScalarFieldMarkers();
				#pragma endregion

				#pragma region InternalUpdateFunctions
				void updateVorticity();

				Vector3 calculateVorticity(uint i, uint j, uint k);
				#pragma endregion

			};
		
	}
	
}

#endif