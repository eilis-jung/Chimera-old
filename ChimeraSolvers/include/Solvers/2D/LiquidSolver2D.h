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

#ifndef __CHIMERA_BASE_LIQUIDS_SOLVER_2D__
#define __CHIMERA_BASE_LIQUIDS_SOLVER_2D__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraAdvection.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"

#include "Solvers/FlowSolver.h"

#include "ChimeraLevelSets.h"

namespace Chimera {

	using namespace Meshes;
	using namespace LevelSets;
	using namespace Advection;

	namespace Solvers {

		/** Base class for the implementation of 2-D liquid Navier-stokes solvers. Each specialized liquid solver class has
		to extend base methods from this class. All liquids representation will have a inherent cut-cell structure,
		even though the base classes do not use the full extend of cut-cells power. */
		class LiquidSolver2D : public FlowSolver<Vector2> {
		public:

			#pragma region Constructors
			/** Standard constructor. Receives params that will configure solver's several characteristics, the underlying
			** structured grid *pGrid, boundary conditions and the liquids original mesh representation. */
			LiquidSolver2D(const FlowSolverParameters &params, StructuredGrid<Vector2> *pGrid, LiquidRepresentation2D *pLiquidRepresentation,
							const vector<BoundaryCondition<Vector2> *> &boundaryConditions = vector<BoundaryCondition<Vector2> *>());
			#pragma endregion Constructors

			#pragma region Functionalities
			/**Overrides FlowSolver's update function in order to perform advection & liquid movement correctly */
			virtual void update(Scalar dt);

			/** Updates thin objects Poisson Matrix. This function is implementation-specific. Returns if the method needs
			* additional entries on the Poisson matrix. */
			virtual bool updatePoissonMatrix() = 0 { return false; };

			/** Updates FLIP particle's tags to identify if they are inside the liquid surface originally or not. */
			void updateParticleTags();

			/** Reinitialization of liquid bounds */
			void reinitializeLiquidBounds();
			#pragma endregion Functionalities

			#pragma region AccessFunctions
			ParticleBasedAdvection<Vector2, Array2D> * getParticleBasedAdvection() {
				return m_pParticleBasedAdvection;
			}

			FORCE_INLINE LiquidRepresentation2D * getLiquidRepresentation() {
				return m_pLiquidRepresentation;
			}

			#pragma endregion AccessFunctions


		protected:

			#pragma region ClassMembers
			/* Particle Based Advection */
			ParticleBasedAdvection<Vector2, Array2D> *m_pParticleBasedAdvection;

			/** Liquid representation */
			LiquidRepresentation2D *m_pLiquidRepresentation;

			/** GridData shortcut*/
			GridData2D *m_pGridData;

			/** Density coefficients */
			

			#pragma endregion ClassMembers

			#pragma region InitializationFunctions
			/** Initialize trajectory integrator parameters */
			void initializeIntegrationParams();

			/** Initializes the Poisson matrix accordingly with Finite Differences formulation */
			PoissonMatrix * createPoissonMatrix();

			#pragma endregion InitializationFunctions

			#pragma region InternalAuxiliaryFunctions
			/** Updates conjugate gradient algorithm to accommodate changes in the Poisson matrix */
			void updateConjugateGradientSolver(bool additionalCells);
			#pragma endregion InternalAuxiliaryFunctions


		};

	}
}

#endif