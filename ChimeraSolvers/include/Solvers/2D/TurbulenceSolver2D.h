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

#ifndef __CHIMERA_TURBULENCE_SOLVER_2D__
#define __CHIMERA_TURBULENCE_SOLVER_2D__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraAdvection.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"

#include "Solvers/2D/RegularGridSolver2D.h"

namespace Chimera {

	using namespace Core;
	using namespace Advection;
	using namespace Solvers;
	using namespace Particles;

	/** Implementation of a experimental solver with a novel turbulence modeling method for unsteady incompressible flows. 
		It inherits a Regular Grid solver, making some modifications on the way that its updated. */

	namespace Solvers {

		class TurbulenceSolver2D : public RegularGridSolver2D {

		public:
			#pragma region Constructors
			TurbulenceSolver2D(const params_t &params, unsigned int numSubdivisions, StructuredGrid<Vector2> *pGrid,
				const vector<BoundaryCondition<Vector2> *> &boundaryConditions = vector<BoundaryCondition<Vector2> *>());
			#pragma endregion 

			#pragma region Functionalities
			/**Overrides FlowSolver's update function in order to perform advection & cloth movement correctly */
			virtual void update(Scalar dt);
			#pragma endregion

			#pragma region AccessFunctions
			Array2D<Scalar> * getStreamfunctionPtr() {
				return &m_streamfunctionGrid;
			}

			Array2D<Vector2> * getFineGridVelocities() {
				return &m_fineGridVelocities;
			}

			Scalar getStreamfunctionDx() {
				return m_pGridData->getScaleFactor(0, 0).x / pow(2, m_numDivisions);
			}

			Interpolant<Scalar, Array2D, Vector2> * getStreamfunctionInterpolant() {
				return m_pStreamfunctionsScalarInterpolant;
			}
			#pragma endregion

		private:

			#pragma region ClassMembers
			typedef Interpolant <Vector2, Array2D, Vector2> VectorInterpolant;
			typedef Interpolant <Scalar, Array2D, Vector2> ScalarInterpolant;

			/*Number of regular-grid subdivisions to create the fine grid that will encode turbulence*/
			unsigned int m_numDivisions;
			/** Regined grid spacing */
			Scalar m_refinedDx;
			/** Streamfunction refined grid: it has 2^(numDivisions) more cells than the original coarse grid. */
			Array2D<Scalar> m_streamfunctionGrid;
			/** Streamfunction refined grid after advection: it has 2^(numDivisions) more cells than the original 
				coarse grid. */
			Array2D<Scalar> m_auxStreamfunctionGrid;
			/** Fine grid velocities have the same number of divisions as the streamfunction grid. */
			Array2D<Vector2> m_fineGridVelocities;
			
			TurbulentInterpolant2D<Vector2> *m_pTurbulenceInterpolant;

			CubicStreamfunctionInterpolant2D<Vector2> *m_pStreamfunctionsVecInterpolant;
			BilinearNodalInterpolant2D<Scalar> *m_pStreamfunctionsScalarInterpolant;

			pair<VectorInterpolant *, VectorInterpolant *> m_velocitiesInterpolants;
			pair<VectorInterpolant *, VectorInterpolant *> m_fineVelocitiesInterpolants;
			pair<ScalarInterpolant *, ScalarInterpolant *> m_streamfunctionInterpolants;
			#pragma endregion

			#pragma region InitializationMethods
			GridToParticles<Vector2, Array2D> * initializeGridToParticles();
			#pragma endregion


			#pragma region TurbulenceStreamfunctions
			void generateTurbulence(Scalar dt);

			void updateCoarseStreamfunctionGrid();
			void updateFineGridVelocities();

			/** Initializes streamfunction values on the high-res grid considering the coarse grid velocities.
			Utilizes regular-grid bilinear interpolation at sttagered velocity positions to calculate fluxes. */
			void velocitiesToStreamfunctions();

			/** Transfers streamfunctions from the fine grid to coarse velocities */
			void streamfunctionsToVelocities();

			/** Projects streamfunction velocities with pressure gradients calculated from the coarse grid. *
			/** After this, recalculates the streamfunction values with projected velocities. */
			void projectFineGridStreamfunctions(Scalar dt);
			#pragma endregion
		};
	}
}

#endif