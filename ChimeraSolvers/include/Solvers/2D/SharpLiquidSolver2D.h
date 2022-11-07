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

#ifndef __CHIMERA_SHARP_LIQUIDS_SOLVER_2D__
#define __CHIMERA_SHARP_LIQUIDS_SOLVER_2D__
#pragma once

/************************************************************************/
/* Chimera Math                                                         */
/************************************************************************/
#include "ChimeraMath.h"

/************************************************************************/
/* Chimera Data			                                                */
/************************************************************************/
#include "ChimeraData.h"

#include "Physics/RigidThinObject2D.h"
#include "Physics/FLIPAdvection2D.h"

#include "Physics/LiquidSolver2D.h"

namespace Chimera {

	using namespace Core;
	using namespace Math;

	/** Implementation of the 2nd order pressure gradient accurate liquid solver */
	class SharpLiquidSolver2D : public LiquidSolver2D {
	public:

		#pragma region ConstructorsDestructors
		/** Standard constructor. Receives params that will configure solver's several characteristics, the underlying
		** structured grid *pGrid, boundary conditions and the liquids original mesh representation. */
		SharpLiquidSolver2D(const FlowSolverParameters &params, StructuredGrid<Vector2> *pGrid,
			const vector<Data::BoundaryCondition<Vector2> *> &boundaryConditions = vector<Data::BoundaryCondition<Vector2> *>(),
			const vector<LineMesh<Vector2>*> &liquidsVec = vector<LineMesh<Vector2>*>()) 
			: LiquidSolver2D(params, pGrid, boundaryConditions, liquidsVec) {
		
		};
		#pragma endregion ConstructorsDestructors

		#pragma region UpdateFunctions
		/** Updates thin objects Poisson Matrix. This function is implementation-specific. Returns if the method needs
		* additional entries on the Poisson matrix. */
		virtual bool updatePoissonMatrix();
		#pragma endregion UpdateFunctions

		#pragma region SimulationFunctions
		/** Divergence free pressure projection. This function is implementation specific */
		virtual void divergenceFree(Scalar dt);

		virtual Scalar calculateFluxDivergent(int i, int j) {
			return 0;
		}
		#pragma endregion SimulationFunctions		
	};

}

#endif