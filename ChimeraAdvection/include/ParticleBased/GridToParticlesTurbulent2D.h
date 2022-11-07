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

#ifndef __CHIMERA_GRID_TO_PARTICLES_TURBULENT_2D_H_
#define __CHIMERA_GRID_TO_PARTICLES_TURBULENT_2D_H_
#pragma once

#include "ChimeraCore.h"
#include "ParticleBased/GridToParticlesFLIP2D.h"

namespace Chimera {

	using namespace Core;

	namespace Advection {


		/** The transfer of turbulent velocities from the grid to particles is made by this class. The idea is that
		  * it updates the velocity and a scalar value (streamfunction or energy) of the particles by considering a 
		  * turbulent deviation from the mean flow (RANS style). To this end, the algorithm updates the particles 
		  * streamfunction and velocities in a FLIP-style manner (by taking the difference of an updated previous grid)
		  * if interpolants are provided (not null). If interpolants are not provided, streamfunction are conserved
		  * constant in the material derivative sense, and the particles velocities will not be deviated. */
		class GridToParticlesTurbulent2D : public GridToParticlesFLIP2D {

			typedef Interpolant <Vector2, Array2D, Vector2> VectorInterpolant;
			typedef Interpolant <Scalar, Array2D, Vector2> ScalarInterpolant;

		public:

			/** Default (regular/cut-cell)-grid constructor. Receives three pairs of interpolants. Each pair contains a 
			  * interpolant for a given quantity and a auxiliary interpolant that interpolates this same quantity after
			  * the splatting from particles to grid. FineVelocity and streamfunction interpolants are defined possibly
			  * in a finer grid than coarse velocity interpolants. */
			GridToParticlesTurbulent2D(const pair<VectorInterpolant*, VectorInterpolant*> &coarseVelocityInterpolants,
										const pair<ScalarInterpolant *, ScalarInterpolant *> &scalarInterpolants,
										const pair<VectorInterpolant*, VectorInterpolant*> &fineVelocityInterpolant,
										const pair<ScalarInterpolant*, ScalarInterpolant*> &streamfunctionInterpolant,
										Scalar mixPic);

			/** Velocity transfer from grid to particles. All subclasses must implement this. */
			virtual void transferVelocityToParticles(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData);

			/** Scalar transfer from grid to particles. */
			virtual void transferScalarAttributesToParticles(GridData<Vector2> *pGridData, ParticlesData<Vector2> *pParticlesData);
		
		protected:

			const pair<VectorInterpolant*, VectorInterpolant*> &m_fineVelocitiesInterpolants;
			const pair<ScalarInterpolant*, ScalarInterpolant*> &m_streamfunctionInterpolants;
		};
	}
}

#endif