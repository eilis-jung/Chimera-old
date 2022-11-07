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

#ifndef _RENDERING_PARTICLE_VISUALIZATION_WINDOW_H
#define _RENDERING_PARTICLE_VISUALIZATION_WINDOW_H

#pragma  once

#include "ChimeraCore.h"
#include "ChimeraResources.h"
#include "Windows/BaseWindow.h"

//Rendering Cross-ref
#include "Particles/ParticleSystem2D.h"
#include "Particles/ParticleSystem3D.h"
#include "Particles/ParticlesRenderer.h"

namespace Chimera {
	using namespace Resources;
	using namespace Rendering;

	namespace Windows {

		template <class VectorT, template <class> class ArrayType>
		class ParticleVisualizationWindow : public BaseWindow {


		private:
			#pragma region ClassMembers
			/** Particle system */
			ParticleSystem2D *m_pParticleSystem2D;

			ParticleSystem3D *m_pParticleSystem3D;

			ParticlesRenderer<VectorT, ArrayType> *m_pParticlesRenderer;
			#pragma endregion

		public:
			
			#pragma region Constructors
			ParticleVisualizationWindow(ParticlesRenderer<VectorT, ArrayType> *pParticlesRenderer);
			#pragma endregion

			#pragma region AccessFunctions
			
			#pragma endregion

			#pragma region Functionalities
			virtual void update();
			#pragma endregion
		};
	}
}

#endif#pragma once
