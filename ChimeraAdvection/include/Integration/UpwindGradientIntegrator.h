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
//#ifndef _MATH_UPWIND_GRADIENT_INTEGRATOR_H_
//#define _MATH_UPWIND_GRADIENT_INTEGRATOR_H_
//#pragma once
//
//#include "ChimeraCore.h"
//
////Math
//#include "Base/Vector2.h"
//#include "Base/Vector3.h"
//#include "Integration/TrajectoryIntegrators.h"
//#include "Interpolation/LinearInterpolation2D.h"
//
//namespace Chimera {
//
//	using namespace Core;
//
//	namespace Advection {
//
//		/** Solves, by upwind, the integration of a scalar field function with its gradient as velocity. */
//		template<class VectorType>
//		class UpwindGradientIntegrator : public Singleton<UpwindGradientIntegrator<VectorType> > {
//			public:
//				/************************************************************************/
//				/* Functionalities														*/
//				/************************************************************************/
//
//				/** Integrates the scalar field function with a timestep dt in a regular grid with fixed 
//				 ** spacing of dx (in both directions). */
//			    void integrateScalarField(Array<Scalar> *pScalarField, Scalar dx, Scalar dt);
//		};
//	}
//}
//
//#endif