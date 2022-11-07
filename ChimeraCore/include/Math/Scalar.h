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

#ifndef _MATH_SCALAR_
#define _MATH_SCALAR_
#pragma once

#include "Config/ChimeraConfig.h"

namespace Chimera {
	namespace Core {

#ifdef USE_DOUBLE_PRECISION
		typedef double Scalar;
#else
		typedef float Scalar;
#endif

#ifdef USE_FLOAT_EXCEPTIONS
		unsigned int fp_control_state = _controlfp(_EM_INEXACT, _MCW_EM);
#endif

		/** Bad defines, bad*/
		#define PI 3.14159265
		#define DEG2RAD  0.0174532925f

		FORCE_INLINE Scalar SquareRoot(Scalar x) { return std::sqrt(x); }

		FORCE_INLINE Scalar ArcCosine(Scalar x) { return std::acos(x); }

		FORCE_INLINE Scalar Absolute(Scalar x) { return std::fabs(x); }

		FORCE_INLINE Scalar Sine(Scalar x) { return std::sin(x); }

		FORCE_INLINE Scalar Cosine(Scalar x) { return std::cos(x); }

		FORCE_INLINE Scalar DegreeToRad(Scalar x) { return DEG2RAD * x; }

		FORCE_INLINE Scalar RadToDegree(Scalar x) { return x/DEG2RAD; }
	}
}

#endif
