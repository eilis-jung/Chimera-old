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

#ifndef _MATH_BILINEAR_APIC_INTERPOLANT_2D_H_
#define _MATH_BILINEAR_APIC_INTERPOLANT_2D_H_

#pragma once

#include "Interpolation/BilinearStaggeredInterpolant2D.h"

namespace Chimera {

	namespace Interpolation {
		/** Bilinear staggered interpolation for angular momentum and inertia tensor based on 
			velocity field of APIC solver*/
		class BilinearAPICStaggeredInterpolant2D : public BilinearStaggeredInterpolant2D<Vector2> {

		public:
			#pragma region Constructors
			BilinearAPICStaggeredInterpolant2D(const Array2D<Vector2> &values, Scalar gridDx)
				: BilinearStaggeredInterpolant2D(values, gridDx) {}
			#pragma endregion
			

			#pragma region Functionalities
			/* Basic interpolation function */
			virtual Vector2 interpolate(const Vector2 &position) { return BilinearStaggeredInterpolant2D<Vector2>::interpolate(position); }
			/* For RPIC */
			virtual Vector2 angualrMomentemInterpolate(const Vector2 &position);
			virtual Matrix3x3 inertiaTensorInverseInterpolate(const Vector2 &position, bool yAxis);
			/* For APIC */
			virtual Matrix2x2 velocityDerivativeInterpolate(const Vector2 &position);
			#pragma endregion
		};
	}
	

}
#endif