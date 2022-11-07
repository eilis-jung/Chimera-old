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

#ifndef _MATH_TURBULENT_INTERPOLANT_2D_H_
#define _MATH_TURBULENT_INTERPOLANT_2D_H_

#pragma once

#include "Interpolation/BilinearStaggeredInterpolant2D.h"
#include "Interpolation/CubicStreamfunctionInterpolant2D.h"

namespace Chimera {

	using namespace Core;

	namespace Interpolation {

		/** Turbulence interpolant which uses a fine-grid streamfunction to perturb velocities. */
		template <class valueType>
		class TurbulentInterpolant2D : public Interpolant<valueType, Array2D, Vector2> {

		public:

			#pragma region Constructors
			TurbulentInterpolant2D(const Array2D<valueType> &values, Array2D<Scalar> *pStreamfunctionValues, Scalar coarseGridDx, Scalar fineGridDx);
			#pragma endregion
			

			#pragma region Functionalities
			/* Basic interpolation function */
			virtual valueType interpolate(const Vector2 &position);
			#pragma endregion

		protected:
			#pragma region ClassMembers
			BilinearStaggeredInterpolant2D<valueType> *m_pBilinearInterpolant;
			CubicStreamfunctionInterpolant2D<Vector2> *m_pCubicInterpolant;
			Scalar m_dx;
			Scalar m_fineGridDx;
			#pragma endregion
		};
	}
	

}
#endif