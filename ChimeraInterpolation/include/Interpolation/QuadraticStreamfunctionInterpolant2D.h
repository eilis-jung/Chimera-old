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

#ifndef _MATH_QUADRATIC_STREAMFUNCTION_INTERPOLANT_2D_H_
#define _MATH_QUADRATIC_STREAMFUNCTION_INTERPOLANT_2D_H_

#pragma once

#include "Interpolation/Interpolant.h"
#include "Interpolation/BilinearStreamfunctionInterpolant2D.h"

namespace Chimera {

	using namespace Core;
	namespace Interpolation {

		/** Bilinear Streamfunction Interpolant: uses the analytic form of the bilinear gradient to recover vectors from
		  * nodal streamfunction values. It does not make sense to use a template different then Vector2 with this class,
		  * but since CHEWBACCA COMES FROM ENDOR AND THAT DOES NOT MAKE SENSE, I WILL LEAVE THE TEMPLATE HERE. 
		  * (Chewbacca Defense, thanks Johnnie Cochran) */
		template <class valueType>
		class QuadraticStreamfunctionInterpolant2D : public BilinearStreamfunctionInterpolant2D<valueType> {

		public:

		#pragma region Constructors
			/** Standard constructor. Use this when the update of the streamfunction values is responsibility of this 
			  * class. */
			QuadraticStreamfunctionInterpolant2D(const Array2D<valueType> &values, Scalar gridDx);
		#pragma endregion


		#pragma region Functionalities
			virtual Scalar interpolateScalarStreamfunction(const Vector2 &position);

			/* Basic interpolation function */
			virtual valueType interpolate(const Vector2 &position);

			virtual valueType interpolate2(const Vector2 &position);

			virtual valueType interpolateApprox(const Vector2 &position);

			virtual valueType interpolateAccurate(const Vector2 &position);
		#pragma endregion

		#pragma region AccessFunctions
			Array<Scalar[4]> * getStreamfunctionValuesArrayPtr() {
				return m_pStreamDiscontinuousValues;
			}
		#pragma endregion

		protected:
		
		#pragma region PrivateFunctionalities
			virtual void calculateCoefficients(dimensions_t cellIndex, Scalar *a, Scalar *b, Scalar *c);
			virtual void calculateCoefficientsScalar(dimensions_t cellIndex, Scalar *a, Scalar *b, Scalar *c);
			virtual Scalar interpolateGradientAccurate(const Vector2 &position, CutCells::faceLocation_t faceLocation, Scalar *a, Scalar *b, Scalar *c);
			virtual Scalar interpolateGradientApprox(const Vector2 &position, CutCells::faceLocation_t faceLocation, Scalar *a, Scalar *b, Scalar *c);
			virtual Scalar interpolateScalar(const Vector2 &position, CutCells::faceLocation_t faceLocation, Scalar *a, Scalar *b, Scalar *c);
		#pragma endregion
		#pragma region ClassMembers
			Scalar m_dx;

			/** Discontinuous streamfunction values per grid cell. */
			Array2D<Scalar[4]> *m_pStreamDiscontinuousValues;

			/** Continuous streamfunction values for the whole grid. */
			Array2D<Scalar> *m_pStreamContinuousValues;

			#pragma endregion
		};
	}


}
#endif