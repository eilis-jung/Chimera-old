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

#ifndef _MATH_UTILS_CORE_
#define _MATH_UTILS_CORE_
#pragma once

#include "Config/ChimeraConfig.h"

/************************************************************************/
/* Chimera Math                                                         */
/************************************************************************/
#include "Math/DoubleScalar.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/Vector2d.h"
#include "Math/Vector3d.h"
#include "Data/Array2D.h"
#include "Data/Array3D.h"

namespace Chimera {

	namespace Core {
		/** Template check for specializations */
		// boolean stuff
		template <bool B>
		struct bool_type {};

		typedef bool_type<true> true_type;
		typedef bool_type<false> false_type;

		// trait stuff
		template <typename T>
		struct isVector2 : false_type
		{
			static const bool value = false;
		};

		template <>
		struct isVector2<Vector2> : true_type
		{
			static const bool value = true;
		};

		template <>
		struct isVector2<Vector2D> : true_type
		{
			static const bool value = true;
		};


		template <>
		struct isVector2<Vector3> : false_type
		{
			static const bool value = false;
		};

		template <>
		struct isVector2<Vector3D> : false_type
		{
			static const bool value = false;
		};

		typedef const true_type& isVector2True;
		typedef const false_type& isVector2False;


		/************************************************************************/
		/* Definitions                                                          */
		/************************************************************************/		
		static const Scalar g_pointProximityLenght = 1e-7f;
		static const Scalar g_epsilon = 1e-6f;

		/************************************************************************/
		/* Floating point utils			                                        */
		/************************************************************************/
		template <typename T> int sign(T val) {
			return (T(0) < val) - (val < T(0));
		}

		template <typename valueT>
		valueT clamp(valueT varValue, valueT minRange, valueT maxRange) {
			return std::min<valueT>(maxRange, std::max(minRange, varValue));
		}

		template <typename valueT>
		valueT roundClamp(valueT varValue, valueT minRange, valueT maxRange) {
			valueT clampedValue = varValue;
			
			if(varValue < minRange)
				clampedValue = maxRange - abs(clampedValue - minRange);
			else if(varValue >= maxRange)
				clampedValue = minRange + abs(clampedValue - maxRange);

			return clampedValue;
		}

		//Checks if a 
		template <class VectorType>
		FORCE_INLINE bool isInBetween(const VectorType &p1, const VectorType &p2, const VectorType &position);
		/************************************************************************/
		/* Linear algebra utils                                                 */
		/************************************************************************/
		/** a,b -> Line parameters 
		 ** c -> Point to check against */
		FORCE_INLINE bool isLeft(Vector2 a, Vector2 b, Vector2 c) {
			return ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)) > 0;
		}

		template <class VectorType>
		bool isTop(VectorType planeOrigin, VectorType planeNormal, VectorType point) {
			VectorType v1 = point - planeOrigin;
			//Testing which side of the plane the point is on
			DoubleScalar dprod = planeNormal.dot(v1);
			if (dprod <= 0)
				return false;
			return true;
		}



		/************************************************************************/
		/* Limiters                                                             */
		/************************************************************************/
		/** Scalar quantities 2D*/
		Scalar getMinLimiter(const Vector2 &position, Core::Array2D<Scalar> &scalarField);
		Scalar getMaxLimiter(const Vector2 &position, Core::Array2D<Scalar> &scalarField);

		/** Velocity 2D */
		Scalar getMinLimiterX(const Vector2 &position, Core::Array2D<Vector2> &velocityField);
		Scalar getMaxLimiterX(const Vector2 &position, Core::Array2D<Vector2> &velocityField);
		Scalar getMinLimiterY(const Vector2 &position, Core::Array2D<Vector2> &velocityField);
		Scalar getMaxLimiterY(const Vector2 &position, Core::Array2D<Vector2> &velocityField);

		/** Scalar quantities 3D*/
		Scalar getMinLimiter(const Vector3 &position, Core::Array3D<Scalar> &scalarField);
		Scalar getMaxLimiter(const Vector3 &position, Core::Array3D<Scalar> &scalarField);

		/** Velocity 3D */
		Scalar getMinLimiterX(const Vector3 &position, Core::Array3D<Vector3> &velocityField);
		Scalar getMaxLimiterX(const Vector3 &position, Core::Array3D<Vector3> &velocityField);
		Scalar getMinLimiterY(const Vector3 &position, Core::Array3D<Vector3> &velocityField);
		Scalar getMaxLimiterY(const Vector3 &position, Core::Array3D<Vector3> &velocityField);
		Scalar getMinLimiterZ(const Vector3 &position, Core::Array3D<Vector3> &velocityField);
		Scalar getMaxLimiterZ(const Vector3 &position, Core::Array3D<Vector3> &velocityField);

	}	
}

#endif