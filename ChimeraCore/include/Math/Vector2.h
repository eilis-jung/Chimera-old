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

#ifndef _MATH_VECTOR_2_
#define _MATH_VECTOR_2_

#include "Math/Scalar.h"

namespace Chimera {
	namespace Core {

		class Vector2 {

		public:
			Scalar x;
			Scalar y;


			FORCE_INLINE Vector2() { x = y = 0;}

			FORCE_INLINE Vector2(const Scalar &x, const Scalar &y) {
				this->x = x;
				this->y = y;
			}

			/************************************************************************/
			/* Operators                                                            */
			/************************************************************************/
			// Array indexing - Useful for matrix operations
			Scalar &operator [] (unsigned int i) {
				assert(i < 2);
				return *(&x+i);
			}

			// Array indexing
			const Scalar &operator [] (unsigned int i) const {
				assert(i<2);
				return *(&x+i);
			}

			FORCE_INLINE Vector2& operator+=(const Vector2& rhs) {
				x += rhs.x; y += rhs.y;
				return *this;
			}

			FORCE_INLINE Vector2& operator+=(const Scalar& s) {
				x += s; y += s;
				return *this;
			}

			FORCE_INLINE Vector2& operator-=(const Vector2& rhs) {
				x -= rhs.x; y -= rhs.y;
				return *this;
			}
			FORCE_INLINE Vector2& operator-=(const Scalar& s) {
				x -= s; y -= s;
				return *this;
			}

			/**@brief Elementwise multiply this vector by the other 
			* @param rhs The other vector */
			FORCE_INLINE Vector2& operator*=(const Vector2& rhs) {
				x *= rhs.x; y *= rhs.y;
				return *this;
			}

			FORCE_INLINE Vector2& operator/=(const Vector2& rhs) {
				x /= rhs.x; y /= rhs.y;
				return *this;
			}

			FORCE_INLINE Vector2& operator*=(const Scalar& s) {
				x *= s; y *= s;
				return *this;
			}

			FORCE_INLINE Vector2& operator/=(const Scalar& s) {
				assert(s != Scalar(0.0));
				x /= s; y /= s;
				return *this;
			}


			FORCE_INLINE bool operator==(const Vector2& rhs) const {
				return (x == rhs.x) && (y == rhs.y);
			}

			FORCE_INLINE bool operator!=(const Vector2& other) const {
				return !(*this == other);
			}

			//// Negate a Vector3f
			FORCE_INLINE friend Vector2 operator - (const Vector2 &a) {
				return Vector2(-a.x, -a.y);
			}

			// Add two Vector3f's
			FORCE_INLINE friend Vector2 operator + (const Vector2 &a, const Vector2 &b) {
				Vector2 ret(a);
				ret += b;
				return ret;
			}
			FORCE_INLINE friend Vector2 operator + (const Vector2 &lhs, Scalar s) {
				Vector2 ret(lhs);
				ret += s;
				return ret;
			}

			// Subtract one vector3 from another
			FORCE_INLINE friend Vector2 operator - (const Vector2 &a, const Vector2 &b) {
				Vector2 ret(a);
				ret -= b;
				return ret;
			}

			FORCE_INLINE friend Vector2 operator - (const Vector2 &lhs, Scalar s) {
				Vector2 ret(lhs);
				ret -= s;
				return ret;
			}

			FORCE_INLINE friend Vector2 operator * (const Vector2 &lhs, Scalar s) {
				Vector2 ret(lhs);
				ret *= s;
				return ret;
			}
			FORCE_INLINE friend Vector2 operator / (const Vector2 &lhs, Scalar s) {
				Vector2 ret(lhs);
				ret /= s;
				return ret;
			}

			/** It is not a dot product, it is rather a component-by-component multiplication. */
			FORCE_INLINE friend Vector2 operator * (const Vector2 &lhs, const Vector2 &rhs) {
				Vector2 ret(lhs);
				ret *= rhs;
				return ret;
			}
			
			/** Component-by-component division.*/
			FORCE_INLINE friend Vector2 operator / (const Vector2 &lhs, const Vector2 &rhs) {
				Vector2 ret(lhs);
				ret /= rhs;
				return ret;
			}

			/************************************************************************/
			/* Comparison operators/Relational operators							*/
			/************************************************************************/
			/** Compares the absolute position of vectors; isn't related to length*/
			/** Priority is given to x-axis. */
			bool Vector2::operator < (const Vector2 & rhs) const {
				if(x < rhs.x)
					return true;
				else if(x == rhs.x) {
					if(y < rhs.y)
						return true;
				}
				return false;
			}
			bool Vector2::operator > (const Vector2 & rhs) const {
				if(x > rhs.x)
					return true;
				else if(x == rhs.x) {
					if(y > rhs.y)
						return true;
				}
				return false;
			}


			/************************************************************************/
			/* Geometric operations                                                 */
			/************************************************************************/
			FORCE_INLINE Scalar dot(const Vector2& rhs) const {
				return x * rhs.x + y * rhs.y;
			}

			/** 2D interpretation of the cross vector, determinant of
			 ** |	i		j		k	|
			 ** |	v1.x	v1.y	0	|
			 ** |	v2.x	v2.y	0	|
			 **/

			FORCE_INLINE Scalar cross(const Vector2 &rhs) const {
				return x*rhs.y - y*rhs.x;
			}

			/**@brief Return the L2 of a a vector */
			FORCE_INLINE Scalar length2() const {
				return dot(*this);
			}

			/**@brief Return the length of the vector */
			FORCE_INLINE Scalar length() const {
				return SquareRoot(length2());
			}

			/**@brief Return the distance squared between the ends of this and another vector
			* This is semantically treating the vector like a point */
			FORCE_INLINE Scalar distance2(const Vector2& v) const {
				return SquareRoot(distance(v));
			};

			/**@brief Return the distance between the ends of this and another vector
			* This is semantically treating the passed vector like a point */
			FORCE_INLINE Scalar distance(const Vector2& point) const {
				Vector2 v(this->perpendicular());
				Vector2 r(-point);
				return (v.dot(r))/length();			
			};

			/**@brief Normalize this vector 
			* x^2 + y^2 + z^2 = 1 */
			FORCE_INLINE Vector2& normalize() {
				if(length() != 0)
					return *this /= length();
				else
					return *this;
			}

			/**@brief Return a normalized version of this vector */
			FORCE_INLINE Vector2 normalized() const { 
				Scalar len = length();
				if(len == 0)
					return Vector2(0,0);

				Vector2 rec(x, y);
				rec /= len;
				return rec;
			};


			/**Relative to one point*/
			FORCE_INLINE Vector2 perpendicular(const Vector2 &rhs) const {
				return Vector2(rhs.y - y, -(rhs.x - x));
			}

			/**Relative to the origin*/
			FORCE_INLINE Vector2 perpendicular() const {
				return Vector2(-y, x);
			}

			/**Relative to the origin*/
			FORCE_INLINE Vector2 perpendicularRight() const {
				return Vector2(y, -x);
			}

			/** Symmetry */
			FORCE_INLINE Vector2 symmetricY() {
				return Vector2(x, -y);
			}

			/** Symmetry */
			FORCE_INLINE Vector2 symmetricX() {
				return Vector2(-x, y);
			}

			/**@brief Rotate this vector 
			* @param angle The angle to rotate by */
			FORCE_INLINE void rotate(const Scalar radiansAngle) {
				Vector2 tempVec(*this);
				x = tempVec.x * Cosine(radiansAngle) - tempVec.y * Sine(radiansAngle);
				y = tempVec.x * Sine(radiansAngle) + tempVec.y * Cosine(radiansAngle);
			};

			/**@brief Return the angle between this and another vector
			* @param v The other vector */
			FORCE_INLINE Scalar angle(const Vector2& rhs) const {
				Scalar s = SquareRoot(length2() * rhs.length2());
				assert(s != Scalar(0.0));
				return ArcCosine(dot(rhs) / s);
			}

			/************************************************************************/
			/* Utils                                                                */
			/************************************************************************/
			FORCE_INLINE void set(Scalar x, Scalar y) {
				this->x = x;
				this->y= y;
			}
			/**@brief Return a vector will the absolute values of each element */
			FORCE_INLINE Vector2 absolute() const {
				return Vector2(Absolute(x), Absolute(y));
			}

			/**@brief Return the axis with the smallest value */
			FORCE_INLINE Scalar minAxis() const {
				return x < y ? x : y;
			}

			/**@brief Return the axis with the largest value */
			FORCE_INLINE Scalar maxAxis() const {
				return x > y ? x : y;
			}

			FORCE_INLINE Scalar furthestAxis() const {
				return absolute().minAxis();
			}

			FORCE_INLINE Scalar closestAxis() const {
				return absolute().maxAxis();
			}
		};
	}
}
#endif