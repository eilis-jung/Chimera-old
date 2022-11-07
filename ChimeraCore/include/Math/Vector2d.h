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

#ifndef _MATH_VECTOR_2D_
#define _MATH_VECTOR_2D_

#include "Math/DoubleScalar.h"

namespace Chimera {
	namespace Core {

		class Vector2D {

		public:
			DoubleScalar x;
			DoubleScalar y;


			FORCE_INLINE Vector2D() { x = y = 0; }

			FORCE_INLINE Vector2D(const DoubleScalar &x, const DoubleScalar &y) {
				this->x = x;
				this->y = y;
			}

			/************************************************************************/
			/* Operators                                                            */
			/************************************************************************/
			// Array indexing - Useful for matrix operations
			DoubleScalar &operator [] (unsigned int i) {
				assert(i < 2);
				return *(&x + i);
			}

			// Array indexing
			const DoubleScalar &operator [] (unsigned int i) const {
				assert(i<2);
				return *(&x + i);
			}

			FORCE_INLINE Vector2D& operator+=(const Vector2D& rhs) {
				x += rhs.x; y += rhs.y;
				return *this;
			}

			FORCE_INLINE Vector2D& operator+=(const DoubleScalar& s) {
				x += s; y += s;
				return *this;
			}

			FORCE_INLINE Vector2D& operator-=(const Vector2D& rhs) {
				x -= rhs.x; y -= rhs.y;
				return *this;
			}
			FORCE_INLINE Vector2D& operator-=(const DoubleScalar& s) {
				x -= s; y -= s;
				return *this;
			}

			/**@brief Elementwise multiply this vector by the other
			* @param rhs The other vector */
			FORCE_INLINE Vector2D& operator*=(const Vector2D& rhs) {
				x *= rhs.x; y *= rhs.y;
				return *this;
			}

			FORCE_INLINE Vector2D& operator/=(const Vector2D& rhs) {
				x /= rhs.x; y /= rhs.y;
				return *this;
			}

			FORCE_INLINE Vector2D& operator*=(const DoubleScalar& s) {
				x *= s; y *= s;
				return *this;
			}

			FORCE_INLINE Vector2D& operator/=(const DoubleScalar& s) {
				assert(s != DoubleScalar(0.0));
				x /= s; y /= s;
				return *this;
			}


			FORCE_INLINE bool operator==(const Vector2D& rhs) const {
				return (x == rhs.x) && (y == rhs.y);
			}

			FORCE_INLINE bool operator!=(const Vector2D& other) const {
				return !(*this == other);
			}

			//// Negate a Vector3f
			FORCE_INLINE friend Vector2D operator - (const Vector2D &a) {
				return Vector2D(-a.x, -a.y);
			}

			// Add two Vector3f's
			FORCE_INLINE friend Vector2D operator + (const Vector2D &a, const Vector2D &b) {
				Vector2D ret(a);
				ret += b;
				return ret;
			}
			FORCE_INLINE friend Vector2D operator + (const Vector2D &lhs, DoubleScalar s) {
				Vector2D ret(lhs);
				ret += s;
				return ret;
			}

			// Subtract one vector3 from another
			FORCE_INLINE friend Vector2D operator - (const Vector2D &a, const Vector2D &b) {
				Vector2D ret(a);
				ret -= b;
				return ret;
			}

			FORCE_INLINE friend Vector2D operator - (const Vector2D &lhs, DoubleScalar s) {
				Vector2D ret(lhs);
				ret -= s;
				return ret;
			}

			FORCE_INLINE friend Vector2D operator * (const Vector2D &lhs, DoubleScalar s) {
				Vector2D ret(lhs);
				ret *= s;
				return ret;
			}
			FORCE_INLINE friend Vector2D operator / (const Vector2D &lhs, DoubleScalar s) {
				Vector2D ret(lhs);
				ret /= s;
				return ret;
			}

			/** It is not a dot product, it is rather a component-by-component multiplication. */
			FORCE_INLINE friend Vector2D operator * (const Vector2D &lhs, const Vector2D &rhs) {
				Vector2D ret(lhs);
				ret *= rhs;
				return ret;
			}

			/** Component-by-component division.*/
			FORCE_INLINE friend Vector2D operator / (const Vector2D &lhs, const Vector2D &rhs) {
				Vector2D ret(lhs);
				ret /= rhs;
				return ret;
			}

			/************************************************************************/
			/* Comparison operators/Relational operators							*/
			/************************************************************************/
			/** Compares the absolute position of vectors; isn't related to length*/
			/** Priority is given to x-axis. */
			bool Vector2D::operator < (const Vector2D & rhs) const {
				if (x < rhs.x)
					return true;
				else if (x == rhs.x) {
					if (y < rhs.y)
						return true;
				}
				return false;
			}
			bool Vector2D::operator >(const Vector2D & rhs) const {
				if (x > rhs.x)
					return true;
				else if (x == rhs.x) {
					if (y > rhs.y)
						return true;
				}
				return false;
			}


			/************************************************************************/
			/* Geometric operations                                                 */
			/************************************************************************/
			FORCE_INLINE DoubleScalar dot(const Vector2D& rhs) const {
				return x * rhs.x + y * rhs.y;
			}

			/** 2D interpretation of the cross vector, determinant of
			** |	i		j		k	|
			** |	v1.x	v1.y	0	|
			** |	v2.x	v2.y	0	|
			**/

			FORCE_INLINE DoubleScalar cross(const Vector2D &rhs) const {
				return x*rhs.y - y*rhs.x;
			}

			/**@brief Return the L2 of a a vector */
			FORCE_INLINE DoubleScalar length2() const {
				return dot(*this);
			}

			/**@brief Return the length of the vector */
			FORCE_INLINE DoubleScalar length() const {
				return SquareRoot(length2());
			}

			/**@brief Return the distance squared between the ends of this and another vector
			* This is semantically treating the vector like a point */
			FORCE_INLINE DoubleScalar distance2(const Vector2D& v) const {
				return SquareRoot(distance(v));
			};

			/**@brief Return the distance between the ends of this and another vector
			* This is semantically treating the passed vector like a point */
			FORCE_INLINE DoubleScalar distance(const Vector2D& point) const {
				Vector2D v(this->perpendicular());
				Vector2D r(-point);
				return (v.dot(r)) / length();
			};

			/**@brief Normalize this vector
			* x^2 + y^2 + z^2 = 1 */
			FORCE_INLINE Vector2D& normalize() {
				if (length() != 0)
					return *this /= length();
				else
					return *this;
			}

			/**@brief Return a normalized version of this vector */
			FORCE_INLINE Vector2D normalized() const {
				DoubleScalar len = length();
				if (len == 0)
					return Vector2D(0, 0);

				Vector2D rec(x, y);
				rec /= len;
				return rec;
			};


			/**Relative to one point*/
			FORCE_INLINE Vector2D perpendicular(const Vector2D &rhs) const {
				return Vector2D(rhs.y - y, -(rhs.x - x));
			}

			/**Relative to the origin*/
			FORCE_INLINE Vector2D perpendicular() const {
				return Vector2D(-y, x);
			}

			/**@brief Rotate this vector
			* @param angle The angle to rotate by */
			FORCE_INLINE void rotate(const DoubleScalar radiansAngle) {
				Vector2D tempVec(*this);
				x = tempVec.x * Cosine(radiansAngle) - tempVec.y * Sine(radiansAngle);
				y = tempVec.x * Sine(radiansAngle) + tempVec.y * Cosine(radiansAngle);
			};

			/**@brief Return the angle between this and another vector
			* @param v The other vector */
			FORCE_INLINE DoubleScalar angle(const Vector2D& rhs) const {
				DoubleScalar s = SquareRoot(length2() * rhs.length2());
				assert(s != DoubleScalar(0.0));
				return ArcCosine(dot(rhs) / s);
			}

			/************************************************************************/
			/* Utils                                                                */
			/************************************************************************/
			FORCE_INLINE void set(DoubleScalar x, DoubleScalar y) {
				this->x = x;
				this->y = y;
			}
			/**@brief Return a vector will the absolute values of each element */
			FORCE_INLINE Vector2D absolute() const {
				return Vector2D(Absolute(x), Absolute(y));
			}

			/**@brief Return the axis with the smallest value */
			FORCE_INLINE DoubleScalar minAxis() const {
				return x < y ? x : y;
			}

			/**@brief Return the axis with the largest value */
			FORCE_INLINE DoubleScalar maxAxis() const {
				return x > y ? x : y;
			}

			FORCE_INLINE DoubleScalar furthestAxis() const {
				return absolute().minAxis();
			}

			FORCE_INLINE DoubleScalar closestAxis() const {
				return absolute().maxAxis();
			}
		};

		//Explicit conversion functions
		Vector2D static convertToVector2D(const Vector2 & vec) {
			Vector2D convertedVec(static_cast<DoubleScalar>(vec.x),
				static_cast<DoubleScalar>(vec.y));
			return convertedVec;
		}

		//Explicit conversion functions
		Vector2 static convertToVector2F(const Vector2D & vec) {
			Vector2 convertedVec(static_cast<Scalar>(vec.x),
				static_cast<Scalar>(vec.y));
			return convertedVec;
		}

	}
}
#endif