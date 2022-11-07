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

#ifndef _MATH_VECTOR3D_H_
#define _MATH_VECTOR3D_H_

#include "Math/DoubleScalar.h"
#include "Math/Vector2D.h"

namespace Chimera {
	namespace Core {
		class Vector3D
		{
		public:

			DoubleScalar x, y, z;

			FORCE_INLINE Vector3D() { x = 0; y = 0; z = 0; }
			FORCE_INLINE Vector3D(DoubleScalar _x, DoubleScalar _y, DoubleScalar _z) { x = _x; y = _y; z = _z; }
			FORCE_INLINE Vector3D(const Vector2D &vec2) { x = vec2.x; y = vec2.y; z = 0; }

			// Operators
			// Array indexing
			FORCE_INLINE DoubleScalar &operator [] (unsigned int i) {
				assert(i < 3);
				return *(&x + i);
			}

			// Array indexing
			FORCE_INLINE const DoubleScalar &operator [] (unsigned int i) const {
				assert(i < 3);
				return *(&x + i);
			}

			// Add a Vector3D to this one
			FORCE_INLINE Vector3D &operator += (const Vector3D &v) {
				x += v.x;
				y += v.y;
				z += v.z;
				return *this;
			}


			// Subtract a Vector3D from this one
			FORCE_INLINE Vector3D &operator -= (const Vector3D &v) {
				x -= v.x;
				y -= v.y;
				z -= v.z;
				return *this;
			}

			FORCE_INLINE Vector3D & operator*=(const Vector3D &rhs) {
				x *= rhs.x; y *= rhs.y; z *= rhs.z;
				return *this;
			}

			FORCE_INLINE Vector3D & operator/=(const Vector3D &rhs) {
				x /= rhs.x; y /= rhs.y; z /= rhs.z;
				return *this;
			}

			// Multiply the Vector3D by a DoubleScalar
			FORCE_INLINE Vector3D &operator *= (DoubleScalar f) {
				x *= f;
				y *= f;
				z *= f;
				return *this;
			}

			// Divide the Vector3D by a DoubleScalar
			FORCE_INLINE Vector3D &operator /= (DoubleScalar f) {
				x /= f;
				y /= f;
				z /= f;
				return *this;
			}

			// Are these two Vector3D's equal?
			FORCE_INLINE friend bool operator == (const Vector3D &a, const Vector3D &b) {
				return((a.x == b.x) && (a.y == b.y) && (a.z == b.z));
			}

			// Are these two Vector3D's not equal?
			FORCE_INLINE friend bool operator != (const Vector3D &a, const Vector3D &b) {
				return((a.x != b.x) || (a.y != b.y) || (a.z != b.z));
			}

			// Negate a Vector3D
			FORCE_INLINE friend Vector3D operator - (const Vector3D &a) {
				return Vector3D(-a.x, -a.y, -a.z);
			}

			// Add two Vector3D's
			FORCE_INLINE friend Vector3D operator + (const Vector3D &a, const Vector3D &b) {
				Vector3D ret(a);
				ret += b;
				return ret;
			}

			// Subtract one Vector3D from another
			FORCE_INLINE friend Vector3D operator - (const Vector3D &a, const Vector3D &b) {
				Vector3D ret(a);
				ret -= b;
				return ret;
			}

			// Multiply Vector3D by a DoubleScalar
			FORCE_INLINE friend Vector3D operator * (const Vector3D &v, DoubleScalar f) {
				return Vector3D(f * v.x, f * v.y, f * v.z);
			}

			// Divide Vector3D by a DoubleScalar
			FORCE_INLINE friend Vector3D operator / (const Vector3D &v, DoubleScalar f) {
				return Vector3D(v.x / f, v.y / f, v.z / f);
			}

			/** It is not a dot product, it is rather a component-by-component multiplication. */
			FORCE_INLINE friend Vector3D operator * (const Vector3D &lhs, const Vector3D &rhs) {
				Vector3D ret(lhs);
				ret *= rhs;
				return ret;
			}

			/** Component-by-component division.*/
			FORCE_INLINE friend Vector3D operator / (const Vector3D &lhs, const Vector3D &rhs) {
				Vector3D ret(lhs);
				ret /= rhs;
				return ret;
			}

			/************************************************************************/
			/* Comparison operators/Relational operators							*/
			/************************************************************************/
			/** Compares the absolute position of vectors; isn't related to length*/
			/** Priority is given to x-axis and y-axis. */
			FORCE_INLINE bool Vector3D::operator < (const Vector3D & rhs) const {
				if (x < rhs.x)
					return true;
				else if (x == rhs.x) {
					if (y < rhs.y)
						return true;
					else if (y == rhs.y) {
						if (z < rhs.z)
							return true;
					}
				}
				return false;
			}
			FORCE_INLINE bool Vector3D::operator >(const Vector3D & rhs) const {
				if (x > rhs.x)
					return true;
				else if (x == rhs.x) {
					if (y > rhs.y)
						return true;
					else if (y == rhs.y) {
						if (z > rhs.z)
							return true;
					}
				}
				return false;
			}

			// Set Values
			FORCE_INLINE void set(DoubleScalar xIn, DoubleScalar yIn, DoubleScalar zIn) {
				x = xIn;
				y = yIn;
				z = zIn;
			}

			// Get length of a Vector3D
			FORCE_INLINE DoubleScalar length() const {
				return(DoubleScalar)sqrt(x*x + y*y + z*z);
			}

			// Get squared length of a Vector3D
			FORCE_INLINE DoubleScalar lengthSqr() const {
				return(x*x + y*y + z*z);
			}

			// Does Vector3D equal (0, 0, 0)?
			FORCE_INLINE bool isZero() const {
				return((x == 0.0F) && (y == 0.0F) && (z == 0.0F));
			}

			// Normalize a Vector3D
			FORCE_INLINE void normalize() {
				DoubleScalar m = length();
				if (m > 0.0F)
					m = 1.0F / m;
				else
					m = 0.0F;

				x *= m;
				y *= m;
				z *= m;
			}


			// Normalize a Vector3D
			FORCE_INLINE Vector3D normalized() const {
				Vector3D vec(*this);
				DoubleScalar m = length();
				if (m > 0.0F)
					m = 1.0F / m;
				else
					m = 0.0F;

				vec.x *= m;
				vec.y *= m;
				vec.z *= m;

				return vec;
			}


			// Cross product of two Vector3D's
			FORCE_INLINE Vector3D cross(const Vector3D &a, const Vector3D &b) const
			{
				return Vector3D(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
			}
			// Cross product of one Vector3D's
			FORCE_INLINE Vector3D cross(const Vector3D &b) const
			{
				return Vector3D(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
			}
			// Dot product of two Vector3D's
			FORCE_INLINE DoubleScalar dot(const Vector3D &b) const
			{
				return x*b.x + y*b.y + z*b.z;
			}

			static bool compare(Vector3D a, Vector3D b) {
				return a < b;
			};


			DoubleScalar angle(const Vector3D &a) {
				DoubleScalar dotProd = dot(a);
				return acos(dotProd / (length()*a.length()));
			}

			/** Static functions */
			static DoubleScalar triple(const Vector3D &a, const Vector3D &b, const Vector3D &c) {
				return a.dot(b.cross(c));
			}

			FORCE_INLINE Vector3D perpendicular() const {
				return Vector3D(x, y, z);
			}
		};


		//Explicit conversion functions
		template <class VectorType>
		Vector3D static convertToVector3D(const VectorType & vec) {
			Vector3D convertedVec(static_cast<DoubleScalar>(vec.x), 
									static_cast<DoubleScalar>(vec.y), 
									static_cast<DoubleScalar>(vec.z));
			return convertedVec;
		}

		//Explicit conversion functions
		template <class VectorType>
		Vector3 static convertToVector3F(const VectorType & vec) {
			Vector3 convertedVec(	static_cast<Scalar>(vec.x),
									static_cast<Scalar>(vec.y),
									static_cast<Scalar>(vec.z));
			return convertedVec;
		}

	} // namespace Core {
}

#endif 
