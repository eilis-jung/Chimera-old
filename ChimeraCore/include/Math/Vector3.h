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

#ifndef _MATH_VECTOR3F_H_
#define _MATH_VECTOR3F_H_

#include "Math/Scalar.h"
#include "Math/Vector2.h"

namespace Chimera {
	namespace Core {
		class Vector3
		{
		public:

			Scalar x, y, z;

			FORCE_INLINE Vector3() { x = 0; y = 0; z = 0; }
			FORCE_INLINE Vector3(Scalar _x, Scalar _y, Scalar _z) { x = _x; y = _y; z = _z; }
			FORCE_INLINE Vector3(const Vector2 &vec2) { x = vec2.x; y = vec2.y; z = 0; } 

			// Operators
			// Array indexing
			FORCE_INLINE Scalar &operator [] (unsigned int i) {
				assert(i<3);
				return *(&x+i);
			}

			// Array indexing
			FORCE_INLINE const Scalar &operator [] (unsigned int i) const {
				assert(i<3);
				return *(&x+i);
			}

			// Add a Vector3 to this one
			FORCE_INLINE Vector3 &operator += (const Vector3 &v) {
				x += v.x;
				y += v.y;
				z += v.z;
				return *this;
			}


			// Subtract a Vector3 from this one
			FORCE_INLINE Vector3 &operator -= (const Vector3 &v) {
				x -= v.x;
				y -= v.y;
				z -= v.z;
				return *this;
			}

			FORCE_INLINE Vector3 & operator*=(const Vector3 &rhs) {
				x *= rhs.x; y *= rhs.y; z *= rhs.z;
				return *this;
			}

			FORCE_INLINE Vector3 & operator/=(const Vector3 &rhs) {
				x /= rhs.x; y /= rhs.y; z /= rhs.z;
				return *this;
			}

			// Multiply the Vector3 by a Scalar
			FORCE_INLINE Vector3 &operator *= (Scalar f) {
				x *= f;
				y *= f;
				z *= f;
				return *this;
			}

			// Divide the Vector3 by a Scalar
			FORCE_INLINE Vector3 &operator /= (Scalar f) {
				x /= f;
				y /= f;
				z /= f;
				return *this;
			}

			// Are these two Vector3's equal?
			FORCE_INLINE friend bool operator == (const Vector3 &a, const Vector3 &b) {
				return((a.x == b.x) && (a.y == b.y) && (a.z == b.z));
			}

			// Are these two Vector3's not equal?
			FORCE_INLINE friend bool operator != (const Vector3 &a, const Vector3 &b) {
				return((a.x != b.x) || (a.y != b.y) || (a.z != b.z));
			}

			// Negate a Vector3
			FORCE_INLINE friend Vector3 operator - (const Vector3 &a) {
				return Vector3(-a.x, -a.y, -a.z);
			}

			// Add two Vector3's
			FORCE_INLINE friend Vector3 operator + (const Vector3 &a, const Vector3 &b) {
				Vector3 ret(a);
				ret += b;
				return ret;
			}

			// Subtract one vector3 from another
			FORCE_INLINE friend Vector3 operator - (const Vector3 &a, const Vector3 &b) {
				Vector3 ret(a);
				ret -= b;
				return ret;
			}

			// Multiply vector3 by a Scalar
			FORCE_INLINE friend Vector3 operator * (const Vector3 &v, Scalar f) {
				return Vector3(f * v.x, f * v.y, f * v.z);
			}

			// Divide vector3 by a Scalar
			FORCE_INLINE friend Vector3 operator / (const Vector3 &v, Scalar f) {
				return Vector3(v.x / f, v.y / f, v.z / f);
			}

			/** It is not a dot product, it is rather a component-by-component multiplication. */
			FORCE_INLINE friend Vector3 operator * (const Vector3 &lhs, const Vector3 &rhs) {
				Vector3 ret(lhs);
				ret *= rhs;
				return ret;
			}

			/** Component-by-component division.*/
			FORCE_INLINE friend Vector3 operator / (const Vector3 &lhs, const Vector3 &rhs) {
				Vector3 ret(lhs);
				ret /= rhs;
				return ret;
			}

			/************************************************************************/
			/* Comparison operators/Relational operators							*/
			/************************************************************************/
			/** Compares the absolute position of vectors; isn't related to length*/
			/** Priority is given to x-axis and y-axis. */
			FORCE_INLINE bool Vector3::operator < (const Vector3 & rhs) const {
				if(x < rhs.x)
					return true;
				else if(x == rhs.x) {
					if(y < rhs.y)
						return true;
					else if(y == rhs.y) {
						if(z < rhs.z)
							return true;
					} 
				}
				return false;
			}
			FORCE_INLINE bool Vector3::operator > (const Vector3 & rhs) const {
				if(x > rhs.x)
					return true;
				else if(x == rhs.x) {
					if(y > rhs.y)
						return true;
					else if(y == rhs.y) {
						if(z > rhs.z)
							return true;
					} 
				}
				return false;
			}

			// Set Values
			FORCE_INLINE void set(Scalar xIn, Scalar yIn, Scalar zIn) {
				x = xIn;
				y = yIn;
				z = zIn;
			}

			// Get length of a Vector3
			FORCE_INLINE Scalar length() const {
				return(Scalar) sqrt(x*x + y*y + z*z);
			}

			// Get squared length of a Vector3
			FORCE_INLINE Scalar lengthSqr() const {
				return(x*x + y*y + z*z);
			}

			// Does Vector3 equal (0, 0, 0)?
			FORCE_INLINE bool isZero() const {
				return((x == 0.0F) && (y == 0.0F) && (z == 0.0F));
			}

			// Normalize a Vector3
			FORCE_INLINE void normalize() {
				Scalar m = length();
				if (m > 0.0F)
					m = 1.0F / m;
				else
					m = 0.0F;

				x *= m;
				y *= m;
				z *= m;
			}


			// Normalize a Vector3
			FORCE_INLINE Vector3 normalized() {
				Vector3 vec(*this);
				Scalar m = length();
				if (m > 0.0F)
					m = 1.0F / m;
				else
					m = 0.0F;

				vec.x *= m;
				vec.y *= m;
				vec.z *= m;

				return vec;
			}


			// Cross product of two Vector3's
			FORCE_INLINE Vector3 cross(const Vector3 &a, const Vector3 &b) const
			{
				return Vector3(a.y*b.z - a.z*b.y, a.z*b.x -  a.x*b.z, a.x*b.y - a.y*b.x);
			}
			// Cross product of one Vector3's
			FORCE_INLINE Vector3 cross(const Vector3 &b) const
			{
				return Vector3(y*b.z - z*b.y, z*b.x -  x*b.z, x*b.y - y*b.x);
			}
			// Dot product of two Vector3's
			FORCE_INLINE Scalar dot(const Vector3 &b) const
			{
				return x*b.x + y*b.y + z*b.z;
			}

			static bool compare(Vector3 a, Vector3 b) { 
				return a < b;
			};


			Scalar angle(const Vector3 &a) {
				Scalar dotProd = dot(a);	
				return acos(dotProd / (length()*a.length()));
			}
			

			/** Static functions */
			static Scalar triple(const Vector3 &a, const Vector3 &b, const Vector3 &c) {
				return a.dot(b.cross(c));
			}

			/**Relative to the origin*/
			/** Compatibel to 2-D vecs */
			FORCE_INLINE Vector3 perpendicular() const {
				return Vector3(x, y, z);
			}


		};



	} // namespace Core {
}

#endif 
