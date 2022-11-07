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
	
#ifndef _MATH_MATRIX_2X2
#define _MATH_MATRIX_2X2

#include "Math/Vector2.h"

namespace Chimera {
	namespace Core {

		class Matrix2x2 {
		public:

			Vector2 column[2];

			/************************************************************************/
			/* ctors and dtors                                                      */
			/************************************************************************/
			//Default ctor
			Matrix2x2() { };

			// Constructor with initializing value
			FORCE_INLINE Matrix2x2(Scalar v) {
				column[0].set(v, v);
				column[1].set(v, v);
			}
			// Constructor with initializing Matrix2x2
			FORCE_INLINE Matrix2x2(const Matrix2x2 &m) {
				column[0] = m[0];
				column[1] = m[1];
			}
			// Constructor with initializing Vector3f's
			FORCE_INLINE Matrix2x2(const Vector2 &v0, const Vector2 &v1) {
				column[0] = v0;
				column[1] = v1;
			}


			/************************************************************************/
			/* Operators                                                            */
			/************************************************************************/

			// Array indexing
			FORCE_INLINE Vector2 &operator [] (unsigned int i) {
				assert (i < 2);
				return(Vector2&)column[i];
			}

			// Array indexing
			FORCE_INLINE const Vector2 &operator [] (unsigned int i) const {
				assert (i < 2);
				return(Vector2&)column[i];
			}

			// Assign
			FORCE_INLINE Matrix2x2 &operator= (const Matrix2x2 &m) {
				column[0] = m[0];
				column[1] = m[1];
				return *this;
			}

			// Add a Matrix2x2 to this one
			FORCE_INLINE Matrix2x2 &operator+= (const Matrix2x2 &m) {
				column[0] += m[0];
				column[1] += m[1];
				return *this;
			}

			// Subtract a Matrix2x2 from this one
			FORCE_INLINE Matrix2x2 &operator-= (const Matrix2x2 &m) {
				column[0] -= m[0];
				column[1] -= m[1];
				return *this;
			}

			// Multiply the Matrix2x2 by another Matrix2x2
			Matrix2x2      &operator *= (const Matrix2x2 &m);

			// Multiply the Matrix2x2 by a float
			Matrix2x2      &operator *= (float f) {
				column[0] *= f;
				column[1] *= f;
				column[2] *= f;
				return *this;
			}

			// Are these two Matrix2x2's equal?
			friend bool       operator == (const Matrix2x2 &a, const Matrix2x2 &b) {
				return((a[0] == b[0]) && (a[1] == b[1]));
			}

			// Are these two Matrix2x2's not equal?
			friend bool       operator != (const Matrix2x2 &a, const Matrix2x2 &b) {
				return((a[0] != b[0]) || (a[1] != b[1]));
			}

			// Add two Matrix2x2's
			friend Matrix2x2   operator + (const Matrix2x2 &a, const Matrix2x2 &b) {
				Matrix2x2 ret(a);
				ret += b;
				return ret;
			}

			// Subtract one Matrix2x2 from another
			friend Matrix2x2   operator - (const Matrix2x2 &a, const Matrix2x2 &b) {
				Matrix2x2 ret(a);
				ret -= b;
				return ret;
			}

			// Multiply Matrix2x2 by another Matrix2x2
			friend Matrix2x2   operator * (const Matrix2x2 &a, const Matrix2x2 &b) {
				Matrix2x2 ret(a);
				ret *= b;
				return ret;
			}

			// Multiply a Vector2 by this Matrix2x2
			friend Vector2    operator * (const Matrix2x2 &m, const Vector2 &v) {
				Vector2 ret;
				ret.x = v.x * m[0][0] + v.y * m[1][0];
				ret.y = v.x * m[0][1] + v.y * m[1][1];
				return ret;
			}

			// Multiply a Vector2 by this Matrix2x2
			friend Vector2    operator * (const Vector2 &v, const Matrix2x2 &m) {
				Vector2 ret;
				ret.x = v.dot(m[0]);
				ret.y = v.dot(m[1]);
				return ret;
			}

			// Multiply Matrix2x2 by a Scalar
			friend Matrix2x2   operator * (const Matrix2x2 &m, Scalar s) {
				Matrix2x2 ret(m);
				ret *= s;
				return ret;
			}

			// Multiply Matrix2x2 by a Scalar
			friend Matrix2x2   operator * (Scalar s, const Matrix2x2 &m) {
				Matrix2x2 ret(m);
				ret *= s;
				return ret;
			}


			/************************************************************************/
			/* Matrix operations                                                    */
			/************************************************************************/

			FORCE_INLINE Scalar determinant() const  {
				return	column[0].x * column[1].y -
					column[0].y * column[1].x;
			}

			FORCE_INLINE void transpose() {
				Scalar tempValue = column[0].y;
				column[0].y = column[1].x;
				column[1].x = tempValue;
			}
			void invert() {
				Scalar inverseDeterminant = 1/determinant();
				Scalar tempValue = column[0].x;

				column[0].x = inverseDeterminant*column[1].y;
				column[1].y = inverseDeterminant*tempValue;
				column[0].y = -inverseDeterminant*column[0].y;
				column[1].x = -inverseDeterminant*column[1].x;
			}

		};
	}
}
#endif