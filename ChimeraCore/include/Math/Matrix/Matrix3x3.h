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

#ifndef _MATH_MATRIX_3x3
#define _MATH_MATRIX_3x3

#include "Math/Vector3.h"
#include "Math/Matrix/MatrixNxN.h"

namespace Chimera {

	namespace Core {

		class Matrix3x3 {
		public:

			Vector3 column[3];

			/************************************************************************/
			/* ctors and dtors                                                      */
			/************************************************************************/
			//Default ctor
			Matrix3x3() { };

			// Constructor with initializing value
			FORCE_INLINE Matrix3x3(Scalar v) {
				column[0].set(v, v, v);
				column[1].set(v, v, v);
				column[2].set(v, v, v);
			}
			// Constructor with initializing Matrix3x3
			FORCE_INLINE Matrix3x3(const Matrix3x3 &m) {
				column[0] = m[0];
				column[1] = m[1];
				column[2] = m[2];
			}
			// Constructor with initializing Vector3f's
			FORCE_INLINE Matrix3x3(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2) {
				column[0] = v0;
				column[1] = v1;
				column[2] = v2;
			}
			//Constructor converting from a 3x3 MatrixNxN
			FORCE_INLINE Matrix3x3(const MatrixNxN &m) {
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						column[j][i] = m(i, j);
					}
				}
			}


			/************************************************************************/
			/* Operators                                                            */
			/************************************************************************/

			// Get column
			FORCE_INLINE Vector3 &operator [] (unsigned int i) {
				assert (i < 3);
				return(Vector3&)column[i];
			}

			// Array indexing
			FORCE_INLINE const Vector3 &operator [] (unsigned int i) const {
				assert (i < 3);
				return(const Vector3&)column[i];
			}

			// Assign
			FORCE_INLINE Matrix3x3 &operator= (const Matrix3x3 &m) {
				column[0] = m[0];
				column[1] = m[1];
				column[2] = m[2];
				return *this;
			}

			// Add a Matrix3x3 to this one
			FORCE_INLINE Matrix3x3 &operator+= (const Matrix3x3 &m) {
				column[0] += m[0];
				column[1] += m[1];
				column[2] += m[2];
				return *this;
			}

			// Subtract a Matrix3x3 from this one
			FORCE_INLINE Matrix3x3 &operator-= (const Matrix3x3 &m) {
				column[0] -= m[0];
				column[1] -= m[1];
				column[2] -= m[2];
				return *this;
			}

			// Multiply the Matrix3x3 by another Matrix3x3
			Matrix3x3      &operator *= (const Matrix3x3 &m) {
				Matrix3x3 tempMatrix(*this);

				for(int i = 0; i < 3; i++) {
					for(int j = 0; j < 3; j++) {
						column[i][j] = 0;
						for(int k = 0; k < 3; k++) {
							column[i][j] += tempMatrix[i][k]*m[k][j];
						}
					}
				}
				return *this;
			};

			// Multiply the Matrix3x3 by a float
			Matrix3x3      &operator *= (float f) {
				column[0] *= f;
				column[1] *= f;
				column[2] *= f;
				return *this;
			}

			// Are these two Matrix3x3's equal?
			friend bool       operator == (const Matrix3x3 &a, const Matrix3x3 &b) {
				return((a[0] == b[0]) && (a[1] == b[1]));
			}

			// Are these two Matrix3x3's not equal?
			friend bool       operator != (const Matrix3x3 &a, const Matrix3x3 &b) {
				return((a[0] != b[0]) || (a[1] != b[1]));
			}

			// Add two Matrix3x3's
			friend Matrix3x3   operator + (const Matrix3x3 &a, const Matrix3x3 &b) {
				Matrix3x3 ret(a);
				ret += b;
				return ret;
			}

			// Subtract one Matrix3x3 from another
			friend Matrix3x3   operator - (const Matrix3x3 &a, const Matrix3x3 &b) {
				Matrix3x3 ret(a);
				ret -= b;
				return ret;
			}

			// Multiply Matrix3x3 by another Matrix3x3
			friend Matrix3x3   operator * (const Matrix3x3 &a, const Matrix3x3 &b) {
				Matrix3x3 ret(a);
				ret *= b;
				return ret;
			}

			// Multiply a Vector3 by this Matrix3x3
			friend Vector3    operator * (const Matrix3x3 &m, const Vector3 &v) {
				Vector3 ret;
				ret.x = v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0];
				ret.y = v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1];
				ret.z = v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2];
				return ret;
			}

			// Multiply a Vector3 by this Matrix3x3
			friend Vector3    operator * (const Vector3 &v, const Matrix3x3 &m) {
				Vector3 ret;
				ret.x = v.dot(m[0]);
				ret.y = v.dot(m[1]);
				ret.z = v.dot(m[2]);
				return ret;
			}

			// Multiply Matrix3x3 by a Scalar
			friend Matrix3x3   operator * (const Matrix3x3 &m, Scalar s) {
				Matrix3x3 ret(m);
				ret *= s;
				return ret;
			}

			// Multiply Matrix3x3 by a Scalar
			friend Matrix3x3   operator * (Scalar s, const Matrix3x3 &m) {
				Matrix3x3 ret(m);
				ret *= s;
				return ret;
			}

			/************************************************************************/
			/* Matrix operations                                                    */
			/************************************************************************/
			Scalar determinant() const  {
				return	column[0].x * column[1].y * column[2].z +
						column[0].y * column[1].z * column[2].x +
						column[0].z * column[1].x * column[2].y -

						column[0].z * column[1].y * column[2].x -
						column[0].y * column[1].x * column[2].z -
						column[0].x * column[1].z * column[2].y;
			}

			void transpose() {
				Matrix3x3 tempMat(*this);
				for(int i = 0; i < 3; i++) {
					for(int j = 0; j < 3; j++) {
						column[i][j] = tempMat[j][i];
					}
				}
			}
			void invert() {
				Scalar inverseDeterminant = 1/determinant();
				Matrix3x3 tempMat(*this);
				

				column[0].x = inverseDeterminant*(tempMat[1].y*tempMat[2].z - tempMat[2].y*tempMat[1].z);
				column[0].y = inverseDeterminant*(tempMat[2].x*tempMat[1].z - tempMat[1].x*tempMat[2].z);
				column[0].z = inverseDeterminant*(tempMat[1].x*tempMat[2].y - tempMat[2].x*tempMat[1].y);

				column[1].x = inverseDeterminant*(tempMat[2].y*tempMat[0].z - tempMat[0].y*tempMat[2].z);
				column[1].y = inverseDeterminant*(tempMat[0].x*tempMat[2].z - tempMat[2].x*tempMat[0].z);
				column[1].z = inverseDeterminant*(tempMat[2].x*tempMat[0].y - tempMat[0].x*tempMat[2].y);

				column[2].x = inverseDeterminant*(tempMat[0].y*tempMat[1].z - tempMat[1].y*tempMat[0].z);
				column[2].y = inverseDeterminant*(tempMat[1].x*tempMat[0].z - tempMat[0].x*tempMat[1].z);
				column[2].z = inverseDeterminant*(tempMat[0].x*tempMat[1].y - tempMat[1].x*tempMat[0].y);

				transpose();

			}


		};
	}
}
#endif