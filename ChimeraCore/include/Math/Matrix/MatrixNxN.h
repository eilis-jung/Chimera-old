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

#ifndef _MATH_MATRIX_NXN
#define _MATH_MATRIX_NXN

#include "Math/Scalar.h"


namespace Chimera {
	namespace Core {

		class MatrixNxN {
		private:
			Scalar *m_pRawMatrix;
			unsigned int m_numColumns;
			unsigned int m_numRows;

		public:



			/************************************************************************/
			/* ctors and dtors                                                      */
			/************************************************************************/
			//Default ctor
			MatrixNxN(unsigned int numRows, unsigned int numColumns) : m_numRows(numRows), m_numColumns(numColumns) {
				m_pRawMatrix = new Scalar[m_numRows*m_numColumns];
			}


			// Constructor with initializing value
			FORCE_INLINE MatrixNxN(Scalar s, unsigned int numRows, unsigned int numColumns)
				: m_numRows(numRows), m_numColumns(numColumns) {
					m_pRawMatrix = new Scalar[m_numRows*m_numColumns];

					for(unsigned int i = 0; i < m_numRows; i++) {
						for(unsigned int j = 0; j < m_numColumns; j++) {
							m_pRawMatrix[i*m_numColumns + j] = s;
						}
					}
			}

			// Constructor with initializing MatrixNxN
			CHIMERA_API MatrixNxN(const MatrixNxN &m);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			FORCE_INLINE unsigned int getNumRows() const {
				return m_numRows;
			}

			FORCE_INLINE unsigned int getNumColumns() const {
				return m_numColumns;
			}

			/************************************************************************/
			/* Operators                                                            */
			/************************************************************************/

			//// Returns a pointer to the ith row
			//FORCE_INLINE Scalar * operator [] (unsigned int i) {
			//	assert (i < m_numRows);
			//	return (Scalar *)m_pRawMatrix[i];
			//}

			//// Returns a pointer to the ith row
			//FORCE_INLINE const Scalar *operator [] (unsigned int i) const {
			//	assert (i < m_numRows);
			//	return (Scalar *)m_pRawMatrix[i];
			//}

			FORCE_INLINE Scalar & operator() (unsigned int i, unsigned int j) {
				assert(i < m_numRows);
				assert(j < m_numColumns);
				return (m_pRawMatrix[i*m_numColumns + j]);
			}

			FORCE_INLINE Scalar operator() (unsigned int i , unsigned int j) const {
				assert(i < m_numRows);
				assert(j < m_numColumns);
				return(m_pRawMatrix[i*m_numColumns + j]);
			}

			// Assign
			FORCE_INLINE MatrixNxN &operator= (const MatrixNxN &rhs) {
				for(unsigned int i = 0; i < m_numRows; i++) {
					for(unsigned int j = 0; j < m_numColumns; j++) {
						m_pRawMatrix[i*m_numColumns + j] = rhs(i, j);
					}
				}
				return *this;
			}

			// Add a MatrixNxN to this one
			FORCE_INLINE MatrixNxN &operator+= (const MatrixNxN &rhs) {
				for(unsigned int i = 0; i < m_numRows; i++) {
					for(unsigned int j = 0; j < m_numColumns; j++) {
						m_pRawMatrix[i*m_numColumns + j] += rhs(i, j);
					}
				}
				return *this;
			}

			// Subtract a MatrixNxN from this one
			FORCE_INLINE MatrixNxN &operator-= (const MatrixNxN &rhs) {
				for(unsigned int i = 0; i < m_numRows; i++) {
					for(unsigned int j = 0; j < m_numColumns; j++) {
						m_pRawMatrix[i*m_numColumns + j] -= rhs(i, j);
					}
				}
				return *this;
			}

			// Multiply the MatrixNxN by another MatrixNxN
			MatrixNxN      &operator *= (const MatrixNxN &m);

			// Multiply the MatrixNxN by a float
			MatrixNxN      &operator *= (Scalar s) {
				for(unsigned int i = 0; i < m_numRows; i++) {
					for(unsigned int j = 0; j < m_numColumns; j++) {
						m_pRawMatrix[i*m_numColumns + j] *= s;
					}
				}

				return *this;
			}

			// Add two MatrixNxN's
			friend MatrixNxN   operator + (const MatrixNxN &a, const MatrixNxN &b) {
				MatrixNxN ret(a);
				ret += b;
				return ret;
			}

			// Subtract one MatrixNxN from another
			friend MatrixNxN   operator - (const MatrixNxN &a, const MatrixNxN &b) {
				MatrixNxN ret(a);
				ret -= b;
				return ret;
			}

			// Multiply MatrixNxN by another MatrixNxN
			friend MatrixNxN   operator * (const MatrixNxN &a, const MatrixNxN &b) {
				MatrixNxN ret(a);
				ret *= b;
				return ret;
			}

			//// Multiply a Vector2 by this MatrixNxN
			//friend Vector2    operator * (const MatrixNxN &m, const Vector2 &v) {
			//	Vector2 ret;
			//	ret.x = v.x * m[0][0] + v.y * m[1][0];
			//	ret.y = v.x * m[0][1] + v.y * m[1][1];
			//	return ret;
			//}

			//// Multiply a Vector2 by this MatrixNxN
			//friend Vector2    operator * (const Vector2 &v, const MatrixNxN &m) {
			//	Vector2 ret;
			//	ret.x = v.dot(m[0]);
			//	ret.y = v.dot(m[1]);
			//	return ret;
			//}

			// Multiply MatrixNxN by a Scalar
			friend MatrixNxN   operator * (const MatrixNxN &m, Scalar s) {
				MatrixNxN ret(m);
				ret *= s;
				return ret;
			}

			// Multiply MatrixNxN by a Scalar
			friend MatrixNxN   operator * (Scalar s, const MatrixNxN &m) {
				MatrixNxN ret(m);
				ret *= s;
				return ret;
			}


			void transpose();

		};
	}

}
#endif