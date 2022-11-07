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

#ifndef _MATH_MATRIX_NXN_D
#define _MATH_MATRIX_NXN_D

#include "Math/DoubleScalar.h"


namespace Chimera {
	namespace Core {

		class MatrixNxND {
		private:
			DoubleScalar *m_pRawMatrix;
			unsigned int m_numColumns;
			unsigned int m_numRows;

		public:



			/************************************************************************/
			/* ctors and dtors                                                      */
			/************************************************************************/
			//Default ctor
			MatrixNxND(unsigned int numRows, unsigned int numColumns) : m_numRows(numRows), m_numColumns(numColumns) {
				m_pRawMatrix = new DoubleScalar[m_numRows*m_numColumns];
			}

			~MatrixNxND() {
				delete m_pRawMatrix;
			}


			// Constructor with initializing value
			FORCE_INLINE MatrixNxND(DoubleScalar s, unsigned int numRows, unsigned int numColumns)
				: m_numRows(numRows), m_numColumns(numColumns) {
				m_pRawMatrix = new DoubleScalar[m_numRows*m_numColumns];

					for(unsigned int i = 0; i < m_numRows; i++) {
						for(unsigned int j = 0; j < m_numColumns; j++) {
							m_pRawMatrix[i*m_numColumns + j] = s;
						}
					}
			}

			// Constructor with initializing MatrixNxN
			CHIMERA_API MatrixNxND(const MatrixNxND &m);

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

			FORCE_INLINE DoubleScalar & operator() (unsigned int i, unsigned int j) {
				assert(i < m_numRows);
				assert(j < m_numColumns);
				return (m_pRawMatrix[i*m_numColumns + j]);
			}

			FORCE_INLINE DoubleScalar operator() (unsigned int i, unsigned int j) const {
				assert(i < m_numRows);
				assert(j < m_numColumns);
				return(m_pRawMatrix[i*m_numColumns + j]);
			}

			// Assign
			FORCE_INLINE MatrixNxND &operator= (const MatrixNxND &rhs) {
				for(unsigned int i = 0; i < m_numRows; i++) {
					for(unsigned int j = 0; j < m_numColumns; j++) {
						m_pRawMatrix[i*m_numColumns + j] = rhs(i, j);
					}
				}
				return *this;
			}

			// Add a MatrixNxNd to this one
			FORCE_INLINE MatrixNxND &operator+= (const MatrixNxND &rhs) {
				for(unsigned int i = 0; i < m_numRows; i++) {
					for(unsigned int j = 0; j < m_numColumns; j++) {
						m_pRawMatrix[i*m_numColumns + j] += rhs(i, j);
					}
				}
				return *this;
			}

			// Subtract a MatrixNxNd from this one
			FORCE_INLINE MatrixNxND &operator-= (const MatrixNxND &rhs) {
				for(unsigned int i = 0; i < m_numRows; i++) {
					for(unsigned int j = 0; j < m_numColumns; j++) {
						m_pRawMatrix[i*m_numColumns + j] -= rhs(i, j);
					}
				}
				return *this;
			}

			// Multiply the MatrixNxNd by another MatrixNxNd
			MatrixNxND      &operator *= (const MatrixNxND &m);

			// Multiply the MatrixNxNd by a float
			MatrixNxND      &operator *= (Scalar s) {
				for(unsigned int i = 0; i < m_numRows; i++) {
					for(unsigned int j = 0; j < m_numColumns; j++) {
						m_pRawMatrix[i*m_numColumns + j] *= s;
					}
				}

				return *this;
			}

			// Add two MatrixNxNd's
			friend MatrixNxND   operator + (const MatrixNxND &a, const MatrixNxND &b) {
				MatrixNxND ret(a);
				ret += b;
				return ret;
			}

			// Subtract one MatrixNxNd from another
			friend MatrixNxND   operator - (const MatrixNxND &a, const MatrixNxND &b) {
				MatrixNxND ret(a);
				ret -= b;
				return ret;
			}

			// Multiply MatrixNxNd by another MatrixNxNd
			friend MatrixNxND   operator * (const MatrixNxND &a, const MatrixNxND &b) {
				MatrixNxND ret(a);
				ret *= b;
				return ret;
			}

			//// Multiply a Vector2 by this MatrixNxNd
			//friend Vector2    operator * (const MatrixNxNd &m, const Vector2 &v) {
			//	Vector2 ret;
			//	ret.x = v.x * m[0][0] + v.y * m[1][0];
			//	ret.y = v.x * m[0][1] + v.y * m[1][1];
			//	return ret;
			//}

			//// Multiply a Vector2 by this MatrixNxNd
			//friend Vector2    operator * (const Vector2 &v, const MatrixNxNd &m) {
			//	Vector2 ret;
			//	ret.x = v.dot(m[0]);
			//	ret.y = v.dot(m[1]);
			//	return ret;
			//}

			// Multiply MatrixNxNd by a Scalar
			friend MatrixNxND   operator * (const MatrixNxND &m, Scalar s) {
				MatrixNxND ret(m);
				ret *= s;
				return ret;
			}

			// Multiply MatrixNxNd by a Scalar
			friend MatrixNxND   operator * (Scalar s, const MatrixNxND &m) {
				MatrixNxND ret(m);
				ret *= s;
				return ret;
			}


			void transpose();

		};
	}

}
#endif