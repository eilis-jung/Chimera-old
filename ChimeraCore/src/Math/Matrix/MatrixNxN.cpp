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

#include "Math/Matrix/MatrixNxN.h"

namespace Chimera {
	namespace Core {
		/************************************************************************/
		/* ctors and dtors                                                      */
		/************************************************************************/
		MatrixNxN::MatrixNxN(const MatrixNxN &rhs) {
			m_numRows = rhs.getNumRows();
			m_numColumns = rhs.getNumColumns();

			m_pRawMatrix = new Scalar[m_numRows*m_numColumns];

			for(unsigned int i = 0; i < m_numRows; i++) {
				for(unsigned int j = 0; j < m_numColumns; j++) {
					m_pRawMatrix[i*m_numColumns + j] = rhs(i, j);
				}
			}
		}

		MatrixNxN & MatrixNxN::operator *=(const MatrixNxN &m) {
			assert(m.getNumRows() == m_numColumns);
			MatrixNxN tempMatrix(*this);

			delete m_pRawMatrix;
			m_numColumns = m.getNumColumns();
			m_pRawMatrix = new Scalar[m_numRows*m_numColumns];

			for (int i = 0; i < m_numRows; i++) {
				for (int j = 0; j < m_numColumns; j++) {
					m_pRawMatrix[i*m_numColumns + j] = 0;
					for (int k = 0; k < m.getNumRows(); k++) {
						m_pRawMatrix[i*m_numColumns + j] += tempMatrix(i, k) * m(k, j);
					}
				}
			}
			return *this;
		}

		void MatrixNxN::transpose() {
			MatrixNxN tempMat(*this);

			delete m_pRawMatrix;
			std::swap(m_numRows, m_numColumns);
			m_pRawMatrix = new Scalar[m_numRows*m_numColumns];

			for (int i = 0; i < tempMat.getNumRows(); i++) {
				for (int j = 0; j < tempMat.getNumColumns(); j++) {
					(*this)(j, i) = tempMat(i, j);
				}
			}
		}
	}
}