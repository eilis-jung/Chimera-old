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

#ifndef __CHIMERA_DOUBLE_BUFFER_H__
#define __CHIMERA_DOUBLE_BUFFER_H__

#pragma once

namespace Chimera {

	namespace Core {

		template <class ValueType, template <class> class ArrayType>
		class DoubleBuffer {

			ArrayType<ValueType> * m_pBuffer1;
			ArrayType<ValueType> * m_pBuffer2;

			dimensions_t m_dimensions;

		public:

			DoubleBuffer(const dimensions_t &gridDimensions) {
				m_dimensions = gridDimensions;
				m_pBuffer1 = new ArrayType<ValueType>(gridDimensions);
				m_pBuffer2 = new ArrayType<ValueType>(gridDimensions);
			}

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			FORCE_INLINE const dimensions_t & getDimensions() const {
				return m_dimensions;
			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			FORCE_INLINE void swapBuffers() {
				ArrayType<ValueType> *pAuxPtr = m_pBuffer1;
				m_pBuffer1 = m_pBuffer2;
				m_pBuffer2 = pAuxPtr;
			}

			ArrayType<ValueType> * getBufferArray1() const {
				return m_pBuffer1;
			}
			
			ArrayType<ValueType> * getBufferArray2() const {
				return m_pBuffer2;
			}

			FORCE_INLINE ValueType getValue(int i, int j) const {
				return (*m_pBuffer1)(i, j);
			}
			FORCE_INLINE ValueType getValue(int i, int j, int k) const {
				return (*m_pBuffer1)(i, j, k);
			}
			
			FORCE_INLINE void setValue(ValueType gValue, int i, int j) {
				(*m_pBuffer2)(i, j) = gValue;
			}
			FORCE_INLINE void setValue(ValueType gValue, int i, int j, int k) {
				(*m_pBuffer2)(i, j, k) = gValue;
			}

			FORCE_INLINE void setValueBothBuffers(ValueType gValue, int i, int j) {
				(*m_pBuffer1)(i, j) = gValue;
				(*m_pBuffer2)(i, j) = gValue;
			}
			FORCE_INLINE void setValueBothBuffers(ValueType gValue, int i, int j, int k) {
				(*m_pBuffer1)(i, j, k) = gValue;
				(*m_pBuffer2)(i, j, k) = gValue;
			}
		};
	}
}

#endif