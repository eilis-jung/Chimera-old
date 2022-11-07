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

#ifndef __CHIMERA_ARRAY_2D_H__
#define __CHIMERA_ARRAY_2D_H__

#include "Data/ChimeraStructures.h"
#include "Data/Array.h"
#include "Data/ReferenceWrapper.h"

#pragma once

namespace Chimera {

	namespace Core {
		template <class ArrayClassT>
		class Array2D : public Array<ArrayClassT> {

		protected:
			/**
			* the value of the given location (non-const).
			*/
			ArrayClassT& Value(int i, int j)
			{
				return  m_rawArray[j*m_dimensions.x + i];
			}

		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			Array2D(const dimensions_t &dimensions, bool initializePointers = false) : Array<ArrayClassT>(dimensions, initializePointers) {
				//m_rawArray.resize(dimensions.x*dimensions.y);
				
				m_arrayOwnership = true;
			}

			Array2D() : Array<ArrayClassT>(dimensions_t(0, 0, 0)) {
				m_arrayOwnership = true;
			}

			/*Array2D(ArrayClassT *gpRawArray, const dimensions_t &dimensions) : Array<ArrayClassT>(gpRawArray, dimensions) {
				m_arrayOwnership = false;
			}*/

			~Array2D() {
				/*if(m_arrayOwnership) {
					if(pRawArray != NULL)
						delete[] pRawArray;
					pRawArray = NULL;
				}*/
			}


			static Array2D<ReferenceWrapper<ArrayClassT>> * createArrayOfReferences(Array2D<ArrayClassT> &array2D) {
				Array2D<ReferenceWrapper<ArrayClassT>> *pArray = new Array2D<ReferenceWrapper<ArrayClassT>>();
				for (int j = 0; j < array2D.getDimensions().y; j++) {
					for (int i = 0; i < array2D.getDimensions().x; i++) {
						ArrayClassT *pRef = array2D.getValuePtr(i, j);
						pArray->getRawData().push_back(ReferenceWrapper<ArrayClassT>(*pRef));
					}
				}
				pArray->setDimensions(dimensions_t(array2D.getDimensions().x, array2D.getDimensions().y, 0));
				return pArray;
			}
			/************************************************************************/
			/* copy                                                                 */
			/************************************************************************/
			/**Copy the contents of the other array to this one. Has ownership, since the object is a full copy */
			Array2D(const Array2D<ArrayClassT> &rArray) :  Array<ArrayClassT>(rArray.getDimensions()) {
				int arrayDimension = m_dimensions.x*m_dimensions.y;
				m_rawArray.resize(arrayDimension);
				m_arrayOwnership = true;

				for(int i = 0; i < m_dimensions.x; i++) {
					for(int j = 0; j < m_dimensions.y; j++) {
						(*this)(i, j) = rArray(i, j);
					}
				}
			}

			/**Copy the contents of the other array to this one. Has ownership, since the object is a full copy */
			Array2D & operator= (const Array2D<ArrayClassT> &rArray) {
				m_dimensions = rArray.getDimensions();
				int arrayDimension = m_dimensions.x*m_dimensions.y;
				m_rawArray.resize(arrayDimension);
				m_arrayOwnership = true;
				
				for(int i = 0; i < m_dimensions.x; i++) {
					for(int j = 0; j < m_dimensions.y; j++) {
						(*this)(i, j) = rArray(i, j);
					}
				}

				return *this;
			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/

			const ArrayClassT & operator()(int i, int j) const {
#ifdef _DEBUG
				if(i >= m_dimensions.x || i < 0 || j >= m_dimensions.y || j < 0)
					throw("Array2D out of bounds exception");
#endif
				return m_rawArray[j*m_dimensions.x + i];
			}

			ArrayClassT & operator()(int i, int j) {
#ifdef _DEBUG
				if (i >= m_dimensions.x || i < 0 || j >= m_dimensions.y || j < 0)
					throw("Array2D out of bounds exception");
#endif
				return Value(i, j);
			}

			const ArrayClassT & operator()(const dimensions_t &dimensions) const {
#ifdef _DEBUG
				if (dimensions.x >= m_dimensions.x || dimensions.x < 0 || dimensions.y >= m_dimensions.y || dimensions.y < 0)
					throw("Array2D out of bounds exception");
#endif
				return m_rawArray[dimensions.y*m_dimensions.x + dimensions.x];
			}

			ArrayClassT & operator()(const dimensions_t &dimensions) {
#ifdef _DEBUG
				if (dimensions.x >= m_dimensions.x || dimensions.x < 0 || dimensions.y >= m_dimensions.y || dimensions.y < 0)
					throw("Array2D out of bounds exception");
#endif
				return m_rawArray[dimensions.y*m_dimensions.x + dimensions.x];
			}



			FORCE_INLINE int getRawPtrIndex(int i, int j) const {
				return j*m_dimensions.x + i;
			}

			/************************************************************************/
			/* 2D access functions                                                  */
			/************************************************************************/
			FORCE_INLINE const ArrayClassT & getValue(int i, int j) const {
				return m_rawArray[j*m_dimensions.x + i];
			}

			FORCE_INLINE ArrayClassT * getValuePtr(int i, int j) {
				return &m_rawArray[j*m_dimensions.x + i];
			}

			FORCE_INLINE void setValue(ArrayClassT gValue, int i, int j) {
				m_rawArray[j*m_dimensions.x + i] = gValue;
			}

		};
	}
}

#endif