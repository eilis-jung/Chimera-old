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

#ifndef __CHIMERA_ARRAY_3D_H__
#define __CHIMERA_ARRAY_3D_H__

#include "Data/ChimeraStructures.h"
#include "Data/Array.h"
#include "Data/Array2D.h"
#include "Data/ReferenceWrapper.h"

#pragma once

namespace Chimera {

	namespace Core {

		template <class ArrayClassT>
		class Array3D : public Array<ArrayClassT> {

		protected:
			/**
			* the value of the given location (non-const).
			*/
			ArrayClassT& Value(int i, int j, int k)
			{
				#ifdef _DEBUG
				if(i >= m_dimensions.x || i < 0 || j >= m_dimensions.y || j < 0 
					|| k >= m_dimensions.z || k < 0)
					throw("Array3D out of bounds exception");
				#endif
				return  m_rawArray[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i];
			}

			const ArrayClassT& Value(int i, int j, int k) const 
			{
				#ifdef _DEBUG
				if (i >= m_dimensions.x || i < 0 || j >= m_dimensions.y || j < 0
					|| k >= m_dimensions.z || k < 0)
					throw("Array3D out of bounds exception");
				#endif
				return  m_rawArray[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i];
			}
		public:

			typedef enum sliceType_t {
				XYSlice,
				XZSlice,
				YZSlice
			} sliceType_t;

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			Array3D(const dimensions_t &dimensions, bool initializePointers = false) : Array<ArrayClassT>(dimensions, initializePointers) {
				//m_rawArray.resize(dimensions.x*dimensions.y*dimensions.z);
				m_arrayOwnership = true;
			}
			/*Array3D(ArrayClassT *gpRawArray, const dimensions_t &dimensions) : Array<ArrayClassT>(gpRawArray, dimensions) {
				m_arrayOwnership = false;
			} */
			~Array3D() {
				/*if(m_arrayOwnership)
					delete pRawArray;*/
			}

			/************************************************************************/
			/* copy                                                                 */
			/************************************************************************/
			/**Copy the contents of the other array to this one. Has ownership, since the object is a full copy */
			Array3D(const Array3D<ArrayClassT> &rArray) :  Array<ArrayClassT>(rArray.getDimensions()) {
				m_dimensions = rArray.getDimensions();
				m_rawArray.resize(m_dimensions.x*m_dimensions.y*m_dimensions.z);
				m_arrayOwnership = true;

				for(int i = 0; i < m_dimensions.x; i++) {
					for(int j = 0; j < m_dimensions.y; j++) {
						for(int k = 0; k < m_dimensions.z; k++) {
							(*this)(i, j, k) = rArray(i, j, k);
						}
					}
				}
			}

			/**Copy the contents of the other array to this one. Has ownership, since the object is a full copy */
			Array3D & operator= (const Array3D<ArrayClassT> &rArray) {
				m_dimensions = rArray.getDimensions();
				m_rawArray.resize(m_dimensions.x*m_dimensions.y*m_dimensions.z);
				m_arrayOwnership = true;
				
				for(int i = 0; i < m_dimensions.x; i++) {
					for(int j = 0; j < m_dimensions.y; j++) {
						for(int k = 0; k < m_dimensions.z; k++) {
							(*this)(i, j, k) = rArray(i, j, k);
						}
					}
				}

				return *this;
			}


			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			const ArrayClassT & operator()(int i, int j, int k) const {
				#ifdef _DEBUG
				if(i >= m_dimensions.x || i < 0 || j >= m_dimensions.y || j < 0 
					|| k >= m_dimensions.z || k < 0)
					throw("Array3D out of bounds exception");
				#endif
				return m_rawArray[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i];
			}

			ArrayClassT & operator()(int i, int j, int k) {
				return Value(i, j, k);
			}

			const ArrayClassT & operator()(const dimensions_t &dim) const {
				return Value(dim.x, dim.y, dim.z);
			}

			ArrayClassT & operator()(const dimensions_t &dim) {
				return Value(dim.x, dim.y, dim.z);
			}

			static Array2D<ReferenceWrapper<ArrayClassT>> * createArraySlice(Array3D<ArrayClassT> &array3D, uint kSlice, sliceType_t sliceType) {
				Array2D<ReferenceWrapper<ArrayClassT>> *pArraySlice = new Array2D<ReferenceWrapper<ArrayClassT>>();
				if (sliceType == XYSlice) {
					//Trick: initialize a zero size array2D, and manually add entries on initialization
					//pArraySlice = new Array2D<reference_wrapper<ArrayClassT>>(dimensions_t(array3D.getDimensions().x, array3D.getDimensions().y));
					//pArraySlice = new Array2D<reference_wrapper<ArrayClassT>>(dimensions_t(0, 0));
					pArraySlice->getRawData().reserve(array3D.getDimensions().x*array3D.getDimensions().y);
					//Carefull! The order of pushing back elements into rawData matters, start from smallest index (i)
					for (int j = 0; j < array3D.getDimensions().y; j++) {
						for (int i = 0; i < array3D.getDimensions().x; i++) {
							ArrayClassT *pRef = array3D.getValuePtr(i, j, kSlice);
							pArraySlice->getRawData().push_back(ReferenceWrapper<ArrayClassT>(*pRef));
							//(*pArraySlice)(i, j) = reference_wrapper<ArrayClassT>(*pRef); //ArrayClassT()/*array3D(i, j, kSlice)*/);
						}
					}
					pArraySlice->setDimensions(dimensions_t(array3D.getDimensions().x, array3D.getDimensions().y, 0));
				} else if(sliceType == XZSlice) {
					//pArraySlice = new Array2D<reference_wrapper<ArrayClassT>>(dimensions_t(array3D.getDimensions().x, array3D.getDimensions().z));
					//pArraySlice = new Array2D<reference_wrapper<ArrayClassT>>(dimensions_t(0, 0));
					pArraySlice->getRawData().reserve(array3D.getDimensions().x*array3D.getDimensions().z);
					for (int j = 0; j < array3D.getDimensions().z; j++) {
						for (int i = 0; i < array3D.getDimensions().x; i++) {
							ArrayClassT *pRef = array3D.getValuePtr(i, kSlice, j);
							pArraySlice->getRawData().push_back(ReferenceWrapper<ArrayClassT>(*pRef));
							//(*pArraySlice)(i, j) = array3D(i, kSlice, j);
						}
					}
					pArraySlice->setDimensions(dimensions_t(array3D.getDimensions().x, array3D.getDimensions().z, 0));
				} else if (sliceType == YZSlice) {
					//pArraySlice = new Array2D<reference_wrapper<ArrayClassT>>(dimensions_t(array3D.getDimensions().z, array3D.getDimensions().y));
					//pArraySlice = new Array2D<reference_wrapper<ArrayClassT>>(dimensions_t(0, 0));
					pArraySlice->getRawData().reserve(array3D.getDimensions().y*array3D.getDimensions().z);
					for (int j = 0; j < array3D.getDimensions().y; j++) {
						for (int i = 0; i < array3D.getDimensions().z; i++) {
							ArrayClassT *pRef = array3D.getValuePtr(kSlice, i, j);
							pArraySlice->getRawData().push_back(ReferenceWrapper<ArrayClassT>(*pRef));
							//(*pArraySlice)(i, j) = array3D(kSlice, i, j);
						}
					}
					pArraySlice->setDimensions(dimensions_t(array3D.getDimensions().z, array3D.getDimensions().y, 0));
				}

				return pArraySlice;

				//return nullptr;
			}

			/** Only works if ArrayClassT is a pointer*/
			static Array2D<ArrayClassT> * createArraySlicePtr(Array3D<ArrayClassT> &array3D, uint kSlice, sliceType_t sliceType) {
				Array2D<ArrayClassT> *pArraySlice = nullptr;
				if (!is_pointer<ArrayClassT>::value) {
					throw ("createArraySlice: cannot use this function on an Array that does uses pointers as primitives");
				}
				if (sliceType == XYSlice) {
					pArraySlice = new Array2D<ArrayClassT>(dimensions_t(array3D.getDimensions().x, array3D.getDimensions().y), false);
					for (int j = 0; j < array3D.getDimensions().y; j++) {
						for (int i = 0; i < array3D.getDimensions().x; i++) {
							(*pArraySlice)(i, j) = array3D.getValue(i, j, kSlice);
						}
					}
				}
				else if (sliceType == XZSlice) {
					pArraySlice = new Array2D<ArrayClassT>(dimensions_t(array3D.getDimensions().x, array3D.getDimensions().z), false);
					for (int j = 0; j < array3D.getDimensions().z; j++) {
						for (int i = 0; i < array3D.getDimensions().x; i++) {
							(*pArraySlice)(i, j) = array3D(i, kSlice, j);
						}
					}
				}
				else if (sliceType == YZSlice) {
					pArraySlice = new Array2D<ArrayClassT>(dimensions_t(array3D.getDimensions().z, array3D.getDimensions().y), false);
					for (int j = 0; j < array3D.getDimensions().y; j++) {
						for (int i = 0; i < array3D.getDimensions().z; i++) {
							(*pArraySlice)(i, j) = array3D(kSlice, j, i);
						}
					}
				}

				return pArraySlice;
			}
			/************************************************************************/
			/* 3D access functions                                                  */
			/************************************************************************/
			FORCE_INLINE const ArrayClassT & getValue(int i, int j, int k) const {
#ifdef _DEBUG
				if(i >= m_dimensions.x || i < 0 || j >= m_dimensions.y || j < 0 
					|| k >= m_dimensions.z || k < 0)
					throw("Array3D out of bounds exception");
#endif
				return m_rawArray[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i];
			}

			FORCE_INLINE ArrayClassT * getValuePtr(int i, int j, int k) {
				return &m_rawArray[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i];
			}

			FORCE_INLINE void setValue(ArrayClassT gValue, int i, int j, int k) {
				#ifdef _DEBUG
				if(i >= m_dimensions.x || i < 0 || j >= m_dimensions.y || j < 0 
					|| k >= m_dimensions.z || k < 0)
					throw("Array3D out of bounds exception");
				#endif
				m_rawArray[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i] = gValue;
			}

			FORCE_INLINE int getLinearIndex(int i, int j, int k) const {
				#ifdef _DEBUG
				if(i >= m_dimensions.x || i < 0 || j >= m_dimensions.y || j < 0 
					|| k >= m_dimensions.z || k < 0)
					throw("Array3D out of bounds exception");
				#endif
				return k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i;
			}

		};
	}
}

#endif