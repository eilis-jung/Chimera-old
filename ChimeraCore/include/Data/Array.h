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

#ifndef __CHIMERA_ARRAY_H__
#define __CHIMERA_ARRAY_H__

#include "Config/ChimeraConfig.h"
#include "Data/ChimeraStructures.h"

#pragma once

namespace Chimera {
	using namespace std;
	namespace Core {

		/* 1D range structure */
		typedef struct range1D_t {
			int initialRange;
			int finalRange;
			range1D_t(int iRange, int fRange) : initialRange(iRange), finalRange(fRange) {}
			range1D_t() {
				initialRange = finalRange = -1;
			}
		} range1D_t;

		template <class ChildT, class ArrayClassT>
		class ArrayBase {


		protected:
			vector<ArrayClassT> m_rawArray;
			dimensions_t m_dimensions;
			/** Array ownership: set to true if the array allocated the data and is responsible for deallocation
			/** This is usually true - the only case that this does not hold is when a an raw array pointer is passed as 
			/** constructor. In this case, the class is not responsible for data allocation, since its just a facilitator
			/** in the process of accessing the raw data. */
			bool m_arrayOwnership;

		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			ArrayBase(const dimensions_t &dimensions) : m_dimensions(dimensions) {
			
			}

			//Array(ArrayClassT *gpRawArray, const dimensions_t &dimensions) : m_dimensions(dimensions), pRawArray(gpRawArray) { }
			virtual ~ArrayBase() { }

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			FORCE_INLINE const dimensions_t & getDimensions() const {
				return m_dimensions;
			} 

			FORCE_INLINE void setDimensions(const dimensions_t &dimensions) {
				m_dimensions = dimensions;
			}

			FORCE_INLINE vector<ArrayClassT> & getRawData() {
				return m_rawArray;
			}

			FORCE_INLINE void * getRawDataPointer() const {
				return (void *)&m_rawArray[0];
			}

			FORCE_INLINE unsigned int size() const {
				if(m_dimensions.y == 0) {
					return m_dimensions.x;
				}
				if(m_dimensions.z == 0) {
					return m_dimensions.x*m_dimensions.y;
				}
				return m_dimensions.x*m_dimensions.y*m_dimensions.z;
			}

			/**Assigns a value to all array members */
			FORCE_INLINE void assign(const ArrayClassT &constantValue) {
				for(int i = 0; i < size(); i++) {
					m_rawArray[i] = constantValue;
				}
			}
			
		};


		template<class ArrayClassT>
		class ArrayT : public ArrayBase<ArrayT<ArrayClassT>, ArrayClassT> {
			public:
			
			ArrayT(const dimensions_t &dimensions, bool initialize = false) : ArrayBase(dimensions) {
				if (dimensions.z == 0)
					m_rawArray.resize(m_dimensions.x*m_dimensions.y);
				else
					m_rawArray.resize(m_dimensions.x*m_dimensions.y*m_dimensions.z);
			}

			void resize(const dimensions_t &dimensions) {
				m_dimensions = dimensions;

				if (dimensions.z == 0)
					m_rawArray.resize(m_dimensions.x*m_dimensions.y);
				else
					m_rawArray.resize(m_dimensions.x*m_dimensions.y*m_dimensions.z);

			}
		};

		template<class ArrayClassT>
		class ArrayT<ArrayClassT *> : public ArrayBase<ArrayT<ArrayClassT *>, ArrayClassT *> {
		public:

			ArrayT(const dimensions_t &dimensions, bool initialize = false) : ArrayBase(dimensions) {
				resize(dimensions, initialize);
			}

			void resize(const dimensions_t &dimensions, bool initialize = false) {
				m_dimensions = dimensions;

				if (dimensions.z == 0)
					m_rawArray.resize(m_dimensions.x*m_dimensions.y);
				else
					m_rawArray.resize(m_dimensions.x*m_dimensions.y*m_dimensions.z);

				if (initialize) {
					for (int i = 0; i < m_rawArray.size(); i++) {
						m_rawArray[i] = new ArrayClassT();
					}
				}
				else {
					for (int i = 0; i < m_rawArray.size(); i++) {
						m_rawArray[i] = nullptr;
					}
				}

			}

		};

		template <typename ArrayClassT>
		using Array = ArrayT<ArrayClassT>;
 	}
}

#endif