#pragma once
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

#ifndef _MATH_INTERPOLANT_2D_H_
#define _MATH_INTERPOLANT_2D_H_

#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraCutCells.h"
#include "ChimeraMesh.h"

namespace Chimera {

	
	namespace Interpolation {

		/** Interpolant base classWorks for scalarFields or vectorFields, just pass it on the template. Pure-virtual 
		 ** class, concrete 2D and 3D, linear and non-linear classes will derive from this. First template argument is
		 ** the type (e.g., scalar or vectors) of the interpolated/original values, the second template is the array 
		 ** type (Array2D<valueType> for 2-D, Array3D<valueType> for 3-D) where the original values will be stored and 
		 ** the last argument is the vector type (Vector2, Vector3, etc.) of the position that will be interpolated.*/
		template <class valueType, template <class> class ArrayType, class VectorT>
		class Interpolant  {

		public:

			#pragma region Constructors
			/** Empty constructor. Useful for fancy interpolators that will not receive an Array2D a-priori. */
			Interpolant() : m_values(dimensions_t(0, 0)) {
				
			}
			/** Cut-cell constructor */
			Interpolant(const ArrayType<valueType> &values) : m_values(values) {
				m_gridDimensions = m_values.getDimensions();
			}
			#pragma endregion
			

			#pragma region Functionalities
			/* Basic interpolation function */
			virtual valueType interpolate(const VectorT &position) = 0;
			#pragma endregion

			#pragma region Access Functions
			/** Get original ArrayType values */
			FORCE_INLINE const ArrayType<valueType> & getValues() const {
				return m_values;
			}

			FORCE_INLINE const dimensions_t & getGridDimensions() const {
				return m_gridDimensions;
			}

			/** Sibling interpolants are useful for algorithms that need interpolation for values before and after 
				advection (e.g., FLIP). Removes the need of the cloning functions. */
			FORCE_INLINE void setSiblingInterpolant(Interpolant<valueType, ArrayType, VectorT> *pInterpolant) {
				m_pSiblingInterpolant = pInterpolant;
			}

			FORCE_INLINE Interpolant<valueType, ArrayType, VectorT> * getSibilingInterpolant() {
				return m_pSiblingInterpolant;
			}
			#pragma endregion
		protected:
			#pragma region ClassMembers
			const ArrayType<valueType> &m_values;

			/** Sibling interpolant */
			Interpolant<valueType, ArrayType, VectorT> *m_pSiblingInterpolant;

			/** Store this explicitly to facilitate access*/
			dimensions_t m_gridDimensions;
			#pragma endregion
		};
	}
	

}
#endif