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

#ifndef __CHIMERA_GRID_DATA__
#define __CHIMERA_GRID_DATA__
#pragma once

#include "ChimeraCore.h"


namespace Chimera {
	using namespace Core;

	namespace Grids { 

		/** Base GridData class - serves as common base for GridData2D and GridData3D. They inherit this class to be casted
		 ** on other 2D and 3D grid implementations on FlowSolvers and Simulation configs.
		** IMPORTANT: GridData class is only a data storage class. It is NOT RESPONSIBLE for initialization of variables,
		** grid loading, metrics or density points/values.*/
		template <class VectorT>
		class GridData {

		public:

			/************************************************************************/
			/* Constructors                                                         */
			/************************************************************************/
			/** Default ctor*/
			FORCE_INLINE GridData(dimensions_t dimensions) : m_dimensions(dimensions) {
				m_dx = 0;
			}

			/************************************************************************/
			/* Utils                                                                */
			/************************************************************************/
			inline const VectorT & getMinBoundary() const {
				return m_minGridBoundary;
			}

			inline const VectorT & getMaxBoundary() const {
				return m_maxGridBoundary;
			}

			inline dimensions_t getDimensions() const {
				return m_dimensions;
			}

			inline Scalar getGridSpacing() const {
				return m_dx;
			}

			inline void setGridSpacing(Scalar gridSpacing) {
				m_dx = gridSpacing;
			}

		protected:
			/************************************************************************/
			/* Initialization functions                                             */
			/************************************************************************/
			virtual void initData() = 0;

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			VectorT			m_minGridBoundary;
			VectorT			m_maxGridBoundary;

			/** Dimensions array */
			dimensions_t m_dimensions;

			Scalar m_dx;

		};
	}
}

#endif