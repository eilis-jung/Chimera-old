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

#pragma once
#include "ChimeraCore.h"

namespace Chimera {

	using namespace Core;
	namespace Grids {

		/** Grid File formats.
		** Brief specification:
		** VRL 97 (2.0): versatile format for 2D structured grid loading. Exports grid points and cell faces. Default file extension: .wrl 
		** UCD: simple format for 3D structured grid loading. Exports grid points and cell faces. Default file extension: .ucd
		**		It is assumed that the vertex numeration is done by the following scheme 
		**		4 5 6 7 0 1 2 3, according to the default hexa grid numeration scheme (see Hexagrid.h); 
		** */   


		/** Grid foundation class. Define the base members needed for grid specification. Used for both unstructured and structured grids. */
		template<class GridPrimitiveT>
		class Grid {

		public:
			typedef struct gridBounds_t {
				GridPrimitiveT lowerBounds;
				GridPrimitiveT upperBounds;
			} gridBounds_t;


			/************************************************************************/
			/* ctors and dtors                                                      */
			/************************************************************************/
			/* Default ctor*/
			Grid() { };

			~Grid() { };

			/************************************************************************/
			/* Access Functions                                                     */
			/************************************************************************/
			inline const GridPrimitiveT & getGridCentroid() const {
				return m_gridCentroid;
			}
			inline void setGridCentroid(const GridPrimitiveT &gridCentroid) {
				m_gridCentroid = gridCentroid;
			}

			inline const gridBounds_t & getBoundingBox() const {
				return m_gridBoundingBox;
			}

		protected:
			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			//Grid cells raw data
			int m_totalGridCells;

			//Grid sizes
			GridPrimitiveT m_gridCentroid;
			gridBounds_t m_gridBoundingBox;
		};
	}
}