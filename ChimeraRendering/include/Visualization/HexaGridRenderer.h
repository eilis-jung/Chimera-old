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


#ifndef _CHIMERA_FV_HEXA_GRID_RENDERER_H_
#define _CHIMERA_FV_HEXA_GRID_RENDERER_H_
#pragma once

/************************************************************************/
/* Data                                                                 */
/************************************************************************/
#include "Grids/HexaGrid.h"

/************************************************************************/
/* Rendering                                                            */
/************************************************************************/
#include "Visualization/ScalarFieldRenderer.h"
#include "Visualization/GridRenderer.h"

namespace Chimera {

	namespace Rendering {

		class HexaGridRenderer : public GridRenderer<Vector3> {
		public:
			/************************************************************************/
			/* Ctors                                                                */
			/************************************************************************/
			HexaGridRenderer(HexaGrid *pHexaGrid);

			/************************************************************************/
			/* Drawing                                                              */
			/************************************************************************/
			void drawXYSlice(int kthSlice) const;
			void drawXZSlice(int kthSlice) const;
			void drawYZSlice(int kthSlice) const;
		private:
			/************************************************************************/
			/* Initialization                                                       */
			/************************************************************************/
			unsigned int initializeGridPointsVBO();
			unsigned int initializeGridCellsIndexVBO();

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			FORCE_INLINE int getGridPointIndex(int i, int j, int k) const {
				return k*(m_gridDimensions.x + 1)*(m_gridDimensions.y + 1) + j*(m_gridDimensions.x + 1) + i;
			}

			/************************************************************************/
			/* Drawing                                                              */
			/************************************************************************/
			void drawGridVertices() const;
			void drawGridCentroids() const;
			void drawGridCells() const;
			void drawGridSolidCells() const;
			void drawGridBoundaries() const;

		};
	}
}

#endif
