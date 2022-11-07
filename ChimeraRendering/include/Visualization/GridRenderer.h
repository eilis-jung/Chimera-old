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


#ifndef _CHIMERA_GRID_RENDERER_H_
#define _CHIMERA_GRID_RENDERER_H_
#pragma once

/************************************************************************/
/* Data                                                                 */
/************************************************************************/
#include "Grids/QuadGrid.h"
#include "Grids/HexaGrid.h"

/************************************************************************/
/* Rendering                                                            */
/************************************************************************/
#include "Visualization/ScalarFieldRenderer.h"
#include "Visualization/VectorFieldRenderer.h"

namespace Chimera {

	namespace Rendering {

		typedef enum gridRenderingMode_t {
			drawVertices,
			drawCells,
			drawSolidCells,
			drawBoundaries,
			drawCentroids
		};

		template <class VectorT>
		class GridRenderer {
		public:
			/************************************************************************/
			/* Ctors                                                                */
			/************************************************************************/
			GridRenderer(StructuredGrid<VectorT> *pGrid);

			/************************************************************************/
			/* Drawing		                                                        */
			/************************************************************************/
			void draw(gridRenderingMode_t gridRenderingMode);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			const ScalarFieldRenderer<VectorT> & getScalarFieldRenderer() const {
				return m_scalarFieldRenderer;
			}
			ScalarFieldRenderer<VectorT> & getScalarFieldRenderer() {
				return m_scalarFieldRenderer;
			}
			const VectorFieldRenderer<VectorT> & getVectorFieldRenderer() const {
				return m_vectorFieldRenderer;
			}
			VectorFieldRenderer<VectorT> & getVectorFieldRenderer() {
				return m_vectorFieldRenderer;
			}

			void setSelectedCell(const dimensions_t &selectedCell) {
				m_selectedCell = selectedCell;
			}

			StructuredGrid<VectorT> * getGrid() {
				return m_pGrid;
			}

			/************************************************************************/
			/* Drawing                                                              */
			/************************************************************************/
			FORCE_INLINE void drawCell(int i, int j) const {
				glBegin(GL_QUADS);
				glVertex2f(m_pGrid->getGridData2D()->getPoint(i, j).x, m_pGrid->getGridData2D()->getPoint(i, j).y);
				glVertex2f(m_pGrid->getGridData2D()->getPoint(i + 1, j).x, m_pGrid->getGridData2D()->getPoint(i + 1, j).y);
				glVertex2f(m_pGrid->getGridData2D()->getPoint(i + 1, j + 1).x, m_pGrid->getGridData2D()->getPoint(i + 1, j + 1).y);
				glVertex2f(m_pGrid->getGridData2D()->getPoint(i, j + 1).x, m_pGrid->getGridData2D()->getPoint(i, j + 1).y);
				glEnd();
			}
			void drawCell(int i, int j, int k) const;

			/** Draw cells based on points description */
			void drawCell(const vector<VectorT> &points) const;

		protected:
			/************************************************************************/
			/* Initialization                                                       */
			/************************************************************************/
			virtual unsigned int initializeGridPointsVBO() = 0;
			virtual unsigned int initializeGridCellsIndexVBO() = 0;

			
			/************************************************************************/
			/* Drawing                                                              */
			/************************************************************************/
			virtual void drawGridVertices() const = 0;
			virtual void drawGridCentroids() const = 0;
			virtual void drawGridCells() const = 0;
			virtual void drawGridSolidCells() const = 0;
			virtual void drawGridBoundaries() const = 0;
			virtual void drawSelectedCell() const { }

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			StructuredGrid<VectorT> *m_pGrid;
			ScalarFieldRenderer<VectorT> m_scalarFieldRenderer;
			VectorFieldRenderer<VectorT> m_vectorFieldRenderer;
			dimensions_t m_selectedCell;
			dimensions_t m_gridDimensions;

			/************************************************************************/
			/* VBOs                                                                 */
			/************************************************************************/
			GLuint *m_pGridPointsVBO;
			GLuint *m_pGridCellsIndexVBO;

			/************************************************************************/
			/* Colors and general configs                                           */
			/************************************************************************/
			Color m_gridLinesColor;
			Color m_gridPointsColor;
			Color m_gridSolidCellColor;
			Scalar m_pointsSize;
			Scalar m_lineWidth;
		};
	}
}

#endif
