
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

#ifndef _RENDERING_SPECIAL_CELL_VISUALIZATION_WINDOW_H
#define _RENDERING_SPECIAL_CELL_VISUALIZATION_WINDOW_H

#pragma  once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraPoisson.h"
#include "ChimeraCutCells.h"

#include "Windows/BaseWindow.h"
#include "Visualization/MeshRenderer.h"
#include "Visualization/PolygonMeshRenderer.h"
#include "Visualization/CutCellsVelocityRenderer2D.h"

namespace Chimera {
	using namespace CutCells;
	using namespace Grids;
	using namespace Poisson;

	namespace Windows {

		class CutCellsWindow : public BaseWindow {

		private:

			#pragma region ClassMembers
			/** Ptr copies */
			QuadGrid *m_pGrid;
			CutCells2D<Vector2> *m_pCutCells2D;
			PoissonMatrix *m_pPoissonMatrix;
			Rendering::PolygonMeshRenderer<Vector2> *m_pCutCellsRenderer;
			Rendering::CutCellsVelocityRenderer2D<Vector2> *m_pCutCellsVelRenderer;

			/**Special cell attributes: */
			dimensions_t m_regularGridIndex;
			Scalar m_cellPressure;
			Scalar m_cellDivergent;
			bool m_drawCellNeighbors;
			
			/**Internal TW management */
			map<string, bool> m_faceAttributesVars;
			#pragma endregion

			/************************************************************************/
			/* Initialization	                                                    */
			/************************************************************************/
			void initializeAntTweakBar();

			/************************************************************************/
			/* Internal updates                                                     */
			/************************************************************************/
			void updateAttributes();
			void updateFaceAttributes();
			void flushFaceAttributes();
		public:

			CutCellsWindow(Rendering::PolygonMeshRenderer<Vector2> *pPolyMeshRenderer, Rendering::CutCellsVelocityRenderer2D<Vector2> *pVelRenderer);
			
			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			CutCells2D<Vector2> * getCutCells2D() {
				return m_pCutCells2D;
			}

			Scalar getPressure() const {
				return m_cellPressure;
			}
			Scalar getDivergent() const {
				return m_cellDivergent;
			}
			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			virtual void update() {
				
			}
			
			FORCE_INLINE void setSelectedCell(int selectedCell) {
				int prevCutCell = m_pCutCellsRenderer->getSelectedCell();
				m_pCutCellsRenderer->setSelectedCutCell(selectedCell);
				if(selectedCell != -1 && prevCutCell != selectedCell) {
					updateAttributes();
					updateFaceAttributes();
				}
			}
		};
	}
}

#endif