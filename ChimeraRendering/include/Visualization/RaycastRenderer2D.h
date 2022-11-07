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


#ifndef _CHIMERA_RAYCAST_SOLVER_RENDERER__
#define _CHIMERA_RAYCAST_SOLVER_RENDERER__
#pragma once

#include "ChimeraCutCells.h"

/************************************************************************/
/* Rendering                                                            */
/************************************************************************/
#include "ChimeraRenderingCore.h"
#include "Windows/GridVisualizationWindow.h"
#include "Visualization/ScalarFieldRenderer.h"
#include "RenderingUtils.h"


namespace Chimera {
	using namespace Core;
	using namespace CutCells;


	namespace Rendering {

		class RaycastRenderer2D {

		public:
			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			RaycastRenderer2D(QuadGrid *pQuadGrid);

			/************************************************************************/
			/* External settings                                                    */
			/************************************************************************/
			void initializeWindowControls(GridVisualizationWindow<Vector2> *pGridVisualizationWindow);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			void setLeftFaceCrossingsPtr(Array2D<Crossing<Vector2>> *pVectorCrossings) {
				m_pLeftFaceCrossings = pVectorCrossings;
			}
			void setBottomFaceCrossingsPtr(Array2D<Crossing<Vector2>> *pVectorCrossings) {
				m_pBottomFaceCrossings = pVectorCrossings;
			}

			void setLeftFacesVisibilityPtr(Array2D<bool> *pVisibility) {
				m_pLeftFaceVisibility = pVisibility;
			}
			void setBottomFacesVisibilityPtr(Array2D<bool> *pVisibility) {
				m_pBottomFaceVisibility = pVisibility;
			}

			void setBoundaryCellsPtr(Array2D<bool> *pBoundaryCellsPtr) {
				m_pBoundaryCells = pBoundaryCellsPtr;
			}

			/************************************************************************/
			/* Functionalities		                                                */
			/************************************************************************/
			void draw();

			/** If the CutCells2D has changed, call this function */
			void update();


		private:
			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			Array2D<Crossing<Vector2>> *m_pLeftFaceCrossings;
			Array2D<Crossing<Vector2>> *m_pBottomFaceCrossings;
			Array2D<bool> *m_pLeftFaceVisibility;
			Array2D<bool> *m_pBottomFaceVisibility;
			Array2D<bool> *m_pBoundaryCells;
			QuadGrid *m_pQuadGrid;

			/** Ant-tweak vars */
			bool m_drawCells;
			bool m_drawFaceVelocities;
			bool m_drawFaceNormals;
			bool m_drawRaycastBoundaries;
			Scalar m_velocityLength;

			//Grid visualization window
			GridVisualizationWindow<Vector2> *m_pGridVisualizationWindow;

			/************************************************************************/
			/* Private drawing functions                                            */
			/************************************************************************/
			void drawCells();
			void drawFaceVelocities();
			void drawFaceNormals();

			void drawOccludedFaces();
			
			FORCE_INLINE void drawCell(int i, int j) const {
				Scalar dx = m_pQuadGrid->getGridData2D()->getScaleFactor(0, 0).x;
				glBegin(GL_QUADS);
				glVertex2f(i*dx, j*dx);
				glVertex2f((i + 1)*dx, j*dx);
				glVertex2f((i + 1)*dx, (j + 1)*dx);
				glVertex2f(i*dx, (j + 1)*dx);
				glEnd();
			}
		};



	}
}
#endif