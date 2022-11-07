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
//	
#include "LevelSets/LevelSet2D.h" 

namespace Chimera {
	namespace LevelSets {

		const int LevelSet2D::maxIsocontourPoints = 10000;

		/************************************************************************/
		/* Intialization                                                        */
		/************************************************************************/
		void LevelSet2D::initializeGrid() {
			if(m_params.initialBoundary.x == m_params.finalBoundary.x &&
				m_params.initialBoundary.y == m_params.finalBoundary.y ) { //No initial and final boundaries set
					calculateBoundaries(m_params.pPolygonPoints, m_params.initialBoundary, m_params.finalBoundary);
			}

			m_pGrid = new QuadGrid(m_params.initialBoundary, m_params.finalBoundary, m_params.dx);
		}

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		/** Updates all level set values accordingly with polygon points. If polygon is convex
			 ** (params.convexPolygon), then the negative distance is stored for points inside
			 ** the polygon.*/
		void LevelSet2D::updateDistanceField() {
			//Upgrades level set accordingly with polygon points 
			for(int i = 0; i < m_pGrid->getDimensions().x; i++) {
				for(int j = 0; j < m_pGrid->getDimensions().y; j++) {

					Vector2 cellCenter = m_pGrid->getGridData2D()->getCenterPoint(i, j);
					Scalar minDistance = FLT_MAX;

					for(unsigned int k = 0; k < m_params.pPolygonPoints->size() - 1; k++) {
						Scalar tempDistance = (cellCenter - m_params.pPolygonPoints->at(k)).length();
						if(tempDistance < minDistance) {
							minDistance = tempDistance;
						}
					}
					if(m_params.convexPolygon) {
						if(isInsidePolygon(cellCenter, *m_params.pPolygonPoints)) {
							minDistance = -minDistance;
						}
					}

					m_pGrid->getGridData2D()->setLevelSetValue(minDistance, i, j);
					m_pGrid->getGridData2D()->getDensityBuffer().setValueBothBuffers(minDistance, i, j);
				}
			}
		}
		/** Updates the distance field in a narrow band around isocontour points */
		void LevelSet2D::updateDistanceField(int bandSize) {
			Scalar dx = m_pGrid->getGridData2D()->getScaleFactor(0, 0).x;
			GridData2D *pGridData = m_pGrid->getGridData2D();

			for(unsigned int k = 0; k < m_params.pPolygonPoints->size(); k++) {
				Vector2 relativePosition = (m_params.pPolygonPoints->at(k) - m_pGrid->getGridOrigin())/dx;

				//First pass only resets all the neighboring cells
				for(int i = floor(relativePosition.x) - bandSize; i <= floor(relativePosition.x) + bandSize; i++) {
					for(int j = floor(relativePosition.y) - bandSize; j <= floor(relativePosition.y) + bandSize; j++) {
						pGridData->setLevelSetValue(FLT_MAX, i, j);
					}
				}
			} 

			for(unsigned int k = 0; k < m_params.pPolygonPoints->size() - 1; k++) {
				Vector2 relativePosition = (m_params.pPolygonPoints->at(k) - m_pGrid->getGridOrigin())/dx;

				//Second pass stores all minimal values
				for(int i = floor(relativePosition.x) - bandSize; i <= floor(relativePosition.x) + bandSize; i++) {
					for(int j = floor(relativePosition.y) - bandSize; j <= floor(relativePosition.y) + bandSize; j++) {
						Vector2 cellCenter = m_pGrid->getGridData2D()->getCenterPoint(i, j);
						Scalar minDistance = (cellCenter - m_params.pPolygonPoints->at(k)).length();
						if(abs(minDistance) < abs(m_pGrid->getGridData2D()->getLevelSetValue(i, j))) {
							if(m_params.convexPolygon && isInsidePolygon(cellCenter, *m_params.pPolygonPoints)) {
								minDistance = -minDistance;
							}
							pGridData->setLevelSetValue(minDistance, i, j);
							pGridData->getDensityBuffer().setValueBothBuffers(minDistance, i, j);
							//Marks the cells as alive - do not recalculate these!
							//This decreases the methods' high-frequency errors near boundaries
							m_pGrid->setSolidCell(true, i, j);
						}
					}
				}
			} 
		}	
	}
}