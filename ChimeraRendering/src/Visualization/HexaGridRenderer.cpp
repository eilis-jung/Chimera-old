#include "Visualization/HexaGridRenderer.h"

namespace Chimera {
	namespace Rendering {

		/************************************************************************/
		/* Ctors                                                                */
		/************************************************************************/
		HexaGridRenderer::HexaGridRenderer(HexaGrid *pHexaGrid) : GridRenderer(pHexaGrid) {
			initializeGridPointsVBO();
			initializeGridCellsIndexVBO();
		}

		/************************************************************************/
		/* Initialization                                                       */
		/************************************************************************/
		unsigned int HexaGridRenderer::initializeGridPointsVBO() {
			m_pGridPointsVBO = new GLuint();
			glGenBuffers(1, m_pGridPointsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridPointsVBO);

			// initialize buffer object
			unsigned int size = (m_gridDimensions.x + 1)*(m_gridDimensions.y + 1)*(m_gridDimensions.z + 1)*sizeof(Vector3);
			glBufferData(GL_ARRAY_BUFFER, size, m_pGrid->getGridData3D()->getGridPointsArray().getRawDataPointer(), GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			return size;
		}

		unsigned int HexaGridRenderer::initializeGridCellsIndexVBO() {
			m_pGridCellsIndexVBO = new GLuint();
			glGenBuffers(1, m_pGridCellsIndexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridCellsIndexVBO);

			// initialize buffer object
			unsigned int size = (m_gridDimensions.x + 1)*(m_gridDimensions.y + 1)*(m_gridDimensions.z + 1)*12;
			int *pIndexes = new int[size];
			int currIndex = 0;
			unsigned int offset = m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z*4;
			for(int i = 0; i < m_gridDimensions.x; i++) {
				for(int j = 0; j < m_gridDimensions.y; j++) {
					for(int k = 0; k < m_gridDimensions.z; k++) {
						currIndex				= getGridPointIndex(i, j, k)*4;
						pIndexes[currIndex]		= getGridPointIndex(i, j, k);
						pIndexes[currIndex + 1] = getGridPointIndex(i + 1, j, k);
						pIndexes[currIndex + 2] = getGridPointIndex(i + 1, j + 1, k);
						pIndexes[currIndex + 3] = getGridPointIndex(i, j + 1, k);

						currIndex				= getGridPointIndex(i, j, k)*4 + offset;
						pIndexes[currIndex]		= getGridPointIndex(i, j, k);
						pIndexes[currIndex + 1] = getGridPointIndex(i + 1, j, k);
						pIndexes[currIndex + 2] = getGridPointIndex(i + 1, j, k + 1);
						pIndexes[currIndex + 3] = getGridPointIndex(i, j, k + 1);

						currIndex				= getGridPointIndex(i, j, k)*4 + offset*2;
						pIndexes[currIndex]		= getGridPointIndex(i, j, k);
						pIndexes[currIndex + 1] = getGridPointIndex(i, j + 1, k);
						pIndexes[currIndex + 2] = getGridPointIndex(i, j + 1, k + 1);
						pIndexes[currIndex + 3] = getGridPointIndex(i, j, k + 1);
					}	
				}
			}

			glBufferData(GL_ARRAY_BUFFER, size*sizeof(int), pIndexes, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			delete[] pIndexes;
			return size*sizeof(int);
		}

		/************************************************************************/
		/* Drawing                                                              */
		/************************************************************************/
		void HexaGridRenderer::drawGridVertices() const {
			glDisable(GL_LIGHTING);
			glPointSize(m_pointsSize);
			glEnable(GL_POINT_SMOOTH);
			glEnableClientState(GL_VERTEX_ARRAY);                 	
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridPointsVBO);
			glVertexPointer(3, GL_FLOAT, 0, 0);	
			glColor3f(m_gridPointsColor.getRed(), m_gridPointsColor.getGreen(), m_gridPointsColor.getBlue());
			glDrawArrays(GL_POINTS, 0, (m_gridDimensions.x + 1)*(m_gridDimensions.y + 1)*(m_gridDimensions.z + 1));
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);                
		}
		void HexaGridRenderer::drawGridCentroids() const {
			glDisable(GL_LIGHTING);
			glPointSize(m_pointsSize);
			glEnable(GL_POINT_SMOOTH);
			glEnableClientState(GL_VERTEX_ARRAY);                 	
			glBindBuffer(GL_ARRAY_BUFFER, m_scalarFieldRenderer.getGridCentroidsVBO());
			glVertexPointer(3, GL_FLOAT, 0, 0);	
			glColor3f(m_gridPointsColor.getRed(), m_gridPointsColor.getGreen(), m_gridPointsColor.getBlue());
			glDrawArrays(GL_POINTS, 0, m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);                
		}

		void HexaGridRenderer::drawGridCells() const {
			glDisable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			glLineWidth(m_lineWidth);
			glEnableClientState(GL_VERTEX_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridPointsVBO);
			glVertexPointer(3, GL_FLOAT, 0, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pGridCellsIndexVBO);
			glColor3f(m_gridLinesColor.getRed(), m_gridLinesColor.getGreen(), m_gridLinesColor.getBlue());
			glDrawElements(GL_QUADS, m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z*12, GL_UNSIGNED_INT, 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);

		}
		void HexaGridRenderer::drawGridSolidCells() const {
			glDisable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glColor3f(m_gridSolidCellColor.getRed(), m_gridSolidCellColor.getGreen(), m_gridSolidCellColor.getBlue());
			for(int i = 0; i < m_gridDimensions.x; i++) {
				for(int j = 0; j < m_gridDimensions.y; j++) {
					for(int k = 0; k < m_gridDimensions.z; k++) {
						if(m_pGrid->isSolidCell(i, j, k)) {
							drawCell(i, j, k);
						}
					}
				}
			}
		}
		void HexaGridRenderer::drawGridBoundaries() const {
			//glDisable(GL_LIGHTING);
			//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			//for(int j = 0; j < m_gridDimensions.y; j++) {
			//	/** West boundary */
			//	glColor3f(0.43f, 0.65f, 0.23f);
			//	drawCell(0, j);

			//	//East boundary
			//	glColor3f(0.8f, 0.2f, 0.2f);
			//	drawCell(m_gridDimensions.x - 1, j);
			//}
			//for(int i = 0; i < m_gridDimensions.x; i++) {
			//	/** South boundary */
			//	glColor3f(0.2f, 0.2f, 0.8f);
			//	drawCell(i, 0);

			//	/** North Boundary */
			//	glColor3f(0.0f, 0.0f, 0.0f);
			//	drawCell(i, m_gridDimensions.y - 1);
			//}
		}

		void HexaGridRenderer::drawXYSlice(int kthSlice) const {
			glLineWidth(1.0f);
			glColor3f(0.35f, 0.35f, 0.35f);
			GridData3D *pGridData = m_pGrid->getGridData3D();
			kthSlice = clamp(kthSlice, 0, pGridData->getDimensions().z);
			glDisable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			for(int i = 0; i < m_gridDimensions.x; i++) {
				for(int j = 0; j < m_gridDimensions.y; j++) {
					glBegin(GL_QUADS);
					glVertex3f(pGridData->getPoint(i, j, kthSlice).x, pGridData->getPoint(i, j, kthSlice).y, pGridData->getPoint(i, j, kthSlice).z);
					glVertex3f(pGridData->getPoint(i + 1, j, kthSlice).x, pGridData->getPoint(i + 1, j, kthSlice).y, pGridData->getPoint(i + 1, j, kthSlice).z);
					glVertex3f(pGridData->getPoint(i + 1, j + 1, kthSlice).x, pGridData->getPoint(i + 1, j + 1, kthSlice).y, pGridData->getPoint(i + 1, j + 1, kthSlice).z);
					glVertex3f(pGridData->getPoint(i, j + 1, kthSlice).x, pGridData->getPoint(i, j + 1, kthSlice).y, pGridData->getPoint(i, j + 1, kthSlice).z);
					glEnd();
				}
			}
			glLineWidth(2.0f);
			glColor3f(0.0f, 0.0f, 0.0f);
			glBegin(GL_QUADS);
			glVertex3f(pGridData->getPoint(0, 0, kthSlice).x, pGridData->getPoint(0, 0, kthSlice).y, pGridData->getPoint(0, 0, kthSlice).z);
			glVertex3f(pGridData->getPoint(m_gridDimensions.x, 0, kthSlice).x, pGridData->getPoint(m_gridDimensions.x, 0, kthSlice).y, 
											pGridData->getPoint(m_gridDimensions.x, 0, kthSlice).z);
			glVertex3f(pGridData->getPoint(m_gridDimensions.x, m_gridDimensions.y, kthSlice).x, pGridData->getPoint(m_gridDimensions.x, m_gridDimensions.y, kthSlice).y,
											pGridData->getPoint(m_gridDimensions.x, m_gridDimensions.y, kthSlice).z);	
			glVertex3f(pGridData->getPoint(0, m_gridDimensions.y, kthSlice).x, pGridData->getPoint(0, m_gridDimensions.y, kthSlice).y,
											pGridData->getPoint(0, m_gridDimensions.y, kthSlice).z);	
			glEnd();

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glEnable(GL_BLEND);
			glColor4f(1.0f, 0.4588f, 0.1839f, 0.25f);
			glLineWidth(1.0f);
			glBegin(GL_QUADS);
			glVertex3f(pGridData->getPoint(0, 0, kthSlice).x, pGridData->getPoint(0, 0, kthSlice).y, pGridData->getPoint(0, 0, kthSlice).z);
			glVertex3f(pGridData->getPoint(m_gridDimensions.x, 0, kthSlice).x, pGridData->getPoint(m_gridDimensions.x, 0, kthSlice).y,
				pGridData->getPoint(m_gridDimensions.x, 0, kthSlice).z);
			glVertex3f(pGridData->getPoint(m_gridDimensions.x, m_gridDimensions.y, kthSlice).x, pGridData->getPoint(m_gridDimensions.x, m_gridDimensions.y, kthSlice).y,
				pGridData->getPoint(m_gridDimensions.x, m_gridDimensions.y, kthSlice).z);
			glVertex3f(pGridData->getPoint(0, m_gridDimensions.y, kthSlice).x, pGridData->getPoint(0, m_gridDimensions.y, kthSlice).y,
				pGridData->getPoint(0, m_gridDimensions.y, kthSlice).z);
			glEnd();
			glDisable(GL_BLEND);
		}

		void HexaGridRenderer::drawXZSlice(int kthSlice) const {
			glLineWidth(1.0f);
			glColor3f(0.35f, 0.35f, 0.35f);
			GridData3D *pGridData = m_pGrid->getGridData3D();
			kthSlice = clamp(kthSlice, 0, pGridData->getDimensions().y);
			glDisable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			for(int i = 0; i < m_gridDimensions.x; i++) {
				for(int k = 0; k < m_gridDimensions.z; k++) {
					glBegin(GL_QUADS);
					glVertex3f(pGridData->getPoint(i, kthSlice, k).x, pGridData->getPoint(i, kthSlice, k).y, pGridData->getPoint(i, kthSlice, k).z);
					glVertex3f(pGridData->getPoint(i + 1, kthSlice, k).x, pGridData->getPoint(i + 1, kthSlice, k).y, pGridData->getPoint(i + 1, kthSlice, k).z);
					glVertex3f(pGridData->getPoint(i + 1, kthSlice, k + 1).x, pGridData->getPoint(i + 1, kthSlice, k + 1).y, pGridData->getPoint(i + 1, kthSlice, k + 1).z);
					glVertex3f(pGridData->getPoint(i, kthSlice, k + 1).x, pGridData->getPoint(i, kthSlice, k + 1).y, pGridData->getPoint(i, kthSlice, k + 1).z);
					glEnd();
				}
			}
			glLineWidth(2.0f);
			glColor3f(0.0f, 0.0f, 0.0f);
			glBegin(GL_QUADS);
			glVertex3f(pGridData->getPoint(0, kthSlice, 0).x, pGridData->getPoint(0, kthSlice, 0).y, pGridData->getPoint(0, kthSlice, 0).z);
			glVertex3f(pGridData->getPoint(m_gridDimensions.x, kthSlice, 0).x, pGridData->getPoint(m_gridDimensions.x, kthSlice, 0).y, 
											pGridData->getPoint(m_gridDimensions.x, kthSlice, 0).z);
			glVertex3f(pGridData->getPoint(m_gridDimensions.x, kthSlice, m_gridDimensions.z).x, pGridData->getPoint(m_gridDimensions.x, kthSlice, m_gridDimensions.z).y, 
											pGridData->getPoint(m_gridDimensions.x, kthSlice, m_gridDimensions.z).z);
			glVertex3f(pGridData->getPoint(0, kthSlice, m_gridDimensions.z).x, pGridData->getPoint(0, kthSlice, m_gridDimensions.z).y, 
											pGridData->getPoint(0, kthSlice, m_gridDimensions.z).z);
			glEnd();

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glEnable(GL_BLEND);
			glColor4f(0.4588f, 1.0f, 0.1839f, 0.25f);
			glLineWidth(1.0f);
			glBegin(GL_QUADS);
			glVertex3f(pGridData->getPoint(0, kthSlice, 0).x, pGridData->getPoint(0, kthSlice, 0).y, pGridData->getPoint(0, kthSlice, 0).z);
			glVertex3f(pGridData->getPoint(m_gridDimensions.x, kthSlice, 0).x, pGridData->getPoint(m_gridDimensions.x, kthSlice, 0).y,
				pGridData->getPoint(m_gridDimensions.x, kthSlice, 0).z);
			glVertex3f(pGridData->getPoint(m_gridDimensions.x, kthSlice, m_gridDimensions.z).x, pGridData->getPoint(m_gridDimensions.x, kthSlice, m_gridDimensions.z).y,
				pGridData->getPoint(m_gridDimensions.x, kthSlice, m_gridDimensions.z).z);
			glVertex3f(pGridData->getPoint(0, kthSlice, m_gridDimensions.z).x, pGridData->getPoint(0, kthSlice, m_gridDimensions.z).y,
				pGridData->getPoint(0, kthSlice, m_gridDimensions.z).z);
			glEnd();
			glDisable(GL_BLEND);
		}

		void HexaGridRenderer::drawYZSlice(int kthSlice) const {
			glLineWidth(1.0f);
			glColor3f(0.35f, 0.35f, 0.35f);
			GridData3D *pGridData = m_pGrid->getGridData3D();
			kthSlice = clamp(kthSlice, 0, pGridData->getDimensions().x);
			glDisable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			for(int j = 0; j < m_gridDimensions.y; j++) {
				for(int k = 0; k < m_gridDimensions.z; k++) {
					glBegin(GL_QUADS);
					glVertex3f(pGridData->getPoint(kthSlice, j, k).x, pGridData->getPoint(kthSlice, j, k).y, pGridData->getPoint(kthSlice, j, k).z);
					glVertex3f(pGridData->getPoint(kthSlice, j + 1, k).x, pGridData->getPoint(kthSlice, j + 1, k).y, pGridData->getPoint(kthSlice, j + 1, k).z);
					glVertex3f(pGridData->getPoint(kthSlice, j + 1, k + 1).x, pGridData->getPoint(kthSlice, j + 1, k + 1).y, pGridData->getPoint(kthSlice, j + 1, k + 1).z);
					glVertex3f(pGridData->getPoint(kthSlice, j, k + 1).x, pGridData->getPoint(kthSlice, j, k + 1).y, pGridData->getPoint(kthSlice, j, k + 1).z);
					glEnd();
				}
			}
			glDisable(GL_BLEND);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glLineWidth(2.0f);
			glColor3f(0.0f, 0.0f, 0.0f);
			glBegin(GL_QUADS);
			glVertex3f(pGridData->getPoint(kthSlice, 0, 0).x, pGridData->getPoint(kthSlice, 0, 0).y, pGridData->getPoint(kthSlice, 0, 0).z);
			glVertex3f(pGridData->getPoint(kthSlice, m_gridDimensions.y, 0).x, pGridData->getPoint(kthSlice, m_gridDimensions.y, 0).y, 
											pGridData->getPoint(kthSlice, m_gridDimensions.y, 0).z);
			glVertex3f(pGridData->getPoint(kthSlice, m_gridDimensions.y, m_gridDimensions.z).x, pGridData->getPoint(kthSlice, m_gridDimensions.y, m_gridDimensions.z).y, 
											pGridData->getPoint(kthSlice, m_gridDimensions.y, m_gridDimensions.z).z);
			glVertex3f(pGridData->getPoint(kthSlice, 0, m_gridDimensions.z).x, pGridData->getPoint(kthSlice, 0, m_gridDimensions.z).y,			
											pGridData->getPoint(kthSlice, 0, m_gridDimensions.z).z);
			glEnd();

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glEnable(GL_BLEND);
			glColor4f(0.1839f, 0.4588f, 1.0f, 0.25f);
			glLineWidth(1.0f);
			glBegin(GL_QUADS);
			glVertex3f(pGridData->getPoint(kthSlice, 0, 0).x, pGridData->getPoint(kthSlice, 0, 0).y, pGridData->getPoint(kthSlice, 0, 0).z);
			glVertex3f(pGridData->getPoint(kthSlice, m_gridDimensions.y, 0).x, pGridData->getPoint(kthSlice, m_gridDimensions.y, 0).y,
				pGridData->getPoint(kthSlice, m_gridDimensions.y, 0).z);
			glVertex3f(pGridData->getPoint(kthSlice, m_gridDimensions.y, m_gridDimensions.z).x, pGridData->getPoint(kthSlice, m_gridDimensions.y, m_gridDimensions.z).y,
				pGridData->getPoint(kthSlice, m_gridDimensions.y, m_gridDimensions.z).z);
			glVertex3f(pGridData->getPoint(kthSlice, 0, m_gridDimensions.z).x, pGridData->getPoint(kthSlice, 0, m_gridDimensions.z).y,
				pGridData->getPoint(kthSlice, 0, m_gridDimensions.z).z);
			glEnd();
			glDisable(GL_BLEND);
		}
	}
}