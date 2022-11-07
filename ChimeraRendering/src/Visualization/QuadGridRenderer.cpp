#include "Visualization/QuadGridRenderer.h"

namespace Chimera {
	namespace Rendering {
		/************************************************************************/
		/* Ctors                                                                */
		/************************************************************************/
		QuadGridRenderer::QuadGridRenderer(QuadGrid *pQuadGrid) : GridRenderer(pQuadGrid) {
			initializeGridPointsVBO();
			initializeGridCellsIndexVBO();
			m_lineWidth = 1.0f;
			m_pointsSize = 2.0f;
		}

		/************************************************************************/
		/* Initialization                                                       */
		/************************************************************************/
		unsigned int QuadGridRenderer::initializeGridPointsVBO() {
			m_pGridPointsVBO = new GLuint();
			glGenBuffers(1, m_pGridPointsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridPointsVBO);

			// initialize buffer object
			unsigned int size = (m_gridDimensions.x + 1)*(m_gridDimensions.y + 1)*sizeof(Vector2);
			glBufferData(GL_ARRAY_BUFFER, size, m_pGrid->getGridData2D()->getGridPointsArray().getRawDataPointer(), GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			return size;
		}

		unsigned int QuadGridRenderer::initializeGridCellsIndexVBO() {
			m_pGridCellsIndexVBO = new GLuint();
			glGenBuffers(1, m_pGridCellsIndexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridCellsIndexVBO);

			// initialize buffer object
			unsigned int size = (m_gridDimensions.x)*(m_gridDimensions.y)*4;
			int *pIndexes = new int[size];
			int currIndex = 0;
			for(int i = 0; i < m_gridDimensions.x; i++) {
				for(int j = 0; j < m_gridDimensions.y; j++) {
					currIndex = (j*m_gridDimensions.x + i)*4;
					pIndexes[currIndex]		= j*(m_gridDimensions.x + 1) + i;
					pIndexes[currIndex + 1] = j*(m_gridDimensions.x + 1) + i + 1;
					pIndexes[currIndex + 2] = (j + 1)*(m_gridDimensions.x + 1) + i + 1;
					pIndexes[currIndex + 3] = (j + 1)*(m_gridDimensions.x + 1) + i;
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
		void QuadGridRenderer::drawGridVertices() const {
			glDisable(GL_LIGHTING);
			glPointSize(m_pointsSize);
			glEnable(GL_POINT_SMOOTH);
			glEnableClientState(GL_VERTEX_ARRAY);                 	
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridPointsVBO);
			glVertexPointer(2, GL_FLOAT, 0, 0);	
			glColor3f(m_gridPointsColor.getRed(), m_gridPointsColor.getGreen(), m_gridPointsColor.getBlue());
			glDrawArrays(GL_POINTS, 0, (m_gridDimensions.x + 1)*(m_gridDimensions.y + 1));
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);                

			glColor3f(1.0f, 0.0f, 0.0f);
			for(int j = 0; j < m_pGrid->getDimensions().y; j++) {
				glBegin(GL_POINTS);
				glVertex2f(m_pGrid->getGridData2D()->getPoint(0, j).x, m_pGrid->getGridData2D()->getPoint(0, j).y);
				glEnd();
			}

			glColor3f(0.0f, 0.0f, 1.0f);
			glPointSize(4.0f);
			for(int j = 0; j < m_pGrid->getDimensions().y; j++) {
				if(j == 0) {
					glBegin(GL_POINTS);
					glVertex2f(m_pGrid->getGridData2D()->getPoint(m_pGrid->getDimensions().x, j).x, m_pGrid->getGridData2D()->getPoint(m_pGrid->getDimensions().x, j).y);
					glEnd();
				}
				
			}

			glPointSize(4.0f);
			glColor3f(0.0f, 1.0f, 0.0f);
			for(int j = 0; j < m_pGrid->getDimensions().y; j++) {
				Vector2 g2 = (m_pGrid->getGridData2D()->getPoint(m_pGrid->getDimensions().x, j) - m_pGrid->getGridData2D()->getPoint(m_pGrid->getDimensions().x - 1, j)).normalized().perpendicular();
				Vector2 finalPoint = m_pGrid->getGridData2D()->getPoint(m_pGrid->getDimensions().x, j)+ g2*0.015;
				glBegin(GL_POINTS);
				glVertex2f(finalPoint.x, finalPoint.y);
				glEnd();
			}

			
			
		}
		void QuadGridRenderer::drawGridCentroids() const {
			glDisable(GL_LIGHTING);
			glPointSize(m_pointsSize);
			glEnable(GL_POINT_SMOOTH);
			glEnableClientState(GL_VERTEX_ARRAY);                 	
			glBindBuffer(GL_ARRAY_BUFFER, m_scalarFieldRenderer.getGridCentroidsVBO());
			glVertexPointer(2, GL_FLOAT, 0, 0);	
			glColor3f(m_gridPointsColor.getRed(), m_gridPointsColor.getGreen(), m_gridPointsColor.getBlue());
			glDrawArrays(GL_POINTS, 0, m_gridDimensions.x*m_gridDimensions.y);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);                
		}
		void QuadGridRenderer::drawGridCells() const {
			glDisable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			glLineWidth(1.0f);
			glEnableClientState(GL_VERTEX_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridPointsVBO);
			glVertexPointer(2, GL_FLOAT, 0, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pGridCellsIndexVBO);
			glColor3f(m_gridLinesColor.getRed(), m_gridLinesColor.getGreen(), m_gridLinesColor.getBlue());
			glDrawElements(GL_QUADS, m_gridDimensions.x*m_gridDimensions.y*4, GL_UNSIGNED_INT, 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);
			
			glPointSize(m_pointsSize);
			glColor3f(0.3f, 1.0f, 0.5f);
			glBegin(GL_POINTS);
			for(int j = 0; j < m_gridDimensions.y; j++) {
				glVertex2f(m_pGrid->getGridData2D()->getPoint(1, j).x, m_pGrid->getGridData2D()->getPoint(1, j).y);
			}
			glEnd();

		}
		void QuadGridRenderer::drawGridSolidCells() const {
			glDisable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glColor3f(m_gridSolidCellColor.getRed(), m_gridSolidCellColor.getGreen(), m_gridSolidCellColor.getBlue());
			for(int i = 0; i < m_gridDimensions.x; i++) {
				for(int j = 0; j < m_gridDimensions.y; j++) {
					if(m_pGrid->isSolidCell(i, j)) {
						drawCell(i, j);
					}
				}
			}
		}
		void QuadGridRenderer::drawGridBoundaries() const {
			glDisable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			for(int j = 0; j < m_gridDimensions.y; j++) {
				/** West boundary */
				glColor3f(0.43f, 0.65f, 0.23f);
				drawCell(0, j);

				//East boundary
				glColor3f(0.8f, 0.2f, 0.2f);
				drawCell(m_gridDimensions.x - 1, j);
			}
			for(int i = 0; i < m_gridDimensions.x; i++) {
				/** South boundary */
				glColor3f(0.2f, 0.2f, 0.8f);
				drawCell(i, 0);

				/** North Boundary */
				glColor3f(0.0f, 0.0f, 0.0f);
				drawCell(i, m_gridDimensions.y - 1);
			}
		}

		void QuadGridRenderer::drawSelectedCell() const {
			if (m_selectedCell.x > 0 && m_selectedCell.x < m_pGrid->getDimensions().x &&
				m_selectedCell.y > 0 && m_selectedCell.y < m_pGrid->getDimensions().y) {
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				glLineWidth(3.0f);
				drawCell(m_selectedCell.x, m_selectedCell.y);
			}
		}
	}
}