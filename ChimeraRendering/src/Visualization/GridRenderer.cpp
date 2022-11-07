#include "Visualization/GridRenderer.h"

namespace Chimera {
	namespace Rendering {

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		template <class VectorT>
		GridRenderer<VectorT>::GridRenderer(StructuredGrid<VectorT> *pGrid) : m_scalarFieldRenderer(pGrid), m_vectorFieldRenderer(pGrid),
				m_gridSolidCellColor(0.5f, 0.5f, 0.5f) /*medium gray*/, m_gridPointsColor(0.3f, 0.3f, 0.3f) /*light gray*/, 
				m_gridLinesColor(0.0f, 0.0f, 0.0f), //black 
				m_pointsSize(2.0f)
		{
				m_pGrid = pGrid;
				m_gridDimensions = pGrid->getDimensions();
			
		}

		/************************************************************************/
		/* Drawing		                                                        */
		/************************************************************************/
		template <>
		void GridRenderer<Vector2>::drawCell(int i, int j, int k) const {
			
		}
		template <>
		void GridRenderer<Vector3>::drawCell(int i, int j, int k) const {
			GridData3D *pGridData3D = m_pGrid->getGridData3D();
			/** Back face */
			glBegin(GL_QUADS);
			glVertex3f(pGridData3D->getPoint(i, j, k).x, pGridData3D->getPoint(i, j, k).y, pGridData3D->getPoint(i, j, k).z);
			glVertex3f(pGridData3D->getPoint(i + 1, j, k).x, pGridData3D->getPoint(i + 1, j, k).y, pGridData3D->getPoint(i + 1, j, k).z);
			glVertex3f(pGridData3D->getPoint(i + 1, j + 1, k).x, pGridData3D->getPoint(i + 1, j + 1, k).y, pGridData3D->getPoint(i + 1, j + 1, k).z);
			glVertex3f(pGridData3D->getPoint(i, j + 1, k).x, pGridData3D->getPoint(i, j + 1, k).y, pGridData3D->getPoint(i, j + 1, k).z);
			glEnd();

			/** Front face */
			glBegin(GL_QUADS);
			glVertex3f(pGridData3D->getPoint(i, j, k + 1).x, pGridData3D->getPoint(i, j, k).y, pGridData3D->getPoint(i, j, k + 1).z);
			glVertex3f(pGridData3D->getPoint(i + 1, j, k + 1).x, pGridData3D->getPoint(i + 1, j, k).y, pGridData3D->getPoint(i + 1, j, k + 1).z);
			glVertex3f(pGridData3D->getPoint(i + 1, j + 1, k + 1).x, pGridData3D->getPoint(i + 1, j + 1, k).y, pGridData3D->getPoint(i + 1, j + 1, k + 1).z);
			glVertex3f(pGridData3D->getPoint(i, j + 1, k + 1).x, pGridData3D->getPoint(i, j + 1, k).y, pGridData3D->getPoint(i, j + 1, k + 1).z);
			glEnd();

			/** Bottom Face */
			glBegin(GL_QUADS);
			glVertex3f(pGridData3D->getPoint(i, j, k).x, pGridData3D->getPoint(i, j, k).y, pGridData3D->getPoint(i, j, k).z);
			glVertex3f(pGridData3D->getPoint(i + 1, j, k).x, pGridData3D->getPoint(i + 1, j, k).y, pGridData3D->getPoint(i + 1, j, k).z);
			glVertex3f(pGridData3D->getPoint(i + 1, j, k + 1).x, pGridData3D->getPoint(i + 1, j, k + 1).y, pGridData3D->getPoint(i + 1, j, k + 1).z);
			glVertex3f(pGridData3D->getPoint(i, j, k + 1).x, pGridData3D->getPoint(i, j, k + 1).y, pGridData3D->getPoint(i, j, k + 1).z);
			glEnd();

			/** Top Face */
			glBegin(GL_QUADS);
			glVertex3f(pGridData3D->getPoint(i, j + 1, k).x, pGridData3D->getPoint(i, j + 1, k).y, pGridData3D->getPoint(i, j + 1, k).z);
			glVertex3f(pGridData3D->getPoint(i + 1, j + 1, k).x, pGridData3D->getPoint(i + 1, j + 1, k).y, pGridData3D->getPoint(i + 1, j + 1, k).z);
			glVertex3f(pGridData3D->getPoint(i + 1, j + 1, k + 1).x, pGridData3D->getPoint(i + 1, j + 1, k + 1).y, pGridData3D->getPoint(i + 1, j + 1, k + 1).z);
			glVertex3f(pGridData3D->getPoint(i, j + 1, k + 1).x, pGridData3D->getPoint(i, j + 1, k + 1).y, pGridData3D->getPoint(i, j + 1, k + 1).z);
			glEnd();


			/** Left Face */
			glBegin(GL_QUADS);
			glVertex3f(pGridData3D->getPoint(i, j, k).x, pGridData3D->getPoint(i, j, k).y, pGridData3D->getPoint(i, j, k).z);
			glVertex3f(pGridData3D->getPoint(i, j + 1, k).x, pGridData3D->getPoint(i, j + 1, k).y, pGridData3D->getPoint(i, j + 1, k).z);
			glVertex3f(pGridData3D->getPoint(i, j + 1, k + 1).x, pGridData3D->getPoint(i, j + 1, k + 1).y, pGridData3D->getPoint(i, j + 1, k + 1).z);
			glVertex3f(pGridData3D->getPoint(i, j, k + 1).x, pGridData3D->getPoint(i, j, k + 1).y, pGridData3D->getPoint(i, j, k + 1).z);
			glEnd();

			/** Right face */
			glBegin(GL_QUADS);
			glVertex3f(pGridData3D->getPoint(i + 1,j, k).x, pGridData3D->getPoint(i + 1,j, k).y, pGridData3D->getPoint(i + 1,j, k).z);
			glVertex3f(pGridData3D->getPoint(i + 1,j + 1, k).x, pGridData3D->getPoint(i + 1,j + 1, k).y, pGridData3D->getPoint(i + 1,j + 1, k).z);
			glVertex3f(pGridData3D->getPoint(i + 1,j + 1, k + 1).x, pGridData3D->getPoint(i + 1,j + 1, k + 1).y, pGridData3D->getPoint(i + 1,j + 1, k + 1).z);
			glVertex3f(pGridData3D->getPoint(i + 1,j, k + 1).x, pGridData3D->getPoint(i + 1,j, k + 1).y, pGridData3D->getPoint(i + 1,j, k + 1).z);
			glEnd();
		}

		/**Point based drawing */
		template<> 
		void GridRenderer<Vector2>::drawCell(const vector<Vector2> &points) const {
			glBegin(GL_QUADS);
			for(int i = 0; i < 4; i++) {
				glVertex2f(points[i].x, points[i].y);
			}
			glEnd();
		}

		template<> 
		void GridRenderer<Vector3>::drawCell(const vector<Vector3> &points) const {
			glBegin(GL_QUADS);
			for(int i = 0; i < 24; i++) {
				glVertex2f(points[i].x, points[i].y);
			}
			glEnd();
		}

		template <class VectorT>
		void GridRenderer<VectorT>::draw(gridRenderingMode_t gridRenderingMode) {
			switch (gridRenderingMode) {
			case Chimera::Rendering::drawVertices:
				drawGridVertices();
				break;
			case Chimera::Rendering::drawCells:
				drawGridCells();
				break;
			case Chimera::Rendering::drawSolidCells:
				drawGridSolidCells();
				break;
			case Chimera::Rendering::drawBoundaries:
				drawGridBoundaries();
				break;
			case Chimera::Rendering::drawCentroids:
				drawGridCentroids();
				break;
			default:
				break;	
			}
			drawSelectedCell();
		}


		/************************************************************************/
		/* Declarations                                                         */
		/************************************************************************/
		template GridRenderer<Vector2>;
		template GridRenderer<Vector3>;
	}
}