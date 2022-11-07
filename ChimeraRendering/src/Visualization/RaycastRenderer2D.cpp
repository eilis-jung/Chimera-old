#include "Visualization/RaycastRenderer2D.h"
#include "Resources/ResourceManager.h"
#include "BaseGLRenderer.h"

namespace Chimera {
	namespace Rendering {
		RaycastRenderer2D::RaycastRenderer2D(QuadGrid *pQuadGrid) {
			m_pQuadGrid = pQuadGrid;
			m_drawCells = true;
			m_drawFaceNormals = false;
			m_velocityLength = 0.01;
		}


		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void RaycastRenderer2D::draw() {
			if (m_drawCells)
				drawCells();

			if (m_drawFaceVelocities)
				drawFaceVelocities();

			if (m_drawFaceNormals) {
				drawFaceNormals();
			}

			if (m_drawRaycastBoundaries) {
				drawOccludedFaces();
			}
		}

		void RaycastRenderer2D::initializeWindowControls(GridVisualizationWindow<Vector2> *pGridVisualizationWindow) {
			TwBar *pTwBar = pGridVisualizationWindow->getBaseTwBar();
			m_pGridVisualizationWindow = pGridVisualizationWindow;

			TwAddVarRW(pTwBar, "drawSpecialCells", TW_TYPE_BOOL8, &m_drawCells, " label='Draw Cells' help='Draw special grid cells' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawRaycastBoundaries", TW_TYPE_BOOL8, &m_drawRaycastBoundaries, " label='Draw Raycast boundaries' help='Draw special grid cells' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawFaceVelocities", TW_TYPE_BOOL8, &m_drawFaceVelocities, " label='Draw Face Velocities' help='Draw face velocities' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawNormals", TW_TYPE_BOOL8, &m_drawFaceNormals, " label='Draw Face Normals' group='Cut Cells'");
			TwAddVarRW(pTwBar, "velocityLenghtScale", TW_TYPE_FLOAT, &m_velocityLength, " label ='Velocity visualization length' group='Cut Cells' step=0.01 min='-0.15' max='0.15'");
			string defName = "'" + m_pQuadGrid->getGridName() + "'/'Cut Cells' group=Visualization";
			TwDefine(defName.c_str());

		}

		void RaycastRenderer2D::update() {

		}

		/************************************************************************/
		/* Private drawing functions                                            */
		/************************************************************************/
		void RaycastRenderer2D::drawCells() {
			GridData2D *pGridData2D = m_pQuadGrid->getGridData2D();
			Scalar dx = pGridData2D->getScaleFactor(0, 0).x;

			glLineWidth(1.0f);
			glPointSize(3.0f);

			for (int i = 0; i < pGridData2D->getDimensions().x; i++) {
				for (int j = 0; j < pGridData2D->getDimensions().y; j++) {
					if ((*m_pBoundaryCells)(i, j)) {
						drawCell(i, j);
					}
				}
			}
		}


		void RaycastRenderer2D::drawFaceVelocities() {
			glLineWidth(1.0f);
			GridData2D *pGridData2D = m_pQuadGrid->getGridData2D();
			Scalar dx = pGridData2D->getScaleFactor(0, 0).x;
			TwBar *pTwBar = m_pGridVisualizationWindow->getBaseTwBar();
			//bool drawIntermediateVel = m_pGridVisualizationWindow->getVelocityDrawingType() == BaseWindow::vectorVisualization_t::drawAuxiliaryVelocity;
			//for (int i = 0; i < m_pQuadGrid->getDimensions().x; i++) {
			//	for (int j = 0; j < m_pQuadGrid->getDimensions().y; j++) {
			//		for (int k = 0; k < m_pSpecialCells->getEdgeVector(dimensions_t(i, j), leftEdge).size(); k++) {
			//			const CutEdge<Vector2> &edge = m_pSpecialCells->getEdgeVector(dimensions_t(i, j), leftEdge)[k];
			//			Vector2 edgeCenter = edge.getCentroid();
			//			Vector2 edgeVelocity;
			//			if (drawIntermediateVel)
			//				edgeVelocity = edge.getIntermediaryVelocity();
			//			else
			//				edgeVelocity = edge.getVelocity();

			//			//edgeVelocity.x = edge.getNormal().dot(edgeVelocity);

			//			glColor3f(1.0f, 0.06666f, 0.1294f);
			//			glBegin(GL_POINTS);
			//			glVertex2f(edgeCenter.x, edgeCenter.y);
			//			glEnd();

			//			RenderingUtils::getInstance()->drawVector(edgeCenter, edgeCenter + edgeVelocity*m_velocityLength);
			//		}

			//		for (int k = 0; k < m_pSpecialCells->getEdgeVector(dimensions_t(i, j), bottomEdge).size(); k++) {
			//			CutEdge<Vector2> edge = m_pSpecialCells->getEdgeVector(dimensions_t(i, j), bottomEdge)[k];
			//			Vector2 edgeCenter = edge.getCentroid();
			//			Vector2 edgeVelocity;
			//			if (drawIntermediateVel)
			//				edgeVelocity = edge.getIntermediaryVelocity();
			//			else
			//				edgeVelocity = edge.getVelocity();

			//			//edgeVelocity.y = edge.getNormal().dot(edgeVelocity);

			//			glColor3f(1.0f, 0.06666f, 0.1294f);
			//			glBegin(GL_POINTS);
			//			glVertex2f(edgeCenter.x, edgeCenter.y);
			//			glEnd();

			//			RenderingUtils::getInstance()->drawVector(edgeCenter, edgeCenter + edgeVelocity*m_velocityLength);
			//		}
			//	}
			//}
		}



		void RaycastRenderer2D::drawFaceNormals() {
			/*if (m_selectedCell != -1) {
				glLineWidth(3.0f);
				const CutFace<Vector2> &currCell = m_pSpecialCells->getSpecialCell(m_selectedCell);
				for (int j = 0; j < currCell.m_cutEdges.size(); j++) {
				Vector2 edgeCenter = currCell.m_cutEdges[j]->getCentroid();

				glColor3f(1.0f, 0.0f, 0.0f);
				glBegin(GL_POINTS);
				glVertex2f(edgeCenter.x, edgeCenter.y);
				glEnd();
				glColor3f(0.0f, 0.0f, 0.0f);
				RenderingUtils::getInstance()->drawVector(edgeCenter, edgeCenter + currCell.getEdgeNormal(j)*m_velocityLength);
				}
				}*/
		}
		
		void RaycastRenderer2D::drawOccludedFaces() {
			Scalar dx = m_pQuadGrid->getGridData2D()->getScaleFactor(0, 0).x;
			glLineWidth(4.0f);
			glColor3f(0, 0, 0);
			for (int i = 0; i < m_pLeftFaceCrossings->getDimensions().x; i++) {
				for (int j = 0; j < m_pLeftFaceCrossings->getDimensions().y; j++) {
					if (!(*m_pLeftFaceVisibility)(i, j)) {
						glBegin(GL_LINES);
						glVertex2f(i*dx, j*dx);
						glVertex2f(i*dx, (j+1)*dx);
						glEnd();
					}
				}
			}

			for (int i = 0; i < m_pBottomFaceCrossings->getDimensions().x; i++) {
				for (int j = 0; j < m_pBottomFaceCrossings->getDimensions().y; j++) {
					if (!(*m_pBottomFaceVisibility)(i, j)) {
						glBegin(GL_LINES);
						glVertex2f(i*dx, j*dx);
						glVertex2f((i + 1)*dx, j*dx);
						glEnd();
					}
				}
			}
		}
	}
}