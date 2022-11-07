#include "Visualization/CutCellsVelocityRenderer2D.h"

namespace Chimera {
	namespace Rendering {

		template<class VectorType>
		void CutCellsVelocityRenderer2D<VectorType>::draw() {
			if (m_drawEdgeVelocities)
				drawEdgeVelocities();

			if (m_drawNodalVelocities)
				drawNodalVelocities();

			if (m_drawFineGridVelocities)
				drawFineGridVelocities();
		}

		template<class VectorType>
		void CutCellsVelocityRenderer2D<VectorType>::drawEdgeVelocities() {
			if (m_drawVelocitiesType == selectedCutCell) {
				drawEdgeVelocities(m_selectedCutCell);
			}
			else {
				for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
					drawEdgeVelocities(i);
				}
			}
		}
		template<class VectorType>
		void CutCellsVelocityRenderer2D<VectorType>::drawEdgeVelocities(uint cellIndex) {
			auto cutCell = m_pCutCells->getCutCell(cellIndex);
			auto halfEdges = cutCell.getHalfEdges();
			for (uint i = 0; i < halfEdges.size(); i++) {
				VectorType initialPoint = halfEdges[i]->getEdge()->getCentroid();
				VectorType finalPoint;

				if (m_mainVelocityType == BaseWindow::drawVelocity)
					finalPoint = halfEdges[i]->getEdge()->getVelocity();
				else if (m_mainVelocityType == BaseWindow::drawAuxiliaryVelocity)
					finalPoint = halfEdges[i]->getEdge()->getAuxiliaryVelocity();
				
				RenderingUtils::getInstance()->drawVector(initialPoint, initialPoint + finalPoint*0.01);
			}
		} 

		template<class VectorType>
		void CutCellsVelocityRenderer2D<VectorType>::drawNodalVelocities() {
			if (m_drawVelocitiesType == selectedCutCell) {
				drawNodalVelocities(m_selectedCutCell);
			}
			else {
				for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
					for (uint i = 0; i < m_pCutCells->getVertices().size(); i++) {
						VectorType initialPoint = m_pCutCells->getVertices()[i]->getPosition();
						VectorType finalPoint;

						if (m_mainVelocityType == BaseWindow::drawVelocity)
							finalPoint = m_pCutCells->getVertices()[i]->getVelocity();
						else if (m_mainVelocityType == BaseWindow::drawAuxiliaryVelocity)
							finalPoint = m_pCutCells->getVertices()[i]->getAuxiliaryVelocity();

						RenderingUtils::getInstance()->drawVector(initialPoint, initialPoint + finalPoint*0.01);
					}
				}
			}	
		}

		template<class VectorType>
		void CutCellsVelocityRenderer2D<VectorType>::drawNodalVelocities(uint cellIndex) {
			auto cutCell = m_pCutCells->getCutCell(cellIndex);
			auto halfEdges = cutCell.getHalfEdges();
			for (uint i = 0; i < halfEdges.size(); i++) {
				VectorType initialPoint = halfEdges[i]->getVertices().first->getPosition();
				VectorType finalPoint;

				if(m_mainVelocityType == BaseWindow::drawVelocity)
					finalPoint = halfEdges[i]->getVertices().first->getVelocity();
				else if(m_mainVelocityType == BaseWindow::drawAuxiliaryVelocity)
					finalPoint = halfEdges[i]->getVertices().first->getAuxiliaryVelocity();

				RenderingUtils::getInstance()->drawVector(initialPoint, initialPoint + finalPoint*0.01);
			}
		}

		template<class VectorType>
		void CutCellsVelocityRenderer2D<VectorType>::drawFineGridVelocities() {
		}


		template class CutCellsVelocityRenderer2D<Vector2>;
		template class CutCellsVelocityRenderer2D<Vector2D>;
	}
}