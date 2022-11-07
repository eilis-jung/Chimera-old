#include "Visualization/PolygonMeshRenderer.h"

namespace Chimera {

	namespace Rendering {

		#pragma region 2DRenderer
		template<class VectorType>
		void MeshRendererT<VectorType, Face, true>::drawElements(uint selectedIndex, Color color /*= Color::BLACK*/) {
			/*if (m_selectedCutCell >= m_pCutCells->getNumberCutCells()) {
				m_selectedCutCell = m_pCutCells->getNumberCutCells() - 1;
			}*/
			if (m_selectedCutCell != -1) {
				glLineWidth(4.0f);
				drawCutCell(m_selectedCutCell);
			}
		
			if (m_drawCutCells) {
				glLineWidth(1.5f);

				for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
					drawCutCell(i);
				}
			}

			if (m_isDrawingEdgeNormals) {
				glLineWidth(1.0f);
				if (m_selectedCutCell != -1) {
					drawEdgeNormals(m_selectedCutCell);
				}
			}
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Face, true>::drawMeshVertices(uint selectedIndex, Color color /*= Color::BLACK*/) {
			glEnable(GL_POINT_SMOOTH);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			glPointSize(6.0f);
			glBegin(GL_POINTS);
			for (int i = 0; i < m_meshes[selectedIndex]->getVertices().size(); i++) {
				if (m_meshes[selectedIndex]->getVertices()[i]->getVertexType() == edgeVertex) {
					glColor3f(0.25f, 0.80f, 0.12f);
				} else if (m_meshes[selectedIndex]->getVertices()[i]->getVertexType() == gridVertex) {
					glColor3f(0.25f, 0.3f, 1.0f);
				}
				else {
					glColor3f(0.8f, 0.8f, 0.8f);
				}
				glVertex2f(m_meshes[selectedIndex]->getVertices()[i]->getPosition().x, m_meshes[selectedIndex]->getVertices()[i]->getPosition().y);
			}
			glEnd();
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Face, true>::drawCutCell(uint cutCellIndex) {
			glBegin(GL_LINES);
			auto edges = m_pCutCells->getCutCell(cutCellIndex).getHalfEdges();
			for (int j = 0; j < edges.size(); j++) {
				const VectorType &v1 = edges[j]->getVertices().first->getPosition();
				const VectorType &v2 = edges[j]->getVertices().second->getPosition();

				glVertex2f(v1.x, v1.y);
				glVertex2f(v2.x, v2.y);
			}
			glEnd();
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Face, true>::drawEdgeNormals(uint cutCellIndex) {
			auto edges = m_pCutCells->getCutCell(cutCellIndex).getHalfEdges();
			for (int j = 0; j < edges.size(); j++) {
				const VectorType &v1 = edges[j]->getVertices().first->getPosition();
				const VectorType &v2 = edges[j]->getVertices().second->getPosition();
				VectorType edgeCentroid = edges[j]->getEdge()->getCentroid();
				RenderingUtils::getInstance()->drawVector(edgeCentroid, edgeCentroid + edges[j]->getNormal()*0.01);
			}
		}
		#pragma endregion

		#pragma region 3DRenderer
		template<class VectorType>
		void MeshRendererT<VectorType, Face, false>::drawElements(uint selectedIndex, Color color /*= Color::BLACK*/) {
			if (m_selectedCutCell != -1) {
				glLineWidth(4.0f);
				drawCutCell(m_selectedCutCell);
			}

			if (m_drawCutCells) {
				glLineWidth(1.5f);

				for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
					drawCutCell(i);
				}
			}

			if (m_isDrawingEdgeNormals) {
				glLineWidth(1.0f);
				if (m_selectedCutCell != -1) {
					drawEdgeNormals(m_selectedCutCell);
				}
			}
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Face, false>::drawMeshVertices(uint selectedIndex, Color color /*= Color::BLACK*/) {
			glEnable(GL_POINT_SMOOTH);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			glPointSize(6.0f);
			glBegin(GL_POINTS);
			for (int i = 0; i < m_meshes[selectedIndex]->getVertices().size(); i++) {
				if (m_meshes[selectedIndex]->getVertices()[i]->getVertexType() == edgeVertex) {
					glColor3f(0.25f, 0.80f, 0.12f);
				}
				else if (m_meshes[selectedIndex]->getVertices()[i]->getVertexType() == gridVertex) {
					glColor3f(0.25f, 0.3f, 1.0f);
				}
				else {
					glColor3f(0.8f, 0.8f, 0.8f);
				}
				glVertex3f(m_meshes[selectedIndex]->getVertices()[i]->getPosition().x, m_meshes[selectedIndex]->getVertices()[i]->getPosition().y, 
							m_meshes[selectedIndex]->getVertices()[i]->getPosition().z);
			}
			glEnd();
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Face, false>::drawCutCell(uint cutCellIndex) {
			glBegin(GL_LINES);
			auto edges = m_pCutCells->getCutCell(cutCellIndex).getHalfEdges();
			for (int j = 0; j < edges.size(); j++) {
				const VectorType &v1 = edges[j]->getVertices().first->getPosition();
				const VectorType &v2 = edges[j]->getVertices().second->getPosition();

				glVertex3f(v1.x, v1.y, v1.z);
				glVertex3f(v2.x, v2.y, v2.z);
			}
			glEnd();
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Face, false>::drawEdgeNormals(uint cutCellIndex) {
			auto edges = m_pCutCells->getCutCell(cutCellIndex).getHalfEdges();
			for (int j = 0; j < edges.size(); j++) {
				const VectorType &v1 = edges[j]->getVertices().first->getPosition();
				const VectorType &v2 = edges[j]->getVertices().second->getPosition();
				VectorType edgeCentroid = edges[j]->getEdge()->getCentroid();
				RenderingUtils::getInstance()->drawVector(edgeCentroid, edgeCentroid + edges[j]->getNormal()*0.01);
			}
		}
		#pragma endregion

		template class MeshRendererT<Vector2, Face, isVector2<Vector2>::value>;
		template class MeshRendererT<Vector2D, Face, isVector2<Vector2D>::value>;
		
		template class MeshRendererT<Vector3, Face, isVector2<Vector3>::value>;
		template class MeshRendererT<Vector3D, Face, isVector2<Vector3D>::value>;
	}
}