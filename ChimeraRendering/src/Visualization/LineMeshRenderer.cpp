#include "Visualization/LineMeshRenderer.h"

namespace Chimera {

	namespace Rendering {

		#pragma region 2DRenderer
		template<class VectorType>
		void MeshRendererT<VectorType, Edge, true>::drawElements(uint selectedIndex, Color color /*= Color::BLACK*/) {
			m_drawLineMeshes = true;
			if (m_drawLineMeshes) {
				glDisable(GL_LIGHTING);

				
				glColor3f(0.1f, 0.1f, 0.1f);
				for (int i = 0; i < m_meshes[selectedIndex]->getElements().size(); i++) {
					glBegin(GL_LINES);
					const VectorType &v1 = m_meshes[selectedIndex]->getElement(i)->getVertex1()->getPosition();
					const VectorType &v2 = m_meshes[selectedIndex]->getElement(i)->getVertex2()->getPosition();
					glVertex2f(v1.x, v1.y);
					glVertex2f(v2.x, v2.y);
					glEnd();
				}

			}
			
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Edge, true>::drawMeshVertices(uint selectedIndex, Color color /*= Color::BLACK*/) {
			glEnable(GL_POINT_SMOOTH);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			glPointSize(8.0f);
			glBegin(GL_POINTS);
			for (int i = 0; i < m_meshes[selectedIndex]->getVertices().size(); i++) {
				if (m_meshes[selectedIndex]->getVertices()[i]->getVertexType() == edgeVertex) {
					glColor3f(0.25f, 0.80f, 0.12f);
				}
				else {
					glColor3f(0.8f, 0.8f, 0.8f);
				}
				glVertex2f(m_meshes[selectedIndex]->getVertices()[i]->getPosition().x, m_meshes[selectedIndex]->getVertices()[i]->getPosition().y);
			}
			glEnd();
		}

		#pragma endregion
		#pragma region 3DRenderer
		template<class VectorType>
		void MeshRendererT<VectorType, Edge, false>::drawElements(uint selectedIndex, Color color /*= Color::BLACK*/) {
			if (m_drawLineMeshes) {
				glDisable(GL_LIGHTING);

				glColor3f(m_elementColor.getRed(), m_elementColor.getGreen(), m_elementColor.getBlue());
				for (int i = 0; i < m_meshes[selectedIndex]->getElements().size(); i++) {
					glBegin(GL_LINES);
					const VectorType &v1 = m_meshes[selectedIndex]->getElement(i)->getVertex1()->getPosition();
					const VectorType &v2 = m_meshes[selectedIndex]->getElement(i)->getVertex2()->getPosition();
					glVertex3f(v1.x, v1.y, v1.z);
					glVertex3f(v2.x, v2.y, v2.z);
					glEnd();
				}
			}
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Edge, false>::drawMeshVertices(uint selectedIndex, Color color /*= Color::BLACK*/) {
			glEnable(GL_POINT_SMOOTH);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			glPointSize(16.0f);
			glBegin(GL_POINTS);
			for (int i = 0; i < m_meshes[selectedIndex]->getVertices().size(); i++) {
				if (m_meshes[selectedIndex]->getVertices()[i]->getVertexType() == edgeVertex) {
					glColor3f(m_vertexColor.getRed(), m_vertexColor.getGreen(), m_vertexColor.getBlue());
				}
				else {
					glColor3f(0.1f, 0.1f, 0.1f);
				}
				glVertex3f(	m_meshes[selectedIndex]->getVertices()[i]->getPosition().x,
							m_meshes[selectedIndex]->getVertices()[i]->getPosition().y,
							m_meshes[selectedIndex]->getVertices()[i]->getPosition().z);
			}
			glEnd();
		}
		#pragma endregion

		template class MeshRendererT<Vector2, Edge, isVector2<Vector2>::value>;
		template class MeshRendererT<Vector2D, Edge, isVector2<Vector2D>::value>;

		template class MeshRendererT<Vector3, Edge, isVector2<Vector3>::value>;
		template class MeshRendererT<Vector3D, Edge, isVector2<Vector3D>::value>;
	}

}