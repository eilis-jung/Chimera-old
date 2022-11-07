#include "Visualization/MeshRenderer.h"

namespace Chimera {
	namespace Rendering {

		#pragma region Constructors
		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		MeshRendererBase<ChildT, VectorType, ElementType, isVector2>::MeshRendererBase(const vector<Mesh<VectorType, ElementType> *> &g_meshes, const Vector3 &cameraPosition) : m_meshes(g_meshes),
																							m_cameraPosition(cameraPosition) {
			m_selectedIndex = 0;
			m_drawMeshVertices = m_drawMeshNormals = false;
			m_draw = true;
			initializeVBOs();
			initializeShaders();
			m_vectorLengthSize = 0.05;

			m_vertexColor = Color(63, 204, 30);
			m_elementColor = Color(35, 35, 35);
		}
		#pragma endregion

		#pragma region Functionalities
		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		void MeshRendererBase<ChildT, VectorType, ElementType, isVector2>::draw() {
			if (m_draw) {
				for (int i = 0; i < m_meshes.size(); i++) {
					if (m_meshes[i] && m_meshes[i]->drawMesh()) {
						drawMesh(i);
					}
				}
			}
 		}		
		#pragma endregion

		#pragma region PrivateFunctionalities
		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		void MeshRendererBase<ChildT, VectorType, ElementType, isVector2>::initializeVBOs() {
			for (int i = 0; i < m_meshes.size(); i++) {
				initializeVBO(i);
			}
		}

		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		void MeshRendererBase<ChildT, VectorType, ElementType, isVector2>::initializeVertexVBO(uint meshIndex) {
			if (m_meshes.size() > 0 && m_meshes[meshIndex] && m_meshes[meshIndex]->getPoints().size() > 0) {
				GLuint *pVertexVBO = new GLuint();
				glGenBuffers(1, pVertexVBO);
				glBindBuffer(GL_ARRAY_BUFFER, *pVertexVBO);
				m_pVerticesVBOs.push_back(pVertexVBO);
				void *pVertices = reinterpret_cast<void *>(&m_meshes[meshIndex]->getPoints()[0]);
				unsigned int sizeVertices = m_meshes[meshIndex]->getPoints().size() * sizeof(VectorType);
				glBufferData(GL_ARRAY_BUFFER, sizeVertices, pVertices, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}
			
		}

		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		void MeshRendererBase<ChildT, VectorType, ElementType, isVector2>::initializeVBO(unsigned int meshIndex) {
			initializeVertexVBO(meshIndex);

		/*	GLuint *pVertexVBO = new GLuint();
			glGenBuffers(1, pVertexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *pVertexVBO);
			m_pVerticesVBOs.push_back(pVertexVBO);
			void *pVertices = reinterpret_cast<void *>(&m_meshes[meshIndex]->getPoints()[0]);
			unsigned int sizeVertices = m_meshes[meshIndex]->getPoints().size() * sizeof(Vector3);
			glBufferData(GL_ARRAY_BUFFER, sizeVertices, pVertices, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			void *pNormals = reinterpret_cast<void *>(&m_meshes[meshIndex]->getPointsNormals()[0]);
			GLuint *pNormalVBO = new GLuint();
			glGenBuffers(1, pNormalVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *pNormalVBO);
			m_pNormalsVBOs.push_back(pNormalVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeVertices, pNormals, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			GLuint *pIndicesVBO = new GLuint();
			glGenBuffers(1, pIndicesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *pIndicesVBO);
			m_pIndicesVBOs.push_back(pIndicesVBO);
*/
			/*vector<int> meshIndices;
			auto faces = m_meshes[meshIndex]->getMeshPolygons();
			for (int i = 0; i < faces.size(); i++) {
				for (int j = 0; j < faces[i].edges.size(); j++) {
					meshIndices.push_back(faces[i].edges[j].first);
				}
			}*/

			/*m_numElementsToDraw.push_back(meshIndices.size());
			unsigned int sizeIndices = meshIndices.size() * sizeof(int);
			void *pIndices = reinterpret_cast<void *>(&meshIndices[0]);

			glBufferData(GL_ARRAY_BUFFER, sizeIndices, pIndices, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);*/
		}

		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		void MeshRendererBase<ChildT, VectorType, ElementType, isVector2>::initializeShaders() {
			m_pWireframeShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/Wireframe.frag");
			m_pPhongShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/PhongShading.frag");
			m_pPhongWireframeShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/PhongShadingWireframe.frag");
			m_lightPosLoc = glGetUniformLocation(m_pPhongShader->getProgramID(), "lightPos");
			m_lightPosLocWire = glGetUniformLocation(m_pPhongWireframeShader->getProgramID(), "lightPos");
		}

		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		void MeshRendererBase<ChildT, VectorType, ElementType, isVector2>::drawMesh(int selectedIndex) {
			drawElements(selectedIndex);
			if (m_drawMeshVertices)
				drawMeshVertices(selectedIndex);
			if (m_drawMeshNormals)
				drawMeshNormals(selectedIndex);

		}
		
		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		void MeshRendererBase<ChildT, VectorType, ElementType, isVector2>::drawTrianglesVBOs(int selectedIndex) {
			glEnableClientState(GL_VERTEX_ARRAY);
			glColor3f(0.21f, 0.54f, 1.0f);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVerticesVBOs[selectedIndex]);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pIndicesVBOs[selectedIndex]);
			glVertexPointer(3, GL_FLOAT, 0, 0);
			glBindBuffer(GL_NORMAL_ARRAY, *m_pNormalsVBOs[selectedIndex]);
			glNormalPointer(3, GL_FLOAT, 0);
			glDrawElements(GL_TRIANGLES, m_numElementsToDraw[selectedIndex], GL_UNSIGNED_INT, 0);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);
		}
		
		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		void MeshRendererBase<ChildT, VectorType, ElementType, isVector2>::drawMeshNormals(uint selectedIndex) {
			for (int i = 0; i < m_meshes[selectedIndex]->getVertices().size(); i++) {
				VectorType iniPoint = m_meshes[selectedIndex]->getVertices()[i]->getPosition();
				VectorType finalPoint = iniPoint + m_meshes[selectedIndex]->getVertices()[i]->getNormal()* m_vectorLengthSize;
				RenderingUtils::getInstance()->drawVector(iniPoint, finalPoint);
			}
		}

		
		template class MeshRendererBase<MeshRendererT<Vector2, Edge, isVector2<Vector2>::value>, Vector2, Edge, isVector2<Vector2>::value>;
		template class MeshRendererBase<MeshRendererT<Vector2D, Edge, isVector2<Vector2D>::value>, Vector2D, Edge, isVector2<Vector2D>::value>;
		template class MeshRendererBase<MeshRendererT<Vector3, Edge, isVector2<Vector3>::value>, Vector3, Edge, isVector2<Vector3>::value>;
		template class MeshRendererBase<MeshRendererT<Vector3D, Edge, isVector2<Vector3D>::value>, Vector3D, Edge, isVector2<Vector3D>::value>;

		template class MeshRendererBase<MeshRendererT<Vector2, Face, isVector2<Vector2>::value>, Vector2, Face, isVector2<Vector2>::value>;
		template class MeshRendererBase<MeshRendererT<Vector2D, Face, isVector2<Vector2D>::value>, Vector2D, Face, isVector2<Vector2D>::value>;
		template class MeshRendererBase<MeshRendererT<Vector3, Face, isVector2<Vector3>::value>, Vector3, Face, isVector2<Vector3>::value>;
		template class MeshRendererBase<MeshRendererT<Vector3D, Face, isVector2<Vector3D>::value>, Vector3D, Face, isVector2<Vector3D>::value>;

		template class MeshRendererBase<MeshRendererT<Vector3, Volume, isVector2<Vector3>::value>, Vector3, Volume, isVector2<Vector3>::value>;
		template class MeshRendererBase<MeshRendererT<Vector3D, Volume, isVector2<Vector3D>::value>, Vector3D, Volume, isVector2<Vector3D>::value>;

		#pragma endregion
	
	}
}