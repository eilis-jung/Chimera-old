#include "Visualization/MeshRenderer3D.h"

namespace Chimera {
	namespace Rendering {

		#pragma region Constructors
		template <class VectorType, template <class> class ElementType>
		MeshRenderer3D<VectorType, ElementType>::MeshRenderer3D(const vector<Mesh<VectorType, ElementType> *> &g_meshes, const Vector3 &cameraPosition) : m_meshes(g_meshes),
																							m_cameraPosition(cameraPosition) {
			m_draw = true;
			m_drawVertices = false;
			m_selectedIndex = 0;
			initializeVBOs();
			initializeShaders();
		}
		#pragma endregion

		#pragma region InitializationFunctions
		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::initializeWindowControls(BaseWindow *pBaseWindow) {
			TwBar *pTwBar = pBaseWindow->getBaseTwBar();
		}

		#pragma endregion

		#pragma region Functionalities
		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::draw() {
			if (m_draw) {
				for (int i = 0; i < m_meshes.size(); i++) {
					if (m_meshes[i]->drawMesh()) {
						drawMesh(i);
					}
				}
			}
			if (m_drawVertices) {
				for (int i = 0; i < m_meshes.size(); i++) {
					drawMeshVertices(i);
				}
			}
			
 		}		
		#pragma endregion

		#pragma region PrivateFunctionalities
		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::initializeVBOs() {
			for (int i = 0; i < m_meshes.size(); i++) {
				initializeVBO(i);
			}
		}

		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::initializeVBO(unsigned int meshIndex) {
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

		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::initializeShaders() {
			m_pWireframeShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/Wireframe.frag");
			m_pPhongShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/PhongShading.frag");
			m_pPhongWireframeShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/PhongShadingWireframe.frag");
			m_lightPosLoc = glGetUniformLocation(m_pPhongShader->getProgramID(), "lightPos");
			m_lightPosLocWire = glGetUniformLocation(m_pPhongWireframeShader->getProgramID(), "lightPos");
		}

		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::drawMesh(int selectedIndex) {
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
			glDepthMask(GL_TRUE);
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT0);
			//m_pPhongShader->applyShader();
			//glUniform3f(m_lightPosLoc, m_cameraPosition.x, m_cameraPosition.y, m_cameraPosition.z);
			GLfloat light_position[] = { m_cameraPosition.x, m_cameraPosition.y, m_cameraPosition.z, 0.0 };
			m_pPhongWireframeShader->applyShader();
			glUniform3f(m_lightPosLocWire, m_cameraPosition.x, m_cameraPosition.y, m_cameraPosition.z);
			glLightfv(GL_LIGHT0, GL_POSITION, light_position);

			//glEnable(GL_LIGHTING);
			//glEnable(GL_LIGHT0);
			glColor3f(0.5, 0.5, 0.5);
			glLineWidth(1.0f);

			drawPolygons(selectedIndex);

			m_pPhongWireframeShader->removeShader();
			//m_pPhongShader->removeShader();
			//drawMeshNormals(selectedIndex);
			//drawMeshFaceNormals(selectedIndex);

			glDisable(GL_LIGHTING);
			glDisable(GL_LIGHT0);

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}

		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::drawMeshVertices(int selectedIndex) {
			glDisable(GL_LIGHTING);
			glDisable(GL_LIGHT0);
			glColor3f(0.0f, 0.0f, 0.0f);
			glPointSize(8.0f);
			glBegin(GL_POINTS);
			for (int i = 0; i < m_meshes[selectedIndex]->getVertices().size(); i++) {
				glVertex3f(m_meshes[selectedIndex]->getVertices()[i]->getPosition().x,
					m_meshes[selectedIndex]->getVertices()[i]->getPosition().y,
					m_meshes[selectedIndex]->getVertices()[i]->getPosition().z);
			}
			glEnd();
		}
		
		template <>
		void MeshRenderer3D<Vector3, HalfFace>::drawPolygons(int selectedIndex) {
			auto polygons = m_meshes[selectedIndex]->getElements();
			//auto vertices = m_meshes[selectedIndex]->getVertices();
			for (int i = 0; i < polygons.size(); i++) {
				glBegin(GL_POLYGON);
				for (int j = 0; j < polygons[i]->getHalfEdges().size(); j++) {
					auto pVertex = polygons[i]->getHalfEdges()[j]->getVertices().first;
					Vector3 vertexNormal = pVertex->getNormal();
					Vector3 vertexPosition = pVertex->getPosition();
					glNormal3f(vertexNormal.x, vertexNormal.y, vertexNormal.z);
					glVertex3f(vertexPosition.x, vertexPosition.y, vertexPosition.z);
				}
				glEnd();
			}
		}

		template <>
		void MeshRenderer3D<Vector3D, HalfFace>::drawPolygons(int selectedIndex) {
			auto polygons = m_meshes[selectedIndex]->getElements();
			//auto vertices = m_meshes[selectedIndex]->getVertices();
			for (int i = 0; i < polygons.size(); i++) {
				glBegin(GL_POLYGON);
				for (int j = 0; j < polygons[i]->getHalfEdges().size(); j++) {
					auto pVertex = polygons[i]->getHalfEdges()[j]->getVertices().first;
					Vector3D vertexNormal = pVertex->getNormal();
					Vector3D vertexPosition = pVertex->getPosition();
					glNormal3f(vertexNormal.x, vertexNormal.y, vertexNormal.z);
					glVertex3f(vertexPosition.x, vertexPosition.y, vertexPosition.z);
				}
				glEnd();
			}
		}


		template <>
		void MeshRenderer3D<Vector3, Face>::drawPolygons(int selectedIndex) {
			auto polygons = m_meshes[selectedIndex]->getElements();
			//auto vertices = m_meshes[selectedIndex]->getVertices();
			for (int i = 0; i < polygons.size(); i++) {
				auto pHalfFace = polygons[i]->getHalfFaces().front();
				glBegin(GL_POLYGON);
				for (int j = 0; j < pHalfFace->getHalfEdges().size(); j++) {
					auto pVertex = pHalfFace->getHalfEdges()[j]->getVertices().first;
					Vector3 vertexNormal = pVertex->getNormal();
					Vector3 vertexPosition = pVertex->getPosition();
					glNormal3f(vertexNormal.x, vertexNormal.y, vertexNormal.z);
					glVertex3f(vertexPosition.x, vertexPosition.y, vertexPosition.z);
				}
				glEnd();
			}
		}

		template <>
		void MeshRenderer3D<Vector3D, Face>::drawPolygons(int selectedIndex) {
			auto polygons = m_meshes[selectedIndex]->getElements();
			//auto vertices = m_meshes[selectedIndex]->getVertices();
			for (int i = 0; i < polygons.size(); i++) {
				auto pHalfFace = polygons[i]->getHalfFaces().front();
				glBegin(GL_POLYGON);
				for (int j = 0; j < pHalfFace->getHalfEdges().size(); j++) {
					auto pVertex = pHalfFace->getHalfEdges()[j]->getVertices().first;
					Vector3D vertexNormal = pVertex->getNormal();
					Vector3D vertexPosition = pVertex->getPosition();
					glNormal3f(vertexNormal.x, vertexNormal.y, vertexNormal.z);
					glVertex3f(vertexPosition.x, vertexPosition.y, vertexPosition.z);
				}
				glEnd();
			}
		}
		
		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::drawTrianglesVBOs(int selectedIndex) {
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
		
		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::drawMeshNormals(int selectedIndex) {

			for (int i = 0; i < m_meshes[selectedIndex]->getVertices().size(); i++) {
				VectorType vertexNormal = m_meshes[selectedIndex]->getVertices()[i]->getNormal();
				VectorType vertexPosition = m_meshes[selectedIndex]->getVertices()[i]->getPosition();
				RenderingUtils::getInstance()->drawVector(vertexPosition, vertexPosition + vertexNormal*0.1);
			}

 			/*auto meshPolygons = m_meshes[selectedIndex]->getMeshPolygons();
			auto meshPoints = m_meshes[selectedIndex]->getPoints();
			auto verticesNormals = m_meshes[selectedIndex]->getPointsNormals();
			for (int i = 0; i < verticesNormals.size(); i++) {
				Vector3 iniPoint = convertToVector3F(meshPoints[i]);
				Vector3 finalPoint = convertToVector3F(meshPoints[i] + verticesNormals[i] * 0.05);
				RenderingUtils::getInstance()->drawVector(iniPoint, finalPoint);
			}*/

		}

		template <class VectorType, template <class> class ElementType>
		void MeshRenderer3D<VectorType, ElementType>::drawMeshFaceNormals(int selectedIndex) {
			auto polygons = m_meshes[selectedIndex]->getElements();
			/*for (int i = 0; i < polygons.size(); i++) {
				RenderingUtils::getInstance()->drawVector(polygons[i]->getCentroid(), polygons[i]->getCentroid() + polygons[i]->getNormal()*0.1);
			}*/
		}



		template class MeshRenderer3D<Vector3, HalfFace>;
		template class MeshRenderer3D<Vector3D, HalfFace>;

		template class MeshRenderer3D<Vector3, Face>;
		template class MeshRenderer3D<Vector3D, Face>;
		#pragma endregion
	
	}
}