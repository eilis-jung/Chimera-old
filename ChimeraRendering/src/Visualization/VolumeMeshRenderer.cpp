#include "Visualization/VolumeMeshRenderer.h"

namespace Chimera {

	namespace Rendering {

		#pragma region DrawingFunctions
		template<class VectorType>
		void MeshRendererT<VectorType, Volume, false>::drawElements(uint selectedIndex, Color color /*= Color::BLACK*/) {
			glColor3f(color.getRed(), color.getGreen(), color.getBlue());
			if (m_selectedCutVoxel != -1 && m_drawSelectedCutVoxel && m_selectedCutVoxel < m_pCutVoxels->getNumberCutVoxels()) {
				glLineWidth(4.0f);
				drawCutVoxel(m_selectedCutVoxel);
				if (m_drawNeighbors) {
					auto cutVoxel = m_pCutVoxels->getCutVoxel(m_selectedCutVoxel);
					auto faces = m_pCutVoxels->getCutVoxel(m_selectedCutVoxel).getHalfFaces();
					for (int i = 0; i < faces.size(); i++) {
						if (faces[i]->getLocation() != geometryHalfFace) {
							const vector<HalfVolume<VectorType> *> &connectedHalfVolumes = faces[i]->getFace()->getConnectedHalfVolumes();
							if (connectedHalfVolumes.size() > 1) {
								uint halfFaceIndex = connectedHalfVolumes[0]->getID() == m_selectedCutVoxel ? connectedHalfVolumes[1]->getID() : connectedHalfVolumes[0]->getID();
								drawCutVoxel(halfFaceIndex);
							}
						}
					}
				}
			}

			if (m_drawCutVoxelNormals) {
				drawCutVoxelNormals();
			}
			if (m_drawCutFacesCentroids) {
				drawCutFacesCentroid(Color(255, 72, 48));
			}

			if (m_drawCutVoxels) {
				glLineWidth(1.5f);
				glColor3f(m_cutVoxelsColor.getRed(), m_cutVoxelsColor.getGreen(), m_cutVoxelsColor.getBlue());

				for (int i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {		
					drawCutVoxel(i);
				}
			}

			if (m_drawVertexHalfFaces) {
				auto cutVoxel = m_pCutVoxels->getCutVoxel(m_selectedCutVoxel);
				auto iter = cutVoxel.getOnEdgeVerticesMap().begin();
				uint tempCounter = 0;
				for (; iter != cutVoxel.getOnEdgeVerticesMap().end(); iter++) {
					if(tempCounter == m_selectedVertex)
						break;
					tempCounter++;
				}
				
			
				
				if (iter != cutVoxel.getOnEdgeVerticesMap().end()) {
					Vertex<VectorType> *pVertex = iter->second;

					glPointSize(12.0f);
					glColor3f(0.0f, 0.0f, 0.0f);
					glBegin(GL_POINTS);
					glVertex3f(pVertex->getPosition().x, pVertex->getPosition().y, pVertex->getPosition().z);
					glEnd();

					glLineWidth(4.0f);
					glColor3f(m_nodeNeighborsColor.getRed(), m_nodeNeighborsColor.getGreen(), m_nodeNeighborsColor.getBlue());
					for (int i = 0; i < pVertex->getConnectedHalfFaces().size(); i++) {
						HalfFace<VectorType> *pHalfFace = pVertex->getConnectedHalfFaces()[i];
						drawCutFace(pHalfFace);
						RenderingUtils::getInstance()->drawVector(pHalfFace->getCentroid(), pHalfFace->getCentroid() + pHalfFace->getNormal()*0.05);
					}

					
				}
			}

			//Draw line meshes added to the object 
			glLineWidth(10.0f);
			for (int i = 0; i < m_lineRenderers.size(); i++) {
				m_lineRenderers[i]->draw();
			}
			glLineWidth(1.0f);

			//Draw cutcells 
			for (int i = 0; i < m_cutCellsRenderers.size(); i++) {
				m_cutCellsRenderers[i]->draw();
			}
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Volume, false>::drawMeshVertices(uint selectedIndex, Color color /*= Color::BLACK*/) {
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
		void MeshRendererT<VectorType, Volume, false>::drawCutVoxelNormals(Color color /*= Color::BLACK*/) {
			if (m_selectedCutVoxel != -1) {
				auto cutVoxel = m_pCutVoxels->getCutVoxel(m_selectedCutVoxel);
				auto faces = m_pCutVoxels->getCutVoxel(m_selectedCutVoxel).getHalfFaces();
				for (int i = 0; i < faces.size(); i++) {

					RenderingUtils::getInstance()->drawVector(faces[i]->getCentroid(), faces[i]->getCentroid() + faces[i]->getNormal()*0.05);
				}
			}
			
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Volume, false>::drawCutFacesCentroid(Color color /*= Color::BLACK*/) {
			if (m_selectedCutVoxel != -1 && m_selectedCutVoxel < m_pCutVoxels->getNumberCutVoxels()) {
				auto cutVoxel = m_pCutVoxels->getCutVoxel(m_selectedCutVoxel);
				auto faces = m_pCutVoxels->getCutVoxel(m_selectedCutVoxel).getHalfFaces();
				glColor3f(color.getRed(), color.getGreen(), color.getBlue());
				glPointSize(6.0f);
				glBegin(GL_POINTS);
				for (int i = 0; i < faces.size(); i++) {
					glVertex3f(faces[i]->getCentroid().x, faces[i]->getCentroid().y, faces[i]->getCentroid().z);
				}
				glEnd();
			}

		}
		#pragma endregion

		#pragma region InitializationFunctions
		template <class VectorType>
		void MeshRendererT<VectorType, Volume, false>::initializeLineRenderers() {
			//XY Line Meshes
			{
				const map<uint, vector<LineMesh<VectorType> *>> &xyLineMeshes = m_pCutVoxels->getXYLineMeshes();
				vector<Mesh <VectorType, Edge> *> meshesVec;
				for (auto iter = xyLineMeshes.begin(); iter != xyLineMeshes.end(); iter++) {
					for (int i = 0; i < iter->second.size(); i++) {
						meshesVec.push_back(iter->second[i]);
					}
				}
				m_lineRenderers.push_back(new LineMeshRenderer<VectorType>(meshesVec, m_cameraPosition));
				m_lineRenderers.back()->setElementColor(Color(255, 211, 2));
				m_lineRenderers.back()->setDrawingVertices(false);
				m_lineRenderers.back()->setDrawing(false);
			}
			
			//XZ Line Meshes 
			{
				const map<uint, vector<LineMesh<VectorType> *>> &xzLineMeshes = m_pCutVoxels->getXZLineMeshes();
				vector<Mesh <VectorType, Edge> *> meshesVec;
				for (auto iter = xzLineMeshes.begin(); iter != xzLineMeshes.end(); iter++) {
					for (int i = 0; i < iter->second.size(); i++) {
						meshesVec.push_back(iter->second[i]);
					}
				}
				m_lineRenderers.push_back(new LineMeshRenderer<VectorType>(meshesVec, m_cameraPosition));
				m_lineRenderers.back()->setElementColor(Color(33, 176, 71));
				m_lineRenderers.back()->setDrawingVertices(true);
				m_lineRenderers.back()->setDrawing(false);
			}

			//YZ Line Meshes 
			{
				const map<uint, vector<LineMesh<VectorType> *>> &yzLineMeshes = m_pCutVoxels->getYZLineMeshes();
				vector<Mesh <VectorType, Edge> *> meshesVec;
				for (auto iter = yzLineMeshes.begin(); iter != yzLineMeshes.end(); iter++) {
					for (int i = 0; i < iter->second.size(); i++) {
						meshesVec.push_back(iter->second[i]);
					}
				}
				m_lineRenderers.push_back(new LineMeshRenderer<VectorType>(meshesVec, m_cameraPosition));
				m_lineRenderers.back()->setElementColor(Color(13, 128, 255));
				m_lineRenderers.back()->setDrawingVertices(false);
				m_lineRenderers.back()->setDrawing(false);
			}
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Volume, false>::initializeCutCellsRenderers() {

			auto xyCutcells = m_pCutVoxels->getXYCutCells();
			for (auto iter = xyCutcells.begin(); iter != xyCutcells.end(); iter++) {
				vector<Mesh <VectorType, Face> *> faceMeshesVec;
				faceMeshesVec.push_back(iter->second);
				PolygonMeshRenderer<VectorType> *pPolyMeshRenderer = new PolygonMeshRenderer<VectorType>(faceMeshesVec, m_cameraPosition);
				m_cutCellsRenderers.push_back(pPolyMeshRenderer);
				pPolyMeshRenderer->setSelectedCutCell(-1);
				pPolyMeshRenderer->setDrawingCutCells(false);
			}

			auto yzCutCells = m_pCutVoxels->getYZCutCells();
			for (auto iter = yzCutCells.begin(); iter != yzCutCells.end(); iter++) {
				vector<Mesh <VectorType, Face> *> faceMeshesVec;
				faceMeshesVec.push_back(iter->second);
				PolygonMeshRenderer<VectorType> *pPolyMeshRenderer = new PolygonMeshRenderer<VectorType>(faceMeshesVec, m_cameraPosition);
				m_cutCellsRenderers.push_back(pPolyMeshRenderer);
				pPolyMeshRenderer->setSelectedCutCell(-1);
				pPolyMeshRenderer->setDrawingCutCells(false);
			}

			auto xzCutcells = m_pCutVoxels->getXZCutCells();
			for (auto iter = xzCutcells.begin(); iter != xzCutcells.end(); iter++) {
				vector<Mesh <VectorType, Face> *> faceMeshesVec;
				faceMeshesVec.push_back(iter->second);
				PolygonMeshRenderer<VectorType> *pPolyMeshRenderer = new PolygonMeshRenderer<VectorType>(faceMeshesVec, m_cameraPosition);
				m_cutCellsRenderers.push_back(pPolyMeshRenderer);
				pPolyMeshRenderer->setSelectedCutCell(-1);
				pPolyMeshRenderer->setDrawingCutCells(false);
			}

		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template <class VectorType>
		void MeshRendererT<VectorType, Volume, false>::drawCutVoxel(uint cutVoxelIndex) {
			auto cutVoxel = m_pCutVoxels->getCutVoxel(cutVoxelIndex);
			auto faces = m_pCutVoxels->getCutVoxel(cutVoxelIndex).getHalfFaces();
			for (int i = 0; i < faces.size(); i++) {
				drawCutFace(faces[i]);
			}
		}

		template <class VectorType>
		void MeshRendererT<VectorType, Volume, false>::drawCutFace(HalfFace<VectorType> *pHalfFace) {
			glBegin(GL_LINES);
			auto edges = pHalfFace->getHalfEdges();
			for (int j = 0; j < edges.size(); j++) {
				const VectorType &v1 = edges[j]->getVertices().first->getPosition();
				const VectorType &v2 = edges[j]->getVertices().second->getPosition();

				glVertex3f(v1.x, v1.y, v1.z);
				glVertex3f(v2.x, v2.y, v2.z);
			}
			glEnd();
		}
		#pragma endregion
		template class MeshRendererT<Vector3, Volume, isVector2<Vector3>::value>;
		template class MeshRendererT<Vector3D, Volume, isVector2<Vector3D>::value>;
	}
}