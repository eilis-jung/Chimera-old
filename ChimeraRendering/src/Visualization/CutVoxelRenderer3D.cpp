//#include "Visualization/CutVoxelRenderer3D.h"
//#include "Resources/ResourceManager.h"
//
//namespace Chimera {
//	namespace Rendering {
//
//		#pragma region Constructors
//		CutVoxelRenderer3D::CutVoxelRenderer3D(CutCells3D *pCutCells, LinearInterpolant3D<Vector3> *pLinearInterpolant, const Scalar * pVelocityScale, const string &gridName) {
//			m_pCutCells = pCutCells;
//			m_pLinearInterpolant = pLinearInterpolant;
//			m_pNodeVelocityField = m_pLinearInterpolant->getParams().pNodeVelocityField;
//			m_pVelocityScale = pVelocityScale;
//			m_gridName = gridName;
//
//			m_drawSelectedVoxel = true;
//			m_selectedCell = 0;
//			m_numSelectedVoxels = 1;
//			m_drawFaceNormals = false;
//			m_drawOnTheFaceLines = false;
//
//			/** Velocities */
//			m_drawInteriorVelocities = false;
//			m_dimVelocityInterpolationSubdivision = 0;
//			m_drawNodeVelocities = m_drawFaceVelocities = m_drawMixedNodeFaces = false;
//			m_drawGeometryVelocities = m_drawAllVelocities = false;
//
//			/** MIced nodes */
//			m_selectedMixedNode = 0;
//			m_drawMixedNodeFaces = false;
//
//		}
//		#pragma endregion
//
//		#pragma region InitializationFunctions
//		void CutVoxelRenderer3D::initializeWindowControls(GridVisualizationWindow<Vector3> *pGridVisualizationWindow) {
//			TwBar *pTwBar = pGridVisualizationWindow->getBaseTwBar();
//			m_pGridVisualizationWindow = pGridVisualizationWindow;
//
//			/** MIced nodes */
//			m_selectedMixedNode = 0;
//			m_drawMixedNodeFaces = false;
//
//			TwAddVarRW(pTwBar, "drawSelectedVoxel", TW_TYPE_BOOL8, &m_drawSelectedVoxel, " label='Draw Selected Voxel' group='Cut-Voxel'");
//			TwAddVarRW(pTwBar, "selectedVoxelID", TW_TYPE_INT32, &m_selectedCell, " label='Selected Voxel Index' group='Cut-Voxel' refresh=0.1");
//			TwAddVarRW(pTwBar, "selectedVoxelNumber", TW_TYPE_INT32, &m_numSelectedVoxels, " label='Number of Selected Voxels' group='Cut-Voxel'");
//			TwAddVarRW(pTwBar, "drawFaceNormals", TW_TYPE_BOOL8, &m_drawFaceNormals, " label='Draw Faces Normals' group='Cut-Voxel'");
//			TwAddVarRW(pTwBar, "drawOnTheFaceLines", TW_TYPE_BOOL8, &m_drawOnTheFaceLines, " label='Draw on Face Lines' group='Cut-Voxel'");
//
//			/** Velocities */
//			TwAddVarRW(pTwBar, "drawFaceVelocities", TW_TYPE_BOOL8, &m_drawFaceVelocities, " label='Draw Face Velocities' help='Draw face velocities' group='Velocity'");
//			TwAddVarRW(pTwBar, "drawInteriorVelocities", TW_TYPE_BOOL8, &m_drawInteriorVelocities, " label='Draw Interior Velocities' group='Velocity'");
//			TwAddVarRW(pTwBar, "drawVelocitiesInterpolationSub", TW_TYPE_INT32, &m_dimVelocityInterpolationSubdivision, " label='Interior Velocities Subdivision' group='Velocity'");
//			TwAddVarRW(pTwBar, "drawNodeVelocities", TW_TYPE_BOOL8, &m_drawNodeVelocities, " label='Draw Node Velocities' help='Draw node velocities' group='Velocity'");
//			TwAddVarRW(pTwBar, "drawGeometryVelocities", TW_TYPE_BOOL8, &m_drawGeometryVelocities, " label='Draw Geometry Velocities' group='Velocity'");
//			TwAddVarRW(pTwBar, "drawAllVelocities", TW_TYPE_BOOL8, &m_drawAllVelocities, " label='Draw All Velocities' group='Velocity'");
//			string defName = "'" + m_gridName + "'/'Velocity' group='Cut-Voxel'";
//			TwDefine(defName.c_str());
//
//			/** Mixed nodes */
//			TwAddVarRW(pTwBar, "drawMixedNodeFaces", TW_TYPE_BOOL8, &m_drawMixedNodeFaces, " label='Draw mixed node faces' group='Mixed-Node'");
//			TwAddVarRW(pTwBar, "drawMixedNodeIthFace", TW_TYPE_INT32, &m_selectedMixedNode, " label='Ith mixed node to draw' group='Mixed-Node'");
//			defName = "'" + m_gridName + "'/'Mixed-Node' group='Cut-Voxel'";
//			TwDefine(defName.c_str());
//		}
//		#pragma endregion
//
//		#pragma region Functionalities
//		void CutVoxelRenderer3D::draw() {
//			if (m_drawSelectedVoxel) {
//				drawSelectedCells();
//			}
//			if (m_drawFaceNormals) {
//				drawFaceNormals();
//			}
//			if (m_drawOnTheFaceLines) {
//				drawOnTheFaceLines();
//			}
//			
//			/** Velocities */
//			if (m_drawFaceVelocities) {
//				drawFaceVelocities();
//			}
//			if (m_drawNodeVelocities) {
//				drawNodeVelocities();
//			}
//			if (m_drawMixedNodeVelocities) {
//				drawMixedNodeVelocities();
//			}
//			if (m_drawGeometryVelocities) {
//				drawGeometryVelocities();
//			}
//			if (m_drawAllVelocities) {
//				drawAllVelocities();
//			}
//
//			if (m_drawInteriorVelocities) {
//				drawInteriorVelocities();
//			}
//			
//			if (m_drawMixedNodeFaces) {
//				int localSelectedMixedNode = -1;
//				int i = 0;
//				const map<int, vector<CutFace<Vector3D> *>> &mixedNodeMap = m_pNodeVelocityField->pMeshes->at(m_selectedCell).getCutFaceToMixedNodeMap();
//				for (auto it = mixedNodeMap.begin(); it != mixedNodeMap.end(); it++) {
//					if (i == m_selectedMixedNode) {
//						localSelectedMixedNode = it->first;
//						break;
//					}
//					i++;
//				}
//				if (localSelectedMixedNode != -1) {
//					drawSelectedMixedNodeFaces(localSelectedMixedNode);
//				}
//			}
//		}
//
//		void CutVoxelRenderer3D::update() {
//			updateVelocityPoints();
//			updateVelocities();
//		}
//		#pragma endregion
//
//		/************************************************************************/
//		/* Private functions			                                        */
//		/************************************************************************/
//
//		#pragma region UpdatingFunctions
//		void CutVoxelRenderer3D::updateVelocities() {
//			Scalar dx = m_pCutCells->getGridSpacing();
//			for (int i = 0; i < m_velocitiesNodes.size(); i++) {
//				velocityNode_t &velNode = m_velocitiesNodes[i];
//				for (int j = 0; j < velNode.velocities.size(); j++) {
//					velNode.velocities[j] = m_pLinearInterpolant->interpolate(velNode.velocityPoints[j] / dx);
//				}
//			}
//		}
//
//		void CutVoxelRenderer3D::updateVelocityPoints() {
//			m_velocitiesNodes.clear();
//			Scalar dx = m_pCutCells->getGridSpacing();
//			int dimSubdivision = m_dimVelocityInterpolationSubdivision;
//			Scalar spacing = 1 / ((Scalar)(dimSubdivision));
//			//Initializing velocities
//			for (int index = 0; index < m_pCutCells->getNumberOfCells(); index++) {
//				CutVoxel cutVoxel = m_pCutCells->getCutVoxel(index);
//				velocityNode_t currVelocityNode;
//				currVelocityNode.specialCell = cutVoxel;
//				for (int i = 1; i <= dimSubdivision; i++) {
//					for (int j = 1; j <= dimSubdivision; j++) {
//						for (int k = 1; k <= dimSubdivision; k++) {
//							Vector3 samplePoint;
//							samplePoint.x = (cutVoxel.regularGridIndex.x + i*spacing)*dx;
//							samplePoint.y = (cutVoxel.regularGridIndex.y + j*spacing)*dx;
//							samplePoint.z = (cutVoxel.regularGridIndex.z + k*spacing)*dx;
//
//							currVelocityNode.velocityPoints.push_back(samplePoint);
//							currVelocityNode.velocities.push_back(Vector3(0, 0, 0));
//						}
//					}
//				}
//				m_velocitiesNodes.push_back(currVelocityNode);
//			}
//		}
//		#pragma endregion
//
//		#pragma region DrawingFunctions
//		/************************************************************************/
//		/* General                                                              */
//		/************************************************************************/
//		void CutVoxelRenderer3D::drawFace(CutFace<Vector3D> *pFace) {
//			glBegin(GL_LINES);
//			Scalar dx = m_pCutCells->getGridSpacing();
//			for (int i = 0; i < pFace->m_cutEdges.size(); i++) {
//				glVertex3f(pFace->m_cutEdges[i]->m_initialPoint.x, pFace->m_cutEdges[i]->m_initialPoint.y, pFace->m_cutEdges[i]->m_initialPoint.z);
//				glVertex3f(pFace->m_cutEdges[i]->m_finalPoint.x, pFace->m_cutEdges[i]->m_finalPoint.y, pFace->m_cutEdges[i]->m_finalPoint.z);
//			}
//			glEnd();
//		}
//
//		void CutVoxelRenderer3D::drawCell(int ithCell) {
//			Scalar dx = m_pCutCells->getGridSpacing();
//			if (ithCell < m_pCutCells->getNumberOfCells()) {
//				CutVoxel cutVoxel = m_pCutCells->getCutVoxel(ithCell);
//
//				glLineWidth(3.0f);
//
//				for (int i = 0; i < cutVoxel.cutFaces.size(); i++) {
//					if (cutVoxel.cutFacesLocations[i] == geometryFace) {
//						glColor3f(1.0f, 0.078f, 0.26f);
//					}
//					else {
//						glColor3f(0.945f, 0.713f, 0.180f);
//					}
//					drawFace(cutVoxel.cutFaces[i]);
//				}
//				glColor3f(0.0f, 0.0f, 0.0f);
//				glLineWidth(1.0f);
//
//				/*glPushMatrix();
//				glColor3f(1.0f, 1.0f, 1.0f);
//				glTranslatef(cutVoxel.centroid.x, cutVoxel.centroid.y, cutVoxel.centroid.z);
//				glutSolidSphere(dx*0.02, 10, 8);
//				glPopMatrix();*/
//			}
//		}
//
//		void CutVoxelRenderer3D::drawSelectedCells() {
//			if (m_selectedCell != -1) {
//				for (int i = m_selectedCell; i < m_selectedCell + m_numSelectedVoxels; i++) {
//					drawCell(i);
//				}
//			}
//			
//		}
//
//		void CutVoxelRenderer3D::drawFaceNormals() {
//			glLineWidth(3.0f);
//			if (m_selectedCell != -1) {
//				const CutVoxel &cutVoxel = m_pCutCells->getCutVoxel(m_selectedCell);
//				for (int j = 0; j < cutVoxel.cutFaces.size(); j++) {
//					if (cutVoxel.cutFacesLocations[j] != geometryFace) {
//						const CutFace<Vector3D> *pCutFace = cutVoxel.cutFaces[j];
//						RenderingUtils::getInstance()->drawVector(convertToVector3F(pCutFace->getCentroid()), convertToVector3F(pCutFace->getCentroid() + pCutFace->m_normal * (*m_pVelocityScale)));
//					}
//				}
//
//				for (unsigned int j = 0; j < cutVoxel.geometryFacesToMesh.size(); j++) {
//					int triangleMeshIndex = cutVoxel.geometryFacesToMesh[j];
//					Vector3D correctedNormal = m_pNodeVelocityField->pMeshes->at(m_selectedCell).getMeshPolygons()[triangleMeshIndex].normal;
//					Vector3D centroid = m_pNodeVelocityField->pMeshes->at(m_selectedCell).getMeshPolygons()[triangleMeshIndex].centroid;
//					RenderingUtils::getInstance()->drawVector(convertToVector3F(centroid), convertToVector3F(centroid + correctedNormal*(*m_pVelocityScale)));
//				}
//			}
//		}
//
//		void CutVoxelRenderer3D::drawOnTheFaceLines() {
//			glLineWidth(3.0f);
//			glColor3f(0.3f, 1.0f, 0.2f);
//			if (m_selectedCell != -1 && m_pNodeVelocityField->pMeshes->size() > m_selectedCell) {
//				const vector<vector<int>> &faceLineIndices = m_pNodeVelocityField->pMeshes->at(m_selectedCell).getFaceLineIndices();
//				for (int i = 0; i < faceLineIndices.size(); i++) {
//					glBegin(GL_LINES);
//					for (int j = 0; j < faceLineIndices[i].size() - 1; j++) {
//						Vector3D currPoint = m_pNodeVelocityField->pMeshes->at(m_selectedCell).getPoints()[faceLineIndices[i][j]];
//						glVertex3f(currPoint.x, currPoint.y, currPoint.z);
//						currPoint = m_pNodeVelocityField->pMeshes->at(m_selectedCell).getPoints()[faceLineIndices[i][j + 1]];
//						glVertex3f(currPoint.x, currPoint.y, currPoint.z);
//					}
//					glEnd();
//				}
//			}
//			glLineWidth(1.0f);
//			glColor3f(0.0f, 0.0f, 0.0f);
//		}
//
//		/************************************************************************/
//		/* Velocities                                                           */
//		/************************************************************************/
//		void CutVoxelRenderer3D::drawFaceVelocities() {
//			glColor3f(0.0f, 0.0f, 0.0f);
//			glLineWidth(1.0f);
//			Scalar dx = m_pCutCells->getGridSpacing();
//			TwBar *pTwBar = m_pGridVisualizationWindow->getBaseTwBar();
//			bool drawIntermediateVel = m_pGridVisualizationWindow->getVelocityDrawingType() == BaseWindow::vectorVisualization_t::drawAuxiliaryVelocity;
//
//			if (m_selectedCell != -1) {
//				const CutVoxel &cutVoxel = m_pCutCells->getCutVoxel(m_selectedCell);
//				for (int i = 0; i < cutVoxel.cutFaces.size(); i++) {
//					CutFace<Vector3D> *pFace = cutVoxel.cutFaces[i];
//					Vector3 faceCenter = convertToVector3F(pFace->getCentroid());
//					if (drawIntermediateVel)
//						RenderingUtils::getInstance()->drawVector(faceCenter, faceCenter + convertToVector3F(pFace->m_velocity*(*m_pVelocityScale)));
//					else
//						RenderingUtils::getInstance()->drawVector(faceCenter, faceCenter + convertToVector3F(pFace->m_velocity*(*m_pVelocityScale)));
//				}
//			}
//		}
//
//		void CutVoxelRenderer3D::drawNodeVelocities() {
//			Scalar dx = m_pCutCells->getGridSpacing();
//			if (m_selectedCell != -1) {
//				const Mesh3D<Vector3D> &currMesh = m_pNodeVelocityField->pMeshes->at(m_selectedCell);
//				const vector<Mesh3D<Vector3D>::nodeType_t> &nodeTypes = currMesh.getNodeTypes();
//				for (int i = 0; i < m_pNodeVelocityField->nodesVelocities[m_selectedCell].size(); i++) {
//					if (nodeTypes[i] == Mesh3D<Vector3D>::gridNode) {
//						Vector3 initialPoint = convertToVector3F(currMesh.getPoints()[i]);
//						Vector3 finalVelPoint = m_pNodeVelocityField->nodesVelocities[m_selectedCell][i] * *m_pVelocityScale + convertToVector3F(currMesh.getPoints()[i]);
//						RenderingUtils::getInstance()->drawVector(initialPoint, finalVelPoint);
//					}
//				}
//			}
//		}
//
//		void CutVoxelRenderer3D::drawMixedNodeVelocities() {
//			Scalar dx = m_pCutCells->getGridSpacing();
//			if (m_selectedCell != -1) {
//				const Mesh3D<Vector3D> &currMesh = m_pNodeVelocityField->pMeshes->at(m_selectedCell);
//				const vector<Mesh3D<Vector3D>::nodeType_t> &nodeTypes = currMesh.getNodeTypes();
//				for (int i = 0; i < m_pNodeVelocityField->nodesVelocities[m_selectedCell].size(); i++) {
//					if (nodeTypes[i] == Mesh3D<Vector3D>::mixedNode) {
//						Vector3 initialPoint = convertToVector3F(currMesh.getPoints()[i]);
//						Vector3 finalVelPoint = m_pNodeVelocityField->nodesVelocities[m_selectedCell][i] * *m_pVelocityScale + convertToVector3F(currMesh.getPoints()[i]);
//						RenderingUtils::getInstance()->drawVector(initialPoint, finalVelPoint);
//					}
//				}
//			}
//		}
//
//		void CutVoxelRenderer3D::drawGeometryVelocities() {
//			Scalar dx = m_pCutCells->getGridSpacing();
//			if (m_selectedCell != -1) {
//				const Mesh3D<Vector3D> &currMesh = m_pNodeVelocityField->pMeshes->at(m_selectedCell);
//				const vector<Mesh3D<Vector3D>::nodeType_t> &nodeTypes = currMesh.getNodeTypes();
//				for (int i = 0; i < m_pNodeVelocityField->nodesVelocities[m_selectedCell].size(); i++) {
//					if (nodeTypes[i] == Mesh3D<Vector3D>::geometryNode) {
//						Vector3 initialPoint = convertToVector3F(currMesh.getPoints()[i]);
//						Vector3 finalVelPoint = m_pNodeVelocityField->nodesVelocities[m_selectedCell][i] * *m_pVelocityScale + convertToVector3F(currMesh.getPoints()[i]);
//						RenderingUtils::getInstance()->drawVector(initialPoint, finalVelPoint);
//					}
//				}
//			}
//		}
//
//		void CutVoxelRenderer3D::drawAllVelocities() {
//			Scalar dx = m_pCutCells->getGridSpacing();
//			if (m_selectedCell != -1) {
//				const Mesh3D<Vector3D> &currMesh = m_pNodeVelocityField->pMeshes->at(m_selectedCell);
//				const vector<Mesh3D<Vector3D>::nodeType_t> &nodeTypes = currMesh.getNodeTypes();
//				for (int i = 0; i < m_pNodeVelocityField->nodesVelocities[m_selectedCell].size(); i++) {
//					Vector3 initialPoint = convertToVector3F(currMesh.getPoints()[i]);
//					Vector3 finalVelPoint = m_pNodeVelocityField->nodesVelocities[m_selectedCell][i] * *m_pVelocityScale + convertToVector3F(currMesh.getPoints()[i]);
//					RenderingUtils::getInstance()->drawVector(initialPoint, finalVelPoint);
//				}
//			}
//		}
//
//		void CutVoxelRenderer3D::drawInteriorVelocities() {
//			if (m_selectedCell != -1) {
//				for (int i = 0; i < m_velocitiesNodes.size(); i++) {
//					if(m_velocitiesNodes[i].specialCell.ID != m_selectedCell)
//						continue;
//					for (int j = 0; j < m_velocitiesNodes[i].velocityPoints.size(); j++) {
//						Vector3 initialPoint = convertToVector3F(m_velocitiesNodes[i].velocityPoints[j]);
//						Vector3 finalVelPoint = m_velocitiesNodes[i].velocities[j]* *m_pVelocityScale + initialPoint;
//						RenderingUtils::getInstance()->drawVector(initialPoint, finalVelPoint);
//					}
//				}
//			}
//		}
//
//		/************************************************************************/
//		/* Mixed node                                                           */
//		/************************************************************************/
//		void CutVoxelRenderer3D::drawSelectedMixedNodeFaces(int selectedMixedNode) {
//			map <int, vector<CutFace<Vector3D> *>> &faceMap = m_pNodeVelocityField->pMeshes->at(m_selectedCell).getCutFaceToMixedNodeMap();
//
//			for (int i = 0; i < faceMap[selectedMixedNode].size(); i++) {
//				drawFace(faceMap[selectedMixedNode][i]);
//			}
//		}
//
//		#pragma endregion
//
//	}
//}