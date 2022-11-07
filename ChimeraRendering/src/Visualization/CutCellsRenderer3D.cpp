//#include "Visualization/CutCellsRenderer3D.h"
//#include "Resources/ResourceManager.h"
//
//namespace Chimera {
//	namespace Rendering {
//
//		#pragma region Constructors
//		CutCellsRenderer3D::CutCellsRenderer3D(CutCells3D *pCutCells, HexaGrid *pHexaGrid, nodeVelocityField3D_t *pNodeVelocityField3D, 
//												HexaGridRenderer *pHexaGridRenderer, LinearInterpolant3D<Vector3> *pLinearInterpolant) {
//			m_pCutCells = pCutCells;
//			m_pHexaGrid = pHexaGrid;
//			m_pCutVoxelRenderer = new CutVoxelRenderer3D(pCutCells, pLinearInterpolant, &pHexaGridRenderer->getVectorFieldRenderer().velocityLength, pHexaGrid->getGridName());
//			
//			m_pHexaGridRenderer = pHexaGridRenderer;
//			m_pNodeVelocityField = pNodeVelocityField3D;
//			
//			m_drawCells = true;
//			m_drawCutSliceLines = false;
//			m_drawAllNodeVelocities = false;
//
//			/** Faces configuration */
//			m_drawFaceLocation = static_cast<faceLocation_t>(-1);
//			m_drawAllFaces = false;
//			m_selectedFace = 0;
//			m_numberOfFacesToDraw = 1;
//
//			m_colorScheme = jet;
//			m_selectedFace = 0;
//
//			m_cutSlicesXY = m_pCutCells->getCutCellsXY();
//			m_cutSlicesXZ = m_pCutCells->getCutCellsXZ();
//			m_cutSlizesYZ = m_pCutCells->getCutCellsYZ();
//
//			if (pCutCells->getNumberOfCells() == 0) {
//				m_pCutVoxelRenderer->setSelectedVoxel(-1);
//				m_selectedFace = -1;
//				m_numberOfFacesToDraw = 0;
//			}
//			
//		}
//		#pragma endregion
//
//		#pragma region InitializationFunctions
//		void CutCellsRenderer3D::initializeWindowControls(GridVisualizationWindow<Vector3> *pGridVisualizationWindow) {
//			TwBar *pTwBar = pGridVisualizationWindow->getBaseTwBar();
//			m_pGridVisualizationWindow = pGridVisualizationWindow;
//			
//			/** General */
//			TwAddVarRW(pTwBar, "drawSpecialCells", TW_TYPE_BOOL8, &m_drawCells, " label='Draw Cells' help='Draw special grid cells' group='Cut-Cells'");
//			TwAddVarRW(pTwBar, "drawCutsliceLines", TW_TYPE_BOOL8, &m_drawCutSliceLines, " label='Draw cut slice lines' group='Cut-Cells'");
//			TwAddVarRW(pTwBar, "drawAllNodeVelocities", TW_TYPE_BOOL8, &m_drawAllNodeVelocities, " label='Draw All Node Velocities' group='Cut-Cells'");
//
//			/** Faces variables */
//			TwEnumVal scalarEV[] = { {-1, "None"}, {faceLocation_t::leftFace, "Left Faces"}, {faceLocation_t::bottomFace, "Bottom Faces"}, {faceLocation_t::backFace, "Back Faces"}};
//			TwType facesType = TwDefineEnum("facesType", scalarEV, 4);
//			TwAddVarRW(pTwBar, "drawFacePerType", facesType, &m_drawFaceLocation, "label='Draw faces per type'  group='Faces'");
//			TwAddVarRW(pTwBar, "drawAllFaces", TW_TYPE_BOOL8, &m_drawAllFaces, " label='Draw all faces' group='Faces'");
//			TwAddVarRW(pTwBar, "selectedFace", TW_TYPE_INT32, &m_selectedFace, " label='Selected face' group='Faces'");
//			TwAddVarRW(pTwBar, "numberOfFacesToDraw", TW_TYPE_INT32, &m_numberOfFacesToDraw, " label='Number of Faces to Draw' group='Faces'");
//
//			string defName = "'" + m_pHexaGrid->getGridName() + "'/'Faces' group='Cut-Cells'";
//			TwDefine(defName.c_str());
//
//			m_pCutVoxelRenderer->initializeWindowControls(pGridVisualizationWindow);
//		}
//		#pragma endregion
//
//		#pragma region Functionalities
//		void CutCellsRenderer3D::draw() {	
//			if(m_drawCells)
//				drawCells();
//
//			drawFacesPerLocation();
//
//			if (m_drawAllNodeVelocities) {
//				drawAllNodesVelocities();
//			}
//
//			if (m_drawCutSliceLines) {
//				glLineWidth(8.0f);
//				glColor3f(1.0f, 0.8274f, 0.01f);
//				for (int i = 0; i < m_cutSlicesXY.size(); i++) {
//					vector<LineMesh<Vector3D> *> lineMeshes = m_cutSlicesXY[i]->getLineMeshes();
//					for (int j = 0; j < lineMeshes.size(); j++) {
//						glBegin(GL_LINES);
//						for (int k = 0; k < lineMeshes[j]->getPoints().size(); k++) {
//							int nextK = roundClamp<int>(k + 1, 0, lineMeshes[j]->getPoints().size());
//							Vector3D currPoint = lineMeshes[j]->getPoints()[k];
//							Vector3D nextPoint = lineMeshes[j]->getPoints()[nextK];
//							glVertex3f(currPoint.x, currPoint.y, currPoint.z);
//							glVertex3f(nextPoint.x, nextPoint.y, nextPoint.z);
//						}
//						glEnd();
//					}
//				}
//				glColor3f(0.13f, 0.69f, 0.28f);
//				for (int i = 0; i < m_cutSlicesXZ.size(); i++) {
//					vector<LineMesh<Vector3D> *> lineMeshes = m_cutSlicesXZ[i]->getLineMeshes();
//					for (int j = 0; j < lineMeshes.size(); j++) {
//						glBegin(GL_LINES);
//						for (int k = 0; k < lineMeshes[j]->getPoints().size(); k++) {
//							int nextK = roundClamp<int>(k + 1, 0, lineMeshes[j]->getPoints().size());
//							Vector3D currPoint = lineMeshes[j]->getPoints()[k];
//							Vector3D nextPoint = lineMeshes[j]->getPoints()[nextK];
//							glVertex3f(currPoint.x, currPoint.y, currPoint.z);
//							glVertex3f(nextPoint.x, nextPoint.y, nextPoint.z);
//						}
//						glEnd();
//					}
//				}
//				glColor3f(0.05f, 0.5f, 1.0f);
//				for (int i = 0; i < m_cutSlizesYZ.size(); i++) {
//					vector<LineMesh<Vector3D> *> lineMeshes = m_cutSlizesYZ[i]->getLineMeshes();
//					for (int j = 0; j < lineMeshes.size(); j++) {
//						glBegin(GL_LINES);
//						for (int k = 0; k < lineMeshes[j]->getPoints().size(); k++) {
//							int nextK = roundClamp<int>(k + 1, 0, lineMeshes[j]->getPoints().size());
//							Vector3D currPoint = lineMeshes[j]->getPoints()[k];
//							Vector3D nextPoint = lineMeshes[j]->getPoints()[nextK];
//							glVertex3f(currPoint.x, currPoint.y, currPoint.z);
//							glVertex3f(nextPoint.x, nextPoint.y, nextPoint.z);
//						}
//						glEnd();
//					}
//				}
//			}
//
//			m_pCutVoxelRenderer->draw();
//		}
//
//		void CutCellsRenderer3D::update() {
//			m_pCutVoxelRenderer->update();
//		}
//		#pragma endregion
//
//		/************************************************************************/
//		/* Private functions			                                        */
//		/************************************************************************/
//
//		#pragma region InitializationFunctions
//		void CutCellsRenderer3D::initializeAuxiliaryDrawingStructures() {
//			m_maxNumberOfVertex = m_pCutCells->getMaxNumberOfCells()*5; //Upper bound
//			m_maxNumberOfIndex = m_maxNumberOfVertex + m_pCutCells->getNumberOfCells() - 1;
//
//			Logger::getInstance()->get() << "	Max number of vertices: " << m_maxNumberOfVertex << endl;
//
//			m_pScalarFieldValues = new Scalar[m_maxNumberOfVertex];
//		}
//
//		void CutCellsRenderer3D::initializeVBOs() {
//			Logger::getInstance()->get() << "	Initializing VBOs" << endl;
//			/**Vertex VBO */
//			m_pVertexVBO = new GLuint();
//			glGenBuffers(1, m_pVertexVBO);
//			glBindBuffer(GL_ARRAY_BUFFER, *m_pVertexVBO);
//			unsigned int sizeVertex = m_maxNumberOfVertex*sizeof(Vector3);
//			glBufferData(GL_ARRAY_BUFFER, sizeVertex, 0, GL_DYNAMIC_DRAW);
//			glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//			/**Index VBO*/
//			m_pIndexVBO = new GLuint();
//			glGenBuffers(1, m_pIndexVBO);
//			glBindBuffer(GL_ARRAY_BUFFER, *m_pIndexVBO);
//			sizeVertex = m_maxNumberOfIndex*sizeof(int);
//			glBufferData(GL_ARRAY_BUFFER, sizeVertex, 0, GL_DYNAMIC_DRAW);
//			glBindBuffer(GL_ARRAY_BUFFER, 0);
//			
//			/**Scalar field VBO */
//			m_pScalarFieldVBO = new GLuint();
//			glGenBuffers(1, m_pScalarFieldVBO);
//			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
//			sizeVertex = m_maxNumberOfVertex*sizeof(Scalar);
//			glBufferData(GL_ARRAY_BUFFER, sizeVertex, 0, GL_DYNAMIC_DRAW);
//			glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//			/**Scalar field Colors VBO */
//			m_pScalarFieldColorsVBO = new GLuint();
//			glGenBuffers(1, m_pScalarFieldColorsVBO);
//			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldColorsVBO);
//			sizeVertex = m_maxNumberOfVertex*sizeof(Vector3);
//			glBufferData(GL_ARRAY_BUFFER, sizeVertex, 0, GL_DYNAMIC_DRAW);
//			glBindBuffer(GL_ARRAY_BUFFER, 0);
//		}
//
//		void CutCellsRenderer3D::initializeShaders() {
//			Logger::getInstance()->get() << "	Initializing Shaders" << endl;
//			/** Jet color shader */
//			{
//				GLchar const * Strings[] = {"rColor", "gColor", "bColor"}; 
//				m_pJetColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, 
//					"Shaders/2D/ScalarColor - wavelength.glsl",
//					3,
//					Strings,
//					GL_INTERLEAVED_ATTRIBS);
//
//				m_jetMinScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "minPressure");
//				m_jetMaxScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "maxPressure");
//				m_jetAvgScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "avgPressure");
//			}
//
//			/** Grayscale color shader */
//			{
//				GLchar const * Strings[] = {"rColor", "gColor", "bColor"}; 
//				m_pGrayScaleColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, 
//					"Shaders/2D/ScalarColor - grayscale.glsl",
//					3,
//					Strings,
//					GL_INTERLEAVED_ATTRIBS);
//
//				m_grayMinScalarLoc = glGetUniformLocation(m_pGrayScaleColorShader->getProgramID(), "minScalar");
//				m_grayMaxScalarLoc = glGetUniformLocation(m_pGrayScaleColorShader->getProgramID(), "maxScalar");
//			}
//		}
//		#pragma endregion
//
//		#pragma region UpdatingFunctions
//		void CutCellsRenderer3D::updateShaderColors() {
//			/** Avg and max pressure calculation */
//			Scalar avgValue = 0.5*(*m_pMinScalarFieldVal + *m_pMaxScalarFieldVal);
//			applyColorShader(*m_pMinScalarFieldVal, *m_pMaxScalarFieldVal, avgValue);
//
//			glEnable(GL_RASTERIZER_DISCARD_NV);
//			glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *m_pScalarFieldColorsVBO);
//
//			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
//			glVertexAttribPointer(0, 1, GL_FLOAT, false, 0, 0);
//			glEnableVertexAttribArray(0);
//
//			glBeginTransformFeedback(GL_POINTS);
//			glDrawArrays(GL_POINTS, 0, m_totalNumberOfVertex);
//			glEndTransformFeedback();
//			glDisableVertexAttribArray(0);
//
//			glDisable(GL_RASTERIZER_DISCARD_NV);
//			removeColorShader();
//		}
//
//		#pragma endregion
//
//		#pragma region DrawingFunctions
//		void CutCellsRenderer3D::drawCells() {
//			glDisable(GL_LIGHTING);
//			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//			glColor4f(1, 0, 0, 0.2f);
//			for (int i = 0; i < m_pHexaGrid->getDimensions().x; i++) {
//				for (int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
//					for (int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
//						if (m_pCutCells->isBoundaryCell(i, j, k)) {
//							glColor4f(1.0f, 0.33f, 0.21f, 0.33f);
//							m_pHexaGridRenderer->drawCell(i, j, k);
//						}
//						else if (m_pCutCells->isSpecialCell(i, j, k)) {
//							glColor4f(1.0f, 0.611f, 0.24f, 0.33f);
//							m_pHexaGridRenderer->drawCell(i, j, k);
//						}
//					}
//				}
//			}
//
//		}
//
//		void CutCellsRenderer3D::drawFace(CutFace<Vector3D> *pFace) {
//			glBegin(GL_LINES);
//			Scalar dx = m_pCutCells->getGridSpacing();
//			for (int i = 0; i < pFace->m_cutEdges.size(); i++) {
//				glVertex3f(pFace->m_cutEdges[i]->m_initialPoint.x, pFace->m_cutEdges[i]->m_initialPoint.y, pFace->m_cutEdges[i]->m_initialPoint.z);
//				glVertex3f(pFace->m_cutEdges[i]->m_finalPoint.x, pFace->m_cutEdges[i]->m_finalPoint.y, pFace->m_cutEdges[i]->m_finalPoint.z);
//			}
//			glEnd();
//
//			glColor3f(0.1f, 1.0f, 0.2f);
//			glPointSize(6.0f);
//			glBegin(GL_POINTS);
//			for (int i = 0; i < pFace->m_cutEdges.size(); i++) {
//				glVertex3f(pFace->m_cutEdges[i]->m_initialPoint.x, pFace->m_cutEdges[i]->m_initialPoint.y, pFace->m_cutEdges[i]->m_initialPoint.z);
//			}
//			glEnd();
//			glColor3f(0.0f, 0.0f, 0.0f);
//
//		}
//
//		void CutCellsRenderer3D::drawCell(int ithCell) {
//			Scalar dx = m_pCutCells->getGridSpacing();
//			if (ithCell < m_pCutCells->getNumberOfCells()) {
//				CutVoxel cutVoxel = m_pCutCells->getCutVoxel(ithCell);
//
//				glLineWidth(6.0f);
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
//				glPushMatrix();
//				glColor3f(1.0f, 1.0f, 1.0f);
//				glTranslatef(cutVoxel.centroid.x, cutVoxel.centroid.y, cutVoxel.centroid.z);
//				glutSolidSphere(dx*0.02, 10, 8);
//				glPopMatrix();
//			}
//		}
//
//		void CutCellsRenderer3D::drawFacesPerLocation() {
//			int currFaceID = 0;
//			glLineWidth(3.0f);
//			glColor3f(0.0f, 0.0f, 0.0f);
//			if (m_drawFaceLocation != -1) {
//				if (m_drawAllFaces) {
//					for (int i = 0; i < m_pHexaGrid->getDimensions().x; i++) {
//						for (int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
//							for (int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
//								dimensions_t currDim(i, j, k);
//								vector<CutFace<Vector3D>*> currFacesVec = m_pCutCells->getFaceVector(currDim, m_drawFaceLocation);
//								for (int l = 0; l < currFacesVec.size(); l++) {
//									if (currFacesVec[l]->getAreaFraction() < 1.0f) //Only draw cut-faces
//										drawFace(currFacesVec[l]);
//								}
//							}
//						}
//					}
//				}
//				else {
//					for (int i = 0; i < m_pHexaGrid->getDimensions().x; i++) {
//						for (int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
//							for (int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
//								dimensions_t currDim(i, j, k);
//								vector<CutFace<Vector3D>*> currFacesVec = m_pCutCells->getFaceVector(currDim, m_drawFaceLocation);
//								bool foundFace = false;
//								for (int l = 0; l < currFacesVec.size(); l++) {
//									if (currFaceID == m_selectedFace) {
//										foundFace = true;
//										drawFace(currFacesVec[l]);
//										return;
//										break;
//									}
//									currFaceID++;
//								}
//
//								if (foundFace) {
//									for (int l = 0; l < currFacesVec.size(); l++) {
//										drawFace(currFacesVec[l]);
//									}
//									return;
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//
//		void CutCellsRenderer3D::drawAllNodesVelocities() {
//			Scalar dx = m_pCutCells->getGridSpacing();
//			for (int i = 0; i < m_pNodeVelocityField->pGridNodesVelocities->getDimensions().x; i++) {
//				for (int j = 0; j < m_pNodeVelocityField->pGridNodesVelocities->getDimensions().y; j++) {
//					for (int k = 0; k < m_pNodeVelocityField->pGridNodesVelocities->getDimensions().z; k++) {
//						Vector3 initialPoint(i*dx, j*dx, k*dx); 
//						Vector3 finalVelPoint = initialPoint + (*m_pNodeVelocityField->pGridNodesVelocities)(i, j, k) * m_pHexaGridRenderer->getVectorFieldRenderer().velocityLength;
//						RenderingUtils::getInstance()->drawVector(initialPoint, finalVelPoint);
//					}
//				}
//			}
//		}
//
//		void CutCellsRenderer3D::setTagColor(int tag) {
//			if (tag == 1) {
//				glColor3f(0.0f, 0.0f, 1.0f);
//			}
//			else if (tag == 0) {
//				glColor3f(1.0f, 0.0f, 0.0f);
//			}
//			else if (tag == 2) {
//				glColor3f(1.0f, 0.847f, 0.0f);
//			}
//		}
//
//		#pragma endregion
//		
//	}
//}