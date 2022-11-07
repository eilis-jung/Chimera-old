#include "Visualization/CutCellsRenderer2D.h"
#include "BaseGLRenderer.h"

namespace Chimera {
	namespace Rendering {
		CutCellsRenderer2D::CutCellsRenderer2D(CutCells2D *pSpecialCells, Interpolant<Vector2, Array2D, Vector2> *pVelocityInterpolant, const Array2D<Vector2> &nodeVelocityField, QuadGrid *pQuadGrid) :
			m_nodeVelocityField(nodeVelocityField) {
			m_pVelocityInterpolant = pVelocityInterpolant;
			m_pSpecialCells = pSpecialCells;
			m_pQuadGrid = pQuadGrid;
			m_drawCells = m_drawSelectedCells = true;
			m_drawPressures = m_drawVelocities = m_drawNodeVelocities = m_drawFaceVelocities = false;
			m_drawFaceNormals = false;
			m_drawCellNeighbors = false;
			m_drawTangentialVelocities = false;
			m_pGridCellPoints = NULL;
			m_pGridCellIndex = NULL;
			m_colorScheme = jet;
			m_selectedCell = -1;
			m_pMinScalarFieldVal = m_pMaxScalarFieldVal = NULL;
			m_velocityLength = 0.01;

			//Logger::getInstance()->get() << "Initializing Special Cells Renderer" << endl;
			initializeAuxiliaryDrawingStructures();
			initializeVBOs();
			initializeShaders();
			//Logger::getInstance()->get() << "	Copying Index to VBOs" << endl;
			copyIndexToVBOs();
			//Logger::getInstance()->get() << "	Copying Vertex to VBOs" << endl;
			copyVertexToVBOs();
			//Logger::getInstance()->get() << "	Copying Scalar Field to VBOs" << endl;
			copyScalarFieldToVBOs();
			//Logger::getInstance()->get() << "Special Cells Renderer successfully initialized" << endl;
		}


		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void CutCellsRenderer2D::draw() {		
			if(m_drawPressures)
				drawPressures();
			if(m_drawVelocities)
				drawVelocities();
			if(m_drawCells)
				drawCells();
			if(m_drawNodeVelocities)
				drawNodeVelocities();
			if(m_drawFaceVelocities)
				drawFaceVelocities();
			if(m_drawTangentialVelocities)
				drawTangentialVelocities();
			if (m_drawFaceNormals) {
				//drawFaceNormals();
				drawCurrentCellFaceVelocities();
			}
			if(m_drawSelectedCells && m_selectedCell != -1) {
				drawSelectedCells();
			}

		}

		void CutCellsRenderer2D::initializeWindowControls(GridVisualizationWindow<Vector2> *pGridVisualizationWindow) {
			TwBar *pTwBar = pGridVisualizationWindow->getBaseTwBar();
			m_pGridVisualizationWindow = pGridVisualizationWindow;
			
			TwAddVarRW(pTwBar, "drawSpecialCells", TW_TYPE_BOOL8, &m_drawCells, " label='Draw Cells' help='Draw special grid cells' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawSelectedCells", TW_TYPE_BOOL8, &m_drawSelectedCells, " label='Draw Selected Cells' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawCellNeighbors", TW_TYPE_BOOL8, &m_drawCellNeighbors, "label='Draw Cell Neighbors' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawSpecialPressures", TW_TYPE_BOOL8, &m_drawPressures, " label='Draw Pressures' help='Draw special grid pressures' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawSpecialVelocities", TW_TYPE_BOOL8, &m_drawVelocities, " label='Draw Velocities' help='Draw special grid velocities' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawNodeVelocities", TW_TYPE_BOOL8, &m_drawNodeVelocities, " label='Draw Node Velocities' help='Draw node velocities' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawFaceVelocities", TW_TYPE_BOOL8, &m_drawFaceVelocities, " label='Draw Face Velocities' help='Draw face velocities' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawTangentialVelocities", TW_TYPE_BOOL8, &m_drawTangentialVelocities, " label='Draw Tangential Velocities' help='Draw face velocities' group='Cut Cells'");
			TwAddVarRW(pTwBar, "drawNormals", TW_TYPE_BOOL8, &m_drawFaceNormals, " label='Draw Face Normals' group='Cut Cells'");
			TwAddVarRW(pTwBar, "velocityLenghtScale", TW_TYPE_FLOAT, &m_velocityLength, " label ='Velocity visualization length' group='Cut Cells' step=0.01 min='-0.15' max='0.15'");
			string defName = "'" + m_pQuadGrid->getGridName() + "'/'Cut Cells' group=Visualization";
			TwDefine(defName.c_str());

		}

		void CutCellsRenderer2D::update() {
			updateGridPoints();
			updateVelocityPoints();
			copyVertexToVBOs();
			copyIndexToVBOs();
			copyScalarFieldToVBOs();
			updateVelocities();
		}
		/************************************************************************/
		/* Initialization				                                        */
		/************************************************************************/
		void CutCellsRenderer2D::initializeAuxiliaryDrawingStructures() {
			m_maxNumberOfVertex = m_pSpecialCells->getMaxNumberOfCells()*5; //Upper bound
			m_maxNumberOfIndex = 2*(m_maxNumberOfVertex + m_pSpecialCells->getNumberOfCells() - 1);

			Logger::getInstance()->get() << "	Max number of vertices: " << m_maxNumberOfVertex << endl;

			m_pGridCellPoints = new Vector2[m_maxNumberOfVertex];
			m_pGridCellIndex = new int[m_maxNumberOfIndex];
			m_pScalarFieldValues = new Scalar[m_maxNumberOfVertex];
			m_totalNumberOfIndex = m_totalNumberOfVertex = 0;
			updateGridPoints();
			updateVelocityPoints();
		}

		void CutCellsRenderer2D::initializeVBOs() {
			Logger::getInstance()->get() << "	Initializing VBOs" << endl;
			/**Vertex VBO */
			m_pVertexVBO = new GLuint();
			glGenBuffers(1, m_pVertexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVertexVBO);
			unsigned int sizeVertex = m_maxNumberOfVertex*sizeof(Vector2);
			glBufferData(GL_ARRAY_BUFFER, sizeVertex, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			/**Index VBO*/
			m_pIndexVBO = new GLuint();
			glGenBuffers(1, m_pIndexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pIndexVBO);
			sizeVertex = m_maxNumberOfIndex*sizeof(int);
			glBufferData(GL_ARRAY_BUFFER, sizeVertex, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			
			/**Scalar field VBO */
			m_pScalarFieldVBO = new GLuint();
			glGenBuffers(1, m_pScalarFieldVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
			sizeVertex = m_maxNumberOfVertex*sizeof(Scalar);
			glBufferData(GL_ARRAY_BUFFER, sizeVertex, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			/**Scalar field Colors VBO */
			m_pScalarFieldColorsVBO = new GLuint();
			glGenBuffers(1, m_pScalarFieldColorsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldColorsVBO);
			sizeVertex = m_maxNumberOfVertex*sizeof(Vector3);
			glBufferData(GL_ARRAY_BUFFER, sizeVertex, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		void CutCellsRenderer2D::initializeShaders() {
			Logger::getInstance()->get() << "	Initializing Shaders" << endl;
			/** Jet color shader */
			{
				GLchar const * Strings[] = {"rColor", "gColor", "bColor"}; 
				m_pJetColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, 
					"Shaders/2D/ScalarColor - wavelength.glsl",
					3,
					Strings,
					GL_INTERLEAVED_ATTRIBS);

				m_jetMinScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "minPressure");
				m_jetMaxScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "maxPressure");
				m_jetAvgScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "avgPressure");
			}

			/** Grayscale color shader */
			{
				GLchar const * Strings[] = {"rColor", "gColor", "bColor"}; 
				m_pGrayScaleColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, 
					"Shaders/2D/ScalarColor - grayscale.glsl",
					3,
					Strings,
					GL_INTERLEAVED_ATTRIBS);

				m_grayMinScalarLoc = glGetUniformLocation(m_pGrayScaleColorShader->getProgramID(), "minScalar");
				m_grayMaxScalarLoc = glGetUniformLocation(m_pGrayScaleColorShader->getProgramID(), "maxScalar");
			}
		}

		/************************************************************************/
		/* Copying functions                                                    */
		/************************************************************************/
		void CutCellsRenderer2D::copyIndexToVBOs() {
			glBindBuffer(GL_ARRAY_BUFFER, *m_pIndexVBO);
			unsigned int sizeIndex = m_totalNumberOfIndex*sizeof(int);
			glBufferData(GL_ARRAY_BUFFER, sizeIndex, m_pGridCellIndex, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		void CutCellsRenderer2D::copyVertexToVBOs() {
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVertexVBO);
			unsigned int sizeVertex = m_totalNumberOfVertex*sizeof(Vector2);
			glBufferData(GL_ARRAY_BUFFER, sizeVertex, m_pGridCellPoints, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		void CutCellsRenderer2D::copyScalarFieldToVBOs() {
			int scalarPointIndex = 0;
			for(int i = 0; i < m_pSpecialCells->getNumberOfCells(); i++) {
				const CutFace<Vector2> &currCell = m_pSpecialCells->getSpecialCell(i);
				for(int j = 0; j < currCell.m_cutEdges.size(); j++) {
					m_pScalarFieldValues[scalarPointIndex++] = m_pSpecialCells->getPressure(i);
				}
			}
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
			unsigned int sizeArrows = m_totalNumberOfVertex*sizeof(Scalar);
			glBufferData(GL_ARRAY_BUFFER, sizeArrows, m_pScalarFieldValues, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

 		/************************************************************************/
		/* Private drawing functions                                            */
		/************************************************************************/
		void CutCellsRenderer2D::drawCells() {
			updateShaderColors();
			GridData2D *pGridData2D = m_pQuadGrid->getGridData2D();
			Scalar dx = pGridData2D->getScaleFactor(0, 0).x;

			glLineWidth(3.0f);
			glPointSize(3.0f);
			
			glEnable(GL_PRIMITIVE_RESTART);
			glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
			glEnableClientState(GL_VERTEX_ARRAY);       
			glColor3f(0.21f, 0.54f, 1.0f);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pIndexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVertexVBO);
			glVertexPointer(2, GL_FLOAT, 0, 0);	
			glDrawElements(GL_POLYGON, m_totalNumberOfIndex, GL_UNSIGNED_INT, 0);
			glPolygonMode( GL_FRONT_AND_BACK, GL_FILL  );
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);
			glDisable(GL_PRIMITIVE_RESTART);
			glLineWidth(1.0f);
		}

		void CutCellsRenderer2D::drawPressures() {
			glLineWidth(1.0f);

			glEnable(GL_PRIMITIVE_RESTART);
			glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
			
			glEnableClientState(GL_VERTEX_ARRAY);       
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVertexVBO);
			glVertexPointer(2, GL_FLOAT, 0, 0);	
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pIndexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldColorsVBO);
			glColorPointer(3, GL_FLOAT, 0, 0);
			
			glDrawElements(GL_POLYGON, m_totalNumberOfIndex, GL_UNSIGNED_INT, 0);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);
			glDisable(GL_PRIMITIVE_RESTART);
		}

		void CutCellsRenderer2D::drawVelocities() {
			glColor3f(0.0f, 0.0f, 0.0f);
			glPointSize(2.0f);
			glBegin(GL_POINTS);
			for(int i = 0; i < m_velocitiesNodes.size(); i++) {
				for(int j = 0; j < m_velocitiesNodes[i].velocityPoints.size(); j++) {
					glVertex2f(m_velocitiesNodes[i].velocityPoints[j].x, m_velocitiesNodes[i].velocityPoints[j].y);
				}
			}
			glEnd();

			glLineWidth(1.0f);
			for(int i = 0; i < m_velocitiesNodes.size(); i++) {
				for(int j = 0; j < m_velocitiesNodes[i].velocityPoints.size(); j++) {
					Vector2 finalVelPoint = m_velocitiesNodes[i].velocities[j]*m_velocityLength + m_velocitiesNodes[i].velocityPoints[j];
					RenderingUtils::getInstance()->drawVector(m_velocitiesNodes[i].velocityPoints[j], finalVelPoint);
				}
			}
		}

		void CutCellsRenderer2D::drawNodeVelocities() {
			glColor3f(0.1764f, 0.3843f, 1.0f);
			glLineWidth(1.0f);
			for(int i = 0; i < m_pQuadGrid->getDimensions().x; i++) {
				for(int j = 0; j < m_pQuadGrid->getDimensions().y; j++) {
					Vector2 finalVelPoint = m_pQuadGrid->getGridData2D()->getPoint(i, j) + m_nodeVelocityField(i, j)*m_velocityLength;
					RenderingUtils::getInstance()->drawVector(m_pQuadGrid->getGridData2D()->getPoint(i, j), finalVelPoint);
					
				}
			}
		} 

		void CutCellsRenderer2D::drawFaceVelocities() {
			glLineWidth(1.0f);
			Scalar dx = m_pSpecialCells->getGridSpacing();
			TwBar *pTwBar = m_pGridVisualizationWindow->getBaseTwBar();
			bool drawIntermediateVel = m_pGridVisualizationWindow->getVelocityDrawingType() == BaseWindow::vectorVisualization_t::drawAuxiliaryVelocity;
			for(int i = 0; i < m_pQuadGrid->getDimensions().x; i++) {
				for(int j = 0; j < m_pQuadGrid->getDimensions().y; j++) {
					for(int k = 0; k < m_pSpecialCells->getEdgeVector(dimensions_t(i, j), leftEdge).size(); k++) {
						const CutEdge<Vector2> &edge = m_pSpecialCells->getEdgeVector(dimensions_t(i, j), leftEdge)[k];
						Vector2 edgeCenter = edge.getCentroid();
						Vector2 edgeVelocity;
						if(drawIntermediateVel)
							edgeVelocity = edge.getIntermediaryVelocity();
						else 
							edgeVelocity = edge.getVelocity();

						//edgeVelocity.x = edge.getNormal().dot(edgeVelocity);

						glColor3f(1.0f, 0.06666f, 0.1294f);
						glBegin(GL_POINTS);
							glVertex2f(edgeCenter.x, edgeCenter.y);
						glEnd();

						RenderingUtils::getInstance()->drawVector(edgeCenter, edgeCenter + edgeVelocity*m_velocityLength);
					}

					for(int k = 0; k < m_pSpecialCells->getEdgeVector(dimensions_t(i, j), bottomEdge).size(); k++) {
						CutEdge<Vector2> edge = m_pSpecialCells->getEdgeVector(dimensions_t(i, j), bottomEdge)[k];
						Vector2 edgeCenter = edge.getCentroid();
						Vector2 edgeVelocity;
						if(drawIntermediateVel)
							edgeVelocity = edge.getIntermediaryVelocity();
						else 
							edgeVelocity = edge.getVelocity();

						//edgeVelocity.y = edge.getNormal().dot(edgeVelocity);

						glColor3f(1.0f, 0.06666f, 0.1294f);
						glBegin(GL_POINTS);
							glVertex2f(edgeCenter.x, edgeCenter.y);
						glEnd();

						RenderingUtils::getInstance()->drawVector(edgeCenter, edgeCenter + edgeVelocity*m_velocityLength);
					}
				}
			}
		}

		void CutCellsRenderer2D::drawCurrentCellFaceVelocities() {
			if (m_selectedCell != -1) {
				glLineWidth(3.0f);
				const CutFace<Vector2> &currCell = m_pSpecialCells->getSpecialCell(m_selectedCell);
				for (int j = 0; j < currCell.m_cutEdges.size(); j++) {
					Vector2 edgeCenter = currCell.m_cutEdges[j]->getCentroid();

					glColor3f(1.0f, 0.0f, 0.0f);
					glBegin(GL_POINTS);
					glVertex2f(edgeCenter.x, edgeCenter.y);
					glEnd();
					glColor3f(0.0f, 0.0f, 0.0f);
					RenderingUtils::getInstance()->drawVector(edgeCenter, edgeCenter + currCell.m_cutEdges[j]->getVelocity()*m_velocityLength);
				}
			}
		}

		void CutCellsRenderer2D::drawFaceNormals() {
			if (m_selectedCell != -1) {
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
			}
		}

		void CutCellsRenderer2D::drawTangentialVelocities() {
			glLineWidth(3.0f);
			for(int i = 0; i < m_pSpecialCells->getNumberOfCells(); i++) {
				const CutFace<Vector2> &currCell = m_pSpecialCells->getSpecialCell(i);
				for(int j = 0; j < currCell.m_cutEdges.size(); j++) {
					if(currCell.m_cutEdgesLocations[j] == geometryEdge) {
						Vector2 edgeCenter = currCell.m_cutEdges[j]->getCentroid();
						
						glColor3f(1.0f, 0.0f, 0.0f);
						glBegin(GL_POINTS);
						glVertex2f(edgeCenter.x, edgeCenter.y);
						glEnd();
						glColor3f(0.0f, 0.0f, 0.0f);
						RenderingUtils::getInstance()->drawVector(edgeCenter, edgeCenter + currCell.m_cutEdges[j]->getVelocity()*m_velocityLength);
					} 
				}
			}
			glLineWidth(1.0f);
		}

		void CutCellsRenderer2D::drawCell(int ithCell, bool drawPoints, bool drawThick) {
			Vector2 centroid = m_pSpecialCells->getSpecialCell(ithCell).getCentroid();
			const CutFace<Vector2> &currCell = m_pSpecialCells->getSpecialCell(ithCell);

			//Normal faces
			if(drawThick) {
				glLineWidth(4.0f);
			}
			glBegin(GL_LINES);
			for(int j = 0; j < currCell.m_cutEdges.size(); j++) {
				if(currCell.m_cutEdgesLocations[j] != geometryEdge) {
					Vector2 initialEdge = currCell.getEdgeInitialPoint(j);
					Vector2 finalEdge = currCell.getEdgeFinalPoint(j);
					glVertex2f(initialEdge.x, initialEdge.y);
					glVertex2f(finalEdge.x, finalEdge.y);
				}
			}
			glEnd();

			//ThinObject faces
			if(drawThick) {
				glLineWidth(8.0f);
			}
			glBegin(GL_LINES);
			for(int j = 0; j < currCell.m_cutEdges.size(); j++) {
				if(currCell.m_cutEdgesLocations[j] == geometryEdge) {
					Vector2 initialEdge = currCell.getEdgeInitialPoint(j);
					Vector2 finalEdge = currCell.getEdgeFinalPoint(j);
					glVertex2f(initialEdge.x, initialEdge.y);
					glVertex2f(finalEdge.x, finalEdge.y);
				}
			}
			glEnd();

			if(drawPoints) {
				glPointSize(6.0f);
				glColor3f(1.0f, 0.0f, 0.0f);
				glBegin(GL_POINTS);
				for(int j = 0; j < currCell.m_cutEdges.size(); j++) {
					Vector2 currPoint = currCell.getEdgeInitialPoint(j);
					glVertex2f(currPoint.x, currPoint.y);
				}
				glEnd();
			}
			
			glColor3f(0.15, 0.23f, 1.0f);
			glBegin(GL_POINTS);
			glVertex2f(centroid.x, centroid.y);
			glEnd();
		}
		void CutCellsRenderer2D::drawSelectedCells() {
			glColor3f(0.0f, 0.0f, 0.0f);
			if(m_selectedCell != -1) {
				const CutFace<Vector2> &currCell = m_pSpecialCells->getSpecialCell(m_selectedCell);
				if(m_drawCellNeighbors) {
					for(int i = 0; i < currCell.m_cutEdges.size(); i++) {
						CutEdge<Vector2> *pEdge = currCell.m_cutEdges[i];
						if(pEdge->m_edgeNeighbors[0] != m_selectedCell && pEdge->m_edgeNeighbors[0] >= 0) {
							glColor3f(0.756f, 0.756f, 0.756f);
							drawCell(pEdge->m_edgeNeighbors[0], false, true);
						} else if(pEdge->m_edgeNeighbors[1] != m_selectedCell && pEdge->m_edgeNeighbors[1] >= 0) {
							glColor3f(0.756f, 0.756f, 0.756f);
							drawCell(pEdge->m_edgeNeighbors[1], false, true);
						}
					}
				}
				glColor3f(0.0f, 0.0f, 0.0f);
				drawCell(m_selectedCell, true);
			}
		}

		void CutCellsRenderer2D::setTagColor(int tag) {
			if(tag == 1) {
				glColor3f(0.0f, 0.0f, 1.0f);
			} else if(tag == 0) {
				glColor3f(1.0f, 0.0f, 0.0f);
			} else if(tag == 2) {
				glColor3f(1.0f, 0.847f, 0.0f);
			}
		}
		/************************************************************************/
		/* Private functionalities                                              */
		/************************************************************************/
		void CutCellsRenderer2D::updateVelocities() {
			Scalar dx = m_pSpecialCells->getGridSpacing();
			for(int i = 0; i < m_velocitiesNodes.size(); i++) {
				velocityNode_t &velNode = m_velocitiesNodes[i];
				for(int j = 0; j < velNode.velocities.size(); j++) {
					velNode.velocities[j] = m_pVelocityInterpolant->interpolate(velNode.velocityPoints[j]);
				}
			}
		}

		void CutCellsRenderer2D::updateGridPoints() {
			//Updating correct number of points and vertexes
			m_totalNumberOfVertex = 0;
			for(int i = 0; i < m_pSpecialCells->getNumberOfCells(); i++) {
				m_totalNumberOfVertex += m_pSpecialCells->getSpecialCell(i).m_cutEdges.size();
			}

			//Number of index = (current number of points in a cell + 1 (index restart primitive)) x per cell
			m_totalNumberOfIndex = m_totalNumberOfVertex + m_pSpecialCells->getNumberOfCells() - 1;
			// enable primitive restart
			glPrimitiveRestartIndex(0xffff);
			int cellPointsIndex = 0;
			int pointsIndex = 0;
			for(int i = 0; i < m_pSpecialCells->getNumberOfCells(); i++) {
				const CutFace<Vector2> &currCell = m_pSpecialCells->getSpecialCell(i);
				for(int j = 0; j < currCell.m_cutEdges.size(); j++) {
					m_pGridCellIndex[pointsIndex++] = cellPointsIndex;
					m_pGridCellPoints[cellPointsIndex++] = currCell.getEdgeInitialPoint(j);
				}
				if(i <  m_pSpecialCells->getNumberOfCells() - 1)
					m_pGridCellIndex[pointsIndex++] = 0xffff;
			}
			if (pointsIndex >= m_maxNumberOfIndex)
				throw(exception("CutCelssRendered2D: max number of indices exceeded"));
			if (cellPointsIndex >= m_maxNumberOfVertex)
				throw(exception("CutCelssRendered2D: max number of vertices exceeded"));
		}

		void CutCellsRenderer2D::updateVelocityPoints() {
			m_velocitiesNodes.clear();
			Scalar dx = m_pSpecialCells->getGridSpacing();
			for(int i = 0; i < m_pQuadGrid->getDimensions().x; i++) {
				for (int j = 0; j < m_pQuadGrid->getDimensions().y; j++) {
					if(m_pSpecialCells->isSpecialCell(i, j)) {
						velocityNode_t currVelocityNode;
						float velocityNodes = 5;
						for (int k = 1; k < velocityNodes; k++) {
							for (int l = 1; l < velocityNodes; l++) {
								Vector2 samplePoint;
								samplePoint.x = (i + k*(1 / velocityNodes))*dx;
								samplePoint.y = (j + l*(1 / velocityNodes))*dx;
								currVelocityNode.velocityPoints.push_back(samplePoint);
								currVelocityNode.velocities.push_back(Vector2(0, 0));
							}
						}
						m_velocitiesNodes.push_back(currVelocityNode);
					}
				}
			}
		}

		void CutCellsRenderer2D::updateShaderColors() {
			if(m_pMinScalarFieldVal == NULL || m_pMaxScalarFieldVal == NULL) {
				m_pMinScalarFieldVal = new Scalar();
				m_pMaxScalarFieldVal = new Scalar();
				*m_pMinScalarFieldVal = *m_pMaxScalarFieldVal = 0.0f;
			}
			/** Avg and max pressure calculation */
			Scalar avgValue = 0.5*(*m_pMinScalarFieldVal + *m_pMaxScalarFieldVal);
			applyColorShader(*m_pMinScalarFieldVal , *m_pMaxScalarFieldVal, avgValue);

			glEnable(GL_RASTERIZER_DISCARD_NV);
			glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *m_pScalarFieldColorsVBO);

			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
			glVertexAttribPointer(0, 1, GL_FLOAT, false, 0, 0);
			glEnableVertexAttribArray(0);

			glBeginTransformFeedback(GL_POINTS);
			glDrawArrays (GL_POINTS, 0, m_totalNumberOfVertex);
			glEndTransformFeedback();
			glDisableVertexAttribArray(0);

			glDisable(GL_RASTERIZER_DISCARD_NV);
			removeColorShader();
		}

	}
}