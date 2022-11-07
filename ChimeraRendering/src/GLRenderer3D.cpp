#include "GLRenderer3D.h"
#include "Resources/ResourceManager.h"

namespace Chimera {
	namespace Rendering {
		#pragma region DrawingFunctions
		void GLRenderer3D::drawGridSlices(int ithGrid) {
			glDisable(GL_LIGHTING);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			glPushMatrix();

			if (m_pGridVisWindows[0]->drawScalarFieldSlices()) {
				glDepthMask(GL_FALSE);
			}
			HexaGridRenderer *pHexaRenderer = dynamic_cast<HexaGridRenderer*>(m_pGridVisWindows[ithGrid]->getGridRenderer());
			dimensions_t ithDimensions = m_pGridVisWindows[ithGrid]->getIthRenderingPlanes();
			dimensions_t gridDimensions = m_pGridVisWindows[ithGrid]->getGridRenderer()->getGrid()->getDimensions();
			if (m_pGridVisWindows[ithGrid]->drawYZPlaneSlice()) {
				for (int i = 0; i < m_pGridVisWindows[ithGrid]->getNumberOfDrawnPlanes(); i++) {
					if (i + ithDimensions.x >= gridDimensions.x) {
						break;
					}
					pHexaRenderer->drawYZSlice(ithDimensions.x + i);
				}
			}
			if (m_pGridVisWindows[ithGrid]->drawXZPlaneSlice()) {
				for (int i = 0; i < m_pGridVisWindows[ithGrid]->getNumberOfDrawnPlanes(); i++) {
					if (i + ithDimensions.y >= gridDimensions.y) {
						break;
					}
					pHexaRenderer->drawXZSlice(ithDimensions.y + i);
				}
			}
			if (m_pGridVisWindows[ithGrid]->drawXYPlaneSlice()) {
				for (int i = 0; i < m_pGridVisWindows[ithGrid]->getNumberOfDrawnPlanes(); i++) {
					if (i + ithDimensions.z >= gridDimensions.z) {
						break;
					}
					pHexaRenderer->drawXYSlice(ithDimensions.z + i);
				}
			}
			glDepthMask(GL_TRUE);

			if (m_pGridVisWindows[0]->drawScalarFieldSlices()) {
				m_gridRenderers[ithGrid]->getScalarFieldRenderer().beginDrawScalarField(m_pGridVisWindows[0]->getScalarFieldType(), m_pGridVisWindows[0]->getIthRenderingPlanes());
				m_gridRenderers[ithGrid]->getScalarFieldRenderer().endDrawScalarField();
			}

			
			glPopMatrix();
		
		}
		void GLRenderer3D::drawVelocitySlices(int ithGrid) {
			glPushMatrix();

			if (m_pGridVisWindows[0]->drawVelocitySlices()) {
				m_gridRenderers[ithGrid]->getVectorFieldRenderer().drawVelocityField(false, m_pGridVisWindows[0]->getIthRenderingPlanes());
			}
		}

		void GLRenderer3D::renderLoop(bool swapBuffers) {
			drawBackground();
			m_pCamera->updateGL();

			// Setup modelview matrix
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

		

			glDisable(GL_BLEND);
			Vector3 cameraPos(m_pCamera->getPosition());
			Vector3 cameraDirection = m_pCamera->getDirection();
			//cameraPos -= cameraDirection.normalized()*2;
			GLfloat light_position[] = { cameraPos.x, cameraPos.y, cameraPos.z, 0.0 };
			glLightfv(GL_LIGHT0, GL_POSITION, light_position);
			glEnable(GL_BLEND);
			for (unsigned int i = 0; i < m_pRenderingObjects.size(); i++) {
				//m_pRenderingObjects[i]->setCameraPosition(cameraPos);
				//Check if the object is supposed to be drawn
				m_pRenderingObjects[i]->draw();
			}
			if (m_pMeshRenderer) {
				m_pMeshRenderer->draw();
			}

			glDisable(GL_BLEND);
			for (int i = 0; i < m_pGridVisWindows.size(); i++) {
				drawGrid(i);
				drawGridSlices(i);
				drawVelocitySlices(i);
				drawVelocityField(i);
			}
			glEnable(GL_BLEND);

			if (m_pParticlesRenderer) {
				m_pParticlesRenderer->draw();
			}

			for (int i = 0; i < m_lineRenderers.size(); i++) {
				m_lineRenderers[i]->draw();
			}

			for (int i = 0; i < m_polyMeshRenderers.size(); i++) {
				m_polyMeshRenderers[i]->draw();
			}

			if (m_pCutVoxelRenderer) {
				m_pCutVoxelRenderer->draw();
			}
			if (m_pCutVoxelVelRenderer) {
				m_pCutVoxelVelRenderer->draw();
			}


			/*if (m_pCutCellsRenderer != NULL) {
				m_pCutCellsRenderer->draw();
			}*/

			// Draw tweak bars
			if(m_drawAntTweakBars)
				TwDraw();

			if (swapBuffers) {
				glutSwapBuffers();
				//glutPostRedisplay();
			}
		}
		#pragma endregion

		#pragma region Callbacks
		void GLRenderer3D::keyboardCallback(unsigned char key, int x, int y) {
			Scalar dx = 0.01;
			if (m_pGridVisWindows.size() > 0)
				dx = m_pGridVisWindows.front()->getGridRenderer()->getGrid()->getGridData3D()->getScaleFactor(0, 0, 0).x;
			BaseGLRenderer::keyboardCallback(key, x, y);
			int selectedCell = 0;
			dimensions_t selectedDim;
			/*if (m_pCutCellsRenderer) {
				selectedCell = m_pCutCellsRenderer->getSelectedCell();
			}
			else *//*if (m_pMeshRenderer) {
				selectedCell = m_pMeshRenderer->getSelectedMesh();
			}*/
			Vector3 cellCentroid, cameraToCellVec;
			switch (key) {
				case 'd':
					m_drawAntTweakBars = !m_drawAntTweakBars;
				break;
				case 'p':
					if(m_pMeshRenderer)
						m_pMeshRenderer->setDrawing(!m_pMeshRenderer->isDrawing());
				break;

			case '+':
				if (m_pCutVoxelRenderer) {
					selectedCell = clamp<int>(m_pCutVoxelRenderer->getSelectedCutVoxel() + 1, 0, m_pCutVoxelRenderer->getCutVoxels()->getNumberCutVoxels() - 1);
					m_pCutVoxelRenderer->setSelectedCutVoxel(selectedCell);
					auto cutVoxel = m_pCutVoxelRenderer->getCutVoxels()->getCutVoxel(selectedCell);
					selectedDim = cutVoxel.getVolume()->getGridCellLocation();
					cout << "Selected cell dim:" << selectedDim.x << " " << selectedDim.y << " " << selectedDim.z << endl;
					getCamera()->setRotationAroundGridMode(Vector3((selectedDim.x + 0.5)*dx, (selectedDim.y + 0.5)*dx, (selectedDim.z + 0.5)*dx));
					//getParticleSystem()->getRenderingParams().selectedVoxelDimension = selectedDim;
					//m_pParticleSystem->updateParticlesTags();
				}
				break;
			case '-':
				if (m_pCutVoxelRenderer) {
					selectedCell = clamp<int>(m_pCutVoxelRenderer->getSelectedCutVoxel() - 1, 0, m_pCutVoxelRenderer->getCutVoxels()->getNumberCutVoxels() - 1);
					m_pCutVoxelRenderer->setSelectedCutVoxel(selectedCell);
					auto cutVoxel = m_pCutVoxelRenderer->getCutVoxels()->getCutVoxel(selectedCell);
					selectedDim = cutVoxel.getVolume()->getGridCellLocation();
					cout << "Selected cell dim:" << selectedDim.x << " " << selectedDim.y << " " << selectedDim.z << endl;
					getCamera()->setRotationAroundGridMode(Vector3((selectedDim.x + 0.5)*dx, (selectedDim.y + 0.5)*dx, (selectedDim.z + 0.5)*dx));
					//getParticleSystem()->getRenderingParams().selectedVoxelDimension = selectedDim;
					//m_pParticleSystem->updateParticlesTags();
				}
				break;
			//	if (m_pMeshRenderer) {
			//		selectedCell = clamp<int>(selectedCell, 0, m_pMeshRenderer->getNodeVelocityField()->pMeshes->size() - 1);
			//		m_pMeshRenderer->setSelectedMesh(selectedCell);
			//		Vector3 centroid;
			//		centroid.x = (m_pMeshRenderer->getNodeVelocityField()->pMeshes->at(selectedCell).getCutCellLocation().x + 0.5)*dx;
			//		centroid.y = (m_pMeshRenderer->getNodeVelocityField()->pMeshes->at(selectedCell).getCutCellLocation().y + 0.5)*dx;
			//		centroid.z = (m_pMeshRenderer->getNodeVelocityField()->pMeshes->at(selectedCell).getCutCellLocation().z + 0.5)*dx;
			//		getParticleSystem()->getRenderingParams().selectedVoxelDimension = m_pMeshRenderer->getNodeVelocityField()->pMeshes->at(selectedCell).getCutCellLocation();
			//		if (m_pParticleSystem->getRenderingParams().drawSelectedVoxelParticles) {
			//			m_pParticleSystem->updateParticlesTags();
			//		}
			//		getCamera()->setRotationAroundGridMode(centroid);
			//	}

			//	break;
			//case '-':
			//	selectedCell--;
			//	if (m_pCutCellsRenderer) {
			//		selectedCell = clamp(selectedCell, 0, m_pCutCellsRenderer->getCutCells3D()->getNumberOfCells() - 1);
			//		m_pCutCellsRenderer->setSelectedCell(selectedCell);
			//	}
			//	if (m_pMeshRenderer) {
			//		selectedCell = clamp<int>(selectedCell, 0, m_pMeshRenderer->getNodeVelocityField()->pMeshes->size() - 1);
			//		m_pMeshRenderer->setSelectedMesh(selectedCell);
			//	}
			//	break;

			case 'o':
			case 'O':
				if (m_pCamera->getType() == cameraTypes_t::orthogonal2D) {
					m_pCamera->setType(cameraTypes_t::perspective3D);
				}
				else {
					m_pCamera->setType(cameraTypes_t::orthogonal2D);
				}
				break;
			case 'l':
			case 'L':
				m_pCamera->setPosition(Vector3(3, 0.5, 4));
				m_pCamera->setDirection(Vector3(0, 0, -1));
				break;

				/*case 't':
				case 'T':
				m_pCamera->setPosition(Vector3(3, 10, 1));
				m_pCamera->setDirection(Vector3(0.0001, -0.9999, 0));
				break;*/

			case 'f':
			case 'F':
				/*if (m_pCutCellsRenderer && (selectedCell = m_pCutCellsRenderer->getSelectedCell()) != -1) {
					cellCentroid = convertToVector3F(m_pCutCellsRenderer->getCutCells3D()->getCutVoxel(selectedCell).centroid);
					cameraToCellVec = (cellCentroid - m_pCamera->getPosition());
					cameraToCellVec.normalize();
					m_pCamera->setPosition(cellCentroid - cameraToCellVec*0.1);
					m_pCamera->setDirection(cameraToCellVec);
				}
				break;*/
			case 'b':
			case 'B':
				m_pCamera->setPosition(Vector3(4, 0.5, 1));
				m_pCamera->setDirection(Vector3(-1, 0, 0));
				break;

			default:
				break;
			}
		}
		#pragma endregion

		#pragma region UpdateFunctions
		void GLRenderer3D::update(Scalar dt) {
			if (m_pCutVoxelVelRenderer) {
				m_pCutVoxelVelRenderer->update();
			}
			if (m_pParticlesRenderer) {
				m_pParticlesRenderer->update(dt);
			}
		}
		#pragma endregion

		#pragma region InitializationFunctions
		void GLRenderer3D::initGL() {
			int argc;  char **argv;
			argc = 0;
			argv = nullptr;
			glutInit(&argc, argv);
			int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
			int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
			glutInitWindowSize(screenWidth, screenHeight);
			glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
			glutCreateWindow("Fluids Renderer");
			GLenum err = glewInit();
			if (GLEW_OK != err) {
				Logger::get() << "GLEW initialization error! " << endl;
				exit(1);
			}

			/**GLUT and GLEW initialization */
			const char* GLVersion = (const char*)glGetString(GL_VERSION);
			Logger::get() << "OpenGL version: " << GLVersion << endl;

			//glEnable(GL_DEPTH_TEST);    // Enable the depth buffer
			glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); // Ask for nicest perspective correction
			//glEnable(GL_CULL_FACE);     // Cull back facing polygons
			//glCullFace(GL_BACK);

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glEnable(GL_POLYGON_SMOOTH);
			glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

			glEnable(GL_POINT_SPRITE); // GL_POINT_SPRITE_ARB if you're
			// using the functionality as an extension.

			glEnable(GL_POINT_SPRITE);
			glEnable(GL_POINT_SMOOTH);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			// Setup default render states
			glClearColor(1.0f, 1.0f, 1.0f, 1.0);
			glEnable(GL_TEXTURE_2D);

			glShadeModel(GL_SMOOTH);

			GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
			GLfloat mat_shininess[] = { 90.0 };

			glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
			glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
		
			GLfloat light_ambient[] = { 0.15, 0.15, 0.15, 1.0 };
			glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);

			glDepthFunc(GL_LEQUAL);
		}

		void GLRenderer3D::initAdditionalControls() {

		}

		void GLRenderer3D::initWindows() {
			addSimulationControlWindow();
			addSimulationStatsWindow();
			initAdditionalControls();
			addCellVisualizationWindows();


			//Adjust sim stats window
			{
				if (m_pSimStatsWindow != NULL && m_pCellVisWindows.front() != NULL) {
					Vector2 windowPosition = m_pCellVisWindows.front()->getWindowPosition();
					windowPosition.y += m_pCellVisWindows.front()->getWindowSize().y;
					m_pSimStatsWindow->setWindowPosition(windowPosition);
				}
			}

		}

		void GLRenderer3D::initCamera() { m_pCamera = new Camera(perspective3D); }
		#pragma endregion

		#pragma region InternalDrawingFunctions
		void GLRenderer3D::drawBackground() {
			glClear(/*GL_COLOR_BUFFER_BIT |*/ GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0, 1, 0, 1, -1, 1);

			glDisable(GL_DEPTH_TEST);
			glDisable(GL_LIGHTING);
			glDepthMask(GL_FALSE);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glBegin(GL_QUADS);
			//Bottom color
			glColor3f(1.0, 1.0, 1.0);
			glVertex2f(-1.0, -1.0);
			glVertex2f(1.0, -1.0);
			//Top color
			glColor3f(1.533, 1.615, 1.698);
			glVertex2f(1.0, 1.0);
			glVertex2f(-1.0, 1.0);
			glEnd();

			glEnable(GL_DEPTH_TEST);
			glDepthMask(GL_TRUE);
		}
		#pragma endregion
	}
}