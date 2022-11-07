//  Copyright (c) 2013, Vinicius Costa Azevedo
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met: 
//
//1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer. 
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution. 
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//	The views and conclusions contained in the software and documentation are those
//	of the authors and should not be interpreted as representing official policies, 
//	either expressed or implied, of the FreeBSD Project.

#include "GLRenderer2D.h"

namespace Chimera {
	namespace Rendering {

		#pragma region Constructors
		GLRenderer2D::GLRenderer2D() {
			m_pCutCellsWindow = nullptr;
			//m_pSpecialCellsRenderer = NULL;
			//m_pRaycastRenderer = NULL; 
			m_isMultigridApplication = false;
			m_drawPressureInterpolationArea = m_drawVelocityInterpolationArea = false;
			m_systemRotationAngle = 0;
			m_multrigridWindowsInitialized = false;
			m_updateCellVisualizationIndex = true;
			m_drawSpecialCells = true;
			m_cameraFollowThinObject = false;
			m_drawLiquidObjects = m_drawSolidObjects = true;
			m_pPolygonMeshRenderer = nullptr;
			m_pLineRenderer = nullptr;
			m_pCutCellsVelRenderer = nullptr;
			m_pCutCells = nullptr;
			m_pCutCellsSolver = nullptr;
			m_pParticlesRenderer = nullptr;
		}
		#pragma endregion

		#pragma region Callbacks
		void GLRenderer2D::keyboardCallback(unsigned char key, int x, int y) {
			BaseGLRenderer::keyboardCallback(key, x, y);
			GridData2D *pGridData = m_pFlowSolver->getGrid()->getGridData2D();
			Interpolant<Vector2, Array2D, Vector2> *pVelocityInterpolant = m_pFlowSolver->getVelocityInterpolant();
			Scalar dx = pGridData->getScaleFactor(1, 1).x;
			//CutCellSolver2D *pCutCell =  dynamic_cast<CutCellSolver2D *>(m_pSimCfgs[0]->getFlowSolver());
			switch (key) {
				//Camera movement:
			case 'i': case 'I':
				m_vectorInterpolatedPosition = m_pCamera->getWorldMousePosition();
				Vector2 interpVec = pVelocityInterpolant->interpolate(m_pCamera->getWorldMousePosition());
				m_pGeneralInfoWin->setVectorInterpolationPosition(interpVec);
				break;
			}
		}

		void GLRenderer2D::mouseCallback(int button, int state, int x, int y) {
			if (!TwEventMouseButtonGLUT(button, state, x, y)) {
				if (m_updateCellVisualizationIndex && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
					Vector3 cameraRayWorld = m_pCamera->getOGLPos(x, y);
					Vector2 cameraRay = Vector2(cameraRayWorld.x, cameraRayWorld.y);
					bool foundCell = false;
					int i, j = 0;
					int selectedGridIndex = 0;
					QuadGrid *pQuadGrid = dynamic_cast<QuadGrid*>(m_pFlowSolver->getGrid());


					Vector2 transformedCamRay = cameraRay / pQuadGrid->getGridData2D()->getScaleFactor(0, 0).x;
					int selectedCellIndex = m_pCellVisWindows[0]->selectCell(transformedCamRay);
					if (m_pPolygonMeshRenderer) {
						CutCells2D<Vector2> *pCutCells = m_pPolygonMeshRenderer->getCutCells();
						dimensions_t selectedCell = dimensions_t(floor(transformedCamRay.x), floor(transformedCamRay.y));
						dimensions_t gridDim = m_gridRenderers.front()->getGrid()->getDimensions();
						if (selectedCell.x < 0 || selectedCell.x > gridDim.x - 1 || selectedCell.y < 0 || selectedCell.y > gridDim.y - 1) {
							selectedCell = dimensions_t(0, 0);
						}
						if (pCutCells->isCutCell(selectedCell)) {
							m_pPolygonMeshRenderer->setSelectedCutCell(pCutCells->getCutCellIndex(transformedCamRay));
							m_pCellVisWindows[0]->switchCellIndex(dimensions_t(0, 0));
							m_gridRenderers[selectedGridIndex]->setSelectedCell(dimensions_t(0, 0));
						}
						else {
							m_pCellVisWindows[0]->switchCellIndex(selectedCell);
							m_gridRenderers[selectedGridIndex]->setSelectedCell(selectedCell);
							m_pPolygonMeshRenderer->setSelectedCutCell(-1);
						}
					}
					else {
						dimensions_t selectedCell = dimensions_t(floor(transformedCamRay.x), floor(transformedCamRay.y));
						dimensions_t gridDim = m_gridRenderers.front()->getGrid()->getDimensions();
						if (selectedCell.x < 0 || selectedCell.x > gridDim.x - 1 || selectedCell.y < 0 || selectedCell.y > gridDim.y - 1) {
							selectedCell = dimensions_t(0, 0);
						}
						m_pCellVisWindows[0]->switchCellIndex(selectedCell);
						m_gridRenderers[selectedGridIndex]->setSelectedCell(selectedCell);
					}
				}
			}
			m_pCamera->mouseCallback(button, state, x, y);
		}
		#pragma endregion

		#pragma region Functionalities
		void GLRenderer2D::renderLoop() {
			updateWindows();

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);


			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			m_pCamera->updateGL();

			// Setup modelview matrix
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glPushMatrix();

			//Find min max scalar field value
			Scalar minScalarFieldVal = 1e10;
			Scalar maxScalarFieldVal = -1e10;
			for (int i = 0; i < m_gridRenderers.size(); i++) {
				//if(m_pSpecialCellsRenderer) {
				//	 m_pGridVisWindows[i]->getGridRenderer()->getScalarFieldRenderer().updateMinMaxScalarField(m_pGridVisWindows[0]->getScalarFieldType());

				//	Scalar specialCellsMin = FLT_MAX;
				//	Scalar specialCellsMax = -FLT_MAX;
				//	/*CutCells2D *pSpecialCells = m_pSpecialCellsRenderer->getCutCells2D();
				//	for(int j = 0; j < pSpecialCells->getNumberOfCells(); j++) {
				//		Scalar currPressure = pSpecialCells->getPressure(j); 
				//		if(currPressure < specialCellsMin) {
				//			specialCellsMin = currPressure;
				//		} 
				//		if(currPressure > specialCellsMax) {
				//			specialCellsMax = currPressure;
				//		}
				//	}*/
				//	if(specialCellsMin < m_pGridVisWindows[i]->getGridRenderer()->getScalarFieldRenderer().m_minScalarFieldVal) {
				//		m_pGridVisWindows[i]->getGridRenderer()->getScalarFieldRenderer().m_minScalarFieldVal = specialCellsMin;
				//	} 
				//	if(specialCellsMax > m_pGridVisWindows[i]->getGridRenderer()->getScalarFieldRenderer().m_maxScalarFieldVal) {
				//		m_pGridVisWindows[i]->getGridRenderer()->getScalarFieldRenderer().m_maxScalarFieldVal = specialCellsMax;
				//	} 
				//}
			}

			//Inverse drawing order
			for (int i = 0; i < m_pGridVisWindows.size(); i++) {
				glPushMatrix();
				glTranslatef(m_pGridVisWindows[i]->getGridRenderer()->getGrid()->getPosition().x, m_pGridVisWindows[i]->getGridRenderer()->getGrid()->getPosition().y, 0);

				Vector2 gridPosition = m_pGridVisWindows[i]->getGridRenderer()->getGrid()->getGridCentroid() - m_pGridVisWindows[i]->getGridRenderer()->getGrid()->getPosition();
				Vector2 gridCentroid = m_pGridVisWindows[i]->getGridRenderer()->getGrid()->getGridCentroid();// - m_pGridVisWindows[i]->getGrid()->getPosition();

				if (i == 1) {
					glTranslatef(gridCentroid.x, gridCentroid.y, 0);
					glRotatef(RadToDegree(m_systemRotationAngle), 0, 0, 1);
					glTranslatef(-gridCentroid.x, -gridCentroid.y, 0);
				}


				/*	m_pGridVisWindows[i]->getGrid()->getFVRenderer()->m_minScalarFieldVal = minScalarFieldVal;
				m_pGridVisWindows[i]->getGrid()->getFVRenderer()->m_maxScalarFieldVal = maxScalarFieldVal;*/

				Vector2 gridPositionV2 = m_pGridVisWindows[i]->getGridRenderer()->getGrid()->getPosition();

				if (m_pCutCells && m_pCutCellsSolver) {
					if (m_pGridVisWindows[0]->getScalarFieldType() == BaseWindow::drawPressure) {
						m_gridRenderers[0]->getScalarFieldRenderer().setCutCells(m_pCutCells, m_pCutCellsSolver->getPressuresVectorPtr());
					}
					else if (m_pGridVisWindows[0]->getScalarFieldType() == BaseWindow::drawDivergent) {
						m_gridRenderers[0]->getScalarFieldRenderer().setCutCells(m_pCutCells, m_pCutCellsSolver->getDivergentsVectorPtr());
					}
				}
				drawFlowVariables(i);

				/*{
				QuadGrid *pQuadGrid = dynamic_cast<QuadGrid *>(m_pGridVisWindows[i]->getGrid());
				pQuadGrid->drawSelectedCell();
				}*/
				drawGrid(i);

				/*if(m_pSpecialCellsRenderer != NULL) {
				m_pSpecialCellsRenderer->draw();
				}*/

				/*if (m_pRaycastRenderer != NULL) {
				m_pRaycastRenderer->draw();
				}*/

				drawVelocityField(i);

				//CutCellSolver2D *pVarSolver = dynamic_cast<CutCellSolver2D*>(m_pSimCfgs[0]->getFlowSolver());
				//if(pVarSolver) {
				//	vector<RigidThinObject2D *> thinObjectVec = pVarSolver->getThinObjectVec();
				//	for(int i = 0; i < thinObjectVec.size(); i++) {
				//		//thinObjectVec[i]->draw();
				//	}
				//	//drawPolygonMesh(pVarSolver->getFlipAdvection()->getPolygonMeshPtr());
				//}

				if (m_pGeneralInfoWin->drawVectorInterpolation()) {
					RenderingUtils::getInstance()->drawVector(m_vectorInterpolatedPosition, m_vectorInterpolatedPosition + m_pGeneralInfoWin->getInterpolatedVector()*0.01, 0.01);
				}
				glPopMatrix();
			}

			if (m_pParticlesRenderer) {
				m_pParticlesRenderer->draw();
			}

			if (m_pCutCellsVelRenderer) {
				m_pCutCellsVelRenderer->draw();
			}
			if (m_pPolygonMeshRenderer)
				m_pPolygonMeshRenderer->draw();

			if (m_pLineRenderer)
				m_pLineRenderer->draw();

			/*for(int i = 0; i < m_pRenderingObjects.size(); i++) {
			m_pRenderingObjects[i]->draw();
			}
			for(int i = 0; i < m_pRenderingObjects.size(); i++) {
			m_pRenderingObjects[i]->draw();
			}

			if(m_drawSolidObjects) {
			for (int j = 0; j < m_thinObjectVec.size(); j++) {
			m_thinObjectVec[j]->draw();
			}
			}
			*/

			if (m_drawLiquidObjects) {
				/*for (int j = 0; j < m_liquidsVec.size(); j++) {
				glLineWidth(6.0f);
				glBegin(GL_LINES);
				glColor3f(0.25f, 0.643f, 0.874f);
				for (int i = 0; i < m_liquidsVec[j]->getPoints().size() - 1; i++) {
				glVertex2f(m_liquidsVec[j]->getPoints()[i].x, m_liquidsVec[j]->getPoints()[i].y);
				glVertex2f(m_liquidsVec[j]->getPoints()[i + 1].x, m_liquidsVec[j]->getPoints()[i + 1].y);
				}
				glEnd();
				glLineWidth(1.0f);
				}*/
			}

			/*GhostLiquidSolver *pGhostSolver = dynamic_cast<GhostLiquidSolver *>(m_pSimCfgs[0]->getFlowSolver());

			if (pGhostSolver && m_pGridVisWindows[0]->drawGridSolidCells()) {
			for (int i = 0; i < pGhostSolver->getGrid()->getDimensions().x; i++) {
			for (int j = 0; j < pGhostSolver->getGrid()->getDimensions().y; j++) {
			glPolygonMode(GL_FRONT, GL_FILL);
			glColor3f(0.789, 0.6465, 0.232f);
			if (pGhostSolver->isBoundaryCell(i, j)) {
			m_gridRenderers.front()->drawCell(i, j);
			}
			}
			}
			}*/



			glPopMatrix();

			// Draw tweak bars
			TwDraw();


			glutSwapBuffers();
			glutPostRedisplay();
		}

		void GLRenderer2D::update(Scalar dt) {
			//Scalar dt = PhysicsCore<Vector2>::getInstance()->getParams()->timestep;
			//if(m_pSpecialCellsRenderer) {
			//	m_pSpecialCellsRenderer->update();
			//}

			/*if (m_pParticleSystem) {
				Scalar minvalue = m_pGridVisWindows[0]->getGridRenderer()->getScalarFieldRenderer().m_minScalarFieldVal;
				Scalar maxvalue = m_pGridVisWindows[0]->getGridRenderer()->getScalarFieldRenderer().m_maxScalarFieldVal;
				m_pParticleSystem->getRenderingParams().minScalarfieldValue = m_pGridVisWindows[0]->getGridRenderer()->getScalarFieldRenderer().m_minScalarFieldVal;
				m_pParticleSystem->getRenderingParams().maxScalarfieldValue = m_pGridVisWindows[0]->getGridRenderer()->getScalarFieldRenderer().m_maxScalarFieldVal;
				m_pParticleSystem->update(PhysicsCore<Vector2>::getInstance()->getParams()->timestep);
			}*/


			/*if(m_cameraFollowThinObject && m_thinObjectVec.size() > 0) {
			Vector3 cameraPosition(m_thinObjectVec[0]->getLineMeshPtr()->getPoints()[0].x, m_pCamera->getPosition().y, m_pCamera->getPosition().z);
			m_pCamera->setPosition(cameraPosition);
			}*/

			if (m_pParticlesRenderer) {
				m_pParticlesRenderer->update(dt);
			}
		}
		#pragma endregion

		#pragma region InitializationFunctions
		void GLRenderer2D::initGL() {
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

			glClearColor(1.0f, 1.0f, 1.0f, 1.0);
		}

		void GLRenderer2D::initWindows() {
			addSimulationControlWindow();
			addSimulationStatsWindow();
			addCellVisualizationWindows();
			//GLrenderer2D windows

			//Adjust sim stats window
			{
				if (m_pSimStatsWindow != NULL && m_pCellVisWindows.front() != NULL) {
					Vector2 windowPosition = m_pCellVisWindows.front()->getWindowPosition();
					windowPosition.y += m_pCellVisWindows.front()->getWindowSize().y;
					m_pSimStatsWindow->setWindowPosition(windowPosition);
				}
			}
			//Initialize and adjust GLRendererWindow
			GLRenderer2DWindow *pGLRendererWindow = nullptr;
			{
				GLRenderer2DWindow::params_t params;
				params.m_pDrawObjectsMeshes = &m_drawSolidObjects;
				params.m_pDrawLiquidMeshes = &m_drawLiquidObjects;
				pGLRendererWindow = new GLRenderer2DWindow(params);
				if (m_pSimStatsWindow != nullptr) {
					Vector2 windowPosition;
					windowPosition.x = getScreenWidth() - m_pSimStatsWindow->getWindowSize().x - 32;
					windowPosition.y = m_pSimStatsWindow->getWindowPosition().y + m_pSimStatsWindow->getWindowSize().y;
					pGLRendererWindow->setWindowPosition(windowPosition);
				}

				addWindow(pGLRendererWindow);
			}

			//Initialize anmd adjust General info window
			{
				m_pGeneralInfoWin = new GeneralInfoWindow(m_pCamera);
				Vector2 windowPosition;
				windowPosition.x = getScreenWidth() - m_pGeneralInfoWin->getWindowSize().x - 32;
				windowPosition.y = pGLRendererWindow->getWindowPosition().y + pGLRendererWindow->getWindowSize().y;
				m_pGeneralInfoWin->setWindowPosition(windowPosition);
				addWindow(m_pGeneralInfoWin);
			}
		}

		void GLRenderer2D::initCamera() {
			m_pCamera = new Camera(orthogonal2D);
		}
		#pragma endregion
	
	}	
}