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

#ifndef _RENDERING_BASE_RENDERER_H_
#define _RENDERING_BASE_RENDERER_H_

#pragma once

#include "ChimeraCore.h"

#include "Primitives/Camera.h"
#include "Visualization/ScalarFieldRenderer.h"

/************************************************************************/
/* Windows                                                              */
/************************************************************************/
#include "Windows/BaseWindow.h"
#include "Windows/SimulationControlWindow.h"
#include "Windows/SimulationStatsWindow.h"
#include "Windows/GridVisualizationWindow.h"
#include "Windows/CellVisualizationWindow.h"
#include "Windows/ObjectVisualizationWindow.h"
#include "Windows/ParticleVisualizationWindow.h"

/************************************************************************/
/* Ant tweak bar                                                        */
/************************************************************************/
#include "AntTweakBar.h"


using namespace std;

namespace Chimera {

	using namespace Windows;
	namespace Rendering { 
		
		template <class VectorT, template <class> class ArrayType, class RenderClassT>
		class BaseGLRenderer : public Singleton<RenderClassT> {

		public:

			/************************************************************************/
			/* ctors, initialization and dtors					                    */
			/************************************************************************/
			//Default constructor
			explicit BaseGLRenderer() {
				m_pSimCtrlWindow = NULL;
				m_pSimStatsWindow = NULL;
				m_pParticlesVisWindow = NULL;
				m_windowWidth = m_windowHeight = 0;
			}

			void initialize(Scalar windowWidth, Scalar windowHeight) {
				initGL();

				m_screenWidth = glutGet(GLUT_SCREEN_WIDTH);
				m_screenHeight = glutGet(GLUT_SCREEN_HEIGHT);

				m_windowWidth = m_screenWidth;
				m_windowHeight = m_screenHeight;

				initAntTweakBar();
				initCamera();
				initWindows();
				initGridVariables();				
			}

			/** Initialization function: Must be called before any other function!*/
			virtual void initGL() { };
			virtual void initWindows() { };
			virtual void initGridVariables() { };
			virtual void initCamera() { };
			virtual void renderLoop() { };

			//Destructor
			virtual ~BaseGLRenderer() {
				
			}
			
			/************************************************************************/
			/* Callbacks                                                            */
			/************************************************************************/

			/**A keyboard callback, that is util when assigning keyboard entries for
			renderization parameters modifications.
			@param key: the pushed key*/
			virtual void keyboardCallback(unsigned char key, int x, int y) {
				if (!TwEventKeyboardGLUT(key, x, y)) {
					m_pCamera->keyboardCallback(key, x, y);
				}
			}

			/**A mouse callback, that is util when assigning mouse callbacks for
			renderization parameters modifications.
			@param button: mouse button ID
			@param state: mouse button state
			@param x: mouse poisition according to the x axis.
			@param y: mouse poisition according to the x axis. */
			virtual void mouseCallback(int button, int state, int x, int y) {
				if(!TwEventMouseButtonGLUT(button, state, x, y))
					m_pCamera->mouseCallback(button, state, x, y);
			}

			/**A mouse motion callback, that is util when assigning mouse callbacks for
			renderization parameters modifications.
			@param x: mouse poisition according to the x axis.
			@param y: mouse poisition according to the x axis. */
			virtual void motionCallback(int x, int y) {
				if(!TwEventMouseMotionGLUT(x, y))
					m_pCamera->motionCallback(x, y);
			}

			/** Reshape callback*/
			void reshapeCallback(int width, int height) {
				m_windowWidth = width;
				m_windowHeight = height;
				m_pCamera->reshapeCallback(width, height);
				glViewport(0, 0, width, height);

				// Send the new window size to AntTweakBar
				TwWindowSize(width, height);
			}

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			FORCE_INLINE int getScreenWidth() const {
				return m_screenWidth;
			}
			
			FORCE_INLINE int getScreenHeight() const {
				return m_screenHeight;
			}

			GridRenderer<VectorT> * getGridRenderer(int i) {
				return m_gridRenderers[i];
			}

			FORCE_INLINE void addWindow(BaseWindow *pWindow) {
				m_windows.push_back(pWindow);
			}

			FORCE_INLINE Camera * getCamera() const {
				return m_pCamera;
			}

			FORCE_INLINE void addObject(PhysicalObject<VectorT> *pObject) {
				m_pRenderingObjects.push_back(pObject);
			}

			FORCE_INLINE GridVisualizationWindow<VectorT> * getGridVisualizationWindow() const {
				if(m_pGridVisWindows.size() > 0) {
					return m_pGridVisWindows[0];
				}
				return NULL;
			}

			FORCE_INLINE SimulationStatsWindow<VectorT, ArrayType> * getSimulationStatsWindow() const {
				return m_pSimStatsWindow;
			}

			FORCE_INLINE CellVisualizationWindow<VectorT, ArrayType> * getCellVisualizationWindow() const {
				return m_pCellVisWindows[0];
			}

			FORCE_INLINE void setFlowSolver(FlowSolver<VectorT, ArrayType> *pFlowSolver) {
				m_pFlowSolver = pFlowSolver;
			}

			/************************************************************************/
			/* Windows customizable features										*/
			/************************************************************************/
			FORCE_INLINE void addSimulationControlWindow() {
				if(m_pFlowSolver != nullptr) {
					m_pSimCtrlWindow = new SimulationControlWindow<VectorT>(m_pFlowSolver);
					addWindow(m_pSimCtrlWindow);
				}
			}

			FORCE_INLINE void addSimulationStatsWindow() {
				if (m_pFlowSolver != nullptr) {
					m_pSimStatsWindow = new SimulationStatsWindow<VectorT, ArrayType>(m_pFlowSolver);
					addWindow(m_pSimStatsWindow);
				}
			}

			virtual FORCE_INLINE void addGridRenderer(StructuredGrid<VectorT> * pQuadGrid) = 0;

			virtual FORCE_INLINE void addGridVisualizationWindow(StructuredGrid<VectorT> *pGrid) {
				if(pGrid != NULL) {
					int gridRendererIndex = m_pGridVisWindows.size();
					addGridRenderer(pGrid);
					m_pGridVisWindows.push_back(new GridVisualizationWindow<VectorT>(m_gridRenderers[gridRendererIndex]));
					if(m_pGridVisWindows.size() > 1) {
						Vector2 newPosition = m_pGridVisWindows[m_pGridVisWindows.size() - 2]->getWindowPosition();
						//Size + windowPadding
						newPosition.x += m_pGridVisWindows[m_pGridVisWindows.size() - 2]->getWindowSize().x; 
						m_pGridVisWindows[m_pGridVisWindows.size() - 1]->setWindowPosition(newPosition);
					}
					if(m_pSimCtrlWindow != NULL) {
						Vector2 windowPosition = m_pGridVisWindows[0]->getWindowPosition();
						windowPosition.y += m_pGridVisWindows[0]->getWindowSize().y;
						m_pSimCtrlWindow->setWindowPosition(windowPosition);
					}
					/*if(m_pSimStatsWindow != NULL) {
						Vector2 windowPosition = m_pSimCtrlWindow->getWindowPosition();
						windowPosition.y += m_pSimCtrlWindow->getWindowSize().y;
						m_pSimStatsWindow->setWindowPosition(windowPosition);
					} */
					addWindow(m_pGridVisWindows[m_pGridVisWindows.size() - 1]);
				}
			}

			FORCE_INLINE void addCellVisualizationWindows() {	
				CellVisualizationWindow<VectorT, ArrayType> *pCellVisualizationWin = new CellVisualizationWindow<VectorT, ArrayType>(m_pFlowSolver);
				
				//Window position
				Vector2 windowPosition;
				windowPosition.x = getScreenWidth() - pCellVisualizationWin->getWindowSize().x - 32;
				windowPosition.y = 16 + 0 * pCellVisualizationWin->getWindowSize().y;
				pCellVisualizationWin->setWindowPosition(windowPosition);

				m_pCellVisWindows.push_back(pCellVisualizationWin);
				addWindow(pCellVisualizationWin);
			}


		protected:
			/** The openGL window width */
			Scalar	m_windowWidth;
			/** The openGL window height */
			Scalar	m_windowHeight;

			/** Current full screen width and size */
			int m_screenWidth;
			int m_screenHeight;

			/************************************************************************/
			/* Specialized window pointers                                          */
			/************************************************************************/
			vector<GridVisualizationWindow<VectorT> *> m_pGridVisWindows;
			vector<CellVisualizationWindow<VectorT, ArrayType> *> m_pCellVisWindows;
			//vector<ObjectVisualizationWindow *> m_pObjectVisWindows;
			SimulationControlWindow<VectorT> *m_pSimCtrlWindow;
			SimulationStatsWindow<VectorT, ArrayType> *m_pSimStatsWindow;
			ParticleVisualizationWindow<VectorT, ArrayType> *m_pParticlesVisWindow;

			/************************************************************************/
			/* Objects                                                              */
			/************************************************************************/
			Camera *m_pCamera;
			vector<BaseWindow *> m_windows;
			vector<PhysicalObject<VectorT> *> m_pRenderingObjects;

			/************************************************************************/
			/* Grid renderers                                                       */
			/************************************************************************/
			vector<GridRenderer<VectorT> *> m_gridRenderers;

			FlowSolver<VectorT, ArrayType> *m_pFlowSolver;

			/************************************************************************/
			/* Initialization functions		                                        */
			/************************************************************************/
			/** Initialization function */
			void initAntTweakBar() {
				//Init antTweak Bar
				if(!TwInit(TW_OPENGL, NULL)) {
					Logger::getInstance()->log(string("AntTweakBarError: ") + TwGetLastError(), Log_HighPriority);
				}
				// Tell the window size to AntTweakBar
				TwWindowSize(m_windowWidth, m_windowHeight);
			}


			/************************************************************************/
			/* Internal drawing and updating functions                              */
			/************************************************************************/
			FORCE_INLINE void updateWindows() {
				for(int i = 0; i < m_windows.size(); i++) {
					m_windows[i]->update();
				}
			}

			FORCE_INLINE void drawGrids() {
				for(int i = 0; i < m_pGridVisWindows.size(); i++) {
					drawGrid(i);
				}
			}

			
			FORCE_INLINE void drawFlowVariables() {
				for(int i = 0; i < m_pGridVisWindows.size(); i++) {
					drawFlowVariables(i);
				}
			}

			FORCE_INLINE void drawFlowVariables(int ithGrid) {
				if(m_pGridVisWindows[ithGrid]->getScalarFieldType() != BaseWindow::drawNoScalarField) {
					m_gridRenderers[ithGrid]->getScalarFieldRenderer().beginDrawScalarField(m_pGridVisWindows[ithGrid]->getScalarFieldType());
					m_gridRenderers[ithGrid]->getScalarFieldRenderer().endDrawScalarField();
					m_gridRenderers[ithGrid]->getScalarFieldRenderer().getIsocontourRenderer().drawIsocontours();
					/*if(m_pGridVisWindows[ithGrid]->drawIsocontours()) {
						m_gridRenderers[ithGrid]->getFVRenderer().drawIsocontours(m_pGridVisWindows[ithGrid]->getScalarFieldType(), m_pGridVisWindows[ithGrid]->getNumberIsocontours());
					}*/
				}	
			}


			FORCE_INLINE void drawGrid(int ithGrid) {
				if(m_pGridVisWindows[ithGrid]->drawGridSolidCells()) {
					m_gridRenderers[ithGrid]->draw(Rendering::drawSolidCells);
				}
				if (m_pGridVisWindows[ithGrid]->drawRegularGridCells()) {
					m_gridRenderers[ithGrid]->draw(Rendering::drawCells);
				}
			}

			FORCE_INLINE void drawVelocityField(int ithGrid) {
				switch(m_pGridVisWindows[ithGrid]->getVelocityDrawingType()) {
				case BaseWindow::drawStaggeredVelocity:
					m_gridRenderers[ithGrid]->getVectorFieldRenderer().drawStaggeredVelocityField();
					break;
				case BaseWindow::drawVelocity:
					m_gridRenderers[ithGrid]->getVectorFieldRenderer().drawVelocityField(false);
					break;

				case BaseWindow::drawAuxiliaryVelocity:
					m_gridRenderers[ithGrid]->getVectorFieldRenderer().drawVelocityField(true);
					break;

				case BaseWindow::drawNodeVelocity:
					m_gridRenderers[ithGrid]->getVectorFieldRenderer().drawNodeVelocityField();
					break;

				case BaseWindow::drawGradients:
					if(m_pGridVisWindows[ithGrid]->getScalarFieldType() != BaseWindow::drawNoScalarField)
						m_gridRenderers[ithGrid]->getVectorFieldRenderer().drawScalarFieldGradients(m_pGridVisWindows[ithGrid]->getScalarFieldType());
					break;
				}
			}
		};
	}
}

#endif
