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

#ifndef _GL_RENDERER_2D_H_
#define _GL_RENDERER_2D_H_

#pragma  once 

#include "ChimeraSolids.h"
#include "ChimeraMesh.h"
#include "ChimeraRendering.h"
#include "ChimeraWindows.h"

namespace Chimera {

	using namespace Solids;
	using namespace Rendering;

	namespace Rendering {
		class GLRenderer2D : public Rendering::BaseGLRenderer<Vector2, Array2D, GLRenderer2D> {

		private:
			/************************************************************************/
			/* Class Members                                                        */
			/************************************************************************/
			ParticleSystem2D *m_pParticleSystem;
			Scalar m_systemRotationAngle;
			vector<RigidObject2D *> m_thinObjectVec;
			vector<LineMesh<Vector2>*> m_liquidsVec;
			vector<LineMesh<Vector2>*> m_lineMeshesVec;

			//CutCellsRenderer2D *m_pSpecialCellsRenderer;
			//RaycastRenderer2D *m_pRaycastRenderer;
			CutCells2D<Vector2> *m_pCutCells;
			CutCellSolver2D *m_pCutCellsSolver;

			CutCellsWindow *m_pSpecialCellsWindow;
			GeneralInfoWindow *m_pGeneralInfoWin;
			CutCellsWindow *m_pCutCellsWindow;
			Vector2 m_vectorInterpolatedPosition;

			LineMeshRenderer<Vector2> *m_pLineRenderer;
			PolygonMeshRenderer<Vector2> *m_pPolygonMeshRenderer;
			CutCellsVelocityRenderer2D<Vector2> *m_pCutCellsVelRenderer;

			bool m_updateCellVisualizationIndex;

			bool m_isMultigridApplication;

			//Drawing functionalities
			bool m_drawPressureInterpolationArea;
			bool m_drawVelocityInterpolationArea;
			bool m_multrigridWindowsInitialized;
			bool m_drawSpecialCells;
			bool m_cameraFollowThinObject;

			//Liquids and solid objects
			bool m_drawSolidObjects;
			bool m_drawLiquidObjects;

			//Distance map: used for drawing and comparision
			//map<int, Scalar> m_minDistanceMap;

			/************************************************************************/
			/* Initialization                                                       */
			/************************************************************************/
			void initGL();
			void initWindows();
			void initCamera();

		public:

			typedef struct params_t {
				bool initializeGridVisualization;

				params_t() {
					initializeGridVisualization = false;
				}
			} params_t;

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			GLRenderer2D();

			/************************************************************************/
			/* Callbacks                                                            */
			/************************************************************************/
			void keyboardCallback(unsigned char key, int x, int y);
			void mouseCallback(int button, int state, int x, int y);

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			void renderLoop();

			void setParticleSystem(ParticleSystem2D *pParticleSystem) {
				m_pParticleSystem = pParticleSystem;
				if (m_pParticlesVisWindow == NULL) {
					m_pParticlesVisWindow = new ParticleVisualizationWindow<Vector2>(m_pParticleSystem);
					Vector2 windowPosition = m_pGridVisWindows[0]->getWindowPosition();
					windowPosition.y += m_pGridVisWindows[0]->getWindowSize().y;
					m_pParticlesVisWindow->setWindowPosition(windowPosition);
				}

				if (m_pSimCtrlWindow != NULL) {
					Vector2 windowPosition = m_pParticlesVisWindow->getWindowPosition();
					windowPosition.y += m_pParticlesVisWindow->getWindowSize().y;
					m_pSimCtrlWindow->setWindowPosition(windowPosition);
				}
				/*if (m_pSimStatsWindow != NULL) {
				Vector2 windowPosition = m_pSimCtrlWindow->getWindowPosition();
				windowPosition.y += m_pSimCtrlWindow->getWindowSize().y;
				m_pSimStatsWindow->setWindowPosition(windowPosition);
				}*/
			}

			FORCE_INLINE void setMultigridApplication(bool multigridApplication) {
				m_isMultigridApplication = multigridApplication;
			}

			FORCE_INLINE void addSimulationConfig(SimulationConfig<Vector2, Array2D> *pSimCfg) {
				m_pSimCfgs.push_back(pSimCfg);
			}

			FORCE_INLINE void addGridRenderer(StructuredGrid<Vector2> *pQuadGrid) {
				if (dynamic_cast<QuadGrid*>(pQuadGrid) != NULL) {
					m_gridRenderers.push_back(new QuadGridRenderer(dynamic_cast<QuadGrid*>(pQuadGrid)));
				}
			}

			FORCE_INLINE void addSpecialCellsRenderer(CutCells2D<Vector2> *pCutCells2D, Interpolant<Vector2, Array2D, Vector2> *pVelocityInterpolant, const Array2D<Vector2> &nodeBasedVelocity, QuadGrid *pQuadGrid,
				GridVisualizationWindow<Vector2> *pGridVisWindow) {
				//m_pSpecialCellsRenderer = new CutCellsRenderer2D(pCutCells2D, pVelocityInterpolant, nodeBasedVelocity, pQuadGrid);
				//m_pSpecialCellsRenderer->initializeWindowControls(pGridVisWindow);
				//
				//m_pSpecialCellsRenderer->setMinMaxScalarFiedValues(m_gridRenderers[0]->getScalarFieldRenderer().getMinScalarFieldValuePtr(), 
				//													m_gridRenderers[0]->getScalarFieldRenderer().getMaxScalarFieldValuePtr());
				//m_pCellVisWindows[0]->setCutCells2D(pCutCells2D);

			}

			/*FORCE_INLINE void addRaycastRenderer(RaycastSolver2D *pRayCastSolver, GridVisualizationWindow<Vector2> *pGridVisWindow) {
			m_pRaycastRenderer = new RaycastRenderer2D(dynamic_cast<QuadGrid*>(pRayCastSolver->getGrid()));
			m_pRaycastRenderer->initializeWindowControls(pGridVisWindow);
			m_pRaycastRenderer->setLeftFaceCrossingsPtr(pRayCastSolver->getLeftCrossingsPtr());
			m_pRaycastRenderer->setBottomFaceCrossingsPtr(pRayCastSolver->getBottomCrossingsPtr());
			m_pRaycastRenderer->setLeftFacesVisibilityPtr(pRayCastSolver->getLeftFacesVisibilityPtr());
			m_pRaycastRenderer->setBottomFacesVisibilityPtr(pRayCastSolver->getBottomFacesVisibilityPtr());
			m_pRaycastRenderer->setBoundaryCellsPtr(pRayCastSolver->getBoundaryCellsPtr());
			}*/

			void update();

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			FORCE_INLINE Scalar getSystemRotationAngle() const {
				return m_systemRotationAngle;
			}

			FORCE_INLINE void setSystemRotationAngle(Scalar angle) {
				m_systemRotationAngle = angle;
			}

			FORCE_INLINE void setLineMeshes(const vector<LineMesh<Vector2>*> &lineMeshes) {
				m_lineMeshesVec = lineMeshes;
				vector<Mesh <Vector2, Edge> *> meshesVec;
				for (int i = 0; i < lineMeshes.size(); i++) {
					meshesVec.push_back(lineMeshes[i]);
				}
				m_pLineRenderer = new LineMeshRenderer<Vector2>(meshesVec, m_pCamera->getPosition());
			}

			FORCE_INLINE void setCutCellsSolver(CutCellSolver2D *pCutCellsSolver) {
				m_pCutCellsSolver = pCutCellsSolver;
				m_pCellVisWindows.front()->setCutCellSolver2D(pCutCellsSolver);
			}
			FORCE_INLINE void setCutCells(CutCells2D<Vector2> * pPlanarMesh) {
				m_pCutCells = pPlanarMesh;
				Mesh <Vector2, Face> *pMesh = pPlanarMesh;
				vector<Mesh <Vector2, Face> *> cuVec; cuVec.push_back(pMesh);
				m_pPolygonMeshRenderer = new PolygonMeshRenderer<Vector2>(cuVec, m_pCamera->getPosition());
				m_pCutCellsVelRenderer = new CutCellsVelocityRenderer2D<Vector2>(pPlanarMesh, m_pPolygonMeshRenderer->getSelectedCell());
				m_pCutCellsWindow = new CutCellsWindow(m_pPolygonMeshRenderer, m_pCutCellsVelRenderer);
				{
					Vector2 windowPosition = m_pGridVisWindows[0]->getWindowPosition();
					windowPosition.y += m_pGridVisWindows[0]->getWindowSize().y;
					m_pCutCellsWindow->setWindowPosition(windowPosition);
				}


				if (m_pParticlesVisWindow != NULL) {
					Vector2 windowPosition = m_pCutCellsWindow->getWindowPosition();
					windowPosition.y += m_pCutCellsWindow->getWindowSize().y;
					m_pParticlesVisWindow->setWindowPosition(windowPosition);
				}

				if (m_pSimCtrlWindow != NULL) {
					Vector2 windowPosition = m_pParticlesVisWindow->getWindowPosition();
					windowPosition.y += m_pParticlesVisWindow->getWindowSize().y;
					m_pSimCtrlWindow->setWindowPosition(windowPosition);
				}
				/*if (m_pSimStatsWindow != NULL) {
				Vector2 windowPosition = m_pSimCtrlWindow->getWindowPosition();
				windowPosition.y += m_pSimCtrlWindow->getWindowSize().y;
				m_pSimStatsWindow->setWindowPosition(windowPosition);
				}*/
			}

			FORCE_INLINE void setThinObjectVector(const vector<RigidObject2D *> &thinObjectVec) {
				m_thinObjectVec = thinObjectVec;
			}

			FORCE_INLINE void setLiquidsVector(const vector<LineMesh<Vector2> *> &liquidsVec) {
				m_liquidsVec = liquidsVec;
			}

			/*FORCE_INLINE CutCellsRenderer2D * getSpecialCellsRendererPtr() {
			return m_pSpecialCellsRenderer;
			}*/
		};
	}

	
}

#endif