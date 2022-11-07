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
#ifndef _GL_RENDERER_3D_H_
#define _GL_RENDERER_3D_H_

#pragma  once 


#include "ChimeraSolids.h"
#include "ChimeraMesh.h"
#include "ChimeraRendering.h"
#include "ChimeraWindows.h"

namespace Chimera {

	namespace Rendering {

		class GLRenderer3D : public BaseGLRenderer<Vector3, Array3D, GLRenderer3D> {
		public:
			#pragma region InternalStructures
			typedef struct params_t {
				bool initializeGridVisualization;

				params_t() {
					initializeGridVisualization = false;
				}
			} params_t;
			#pragma endregion

			#pragma region Constructors
			GLRenderer3D() {
				m_ithPlane = 0;
				m_gridPlane = HexaGrid::XY_Plane;
				m_pMeshRenderer = nullptr;
				m_drawAntTweakBars = true;
				m_pCutVoxelRenderer = nullptr;
				m_pCutVoxels = nullptr;
				m_pCutVoxelsWindow = nullptr;
				m_pCutVoxelVelRenderer = nullptr;
				m_pParticlesRenderer = nullptr;
			};
			#pragma endregion		

			#pragma region DrawingFunctions
			void drawGridSlices(int ithGrid);
			void drawVelocitySlices(int ithGrid);
			void renderLoop(bool swapBuffers = true);

			FORCE_INLINE void renderIthPlane(HexaGrid::gridPlanes_t g_gridPlane, int g_ithPlane) {
				m_gridPlane = g_gridPlane;
				m_ithPlane = g_ithPlane;
			}

			#pragma endregion

			#pragma region Callbacks
			virtual void keyboardCallback(unsigned char key, int x, int y);
			#pragma endregion

			#pragma region AccessFunctions
			FORCE_INLINE void addObject(PhysicalObject<Vector3> *pObject, bool addVisWindow = true) {
				m_pRenderingObjects.push_back(pObject);
				/*if (addVisWindow)
					addObjectVisualizationWindow(pObject);*/
			}
			FORCE_INLINE void addLineMeshRenderer(LineMeshRenderer<Vector3> *pLineMeshRenderer) {
				m_lineRenderers.push_back(pLineMeshRenderer);
			}

			const vector<LineMeshRenderer<Vector3> *> & getLineMeshRenderers() const {
				return m_lineRenderers;
			}
		
			FORCE_INLINE void addPolygonMeshRenderer(PolygonMeshRenderer<Vector3> *pPolyMeshRenderer) {
				m_polyMeshRenderers.push_back(pPolyMeshRenderer);
			}

			const vector<PolygonMeshRenderer<Vector3> *> & getPolygonMeshRenderers() const {
				return m_polyMeshRenderers;
			}

		
			FORCE_INLINE void setParticlesRenderer(ParticlesRenderer<Vector3, Array3D> *pParticlesRenderer) {
				m_pParticlesRenderer = pParticlesRenderer;
				if (m_pParticlesVisWindow == NULL) {
					m_pParticlesVisWindow = new ParticleVisualizationWindow<Vector3, Array3D>(m_pParticlesRenderer);
					Vector2 windowPosition = m_pGridVisWindows[0]->getWindowPosition();
					windowPosition.y += m_pGridVisWindows[0]->getWindowSize().y;
					m_pParticlesVisWindow->setWindowPosition(windowPosition);
				}

				if (m_pSimCtrlWindow != NULL) {
					Vector2 windowPosition = m_pParticlesVisWindow->getWindowPosition();
					windowPosition.y += m_pParticlesVisWindow->getWindowSize().y;
					m_pSimCtrlWindow->setWindowPosition(windowPosition);
				}

			}

			FORCE_INLINE ParticlesRenderer<Vector3, Array3D> * getParticlesRenderer() {
				return m_pParticlesRenderer;
			}

			FORCE_INLINE void addGridRenderer(StructuredGrid<Vector3> * pGrid) {
				HexaGrid *pHexaGrid = dynamic_cast<HexaGrid*>(pGrid);
				if (pHexaGrid)
					m_gridRenderers.push_back(new HexaGridRenderer(pHexaGrid));
			}

			FORCE_INLINE MeshRenderer3D<Vector3, Face> * getMeshRenderer() {
				return m_pMeshRenderer;
			}

			FORCE_INLINE void addMeshRenderer(const vector<Meshes::Mesh<Vector3, Face> *> &meshes) {
				m_pMeshRenderer = new MeshRenderer3D<Vector3, Face>(meshes, m_pCamera->getPosition());
			}

		
			FORCE_INLINE void setCutVoxels(CutVoxels3D<Vector3> *pCutVoxels3D, MeanValueInterpolant3D<Vector3> *pMeanValueInterpolant = nullptr) {
				m_pCutVoxels = pCutVoxels3D;
				//Initializing volumeMeshRenderer
				vector<Mesh <Vector3, Volume> *> volumeMeshesVec;
				volumeMeshesVec.push_back(pCutVoxels3D);
				m_pCutVoxelRenderer = new VolumeMeshRenderer<Vector3>(volumeMeshesVec, getCamera()->getPosition());
				m_pCutVoxelVelRenderer = new CutVoxelsVelocityRenderer3D<Vector3>(m_pCutVoxels, m_pCutVoxelRenderer->getSelectedCutVoxel(), pMeanValueInterpolant);
				m_pCutVoxelsWindow = new CutVoxelsWindow<Vector3>(m_pCutVoxelRenderer, m_pCutVoxelVelRenderer);

				addWindow(m_pCutVoxelsWindow);

				/** Adjusting windows below it */
				if(m_pGridVisWindows.size() > 0 &&  m_pGridVisWindows.front() != nullptr)  {
					Vector2 windowPosition = m_pGridVisWindows.front()->getWindowPosition();
					windowPosition.y += m_pGridVisWindows.front()->getWindowSize().y;
					m_pCutVoxelsWindow->setWindowPosition(windowPosition);
				}
				if (m_pParticlesVisWindow != NULL) {
					Vector2 windowPosition = m_pCutVoxelsWindow->getWindowPosition();
					windowPosition.y += m_pCutVoxelsWindow->getWindowSize().y;
					m_pParticlesVisWindow->setWindowPosition(windowPosition);
				}

				if (m_pSimCtrlWindow != NULL) {
					Vector2 windowPosition = m_pParticlesVisWindow->getWindowPosition();
					windowPosition.y += m_pParticlesVisWindow->getWindowSize().y;
					m_pSimCtrlWindow->setWindowPosition(windowPosition);
				}

			}

		#pragma endregion

			#pragma region UpdateFunctions
			void update(Scalar dt);
			#pragma endregion

		protected:
			#pragma region ClassMembers
			/*Objects rendering*/
			vector<PhysicalObject<Vector3> *> m_pRenderingObjects;
			ParticlesRenderer<Vector3, Array3D> *m_pParticlesRenderer;
			MeshRenderer3D<Vector3, Face> *m_pMeshRenderer;
			CutVoxels3D<Vector3> *m_pCutVoxels;

			/*Slices rendering*/
			HexaGrid::gridPlanes_t m_gridPlane;
			vector<int> m_ithPlanes;
			int m_ithPlane;

			bool m_drawAntTweakBars;

			CutVoxelsWindow<Vector3> *m_pCutVoxelsWindow;
			vector<LineMeshRenderer<Vector3> *> m_lineRenderers;
			vector<PolygonMeshRenderer<Vector3> *> m_polyMeshRenderers;
			VolumeMeshRenderer<Vector3> *m_pCutVoxelRenderer;
			CutVoxelsVelocityRenderer3D<Vector3> *m_pCutVoxelVelRenderer;

			#pragma endregion

			#pragma region InitilizationFunctions
			void initGL();
			void initAdditionalControls();
			void initWindows();
			void initCamera();
			#pragma endregion

			#pragma region InternalDrawingFunctions
			FORCE_INLINE void drawFlowVariables(int ithGrid) {
				if (m_pGridVisWindows[ithGrid]->getScalarFieldType() != BaseWindow::drawNoScalarField) {
					m_gridRenderers[ithGrid]->getScalarFieldRenderer().beginDrawScalarField(m_pGridVisWindows[ithGrid]->getScalarFieldType());
					m_gridRenderers[ithGrid]->getScalarFieldRenderer().endDrawScalarField();
				}
			}

			void drawBackground();
			#pragma endregion
		};
	}
}

#endif