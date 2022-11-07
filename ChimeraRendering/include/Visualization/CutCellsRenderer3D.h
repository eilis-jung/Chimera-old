////  Copyright (c) 2013, Vinicius Costa Azevedo
////	All rights reserved.
////
////	Redistribution and use in source and binary forms, with or without
////	modification, are permitted provided that the following conditions are met: 
////
////1. Redistributions of source code must retain the above copyright notice, this
////	list of conditions and the following disclaimer. 
////	2. Redistributions in binary form must reproduce the above copyright notice,
////	this list of conditions and the following disclaimer in the documentation
////	and/or other materials provided with the distribution. 
////
////	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
////	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
////	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
////	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
////	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
////	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
////LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
////	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
////	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
////	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////
////	The views and conclusions contained in the software and documentation are those
////	of the authors and should not be interpreted as representing official policies, 
////	either expressed or implied, of the FreeBSD Project.
//
//
//#ifndef _CHIMERA_CUT_CELLS_RENDERER_3D_
//#define _CHIMERA_CUT_CELLS_RENDERER_3D_
//#pragma once
//
///************************************************************************/
///* Data                                                                 */
///************************************************************************/
//#include "Grids/CutCells3D.h"
//
///************************************************************************/
///* Rendering                                                            */
///************************************************************************/
//#include "ChimeraRenderingCore.h"
//#include "Windows/GridVisualizationWindow.h"
//#include "Visualization/ScalarFieldRenderer.h"
//#include "RenderingUtils.h"
//#include "Visualization/CutCellsRenderer2D.h"
//#include "Visualization/HexaGridRenderer.h"
//#include "Visualization/CutVoxelRenderer3D.h"
//
//namespace Chimera {
//	using namespace Data;
//
//	namespace Rendering {
//
//		class CutCellsRenderer3D {
//
//		public:
//			#pragma region Constructors
//			CutCellsRenderer3D(CutCells3D *pCutCells, HexaGrid *pHexaGrid, nodeVelocityField3D_t *pNodeVelocityField3D, 
//								HexaGridRenderer *pHexaGridRenderer, LinearInterpolant3D<Vector3> *pLinearInterpolant);
//			#pragma endregion
//
//			#pragma region InitializationFunctions
//			void initializeWindowControls(GridVisualizationWindow<Vector3> *pGridVisualizationWindow);
//			#pragma endregion
//
//			#pragma region AccessFunctions
//			FORCE_INLINE void setMinMaxScalarFiedValues(Scalar *pMin, Scalar *pMax) {
//				m_pMinScalarFieldVal = pMin;
//				m_pMaxScalarFieldVal = pMax;
//			}
//
//			void setCutCells3D(CutCells3D *pCutCells3D) {
//				m_pCutCells = pCutCells3D;
//			}
//			CutCells3D * getCutCells3D() {
//				return m_pCutCells;
//			}			
//
//			int getSelectedCell() const {
//				return m_pCutVoxelRenderer->getSelectedVoxel();
//			}
//			void setSelectedCell(int selectedCell) {
//				m_pCutVoxelRenderer->setSelectedVoxel(selectedCell);
//			}
//
//			dimensions_t getSelectedCellDimensions() const {
//				return m_pCutVoxelRenderer->getSelectedCellDimensions();
//			}
//			#pragma endregion
//
//			#pragma region Functionalities
//			void draw();
//			void update();
//			#pragma endregion
//
//		private:
//
//			#pragma region ClassMembers
//			CutCells3D *m_pCutCells;
//			HexaGrid *m_pHexaGrid;
//			HexaGridRenderer *m_pHexaGridRenderer;
//			CutVoxelRenderer3D *m_pCutVoxelRenderer;
//			GridData3D *m_pGridData3D;
//
//			/*Cutslices by location*/
//			vector<CutSlice3D *> m_cutSlicesXY;
//			vector<CutSlice3D *> m_cutSlicesXZ;
//			vector<CutSlice3D *> m_cutSlizesYZ;
//			
//			/*Node velocity field variables*/
//			nodeVelocityField3D_t *m_pNodeVelocityField;
//
//			/**OpenGL VBOs*/
//			GLuint *m_pVertexVBO;
//			GLuint *m_pIndexVBO;
//			GLuint *m_pScalarFieldVBO;
//			GLuint *m_pScalarFieldColorsVBO;
//
//			// Cell ScalarField packed in an array of Scalar
//			Scalar *m_pScalarFieldValues;
//			// Pressure scalars
//			Scalar *m_pMinScalarFieldVal, *m_pMaxScalarFieldVal;
//
//			/** Drawing vars */
//			int m_maxNumberOfVertex;
//			int m_maxNumberOfIndex;
//			int m_totalNumberOfVertex;
//			int m_totalNumberOfIndex;
//			
//			/** General drawing config */
//			bool m_drawCells;
//			bool m_drawCutSliceLines;
//			bool m_drawAllNodeVelocities;
//
//			/** Faces drawing config */
//			faceLocation_t m_drawFaceLocation;
//			bool m_drawAllFaces;
//			int m_selectedFace;
//			int m_numberOfFacesToDraw;
//			
//			scalarColorScheme_t m_colorScheme; 
//			//Jet scalar color shader
//			shared_ptr<GLSLShader> m_pJetColorShader;
//			GLuint m_jetMinScalarLoc;
//			GLuint m_jetMaxScalarLoc;
//			GLuint m_jetAvgScalarLoc;
//
//			//Grayscale scalar color shader
//			shared_ptr<GLSLShader> m_pGrayScaleColorShader;
//			GLuint m_grayMinScalarLoc;
//			GLuint m_grayMaxScalarLoc;
//
//			//Grid visualization window
//			GridVisualizationWindow<Vector3> *m_pGridVisualizationWindow;
//			#pragma endregion
//
//			#pragma region InitializationFunctions
//			void initializeAuxiliaryDrawingStructures();
//			void initializeVBOs();
//			void initializeShaders();
//			#pragma endregion
//
//			#pragma region UpdatingFunctions
//			void updateShaderColors();
//			void setTagColor(int tag);
//			#pragma endregion
//
//			#pragma region DrawingFunctions
//			void drawFace(CutFace<Vector3D> *pFace);
//			void drawCells();
//			void drawCell(int ithCell);
//			void drawFacesPerLocation();
//			void drawAllNodesVelocities();
//
//			FORCE_INLINE void drawCell(int i, int j, int k) const {
//				glBegin(GL_QUADS);
//				glVertex3f(m_pGridData3D->getPoint(i, j, k).x, m_pGridData3D->getPoint(i, j, k).y, m_pGridData3D->getPoint(i, j, k).z);
//				glVertex3f(m_pGridData3D->getPoint(i + 1, j, k).x, m_pGridData3D->getPoint(i + 1, j, k).y, m_pGridData3D->getPoint(i + 1, j, k).z);
//				glVertex3f(m_pGridData3D->getPoint(i + 1, j + 1, k).x, m_pGridData3D->getPoint(i + 1, j + 1, k).y, m_pGridData3D->getPoint(i + 1, j + 1, k).z);
//				glVertex3f(m_pGridData3D->getPoint(i, j + 1, k).x, m_pGridData3D->getPoint(i, j + 1, k).y, m_pGridData3D->getPoint(i, j + 1, k).z);
//				glEnd();
//			}
//
//			FORCE_INLINE void applyColorShader(Scalar minValue, Scalar maxValue, Scalar avgValue) const {
//				switch(m_colorScheme) {
//				case jet:
//					m_pJetColorShader->applyShader();
//					glUniform1f(m_jetMinScalarLoc, minValue);
//					glUniform1f(m_jetMaxScalarLoc, maxValue);
//					glUniform1f(m_jetAvgScalarLoc, avgValue);
//					break;
//
//				case grayscale:
//					m_pGrayScaleColorShader->applyShader();
//					glUniform1f(m_grayMinScalarLoc, minValue);
//					glUniform1f(m_grayMaxScalarLoc, maxValue);
//					break;
//				}
//			}
//
//			FORCE_INLINE void removeColorShader() const {
//				switch(m_colorScheme) {
//				case jet:
//					m_pJetColorShader->removeShader();
//					break;
//				case grayscale:
//					m_pGrayScaleColorShader->removeShader();
//					break;
//				}
//			}
//
//			#pragma endregion
//		};
//
//
//
//	}
//}
//#endif