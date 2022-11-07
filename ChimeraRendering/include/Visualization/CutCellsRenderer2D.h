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


#ifndef _CHIMERA_SPECIAL_CELLS_RENDERER__
#define _CHIMERA_SPECIAL_CELLS_RENDERER__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraCutCells.h"
#include "ChimeraResources.h"
#include "ChimeraInterpolation.h"

#include "Visualization/ScalarFieldRenderer.h"
#include "RenderingUtils.h"

namespace Chimera {

	namespace Windows {
		template <class VectorT>
		class GridVisualizationWindow;
	}
	using namespace Core;
	using namespace CutCells;
	using namespace Resources;
	using namespace Interpolation;

	namespace Rendering {

		class CutCellsRenderer2D {

		public:
			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			CutCellsRenderer2D(CutCells2D<Vector2> *pSpecialCells, Interpolant<Vector2, Array2D, Vector2> *pVelocityInterpolant, const Array2D<Vector2> &nodeVelocityField, QuadGrid *pQuadGrid);
			
			/************************************************************************/
			/* External settings                                                    */
			/************************************************************************/
			void initializeWindowControls(GridVisualizationWindow<Vector2> *pGridVisualizationWindow);
			FORCE_INLINE void setMinMaxScalarFiedValues(Scalar *pMin, Scalar *pMax) {
				m_pMinScalarFieldVal = pMin;
				m_pMaxScalarFieldVal = pMax;
			}

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/			
			void setCutCells2D(CutCells2D<Vector2> *pCutCells2D) {
				m_pSpecialCells = pCutCells2D;
			}
			CutCells2D<Vector2> * getCutCells2D() {
				return m_pSpecialCells;
			}			

			void setSelectedCell(int selectedCell) {
				m_selectedCell = selectedCell;
			}
			/************************************************************************/
			/* Functionalities		                                                */
			/************************************************************************/
			void draw();

			/** If the CutCells2D has changed, call this function */
			void update();


		private:

			/************************************************************************/
			/* Private structures                                                   */
			/************************************************************************/
			typedef struct velocityNode_t {
				int specialCellID;
				vector<Vector2> velocityPoints;
				vector<Vector2> velocities;
			} velocityNode_t;

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			CutCells2D<Vector2> *m_pSpecialCells;
			Interpolant<Vector2, Array2D, Vector2> *m_pVelocityInterpolant;
			const Array2D<Vector2> &m_nodeVelocityField;
			QuadGrid *m_pQuadGrid;

			/**OpenGL VBOs*/
			GLuint *m_pVertexVBO;
			GLuint *m_pIndexVBO;
			GLuint *m_pScalarFieldVBO;
			GLuint *m_pScalarFieldColorsVBO;

			/** Auxiliary drawing structures */
			// Cell points packed in an array of Vector2
			Vector2 *m_pGridCellPoints;
			// Cell indexes packed in an array of ints
			int *m_pGridCellIndex;
			// Cell ScalarField packed in an array of Scalar
			Scalar *m_pScalarFieldValues;
			// Pressure scalars
			Scalar *m_pMinScalarFieldVal, *m_pMaxScalarFieldVal;
			// Special cells velocities
			vector<velocityNode_t> m_velocitiesNodes;
			

			/** Drawing vars */
			int m_maxNumberOfVertex;
			int m_maxNumberOfIndex;
			int m_totalNumberOfVertex;
			int m_totalNumberOfIndex;
			int m_selectedCell;

			/** Ant-tweak vars */
			bool m_drawCells;
			bool m_drawPressures;
			bool m_drawVelocities;
			bool m_drawNodeVelocities;
			bool m_drawFaceVelocities;
			bool m_drawFaceNormals;
			bool m_drawSelectedCells;
			bool m_drawCellNeighbors;
			bool m_drawTangentialVelocities;
			Scalar m_velocityLength;


			scalarColorScheme_t m_colorScheme; 
			//Jet scalar color shader
			shared_ptr<GLSLShader> m_pJetColorShader;
			GLuint m_jetMinScalarLoc;
			GLuint m_jetMaxScalarLoc;
			GLuint m_jetAvgScalarLoc;

			//Grayscale scalar color shader
			shared_ptr<GLSLShader> m_pGrayScaleColorShader;
			GLuint m_grayMinScalarLoc;
			GLuint m_grayMaxScalarLoc;

			//Grid visualization window
			GridVisualizationWindow<Vector2> *m_pGridVisualizationWindow;

			/************************************************************************/
			/* Initialization			                                            */
			/************************************************************************/
			void initializeAuxiliaryDrawingStructures();
			void initializeVBOs();
			void initializeShaders();

			/************************************************************************/
			/* Copying functions                                                    */
			/************************************************************************/
			void copyIndexToVBOs();
			void copyVertexToVBOs();
			void copyScalarFieldToVBOs();

			/************************************************************************/
			/* Private functionalities                                              */
			/************************************************************************/
			void updateShaderColors();
			void updateVelocities();
			void updateGridPoints();
			void updateVelocityPoints();
			void setTagColor(int tag);

			/************************************************************************/
			/* Private drawing functions                                            */
			/************************************************************************/
			void drawCells();
			void drawPressures();
			void drawVelocities();
			void drawNodeVelocities();
			void drawFaceVelocities();
			void drawCurrentCellFaceVelocities();
			void drawFaceNormals();
			void drawSelectedCells();
			void drawTangentialVelocities();
			void drawCell(int ithCell, bool drawPoints = false, bool drawThick = true);
			void drawVelocity(const Vector2 &velocity, const Vector2 &velocityPosition);

			FORCE_INLINE void drawCell(int i, int j) const {
				glBegin(GL_QUADS);
				glVertex2f(m_pQuadGrid->getGridData2D()->getPoint(i, j).x,			m_pQuadGrid->getGridData2D()->getPoint(i, j).y);
				glVertex2f(m_pQuadGrid->getGridData2D()->getPoint(i + 1, j).x,		m_pQuadGrid->getGridData2D()->getPoint(i + 1, j).y);
				glVertex2f(m_pQuadGrid->getGridData2D()->getPoint(i + 1, j + 1).x,	m_pQuadGrid->getGridData2D()->getPoint(i + 1, j + 1).y);
				glVertex2f(m_pQuadGrid->getGridData2D()->getPoint(i, j + 1).x,		m_pQuadGrid->getGridData2D()->getPoint(i, j + 1).y);
				glEnd();
			}

			FORCE_INLINE void applyColorShader(Scalar minValue, Scalar maxValue, Scalar avgValue) const {
				switch(m_colorScheme) {
				case jet:
					m_pJetColorShader->applyShader();
					glUniform1f(m_jetMinScalarLoc, minValue);
					glUniform1f(m_jetMaxScalarLoc, maxValue);
					glUniform1f(m_jetAvgScalarLoc, avgValue);
					break;

				case grayscale:
					m_pGrayScaleColorShader->applyShader();
					glUniform1f(m_grayMinScalarLoc, minValue);
					glUniform1f(m_grayMaxScalarLoc, maxValue);
					break;
				}
			}

			FORCE_INLINE void removeColorShader() const {
				switch(m_colorScheme) {
				case jet:
					m_pJetColorShader->removeShader();
					break;
				case grayscale:
					m_pGrayScaleColorShader->removeShader();
					break;
				}
			}

		};

		
		
	}
}
#endif