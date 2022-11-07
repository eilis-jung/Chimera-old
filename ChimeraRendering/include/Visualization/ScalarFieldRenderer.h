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


#ifndef _CHIMERA_FV_RENDERER_H_
#define _CHIMERA_FV_RENDERER_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraMesh.h"
#include "ChimeraCutCells.h"
#include "ChimeraResources.h"

#include "Windows/BaseWindow.h"
#include "Visualization/IsocontourRenderer.h"

namespace Chimera {
	using namespace Meshes;
	using namespace CutCells;

	namespace Rendering {

		using namespace Resources;
		typedef enum scalarColorScheme_t {
			singleColor,
			viridis,
			jet,
			grayscale,
			randomColors
		} scalarColorScheme_t;

		/** Flow Variables renderer base class */
		template <class VectorT>
		class ScalarFieldRenderer {
			
		public:
			
			/************************************************************************/
			/* Ant-tweak bar vars                                                   */
			/************************************************************************/
			scalarColorScheme_t m_colorScheme;
			/**Isolines visualization*/
			Scalar m_initialIsoValue;
			Scalar m_isoStepVertical;
			Scalar m_isoStepHorizontal;
			bool m_drawIsoPoints;
			bool m_triangulateQuads;
			bool m_drawFineGridCells;

		protected:
			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			int m_totalGridVertices;
			dimensions_t m_gridDimensions;
			IsocontourRenderer m_isoContourRenderer;

			/** Grid that will be rendered*/
			StructuredGrid<VectorT> *m_pGrid;
			
			/** Triangulated scalar field aux variable*/
			Scalar *m_pTriangulatedScalarField;

			/** Cut-Cells */
			CutCells2D<VectorT> *m_pCutCells;
			vector<Scalar> *m_pCutCellsScalarField;

			/** 2-D & 3-D fine grid streamfunctions*/
			Array2D<Scalar> *m_pFineGridScalarField2D;
			Array3D<Scalar> *m_pFineGridStreamfunction3D;
			Scalar m_fineGridDx;

			/** Streamfunction fine field to be rendered */
			/************************************************************************/
			/* VBOs                                                                 */
			/************************************************************************/
			/** Scalar values VBO */
			GLuint *m_pScalarFieldVBO;
			/** Scalar colors VBO */
			GLuint *m_pScalarColorsVBO;
			/** Scalar indexes */
			GLuint *m_pScalarIndexXYVBO;
			GLuint *m_pScalarIndexYZVBO;
			GLuint *m_pScalarIndexXZVBO;
			/** Grid centroids vbo */
			GLuint *m_pGridCentroidsVBO;
			/** Streamfunction vertices vbo */
			GLuint *m_pFineGridVerticesVBO;
			/** Streamfunction values vbo */
			GLuint *m_pFineGridValuesVBO;
			/** Streamfunction coilors vbo */
			GLuint *m_pFineGridColorsVBO;
			/** Streamfunction indices*/
			GLuint *m_pFineGridIndexVBO;

			/************************************************************************/
			/* Shaders                                                              */
			/************************************************************************/
			//Viridis scalar color shader
			shared_ptr<GLSLShader> m_pViridisColorShader;
			GLuint m_virMinScalarLoc;
			GLuint m_virMaxScalarLoc;
			GLuint m_virAvgScalarLoc;

			//Jet scalar color shader
			shared_ptr<GLSLShader> m_pJetColorShader;
			GLuint m_jetMinScalarLoc;
			GLuint m_jetMaxScalarLoc;
			GLuint m_jetAvgScalarLoc;

			//Grayscale scalar color shader
			shared_ptr<GLSLShader> m_pGrayScaleColorShader;
			GLuint m_grayMinScalarLoc;
			GLuint m_grayMaxScalarLoc;

			/************************************************************************/
			/* VBO initialization                                                   */
			/************************************************************************/
			void initializeVBOs() {
				Core::Logger::get() << "ScalarField Renderer: Initializing VBOs..." << endl;
				unsigned int totalVBOBytes = 0;
				totalVBOBytes += initializeGridCentroidsVBO();
				totalVBOBytes += initializeScalarFieldVBO();
				totalVBOBytes += initializeScalarIndexVBO();
				Core::Logger::get() << "ScalarField Renderer: VBOs allocated successfully. Total occupied video memory: " << 
					totalVBOBytes / (1024.0f*1024) << " Mb."<< endl;
			}
			virtual unsigned int initializeGridCentroidsVBO();
			virtual unsigned int initializeScalarFieldVBO();
			virtual unsigned int initializeScalarIndexVBO();
			
			void initializeFineGridScalarVBOs();

			/************************************************************************/
			/* Shaders initialization			                                    */
			/************************************************************************/
			virtual void initializeShaders();

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			FORCE_INLINE void applyColorShader(Scalar minValue, Scalar maxValue, Scalar avgValue) const {
				switch(m_colorScheme) {
					case viridis:
						m_pViridisColorShader->applyShader();
						glUniform1f(m_virMinScalarLoc, minValue);
						glUniform1f(m_virMaxScalarLoc, maxValue);
						glUniform1f(m_virAvgScalarLoc, avgValue);
					break;
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
					case viridis:
						m_pViridisColorShader->removeShader();
					break;
					case jet:
						m_pJetColorShader->removeShader();
					break;
					case grayscale:
						m_pGrayScaleColorShader->removeShader();
					break;
				}
			}

			/** Find Scalar Field min max values*/
			void findMinMax(const Array<Scalar> &scalarField, Scalar &minValue, Scalar &maxValue) const;

			void updateTriangulatedScalarField(Scalar *pScalarField);

			FORCE_INLINE unsigned int getRegularGridIndex(unsigned int i, unsigned int j) {
				return m_gridDimensions.x*j + i;
			}

			/** CPU-implemented shaders */
			Vector3 viridisColorMap(Scalar scalarValue);
			Vector3 jetColorMap(Scalar scalarValue);
			Vector3 grayScaleColorMap(Scalar scalarValue);
		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			ScalarFieldRenderer(StructuredGrid<VectorT> *pGrid);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			FORCE_INLINE void setCutCells(CutCells2D<VectorT> *pCutCells, vector<Scalar> *pScalarField) {
				m_pCutCells = pCutCells;
				m_pCutCellsScalarField = pScalarField;
			}


			FORCE_INLINE GLuint getGridCentroidsVBO() const {
				return *m_pGridCentroidsVBO;
			}

			FORCE_INLINE void enableMinMaxUpdate(bool enable) {
				m_updateScalarMinMax = enable;
			}
			FORCE_INLINE Scalar * getMinScalarFieldValuePtr() const {
				return &m_minScalarFieldVal;
			}
			FORCE_INLINE Scalar * getMaxScalarFieldValuePtr() const {
				return &m_maxScalarFieldVal;
			}

			FORCE_INLINE void setFineGridScalarValues2D(Array2D<Scalar> *pFineGridStreamfunctions, Scalar fineGridDx) {
				m_fineGridDx = fineGridDx;
				m_pFineGridScalarField2D = pFineGridStreamfunctions;
				if(pFineGridStreamfunctions != nullptr)
					initializeFineGridScalarVBOs();
			}

			FORCE_INLINE void setFineGridScalarValues3D(Array3D<Scalar> *pFineGridStreamfunctions, Scalar fineGridDx) {
				m_fineGridDx = fineGridDx;
				m_pFineGridStreamfunction3D = pFineGridStreamfunctions;
				initializeFineGridScalarVBOs();
			}
		
			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			//Updates the minimum and maximum scalar values of the scalar field
			void updateMinMaxScalarField(const BaseWindow::scalarVisualization_t &visualizationType = BaseWindow::drawPressure) const;
			FORCE_INLINE GLuint * getScalarColorsVBO() const {
				return m_pScalarColorsVBO;
			}

			/************************************************************************/
			/* Ant tweak bar                                                        */
			/************************************************************************/
			mutable Scalar m_minScalarFieldVal;
			mutable Scalar m_maxScalarFieldVal;
			bool m_updateScalarMinMax;

			void updateValueColor(const BaseWindow::scalarVisualization_t &visualizationType);
			

			/************************************************************************/
			/* Drawing                                                              */
			/************************************************************************/
			/** Scalar field */
			void beginDrawScalarField(BaseWindow::scalarVisualization_t visualizationType, dimensions_t kthSlices = dimensions_t(-1, -1, -1));
			void endDrawScalarField() const;	

			void drawCutCellsScalarField(BaseWindow::scalarVisualization_t visualizationType);

			/************************************************************************/
			/* Isocontours                                                         */
			/************************************************************************/
			FORCE_INLINE IsocontourRenderer & getIsocontourRenderer()  {
				return m_isoContourRenderer;
			}
			
		};

	}
}
#endif