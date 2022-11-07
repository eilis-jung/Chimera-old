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


#ifndef _CHIMERA_VECTOR_FIELD_RENDERER_H_
#define _CHIMERA_VECTOR_FIELD_RENDERER_H_
#pragma once


#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraResources.h"
//Windows cross reference
#include "Windows/BaseWindow.h"
#include "Visualization/IsocontourRenderer.h"

namespace Chimera {

	namespace Rendering {
		
		using namespace Resources;

		typedef struct triangles2D_t {
			Vector2 p[3];
		} triangles2D_t;

		typedef struct triangles3D_t {
			Vector3 p[3];
		} triangles3D_t;

		/** Flow Variables renderer base class */
		template <class VectorT>
		class VectorFieldRenderer {

		public:

			/************************************************************************/
			/* Ant-tweak bar vars                                                   */
			/************************************************************************/
			Scalar velocityLength;
			bool m_drawDenseVelocityField;
			bool m_drawFineGridVelocities;

		protected:
			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			dimensions_t m_gridDimensions;
			int m_totalGridVertices;

			/** Grid that will be rendered*/
			StructuredGrid<VectorT> *m_pGrid;

			/** Fine grid velocities */
			Array2D<Vector2> *m_pFineGridVelocities;
			Scalar m_fineGridDx;


			/************************************************************************/
			/* VBOs                                                                 */
			/************************************************************************/
			/** Simulation space velocity buffer object */
			GLuint *m_pVelocityVBO;
			/** Grid centroids vbo */
			GLuint *m_pGridCentroidsVBO;
			/** Grid space velocity buffer object */
			GLuint *m_pGSVelocityVBO;
			/** Grid space velocity arrows */
			GLuint *m_pGSVelocityArrowsVBO;
			/** Velocity index */
			GLuint *m_pVelocityIndexXYVBO;
			GLuint *m_pVelocityIndexYZVBO;
			GLuint *m_pVelocityIndexXZVBO;

			/************************************************************************/
			/* VAOs                                                                 */
			/************************************************************************/
			/** Grid Velocity array object : binds centers and GSVelocities together */
			GLuint *m_pVelocityVAO;
			/** Grid velocity transform feedback array object */
			GLuint *m_pVelocityFeedbackVAO;
			/** Grid Velocity arrows array object : lump velocity arrows*/
			GLuint *m_pVelocityArrowsVAO;

			/************************************************************************/
			/* Shaders                                                              */
			/************************************************************************/
			//Add vectors shader
			shared_ptr<GLSLShader> m_pAddVectorsShader;
			GLuint m_vectorLengthLoc;
			//Velocity arrows shader			
			shared_ptr<GLSLShader> m_pVelocityArrowsShader;
			//Length arrow shader location
			GLuint m_arrowLengthLoc;
			GLuint m_rotationVecLoc;

			/************************************************************************/
			/* VBO initialization                                                   */
			/************************************************************************/
			void initializeVBOs() {
				Core::Logger::get() << "Vector Field Renderer: Initializing VBOs..." << endl;
				unsigned int totalVBOBytes = 0;

				totalVBOBytes += initializeVelocityVBO();
				totalVBOBytes += initializeGridCentroidsVBO();
				totalVBOBytes += initializeGSVelocityVBO();
				totalVBOBytes += initializeVelocityIndexVBO();
				Core::Logger::get() << "Vector Field Renderer: VBOs allocated successfully. Total occupied video memory: " << 
					totalVBOBytes / (1024.0f*1024) << " Mb."<< endl;
			}

			virtual unsigned int initializeVelocityVBO();
			virtual unsigned int initializeGridCentroidsVBO();
			virtual unsigned int initializeGSVelocityVBO();
			virtual unsigned int initializeVelocityIndexVBO();

			/************************************************************************/
			/* VAO initialization                                                   */
			/************************************************************************/
			virtual void initializeVAOs() {
				initializeVelocityVAO();
				initializeTFVelocityVAO();
			}
			virtual void initializeVelocityVAO();
			virtual void initializeTFVelocityVAO();

			/************************************************************************/
			/* Shaders initialization			                                    */
			/************************************************************************/
			virtual void initializeShaders();

			/************************************************************************/
			/* Drawing                                                              */
			/************************************************************************/
			void drawGridArrows(int kthSlice = -1) const;

			void drawDenseVelocityField() const;

			void drawFineGridVelocities() const;


			/************************************************************************/
			/* Updating                                                             */
			/************************************************************************/
			void updateVelocity(bool auxiliaryVel = false) const ;
			void updateVelocityArrows() const;
			void updateDenseVelocityField();

		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			VectorFieldRenderer(StructuredGrid<VectorT> *pGrid);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			FORCE_INLINE GLuint getGridCentroidsVBO() const {
				return *m_pGridCentroidsVBO;
			}

			FORCE_INLINE void setFineGridVelocities(Array2D<Vector2> *pVelocities, Scalar fineGridDx) {
				m_pFineGridVelocities = pVelocities;
				m_fineGridDx = fineGridDx;
			}

			void update();

			/************************************************************************/
			/* Drawing                                                              */
			/************************************************************************/
			/** Velocity field */
			void drawVelocityField(bool auxVelocity = false, dimensions_t kthSlices = dimensions_t(-1, -1, -1)) const;
			void drawScalarFieldGradients(const BaseWindow::scalarVisualization_t &visualizationType = BaseWindow::drawPressure) const;
			void drawNodeVelocityField() const;
			void drawStaggeredVelocityField() const;
		};

	}
}
#endif