#include "Visualization/VectorFieldRenderer.h"
#include "RenderingUtils.h"

namespace Chimera {
	namespace Rendering {
		template<>
		VectorFieldRenderer<Vector2>::VectorFieldRenderer(StructuredGrid<Vector2> *pGrid)  {
			velocityLength = 0.01;
			GridData2D *pGridData2D = pGrid->getGridData2D();
			m_gridDimensions = pGridData2D->getDimensions();
			m_pGrid = pGrid;
			m_totalGridVertices = m_gridDimensions.x*m_gridDimensions.y;
			m_drawDenseVelocityField = false; 
			m_pFineGridVelocities = NULL;

			/** Initialization */
			initializeVBOs();
			initializeVAOs();
			initializeShaders();
		}

		template<>
		VectorFieldRenderer<Vector3>::VectorFieldRenderer(StructuredGrid<Vector3> *pGrid)  {
			velocityLength = 0.01;
			m_drawDenseVelocityField = false;

			GridData3D *pGridData3D = pGrid->getGridData3D();
			m_gridDimensions = pGridData3D->getDimensions();
			m_pGrid = pGrid;
			
			m_totalGridVertices = m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z;

			/** Initialization */
			initializeVBOs();
			initializeVAOs();
			initializeShaders();
		}

		/************************************************************************/
		/* Initialization                                                       */
		/************************************************************************/
		/**Velocity VBO - represents the real simulation velocity */
		template <class VectorT>
		unsigned int VectorFieldRenderer<VectorT>::initializeVelocityVBO() {
			void *pVelocityPtr = NULL;
			
			m_pVelocityVBO = new GLuint();
			glGenBuffers(1, m_pVelocityVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVelocityVBO);
			unsigned int sizeVelocity = m_totalGridVertices*sizeof(VectorT);

			if(m_gridDimensions.z == 0) {
				pVelocityPtr = reinterpret_cast<void *>(m_pGrid->getGridData2D()->getVelocityArray().getRawDataPointer());
			} else {
				pVelocityPtr = reinterpret_cast<void *>(m_pGrid->getGridData3D()->getVelocityArray().getRawDataPointer());
			}
			
			glBufferData(GL_ARRAY_BUFFER, sizeVelocity, pVelocityPtr, GL_DYNAMIC_DRAW);
			return sizeVelocity;
		}

		template <class VectorT>
		unsigned int VectorFieldRenderer<VectorT>::initializeGridCentroidsVBO() {
			void *pGridCentroidsPtr = NULL;

			m_pGridCentroidsVBO = new GLuint();
			glGenBuffers(1, m_pGridCentroidsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridCentroidsVBO);
			unsigned int sizeVelocity = m_totalGridVertices*sizeof(VectorT);

			if(m_gridDimensions.z == 0) {
				pGridCentroidsPtr = reinterpret_cast<void *>(m_pGrid->getGridData2D()->getGridCentersArray().getRawDataPointer());
			} else {
				pGridCentroidsPtr = reinterpret_cast<void *>(m_pGrid->getGridData3D()->getGridCentersArray().getRawDataPointer());
			}

			glBufferData(GL_ARRAY_BUFFER, sizeVelocity, pGridCentroidsPtr, GL_DYNAMIC_DRAW);
			return sizeVelocity;
		}

		/**Grid space velocity VBO - Simulation velocity summed with correspondent grid center
			 **This VBO will store both velocity vectors and grid centers, in order to update velocities with 
			 **transform feedback functionality. */
		template <class VectorT>
		unsigned int VectorFieldRenderer<VectorT>::initializeGSVelocityVBO() {
			void *pVelocityPtr, *pGridCentersPtr = NULL;
			m_pGSVelocityVBO = new GLuint();

			glGenBuffers(1, m_pGSVelocityVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGSVelocityVBO);

			unsigned int sizeVelocity = m_totalGridVertices*sizeof(VectorT)*2;
			if(m_gridDimensions.z == 0) {
				pVelocityPtr = reinterpret_cast<void *>(m_pGrid->getGridData2D()->getVelocityArray().getRawDataPointer());
				pGridCentersPtr = reinterpret_cast<void *>(m_pGrid->getGridData2D()->getGridCentersArray().getRawDataPointer());
			} else {
				pVelocityPtr = reinterpret_cast<void *>(m_pGrid->getGridData3D()->getVelocityArray().getRawDataPointer());
				pGridCentersPtr = reinterpret_cast<void *>(m_pGrid->getGridData3D()->getGridCentersArray().getRawDataPointer());
			}

			glBufferData(GL_ARRAY_BUFFER, sizeVelocity, 0, GL_STATIC_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeVelocity/2, pVelocityPtr);
			glBufferSubData(GL_ARRAY_BUFFER, sizeVelocity/2, sizeVelocity/2, pGridCentersPtr);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			/**VelocityArrows VBO */
			m_pGSVelocityArrowsVBO = new GLuint();
			glGenBuffers(1, m_pGSVelocityArrowsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGSVelocityArrowsVBO);
			unsigned int sizeArrows = 0;
			if(m_gridDimensions.z == 0)
				sizeArrows = m_totalGridVertices*sizeof(triangles2D_t);
			else
				sizeArrows = m_totalGridVertices*sizeof(triangles3D_t)*2;
			glBufferData(GL_ARRAY_BUFFER, sizeArrows, 0, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			return sizeVelocity + sizeArrows;
		}

		unsigned int VectorFieldRenderer<Vector2>::initializeVelocityIndexVBO() {
			/**Velocity Index VBO - stores the connection between velocities and grid centers */
			m_pVelocityIndexXYVBO = new GLuint();
			glGenBuffers(1, m_pVelocityIndexXYVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVelocityIndexXYVBO);
			unsigned int size = m_totalGridVertices*2;
			int *pIndexes = new int[size];
			int currIndex = 0;
			
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					currIndex = (j*m_gridDimensions.x + i) * 2;
					pIndexes[currIndex] = j*m_gridDimensions.x + i;
					pIndexes[currIndex + 1] = m_gridDimensions.x*m_gridDimensions.y + j*m_gridDimensions.x + i;
				}
			}

			glBufferData(GL_ARRAY_BUFFER, size*sizeof(int), pIndexes, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			delete [] pIndexes;
			return size*sizeof(int);
		}

		unsigned int VectorFieldRenderer<Vector3>::initializeVelocityIndexVBO() {
			/**Velocity Index VBO - stores the connection between velocities and grid centers */
			unsigned int totalSize = 0;

			/************************************************************************/
			/* XY indices initialization                                            */
			/************************************************************************/
			m_pVelocityIndexXYVBO = new GLuint();
			glGenBuffers(1, m_pVelocityIndexXYVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVelocityIndexXYVBO);
			unsigned int size = m_totalGridVertices * 2;
			int *pIndexes = new int[size];
			int currIndex = 0;

			for (int k = 0; k < m_gridDimensions.z; k++) {
				for (int i = 0; i < m_gridDimensions.x; i++) {
					for (int j = 0; j < m_gridDimensions.y; j++) {	
						int tempIndex = k*m_gridDimensions.x*m_gridDimensions.y + j*m_gridDimensions.x + i;
						pIndexes[currIndex++] = tempIndex;
						pIndexes[currIndex++] = m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z + tempIndex;
					}
				}
			}
			glBufferData(GL_ARRAY_BUFFER, size*sizeof(int), pIndexes, GL_STATIC_DRAW);
			totalSize += size;
			delete[] pIndexes;

			/************************************************************************/
			/* YZ indices initialization                                            */
			/************************************************************************/
			m_pVelocityIndexYZVBO = new GLuint();
			glGenBuffers(1, m_pVelocityIndexYZVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVelocityIndexYZVBO);
			size = m_totalGridVertices * 2;
			pIndexes = new int[size];
			currIndex = 0;
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int k = 0; k < m_gridDimensions.z; k++) {
					for (int j = 0; j < m_gridDimensions.y; j++) {
						int tempIndex = k*m_gridDimensions.x*m_gridDimensions.y + j*m_gridDimensions.x + i;
						pIndexes[currIndex++] = tempIndex;
						pIndexes[currIndex++] = m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z + tempIndex;
					}
				}
			}
			glBufferData(GL_ARRAY_BUFFER, size*sizeof(int), pIndexes, GL_STATIC_DRAW);
			totalSize += size;
			delete[] pIndexes;

			/************************************************************************/
			/* XZ indices initialization                                            */
			/************************************************************************/
			m_pVelocityIndexXZVBO = new GLuint();
			glGenBuffers(1, m_pVelocityIndexXZVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVelocityIndexXZVBO);
			size = m_totalGridVertices * 2;
			pIndexes = new int[size];
			currIndex = 0;
			for (int j = 0; j < m_gridDimensions.y; j++) {
				for (int k = 0; k < m_gridDimensions.z; k++) {
					for (int i = 0; i < m_gridDimensions.x; i++) {
						int tempIndex = k*m_gridDimensions.x*m_gridDimensions.y + j*m_gridDimensions.x + i;
						pIndexes[currIndex++] = tempIndex;
						pIndexes[currIndex++] = m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z + tempIndex;
					}
				}
			}
			glBufferData(GL_ARRAY_BUFFER, size*sizeof(int), pIndexes, GL_STATIC_DRAW);
			totalSize += size;
			delete[] pIndexes;

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			return size*sizeof(int);
		}


		/************************************************************************/
		/* VAO initialization	                                                */
		/************************************************************************/
		template <class VectorT>
		void VectorFieldRenderer<VectorT>::initializeVelocityVAO() {
			m_pVelocityVAO = new GLuint();
			glGenVertexArrays(1, m_pVelocityVAO);
			glBindVertexArray(*m_pVelocityVAO);
				glBindBuffer(GL_ARRAY_BUFFER, *m_pGSVelocityVBO);
				if(m_gridDimensions.z == 0)
					glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
				else 
					glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
				glEnableVertexAttribArray(0);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}
		template <class VectorT>
		void VectorFieldRenderer<VectorT>::initializeTFVelocityVAO() {
			m_pVelocityFeedbackVAO = new GLuint();
			glGenVertexArrays(1, m_pVelocityFeedbackVAO);
			glBindVertexArray(*m_pVelocityFeedbackVAO);
				glBindBuffer(GL_ARRAY_BUFFER, *m_pVelocityVBO);
				if(m_gridDimensions.z == 0)
					glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
				else 
					glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
				glEnableVertexAttribArray(0);

				glBindBuffer(GL_ARRAY_BUFFER, *m_pGridCentroidsVBO);
				if(m_gridDimensions.z == 0)
					glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
				else 
					glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
				glEnableVertexAttribArray(1);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}

		/************************************************************************/
		/* Shaders initializtion	                                            */
		/************************************************************************/
		template <class VectorT>
		void VectorFieldRenderer<VectorT>::initializeShaders() {
			/** Add vectors shader*/
			{
				if(m_gridDimensions.z == 0)
					m_pAddVectorsShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, "Shaders/2D/AddVectors.glsl");
				else
					m_pAddVectorsShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, "Shaders/3D/AddVectors.glsl");
				m_vectorLengthLoc = glGetUniformLocation(m_pAddVectorsShader->getProgramID(), "scaleFactor");
			}


			/** Velocity arrows shader */
			{
				GLchar const * Strings[] = {"v1Out", "v2Out", "v3Out"}; 
				if(m_gridDimensions.z == 0) {
					m_pVelocityArrowsShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, 
						"Shaders/2D/VectorArrow.glsl",
						3,
						Strings,
						GL_INTERLEAVED_ATTRIBS);
				} else {
					m_pVelocityArrowsShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, 
						"Shaders/3D/VectorArrow.glsl",
						3,
						Strings,
						GL_INTERLEAVED_ATTRIBS);
				}

				m_arrowLengthLoc = glGetUniformLocation(m_pVelocityArrowsShader->getProgramID(), "triangleLength");
				if(m_gridDimensions.z != 0)
					m_rotationVecLoc = glGetUniformLocation(m_pVelocityArrowsShader->getProgramID(), "rotationVec");
			}
		}

		
		/************************************************************************/
		/* Update                                                               */
		/************************************************************************/
		/** Velocity */
		template <class VectorT>
		void VectorFieldRenderer<VectorT>::updateVelocity(bool auxiliaryVel) const {
			glBindBuffer(GL_ARRAY_BUFFER, *m_pVelocityVBO);
			void *pVelocityPtr;

			if(m_gridDimensions.z == 0) {
				if(!auxiliaryVel)
					pVelocityPtr = reinterpret_cast<void *>(m_pGrid->getGridData2D()->getVelocityArray().getRawDataPointer());
				else
					pVelocityPtr = reinterpret_cast<void *>(m_pGrid->getGridData2D()->getAuxVelocityArray().getRawDataPointer());
			} else {
				if(!auxiliaryVel)
					pVelocityPtr = reinterpret_cast<void *>(m_pGrid->getGridData3D()->getVelocityArray().getRawDataPointer());
				else
					pVelocityPtr = reinterpret_cast<void *>(m_pGrid->getGridData3D()->getAuxVelocityArray().getRawDataPointer());
			}
			glBufferData(GL_ARRAY_BUFFER, m_totalGridVertices*sizeof(VectorT), pVelocityPtr, GL_DYNAMIC_DRAW);
			
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glEnable(GL_RASTERIZER_DISCARD_NV);
			/************************************************************************/
			/* Adding velocities                                                    */
			/************************************************************************/
			m_pAddVectorsShader->applyShader();
			glUniform1fv(m_vectorLengthLoc, 1, &velocityLength);

			glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *m_pGSVelocityVBO);
			glBindVertexArray(*m_pVelocityFeedbackVAO);

			glBeginTransformFeedback(GL_POINTS);
			glDrawArrays(GL_POINTS, 0, m_totalGridVertices);
			glEndTransformFeedback();

			glBindVertexArray(0);

			m_pAddVectorsShader->removeShader();

			updateVelocityArrows();

			glDisable(GL_RASTERIZER_DISCARD_NV);
		}

		template <class VectorT>
		void VectorFieldRenderer<VectorT>::updateVelocityArrows() const {
			/************************************************************************/
			/* Updating velocity arrows                                             */
			/************************************************************************/
			m_pVelocityArrowsShader->applyShader();
			Scalar triangleLength = 0.02;
			glUniform1fv(m_arrowLengthLoc, 1, &triangleLength);
			if(m_gridDimensions.z != 0) {
				Vector3 rotationVec = Vector3(0, 1, 0);
				glUniform3f(m_rotationVecLoc, rotationVec.x, rotationVec.y, rotationVec.z);
			}
			

			glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *m_pGSVelocityArrowsVBO);

			glBindBuffer(GL_ARRAY_BUFFER, *m_pGSVelocityVBO);
			if(m_gridDimensions.z == 0)
				glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, 0);
			else
				glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridCentroidsVBO);
			if(m_gridDimensions.z == 0)
				glVertexAttribPointer(4, 2, GL_FLOAT, false, 0, 0);
			else
				glVertexAttribPointer(4, 3, GL_FLOAT, false, 0, 0);
			glEnableVertexAttribArray(4);

			glBeginTransformFeedback(GL_POINTS);
			glDrawArrays (GL_POINTS, 0, m_totalGridVertices);
			glEndTransformFeedback();

			if(m_gridDimensions.z != 0) {
				Vector3 rotationVec = Vector3(0, 0, 1);
				glUniform3f(m_rotationVecLoc, rotationVec.x, rotationVec.y, rotationVec.z);

				int sizeArrows = m_totalGridVertices*sizeof(triangles3D_t)* 2;
				glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *m_pGSVelocityArrowsVBO, sizeArrows/2, sizeArrows/2);

				glBeginTransformFeedback(GL_POINTS);
				glDrawArrays(GL_POINTS, 0, m_totalGridVertices);
				glEndTransformFeedback();
			}

			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(4);

			m_pVelocityArrowsShader->removeShader();

		}

		/************************************************************************/
		/* Drawing                                                              */
		/************************************************************************/
		template <class VectorT>
		void VectorFieldRenderer<VectorT>::drawVelocityField(bool auxVelocity /* = false */, dimensions_t kthSlices /* = -1 */)  const {
			if (m_drawFineGridVelocities) {
				drawFineGridVelocities();
			}

			if(m_drawDenseVelocityField) {
				drawDenseVelocityField();
			} else {
				updateVelocity(auxVelocity);

				glLineWidth(0.5f);
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				glColor3f(0.0f, 0.0f, 0.0f);
				glBindVertexArray(*m_pVelocityVAO);
				if(kthSlices.x != -1) {
					//XY
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pVelocityIndexXYVBO);
					glPushMatrix();
					int kthSlice = clamp(kthSlices.z, 0, m_gridDimensions.z);
					size_t initialIndex = (m_gridDimensions.x)*(m_gridDimensions.y)*kthSlice*sizeof(int) * 2;
					glTranslatef(0, 0, 1e-5f);
					glDrawElements(GL_LINES, (m_gridDimensions.x)*(m_gridDimensions.y)*2, GL_UNSIGNED_INT, (void *) initialIndex); 
					glPopMatrix();

					//YZ
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pVelocityIndexYZVBO);
					glPushMatrix();
					kthSlice = clamp(kthSlices.x, 0, m_gridDimensions.x);
					initialIndex = (m_gridDimensions.y)*(m_gridDimensions.z)*kthSlice*sizeof(int)* 2;
					glTranslatef(0, 0, 1e-5f);
					glDrawElements(GL_LINES, (m_gridDimensions.y)*(m_gridDimensions.z) * 2, GL_UNSIGNED_INT, (void *)initialIndex);
					glPopMatrix();

					//XZ
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pVelocityIndexXZVBO);
					glPushMatrix();
					kthSlice = clamp(kthSlices.y, 0, m_gridDimensions.y);
					initialIndex = (m_gridDimensions.x)*(m_gridDimensions.z)*kthSlice*sizeof(int)* 2;
					glTranslatef(0, 0, 1e-5f);
					glDrawElements(GL_LINES, (m_gridDimensions.x)*(m_gridDimensions.z) * 2, GL_UNSIGNED_INT, (void *)initialIndex);
					glPopMatrix();
				} else {
					//XY
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pVelocityIndexXYVBO);
					glDrawElements(GL_LINES, m_totalGridVertices*2, GL_UNSIGNED_INT, 0); //draw all elements
				}
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
				glBindVertexArray(0);

				drawGridArrows(kthSlices.z);

			}
			
		}

		template <>
		void VectorFieldRenderer<Vector2>::drawNodeVelocityField() const {
			
		}

		template <>
		void VectorFieldRenderer<Vector3>::drawNodeVelocityField() const {
			Scalar dx = m_pGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
		}


		template <>
		void VectorFieldRenderer<Vector2>::drawStaggeredVelocityField() const {
			Scalar dx = m_pGrid->getGridData2D()->getScaleFactor(0, 0).x;
			
			for (int i = 0; i < m_pGrid->getGridData2D()->getDimensions().x; i++) {
				for (int j = 0; j < m_pGrid->getGridData2D()->getDimensions().y; j++) {
					Vector2 initialPoint((i + 0.5)*dx, j*dx);
					Vector2 finalVelPoint = initialPoint;
					finalVelPoint.y += m_pGrid->getGridData2D()->getVelocity(i, j).y*velocityLength;
					RenderingUtils::getInstance()->drawVector(initialPoint, finalVelPoint);

					initialPoint = Vector2(i*dx, (j + 0.5)*dx);
					finalVelPoint = initialPoint;
					finalVelPoint.x += m_pGrid->getGridData2D()->getVelocity(i, j).x*velocityLength;
					RenderingUtils::getInstance()->drawVector(initialPoint, finalVelPoint);
				}
			}
		}

		template <>
		void VectorFieldRenderer<Vector3>::drawStaggeredVelocityField() const {
		}


		template <class VectorT>
		void VectorFieldRenderer<VectorT>::drawGridArrows(int kthSlice /* = -1 */) const {
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGSVelocityArrowsVBO);
			if(m_gridDimensions.z == 0)
				glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
			else
				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(0);
			if(m_gridDimensions.z == 0)
				glDrawArrays(GL_TRIANGLES, 0, m_totalGridVertices*3);
			else if (kthSlice == -1) {
				glDrawArrays(GL_TRIANGLES, 0, m_totalGridVertices * 3 *2);
			}
			else {
				size_t initialIndex = m_gridDimensions.x*m_gridDimensions.y*kthSlice*3;
				glDrawArrays(GL_TRIANGLES, initialIndex, m_gridDimensions.x*m_gridDimensions.y*3);

				initialIndex = m_gridDimensions.x*m_gridDimensions.y*kthSlice*3 + m_totalGridVertices*3;
				glDrawArrays(GL_TRIANGLES, initialIndex, m_gridDimensions.x*m_gridDimensions.y * 3);
			}
				
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glDisableVertexAttribArray(0);
		}


		template<>
		void VectorFieldRenderer<Vector2>::drawFineGridVelocities() const {
			if (m_pFineGridVelocities) {
				glColor3f(0.0f, 0.0f, 0.0f);
				glLineWidth(1.0f);
				for (int i = 1; i < m_pFineGridVelocities->getDimensions().x - 1; i++) {
					for (int j = 1; j < m_pFineGridVelocities->getDimensions().y - 1; j++) {

						Vector2 samplePoint((i + 0.5)*m_fineGridDx, (j + 0.5)*m_fineGridDx);

						RenderingUtils::getInstance()->drawVector(samplePoint, samplePoint + (*m_pFineGridVelocities)(i, j)*velocityLength);

					}
				}
			}
		}

		template<>
		void VectorFieldRenderer<Vector3>::drawFineGridVelocities() const {

		}

		template<>
		void VectorFieldRenderer<Vector2>::drawDenseVelocityField() const {
			//Scalar dx = m_pGrid->getGridData2D()->getScaleFactor(0, 0).x;
			//for (int i = 1; i < m_gridDimensions.x - 1; i++) {
			//	for (int j = 1; j < m_gridDimensions.y - 1; j++) {
			//		for (int k = 0; k < 10; k++) {
			//			for (int l = 0; l < 10; l++) {
			//				Vector2 samplePoint;
			//				samplePoint.x = (i + k*0.1)*dx;
			//				samplePoint.y = (j + l*0.1)*dx;
			//				
			//				/*Vector2 velocitySample = bilinearInterpolation(samplePoint/dx, m_pGrid->getGridData2D()->getVelocityArray());
			//				RenderingUtils::getInstance()->drawVector(samplePoint, samplePoint + velocitySample*velocityLength);*/
			//			}
			//		}
			//	}
			//}
		}

		template<>
		void VectorFieldRenderer<Vector3>::drawDenseVelocityField() const {

		}

		template<>
		void VectorFieldRenderer<Vector2>::drawScalarFieldGradients(const BaseWindow::scalarVisualization_t & visualizationType) const {
			GridData2D *pGridData2D = m_pGrid->getGridData2D();
			auto currScalarField = RenderingUtils::getInstance()->switchScalarField2D(visualizationType, pGridData2D);
			DoubleScalar greaterGradSize = 0;
			for (int i = 1; i < m_gridDimensions.x - 1; i++) {
				for (int j = 1; j < m_gridDimensions.y - 1; j++) {
					Vector2 interpolatedGradient;
					//interpolatedGradient.x = (currScalarField(i + 1, j) - 2*currScalarField(i, j) + currScalarField(i - 1, j)) / (2 * pGridData2D->getGridSpacing());
					interpolatedGradient.y = (currScalarField(i, j + 1) - 2*currScalarField(i, j) + currScalarField(i, j - 1)) / (2 * pGridData2D->getGridSpacing());
					
					interpolatedGradient.x = (currScalarField(i + 1, j) - currScalarField(i - 1, j)) / (2 * pGridData2D->getGridSpacing());
					interpolatedGradient.y = (currScalarField(i, j + 1) - currScalarField(i, j - 1)) / (2 * pGridData2D->getGridSpacing());


					if (interpolatedGradient.length() > greaterGradSize) {
						greaterGradSize = interpolatedGradient.length();
					}
				}
			}
			for (int i = 1; i < m_gridDimensions.x - 1; i++) {
				for (int j = 1; j < m_gridDimensions.y - 1; j++) {
					Vector2 interpolatedGradient;
					//interpolatedGradient.x = (currScalarField(i + 1, j) - 2 * currScalarField(i, j) + currScalarField(i - 1, j)) / (2 * pGridData2D->getGridSpacing());
					//interpolatedGradient.y = (currScalarField(i, j + 1) - 2 * currScalarField(i, j) + currScalarField(i, j - 1)) / (2 * pGridData2D->getGridSpacing());
					interpolatedGradient.x = -(currScalarField(i + 1, j) - currScalarField(i - 1, j)) / (2 * pGridData2D->getGridSpacing());
					interpolatedGradient.y = -(currScalarField(i, j + 1) - currScalarField(i, j - 1)) / (2 * pGridData2D->getGridSpacing());

					interpolatedGradient /= greaterGradSize;
					//interpolatedGradient.normalize();

					Vector2 gridCenter = pGridData2D->getCenterPoint(i, j);
					RenderingUtils::getInstance()->drawVector(gridCenter, gridCenter + interpolatedGradient*velocityLength);
				}
			}
		}

		template <>
		void VectorFieldRenderer<Vector3>::drawScalarFieldGradients(const BaseWindow::scalarVisualization_t &visualizationType) const {
			visualizationType;
		}

		/************************************************************************/
		/* FVRenderer declarations - Linking time                               */
		/************************************************************************/
		template VectorFieldRenderer<Vector2>;
		template VectorFieldRenderer<Vector3>;

	}
}