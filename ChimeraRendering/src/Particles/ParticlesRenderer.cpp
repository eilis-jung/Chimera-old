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

#include "Particles/ParticlesRenderer.h"
#include "Physics/PhysicsCore.h"

namespace Chimera {

	namespace Rendering {
		template<class VectorType, template <class> class ArrayType>
		const int ParticlesRenderer<VectorType, ArrayType>::s_maxTrailSize = 8;
		
		template<class VectorType, template <class> class ArrayType>
		void ParticlesRenderer<VectorType, ArrayType>::initializeInternalStructs() {
			for (int k = 0; k < s_maxTrailSize; k++) {
				for (int i = 0; i < m_pParticlesData->getPositions().size(); i++) {
					(*m_pParticlesTrails)(i, k) = m_pParticlesData->getPositions()[i];
				}
			}
			
			
			//Initializing trails indexes
			m_pParticlesTrailsIndexes = new int[m_pParticlesData->getPositions().size()*s_maxTrailSize * 2];

			int currIndex = 0;
			for (int j = 0; j < s_maxTrailSize - 1; j++) {
				for (int i = 0; i < m_pParticlesData->getPositions().size(); i++) {
					m_pParticlesTrailsIndexes[currIndex++] = m_pParticlesTrails->getRawPtrIndex(i, j);
					m_pParticlesTrailsIndexes[currIndex++] = m_pParticlesTrails->getRawPtrIndex(i, j + 1);
				}
			}
		}
		
		template<class VectorType, template <class> class ArrayType>
		void ParticlesRenderer<VectorType, ArrayType>::initializeVBOs() {
			m_pParticlesVBO = new GLuint();
			unsigned int sizeParticlesVBO = m_pParticlesData->getPositions().size() * sizeof(VectorType);

			glGenBuffers(1, m_pParticlesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_pParticlesTrails->getRawDataPointer(), GL_DYNAMIC_DRAW);

			m_pParticlesColorVBO = new GLuint;
			unsigned int sizeParticlesColorVBO = m_pParticlesData->getPositions().size() * sizeof(float) * 4;
			glGenBuffers(1, m_pParticlesColorVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, &m_particlesColors[0], GL_DYNAMIC_DRAW);

			/** Initializing particles scalar field attributes */
			m_pScalarAttributesVBO = new GLuint();
			unsigned int scalarVBOSize = m_pParticlesData->getPositions().size() * sizeof(Scalar);
			glGenBuffers(1, m_pScalarAttributesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarAttributesVBO);
			glBufferData(GL_ARRAY_BUFFER, scalarVBOSize, 0, GL_DYNAMIC_DRAW);

			/** Initializing particles trails */
			m_pParticleTrailsVBO = new GLuint;
			sizeParticlesVBO = m_pParticlesData->getPositions().size()*s_maxTrailSize * sizeof(VectorType);
			glGenBuffers(1, m_pParticleTrailsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_pParticlesTrails->getRawDataPointer(), GL_DYNAMIC_DRAW);

			/** Initializing particles trails indexes */
			m_pParticleTrailsIndexesVBO = new GLuint;
			sizeParticlesVBO = m_pParticlesData->getPositions().size()*s_maxTrailSize * sizeof(int) * 2;
			glGenBuffers(1, m_pParticleTrailsIndexesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsIndexesVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_pParticlesTrailsIndexes, GL_STATIC_DRAW);

		}

		template<class VectorType, template <class> class ArrayType>
		void ParticlesRenderer<VectorType, ArrayType>::initializeShaders() {
			/** Viridis color shader */
			{
				GLchar const * Strings[] = { "rColor", "gColor", "bColor", "aColor" };
				m_pViridisColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER,
					"Shaders/2D/ScalarColor - viridis - alpha.glsl",
					4,
					Strings,
					GL_INTERLEAVED_ATTRIBS);

				m_virMinScalarLoc = glGetUniformLocation(m_pViridisColorShader->getProgramID(), "minPressure");
				m_virMaxScalarLoc = glGetUniformLocation(m_pViridisColorShader->getProgramID(), "maxPressure");
				m_virAvgScalarLoc = glGetUniformLocation(m_pViridisColorShader->getProgramID(), "avgPressure");
			}

			/** Jet color shader */
			{
				GLchar const * Strings[] = { "rColor", "gColor", "bColor", "aColor" };
				m_pJetColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER,
					"Shaders/2D/ScalarColor - wavelength - alpha.glsl",
					4,
					Strings,
					GL_INTERLEAVED_ATTRIBS);

				m_jetMinScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "minPressure");
				m_jetMaxScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "maxPressure");
				m_jetAvgScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "avgPressure");
			}

			/** Grayscale color shader */
			{
				GLchar const * Strings[] = { "rColor", "gColor", "bColor" , "aColor" };
				m_pGrayScaleColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER,
					"Shaders/2D/ScalarColor - grayscale - alpha.glsl",
					4,
					Strings,
					GL_INTERLEAVED_ATTRIBS);

				m_grayMinScalarLoc = glGetUniformLocation(m_pGrayScaleColorShader->getProgramID(), "minScalar");
				m_grayMaxScalarLoc = glGetUniformLocation(m_pGrayScaleColorShader->getProgramID(), "maxScalar");
			}
		}


		template<class VectorType, template <class> class ArrayType>
		void ParticlesRenderer<VectorType, ArrayType>::updateTrails() {
			if (m_currentTrailSize < s_maxTrailSize - 1)
				m_currentTrailSize++;

			
			for (int j = m_currentTrailSize; j >= 0; j--) {
				for (int i = 0; i < m_pParticlesData->getPositions().size(); i++) {
					if (j == 0) {
						//If the particle was resampled
						if (m_pParticlesData->getResampledParticles()[i]) {
							for (int k = 0; k <= m_currentTrailSize; k++) {
								(*m_pParticlesTrails)(i, k) = m_pParticlesData->getPositions()[i];
							}
						}
						else {
							(*m_pParticlesTrails)(i, 0) = m_pParticlesData->getPositions()[i];
						}
					}
					else {
						(*m_pParticlesTrails)(i, j) = (*m_pParticlesTrails)(i, j - 1);
					}
				}
			}
		}
		
		template<class VectorType, template <class> class ArrayType>
		void ParticlesRenderer<VectorType, ArrayType>::updateVBOs() {
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
			glBufferData(GL_ARRAY_BUFFER, m_pParticlesData->getPositions().size() * sizeof(VectorType), m_pParticlesTrails->getRawDataPointer(), GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
			glBufferData(GL_ARRAY_BUFFER, m_pParticlesData->getPositions().size() *s_maxTrailSize * sizeof(VectorType), m_pParticlesTrails->getRawDataPointer(), GL_DYNAMIC_DRAW);
			
			/** Copying scalar based attribute first to VBO buffer*/
			if (m_pParticlesData->hasScalarBasedAttribute(m_visAttribute)) {
				vector<Scalar> scalarAttributes = m_pParticlesData->getScalarBasedAttribute(m_visAttribute);
				glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarAttributesVBO);
				glBufferData(GL_ARRAY_BUFFER, m_pParticlesData->getPositions().size() * sizeof(Scalar), &scalarAttributes[0], GL_DYNAMIC_DRAW);
			}

			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		template<class VectorType, template <class> class ArrayType>
		void ParticlesRenderer<VectorType, ArrayType>::updateParticlesColorsGridSlice() {
			/*int numTrailsRendered = clamp(m_renderingParams.trailSize, 0, m_currentTrailSize);
			for (int j = 0; j < numTrailsRendered + 1; j++) {
				for (int i = 0; i < m_numInitializedParticles; i++) {
					if (floor(m_pParticlesPosition->at(i).z) == m_renderingParams.gridSliceDimension) {
						m_pParticlesColor[j*m_maxNumberOfParticles + i * 4 + 3] = 1.0f;
					}
					else {
						m_pParticlesColor[j*m_maxNumberOfParticles + i * 4 + 3] = 0.0f;
					}
				}
			}

			unsigned int sizeParticlesColorVBO = m_maxNumberOfParticles * sizeof(float) * 4 * (numTrailsRendered + 1);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, m_pParticlesColor, GL_DYNAMIC_DRAW);*/
		}

		template<class VectorType, template <class> class ArrayType>
		Vector3 ParticlesRenderer<VectorType, ArrayType>::jetShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue) {
			Vector3 particleColor;
			float totalDist = maxValue - minValue;
			float wavelength = 420 + ((scalarFieldValue - minValue) / totalDist) * 360;

			if (wavelength <= 439) {
				particleColor.x = -(wavelength - 440) / (440.0f - 350.0f);
				particleColor.y = 0.0;
				particleColor.z = 1.0;
			}
			else if (wavelength <= 489) {
				particleColor.x = 0.0;
				particleColor.y = (wavelength - 440) / (490.0f - 440.0f);
				particleColor.z = 1.0;
			}
			else if (wavelength <= 509) {
				particleColor.x = 0.0;
				particleColor.y = 1.0;
				particleColor.z = -(wavelength - 510) / (510.0f - 490.0f);
			}
			else if (wavelength <= 579) {
				particleColor.x = (wavelength - 510) / (580.0f - 510.0f);
				particleColor.y = 1.0;
				particleColor.z = 0.0;
			}
			else if (wavelength <= 644) {
				particleColor.x = 1.0;
				particleColor.y = -(wavelength - 645) / (645.0f - 580.0f);
				particleColor.z = 0.0;
			}
			else if (wavelength <= 780) {
				particleColor.x = 1.0;
				particleColor.y = 0.0;
				particleColor.x = 0.0;
			}
			else {
				particleColor.x = 0.0;
				particleColor.y = 0.0;
				particleColor.z = 0.0;
			}

			return particleColor;
		}

		template<class VectorType, template <class> class ArrayType>
		Vector3 ParticlesRenderer<VectorType, ArrayType>::grayScaleShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue) {
			Vector3 particleColor;
			float totalDist = maxValue - minValue;
			particleColor.x = particleColor.y = particleColor.z = (scalarFieldValue - minValue) / totalDist;
			return particleColor;
		}

		template<class VectorType, template <class> class ArrayType>
		void ParticlesRenderer<VectorType, ArrayType>::updateParticlesColorsCPU(Scalar minScalarValue, Scalar maxScalarValue) {
			if (m_pParticlesData->hasScalarBasedAttribute(m_visAttribute)) {
				vector<Scalar> scalarAttributes = m_pParticlesData->getScalarBasedAttribute(m_visAttribute);
				#pragma omp parallel for
				for (int i = 0; i < m_pParticlesData->getPositions().size(); i++) {
					Scalar scalarFieldValue = 0;
					Vector3 color;
					
					switch (m_renderingParams.colorScheme) {
						case jet:
							color = jetShading(i, scalarAttributes[i], minScalarValue, maxScalarValue);
						break;

						case viridis:
							color = jetShading(i, scalarAttributes[i], minScalarValue, maxScalarValue);
						break;
						
						case grayscale:
							color = grayScaleShading(i, scalarAttributes[i], minScalarValue, maxScalarValue);
						break;
					}	
					m_particlesColors[i * 4] = color.x;
					m_particlesColors[i * 4 + 1] = color.y;
					m_particlesColors[i * 4 + 2] = color.z;
					if (m_renderingParams.colorScheme == grayscale) {
						m_particlesColors[i * 4 + 3] = (1.5*color.x - pow(color.x, 1.5)) * 2;
						/*if (m_particlesColors[i * 4 + 3] < 0.2) {
							m_particlesColors[i * 4 + 3] = 0;
						}*/
						//m_particlesColors[i * 4 + 3] = 1.0f;
					} else
						m_particlesColors[i * 4 + 3] = 1.0f;
				}
			}
			unsigned int sizeParticlesColorVBO = m_particlesColors.size()* sizeof(float);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, &m_particlesColors[0], GL_DYNAMIC_DRAW);
		}

		template<class VectorType, template <class> class ArrayType>
		void ParticlesRenderer<VectorType, ArrayType>::updateParticlesColorsShaders(Scalar minScalarValue, Scalar maxScalarValue) {

			/** Avg and max pressure calculation */
			Scalar avgValue = 0.5*(minScalarValue + maxScalarValue);
			applyColorShader(minScalarValue, maxScalarValue, avgValue);

			glEnable(GL_RASTERIZER_DISCARD_NV);

			glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *m_pParticlesColorVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarAttributesVBO);

			glVertexAttribPointer(0, 1, GL_FLOAT, false, 0, 0);
			glEnableVertexAttribArray(0);

			glBeginTransformFeedback(GL_POINTS);
			glDrawArrays(GL_POINTS, 0, m_pParticlesData->getPositions().size()*2);
			
			glEndTransformFeedback();
			glDisableVertexAttribArray(0);

			glDisable(GL_RASTERIZER_DISCARD_NV);
			removeColorShader();
		}

		template<class VectorType, template <class> class ArrayType>
		void ParticlesRenderer<VectorType, ArrayType>::update(Scalar dt) {
			/** Update min max*/
			if (m_pParticlesData->hasScalarBasedAttribute(m_visAttribute)) {
				m_renderingParams.minScalarfieldValue = FLT_MAX;
				m_renderingParams.maxScalarfieldValue = -FLT_MAX;
				const vector<Scalar> &particleAttributes = m_pParticlesData->getScalarBasedAttribute(m_visAttribute);
				for (int i = 0; i < m_pParticlesData->getPositions().size(); i++) {
					if (particleAttributes[i] < m_renderingParams.minScalarfieldValue)
						m_renderingParams.minScalarfieldValue = particleAttributes[i];

					if (particleAttributes[i] > m_renderingParams.maxScalarfieldValue)
						m_renderingParams.maxScalarfieldValue = particleAttributes[i];
				}
			}
			updateVBOs();
			if(m_renderingParams.selectedScalarParam != -1)
				updateParticlesColorsShaders(m_renderingParams.minScalarfieldValue, m_renderingParams.maxScalarfieldValue);
			//updateParticlesColorsCPU(m_renderingParams.minScalarfieldValue, m_renderingParams.maxScalarfieldValue);
			updateTrails();
			
		}

		template<>
		void ParticlesRenderer<Vector3, Array3D>::draw() {
			glPointSize(m_renderingParams.particleSize);

			if (m_renderingParams.draw) {
				updateVBOs();
				if (m_renderingParams.selectedScalarParam != -1)
					updateParticlesColorsShaders(m_renderingParams.minScalarfieldValue, m_renderingParams.maxScalarfieldValue);
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				glDepthMask(GL_FALSE);
				if (m_renderingParams.drawSelectedVoxelParticles) {
					for (int i = 0; i < m_pParticlesData->getPositions().size(); i++) {
						dimensions_t currParticleDim(	floor(m_pParticlesData->getPositions()[i].x/m_gridSpacing),
														floor(m_pParticlesData->getPositions()[i].y/m_gridSpacing),
														floor(m_pParticlesData->getPositions()[i].z/m_gridSpacing));

						if (currParticleDim == m_renderingParams.selectedVoxelDimension) {
							/*if (m_renderingParams.pParticlesTags != NULL) {
								if (m_renderingParams.pParticlesTags[i] == 0) {
									glColor3f(1.0f, 0.0f, 0.0f);
								}
								else if (m_renderingParams.pParticlesTags[i] == 1) {
									glColor3f(0.0f, 1.0f, 0.0f);
								}
								else if (m_renderingParams.pParticlesTags[i] == 2) {
									glColor3f(0.0f, 0.0f, 1.0f);
								}
								else if (m_renderingParams.pParticlesTags[i] == 3) {
									glColor3f(0.0f, 1.0f, 1.0f);
								}
								else if (m_renderingParams.pParticlesTags[i] == 4) {
									glColor3f(1.0f, 1.0f, 0.0f);
								}
								else {
									glColor3f(0.0f, 0.0f, 0.0f);
								}
							}*/
							glBegin(GL_POINTS);
							glVertex3f(m_pParticlesData->getPositions()[i].x, m_pParticlesData->getPositions()[i].y, m_pParticlesData->getPositions()[i].z);
							glEnd();
						}
					}
				}
				else {
					glEnableClientState(GL_VERTEX_ARRAY);
					if (m_renderingParams.selectedScalarParam != -1) {
						glEnableClientState(GL_COLOR_ARRAY);
						glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
						glColorPointer(4, GL_FLOAT, 0, 0);
					}
					else {
						glColor4f(0.0, 0.0, 0.0, 1.0f);
					}
						
					glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
					glVertexPointer(3, GL_FLOAT, 0, 0);

					glEnable(GL_POINT_SMOOTH);
					glDrawArrays(GL_POINTS, 0, m_pParticlesData->getPositions().size());

					int numTrailsRendered = clamp(m_renderingParams.trailSize, 0, m_currentTrailSize);
					glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
					glVertexPointer(3, GL_FLOAT, 0, 0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pParticleTrailsIndexesVBO);
					glDrawElements(GL_LINES, m_pParticlesData->getPositions().size()*numTrailsRendered * 2, GL_UNSIGNED_INT, 0);

					glBindBuffer(GL_ARRAY_BUFFER, 0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

					glDisableClientState(GL_VERTEX_ARRAY);
					glDisableClientState(GL_COLOR_ARRAY);
					glDisable(GL_POINT_SMOOTH);
				}
				glDepthMask(GL_TRUE);		
				glDisable(GL_BLEND);
			}
		}

		template<>
		void ParticlesRenderer<Vector2, Array2D>::draw() {
			glPointSize(m_renderingParams.particleSize);
			
			if (m_renderingParams.draw) {
				updateVBOs();
				if (m_renderingParams.selectedScalarParam != -1)
					updateParticlesColorsShaders(m_renderingParams.minScalarfieldValue, m_renderingParams.maxScalarfieldValue);

				glDepthMask(GL_FALSE);
				if (m_renderingParams.drawSelectedVoxelParticles) {
					for (int i = 0; i < m_pParticlesData->getPositions().size(); i++) {
						dimensions_t currParticleDim(floor(m_pParticlesData->getPositions()[i].x / m_gridSpacing),
													floor(m_pParticlesData->getPositions()[i].y / m_gridSpacing));

						if (currParticleDim == m_renderingParams.selectedVoxelDimension) {
							/*if (m_renderingParams.pParticlesTags != NULL) {
							if (m_renderingParams.pParticlesTags[i] == 0) {
							glColor3f(1.0f, 0.0f, 0.0f);
							}
							else if (m_renderingParams.pParticlesTags[i] == 1) {
							glColor3f(0.0f, 1.0f, 0.0f);
							}
							else if (m_renderingParams.pParticlesTags[i] == 2) {
							glColor3f(0.0f, 0.0f, 1.0f);
							}
							else if (m_renderingParams.pParticlesTags[i] == 3) {
							glColor3f(0.0f, 1.0f, 1.0f);
							}
							else if (m_renderingParams.pParticlesTags[i] == 4) {
							glColor3f(1.0f, 1.0f, 0.0f);
							}
							else {
							glColor3f(0.0f, 0.0f, 0.0f);
							}
							}*/
							glBegin(GL_POINTS);
							glVertex2f(m_pParticlesData->getPositions()[i].x, m_pParticlesData->getPositions()[i].y);
							glEnd();
						}
					}
				}
				else {
					glEnableClientState(GL_VERTEX_ARRAY);
					glEnableClientState(GL_COLOR_ARRAY);
					glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
					glColorPointer(4, GL_FLOAT, 0, 0);

					glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
					glVertexPointer(2, GL_FLOAT, 0, 0);

					glEnable(GL_POINT_SMOOTH);
					glDrawArrays(GL_POINTS, 0, m_pParticlesData->getPositions().size());

					int numTrailsRendered = clamp(m_renderingParams.trailSize, 0, m_currentTrailSize);
					glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
					glVertexPointer(2, GL_FLOAT, 0, 0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pParticleTrailsIndexesVBO);
					glDrawElements(GL_LINES, m_pParticlesData->getPositions().size()*numTrailsRendered * 2, GL_UNSIGNED_INT, 0);

					glBindBuffer(GL_ARRAY_BUFFER, 0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

					glDisableClientState(GL_VERTEX_ARRAY);
					glDisableClientState(GL_COLOR_ARRAY);
					glDisable(GL_POINT_SMOOTH);
				}
				glDepthMask(GL_TRUE);
			}
		}


		template class ParticlesRenderer<Vector2, Array2D>;
		template class ParticlesRenderer<Vector3, Array3D>;
	}
}