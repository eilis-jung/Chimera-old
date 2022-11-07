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

#include "Particles/ParticleSystem2D.h"

namespace Chimera {

	namespace Rendering {
		const int ParticleSystem2D::s_maxTrailSize = 20;

		void ParticleSystem2D::initializeParticles() {
			if (m_params.emitterType == noEmitter && m_params.pExternalParticlePositions != NULL) {
				m_pParticlesPosition = &(*m_params.pExternalParticlePositions)[0];
				if (m_params.pExternalVelocities) {
					m_pParticlesVelocity = &(*m_params.pExternalVelocities)[0];
				}
				m_pParticlesColor = new Vector3[m_params.numParticles];
				/** Velocity, vorticity and life time are not used in this configuration */
				m_pParticlesVorticity = NULL;
				m_pParticlesLifeTime = NULL;
			}
			else {
				m_pParticlesPosition = new Vector2[m_params.numParticles];
				m_pParticlesVelocity = new Vector2[m_params.numParticles];
				m_pParticlesLifeTime = new Scalar[m_params.numParticles];
				m_pParticlesVorticity = new Scalar[m_params.numParticles];
				m_pParticlesColor = new Vector3[m_params.numParticles];
				m_params.pResampledParticles = new vector<bool>(m_params.numParticles);

				for (int i = 0; i < m_params.numParticles; i++) {
					m_pParticlesPosition[i].x = FLT_MIN;
					m_pParticlesPosition[i].y = FLT_MIN;
					m_pParticlesVelocity[i] = Vector2(0, 0);
					m_pParticlesLifeTime[i] = FLT_MAX;
					m_pParticlesVorticity[i] = 0;
					m_pParticlesColor[i] = Vector3(1, 0, 0); //All red, just for testing
					m_params.pResampledParticles->at(i) = false;
				}
			}
			if (m_params.emitterType == squareRandomEmitter) {
				for (int i = 0; i < m_params.initialNumParticles; i++) {
					m_pParticlesPosition[i].x = m_params.emitterPosition.x + m_params.emitterSize.x*(rand() / (float)RAND_MAX);
					m_pParticlesPosition[i].y = m_params.emitterPosition.y + m_params.emitterSize.y*(rand() / (float)RAND_MAX);
					m_pParticlesVelocity[i] = Vector2(0, 0);
					m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand() / (float)RAND_MAX);
					m_pParticlesVorticity[i] = 0;
				}
			}
			else if (m_params.emitterType == circleRandomEmitter) {
				for (int i = 0; i < m_params.initialNumParticles; i++) {
					m_pParticlesPosition[i].x = m_params.emitterPosition.x + m_params.emitterSize.x*(rand() / (float)RAND_MAX);
					m_pParticlesPosition[i].y = m_params.emitterPosition.y + m_params.emitterSize.x*(rand() / (float)RAND_MAX);
					m_pParticlesVelocity[i] = Vector2(0, 0);
					m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand() / (float)RAND_MAX);
					m_pParticlesVorticity[i] = 0;
				}
			}
			numInitializedParticles = m_params.initialNumParticles;
			Scalar dx = m_pQuadGrid->getGridData2D()->getScaleFactor(0, 0).x;

			if (m_params.emitterType == noEmitter) {
				/** Initializing first particles in trails */
				for (int i = 0; i < m_params.numParticles; i++) {
					m_particlesTrails(i, 0) = m_pParticlesPosition[i];
				}
			}
			else {
				/** Initializing first particles in trails */
				for (int i = 0; i < m_params.numParticles; i++) {
					m_particlesTrails(i, 0) = m_pParticlesPosition[i];
				}
			}

			//Initializing trails indexes
			m_pParticlesTrailsIndexes = new int[m_params.numParticles*s_maxTrailSize * 2];

			int currIndex = 0;
			for (int j = 0; j < s_maxTrailSize - 1; j++) {
				for (int i = 0; i < m_params.numParticles; i++) {
					int testIndex = m_particlesTrails.getRawPtrIndex(i, j);
					m_pParticlesTrailsIndexes[currIndex++] = m_particlesTrails.getRawPtrIndex(i, j);
					m_pParticlesTrailsIndexes[currIndex++] = m_particlesTrails.getRawPtrIndex(i, j + 1);
				}
			}
		}
		void ParticleSystem2D::initializeVBOs() {
			m_pParticlesVBO = new GLuint();
			unsigned int sizeParticlesVBO = m_params.numParticles * sizeof(Vector2);

			glGenBuffers(1, m_pParticlesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_particlesTrails.getRawDataPointer(), GL_DYNAMIC_DRAW);

			m_pParticlesColorVBO = new GLuint;
			unsigned int sizeParticlesColorVBO = m_params.numParticles * sizeof(Vector3);
			glGenBuffers(1, m_pParticlesColorVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, m_pParticlesColor, GL_DYNAMIC_DRAW);

			/** Initializing particles trails */
			m_pParticleTrailsVBO = new GLuint;
			sizeParticlesVBO = m_params.numParticles*s_maxTrailSize * sizeof(Vector2);
			glGenBuffers(1, m_pParticleTrailsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_particlesTrails.getRawDataPointer(), GL_DYNAMIC_DRAW);

			/** Initializing particles trails indexes */
			m_pParticleTrailsIndexesVBO = new GLuint;
			sizeParticlesVBO = m_params.numParticles*s_maxTrailSize * sizeof(int) * 2;
			glGenBuffers(1, m_pParticleTrailsIndexesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsIndexesVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_pParticlesTrailsIndexes, GL_STATIC_DRAW);
		}
		void ParticleSystem2D::initializeShaders() {
			GLchar const * Strings[] = { "rColor", "gColor", "bColor" };
			m_pFiftyShadersOfGray = ResourceManager::getInstance()->loadGLSLShader("Shaders/2D/FiftyShadersOfGray.glsl", "Shaders/2D/FiftyShadersOfGray.frag");
			//m_pFiftyShadersOfGray = ResourceManager::getInstance()->loadGLSLShader("Shaders/2D/FiftyShadersOfGray.glsl", "Shaders/2D/FiftyShadersOfGray.glsl");

			m_trailSizeLoc = glGetUniformLocation(m_pFiftyShadersOfGray->getProgramID(), "trailSize");
			m_maxNumberOfParticlesLoc = glGetUniformLocation(m_pFiftyShadersOfGray->getProgramID(), "maxNumberOfParticles");
		}

		void ParticleSystem2D::updateTrails() {
			Scalar dx = m_pQuadGrid->getGridData2D()->getScaleFactor(0, 0).x;
			if (m_currentTrailSize < s_maxTrailSize - 1)
				m_currentTrailSize++;

			for (int j = m_currentTrailSize; j >= 0; j--) {
				for (int i = 0; i < m_params.numParticles; i++) {
					if (j == 0) {
						if (m_params.pResampledParticles && m_params.pResampledParticles->at(i)) {
							for (int k = 0; k <= m_currentTrailSize; k++) {
								if (m_params.emitterType == noEmitter)
									m_particlesTrails(i, k) = m_pParticlesPosition[i];
								else
									m_particlesTrails(i, k) = m_pParticlesPosition[i];
							}
						}
						else {
							if (m_params.emitterType == noEmitter)
								m_particlesTrails(i, 0) = m_pParticlesPosition[i];
							else
								m_particlesTrails(i, 0) = m_pParticlesPosition[i];
						}
					}
					else {
						m_particlesTrails(i, j) = m_particlesTrails(i, j - 1);
					}
				}
			}
			if (m_pCutCells2D && m_renderingParams.m_clipParticlesTrails) {
				for (int i = 0; i < m_params.numParticles; i++) {
					for (int j = 0; j < s_maxTrailSize - 1; j++) {
						Vector2 transformedParticlePos = m_particlesTrails(i, j) / dx;
						if (m_pCutCells2D->isCutCellAt(transformedParticlePos.x, transformedParticlePos.y)) {
							uint specialCellIndex = m_pCutCells2D->getCutCellIndex(transformedParticlePos);
							auto cutFace = m_pCutCells2D->getCutCell(specialCellIndex);
							Vector2 p1 = m_particlesTrails(i, j);
							Vector2 p2 = m_particlesTrails(i, j + 1);
							Vector2 crossingPoint;
							bool crossed = cutFace.crossedThroughGeometry(p1, p2, crossingPoint);
							if (crossed) {
								while (j < s_maxTrailSize - 1) {
									j++;
									m_particlesTrails(i, j) = crossingPoint;
								}
							}
						}
					}
				}
			}

		}
		void ParticleSystem2D::updateVBOs() {
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
			glBufferData(GL_ARRAY_BUFFER, m_params.numParticles * sizeof(Vector2), m_particlesTrails.getRawDataPointer(), GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
			glBufferData(GL_ARRAY_BUFFER, m_params.numParticles*s_maxTrailSize * sizeof(Vector2), m_particlesTrails.getRawDataPointer(), GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		void ParticleSystem2D::updateLocalAxis() {
			//Updating grid position
			Vector2 oldGridPos = m_gridOriginPosition;
			m_gridOriginPosition.x += -m_rotationPoint.x*cos(m_gridOrientation) + m_rotationPoint.y*sin(m_gridOrientation);
			m_gridOriginPosition.y += -m_rotationPoint.x*sin(m_gridOrientation) - m_rotationPoint.y*cos(m_gridOrientation);

			m_gridOriginPosition += m_rotationPoint;

			Vector2 tempLocalAxisX = Vector2(1, 0);// - m_rotationPoint;
			m_localAxisX.x = tempLocalAxisX.x*cos(m_gridOrientation) - tempLocalAxisX.y*sin(m_gridOrientation);
			m_localAxisX.y = tempLocalAxisX.x*sin(m_gridOrientation) + tempLocalAxisX.y*cos(m_gridOrientation);
			m_localAxisX.normalize();

			Vector2 tempLocalAxisY = Vector2(0, 1);// - m_rotationPoint;
			m_localAxisY.x = tempLocalAxisY.x*cos(m_gridOrientation) - tempLocalAxisY.y*sin(m_gridOrientation);
			m_localAxisY.y = tempLocalAxisY.x*sin(m_gridOrientation) + tempLocalAxisY.y*cos(m_gridOrientation);
			m_localAxisY.normalize();
		}

		void ParticleSystem2D::updateEmission(Scalar dt) {
			int totalSpawnedParticles = numInitializedParticles;
			for (int i = numInitializedParticles; numInitializedParticles - totalSpawnedParticles < m_params.emitterSpawnRatio*dt; i++, numInitializedParticles++) {
				if (i >= m_params.numParticles)
					i = 0;

				m_pParticlesPosition[i].x = m_params.emitterPosition.x + m_params.emitterSize.x*(rand() / (float)RAND_MAX);
				m_pParticlesPosition[i].y = m_params.emitterPosition.y + m_params.emitterSize.y*(rand() / (float)RAND_MAX);
				m_pParticlesVelocity[i] = Vector2(0, 0);
				m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand() / (float)RAND_MAX);
				m_pParticlesVorticity[i] = 0;
				m_params.pResampledParticles->at(i) = true;
			}
		}
		void ParticleSystem2D::resetParticleSystem() {
			for (int i = 0; i < m_params.numParticles; i++) {
				m_pParticlesPosition[i].x = FLT_MIN;
				m_pParticlesPosition[i].y = FLT_MIN;
				m_pParticlesVelocity[i] = Vector2(0, 0);
				m_pParticlesLifeTime[i] = FLT_MAX;
				m_pParticlesVorticity[i] = 0;
			}

			if (m_params.emitterType == squareRandomEmitter) {
				for (int i = 0; i < m_params.initialNumParticles; i++) {
					m_pParticlesPosition[i].x = m_params.emitterPosition.x + m_params.emitterSize.x*(rand() / (float)RAND_MAX);
					m_pParticlesPosition[i].y = m_params.emitterPosition.y + m_params.emitterSize.y*(rand() / (float)RAND_MAX);
					m_pParticlesVelocity[i] = Vector2(0, 0);
					m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand() / (float)RAND_MAX);
					m_pParticlesVorticity[i] = 0;
				}
			}
			else if (m_params.emitterType == circleRandomEmitter) {
				for (int i = 0; i < m_params.initialNumParticles; i++) {
					m_pParticlesPosition[i].x = m_params.emitterPosition.x + m_params.emitterSize.x*(rand() / (float)RAND_MAX);
					m_pParticlesPosition[i].y = m_params.emitterPosition.y + m_params.emitterSize.x*(rand() / (float)RAND_MAX);
					m_pParticlesVelocity[i] = Vector2(0, 0);
					m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand() / (float)RAND_MAX);
					m_pParticlesVorticity[i] = 0;
				}
			}
			numInitializedParticles = m_params.initialNumParticles;
		}

		Vector3 ParticleSystem2D::jetShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue) {
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
		void ParticleSystem2D::updateParticlesColors(Scalar minScalarValue, Scalar maxScalarValue) {
			dimensions_t gridDimensions = m_pGridData->getDimensions();
			if (m_renderingParams.pParticlesTags != NULL) {
				#pragma omp parallel for
				for (int i = 0; i < m_params.numParticles; i++) {
					if (m_renderingParams.pParticlesTags->at(i) == 0) {
						m_pParticlesColor[i].x = m_pParticlesColor[i].y = m_pParticlesColor[i].z = 0.0f;
					}
					else if (m_renderingParams.pParticlesTags->at(i) == 1) {
						m_pParticlesColor[i].x = m_pParticlesColor[i].y = 0.0f;
						m_pParticlesColor[i].z = 1.0f;
					}
					else {
						m_pParticlesColor[i].x = m_pParticlesColor[i].y = m_pParticlesColor[i].z = 0.0f;
					}
				}
			} else if(m_params.pParticlesData && m_params.pParticlesData->hasIntegerBasedAttribute("liquid")) {
				const vector<int> &particleTags = m_params.pParticlesData->getIntegerBasedAttribute("liquid");
				for (int i = 0; i < particleTags.size(); i++) {
					if (particleTags[i] == 0) {
						m_pParticlesColor[i].x = m_pParticlesColor[i].y = m_pParticlesColor[i].z = 0.0f;
					}
					else if (particleTags[i] == 1) {
						m_pParticlesColor[i].z = 1.0f;
					}
					else {
						m_pParticlesColor[i].x = m_pParticlesColor[i].y = m_pParticlesColor[i].z = 0.0f;
					}
				}
			} else {
				#pragma omp parallel for
				for (int i = 0; i < numInitializedParticles; i++) {
					Scalar scalarFieldValue = 0;
					Vector2 particleLocalPosition;

					/*particleLocalPosition = (m_pParticlesPosition[i] - m_pQuadGrid->getGridOrigin()) / m_dx;
					if (particleLocalPosition.x > 0 && particleLocalPosition.x < gridDimensions.x
						&& particleLocalPosition.y > 0 && particleLocalPosition.y < gridDimensions.y) {
						scalarFieldValue = m_pGridData->getPressure(particleLocalPosition.x, particleLocalPosition.y);
					}*/
					scalarFieldValue = clamp(scalarFieldValue, minScalarValue, maxScalarValue);
					//m_pParticlesColor[i] = jetShading(i, scalarFieldValue, minScalarValue, maxScalarValue);
				}

			}

			unsigned int sizeParticlesColorVBO = m_params.numParticles * sizeof(Vector3);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, m_pParticlesColor, GL_DYNAMIC_DRAW);
		}

		void ParticleSystem2D::update(Scalar dt) {
			if (m_params.emitterType != noEmitter) { //Particles are being updated from an external source
				updateLocalAxis();

				updateEmission(dt);

				Vector2 emitterTotalSize = m_params.emitterPosition + m_params.emitterSize;
				Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
				dimensions_t gridDimensions = m_pGridData->getDimensions();

				#pragma omp parallel for
				for (int i = 0; i < m_params.numParticles; i++) {
					Vector2 particleLocalPosition, oldParticleLocalPosition;

					particleLocalPosition = m_pParticlesPosition[i] - m_gridOriginPosition;
					oldParticleLocalPosition = particleLocalPosition;
					particleLocalPosition.x = oldParticleLocalPosition.dot(m_localAxisX);
					particleLocalPosition.y = oldParticleLocalPosition.dot(m_localAxisY);


					if (m_pParticlesPosition[i].x > m_params.particlesMaxBounds.x ||
						m_pParticlesPosition[i].x < m_params.particlesMinBounds.x ||
						m_pParticlesPosition[i].y > m_params.particlesMaxBounds.y ||
						m_pParticlesPosition[i].y < m_params.particlesMinBounds.y ||
						m_pParticlesLifeTime[i] < 0) {
						m_pParticlesPosition[i].x = m_params.emitterPosition.x + m_params.emitterSize.x*(rand() / (float)RAND_MAX);
						m_pParticlesPosition[i].y = m_params.emitterPosition.y + m_params.emitterSize.y*(rand() / (float)RAND_MAX);
						m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand() / (float)RAND_MAX);
						m_params.pResampledParticles->at(i) = true;
						continue;
					}

					if (m_pVelocityInterpolant) {
						if (particleLocalPosition.x / m_dx < gridDimensions.x && particleLocalPosition.x / m_dx > 0 &&
							particleLocalPosition.y / m_dx < gridDimensions.y && particleLocalPosition.y / m_dx > 0) {
							Vector2 interpVel;
							interpVel = m_pVelocityInterpolant->interpolate(particleLocalPosition);
							Vector2 tempPosition = particleLocalPosition + interpVel*dt*0.5;
							m_pParticlesVelocity[i] = m_pVelocityInterpolant->interpolate(tempPosition);
						}
					}


					m_pParticlesPosition[i] += m_pParticlesVelocity[i] * dt;
					m_pParticlesLifeTime[i] -= dt;

					if (m_pParticlesPosition[i].x > m_params.particlesMaxBounds.x ||
						m_pParticlesPosition[i].x < m_params.particlesMinBounds.x ||
						m_pParticlesPosition[i].y > m_params.particlesMaxBounds.y ||
						m_pParticlesPosition[i].y < m_params.particlesMinBounds.y ||
						m_pParticlesLifeTime[i] < 0) {
						m_params.pResampledParticles->at(i) = true;
					}
					else {
						m_params.pResampledParticles->at(i) = false;
					}
				}
			}
			updateParticlesColors(m_renderingParams.minScalarfieldValue, m_renderingParams.maxScalarfieldValue);
			updateTrails();
			updateVBOs();
		}

		void ParticleSystem2D::drawLocalAxis() {
			glPushMatrix();
			glLoadIdentity();

			glLineWidth(2.0f);
			glColor3f(0.0f, 1.0f, 0.0f);
			glBegin(GL_LINES);
			glVertex2f(m_gridOriginPosition.x, m_gridOriginPosition.y);
			glVertex2f(m_gridOriginPosition.x + m_localAxisX.x, m_gridOriginPosition.y + m_localAxisX.y);
			glEnd();

			glColor3f(1.0f, 0.0f, 0.0f);
			glBegin(GL_LINES);
			glVertex2f(m_gridOriginPosition.x, m_gridOriginPosition.y);
			glVertex2f(m_gridOriginPosition.x + m_localAxisY.x, m_gridOriginPosition.y + m_localAxisY.y);
			glEnd();

			glPopMatrix();
		}
		void ParticleSystem2D::draw() {
			Scalar dx = m_pQuadGrid->getGridData2D()->getScaleFactor(0, 0).x;
			if (m_renderingParams.m_draw) {
				glEnableClientState(GL_VERTEX_ARRAY);
				if (m_renderingParams.colorScheme != singleColor || m_renderingParams.pParticlesTags != NULL || (m_params.pParticlesData && m_params.pParticlesData->hasIntegerBasedAttribute("liquid"))) {
					glEnableClientState(GL_COLOR_ARRAY);
					glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO); //glBindBuffer(GL_COLOR_ARRAY_BUFFER_BINDING, *m_pParticlesColorVBO);
					glColorPointer(3, GL_FLOAT, 0, (void *)0);
				}
				else {
					glDisableClientState(GL_COLOR_ARRAY);
					glColor3f(m_renderingParams.particleColor[0] * 0.75, m_renderingParams.particleColor[1] * 0.75, m_renderingParams.particleColor[2] * 0.75); //Drawing particles in a darker tone
				}

				glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
				glVertexPointer(2, GL_FLOAT, 0, 0);

				glPointSize(m_renderingParams.particleSize);
				glEnable(GL_POINT_SMOOTH);
				if (m_renderingParams.m_drawParticles)
					glDrawArrays(GL_POINTS, 0, m_params.numParticles);

				if (m_renderingParams.colorScheme == singleColor) { //Drawing 
					//glColor3f(m_renderingParams.particleColor[0], m_renderingParams.particleColor[1], m_renderingParams.particleColor[2]);
				}

				//Drawing trails
				glLineWidth(m_renderingParams.particleSize);
				//glGetFloatv(GL_MODELVIEW_MATRIX, m_modelViewMatrix);

				int numTrailsRendered = clamp(m_renderingParams.trailSize, 0, m_currentTrailSize);
				//if (m_renderingParams.colorScheme == grayscale) { //Drawing 
				m_pFiftyShadersOfGray->applyShader();
				//}
				glUniform1i(m_trailSizeLoc, numTrailsRendered);
				glUniform1i(m_maxNumberOfParticlesLoc, m_params.numParticles);

				glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
				glVertexPointer(2, GL_FLOAT, 0, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pParticleTrailsIndexesVBO);

				glDrawElements(GL_LINES, m_params.numParticles*numTrailsRendered * 2, GL_UNSIGNED_INT, 0); //draw all elements
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

				glDisableClientState(GL_VERTEX_ARRAY);
				glDisableClientState(GL_COLOR_ARRAY);
				glDisable(GL_POINT_SMOOTH);

				//if (m_renderingParams.colorScheme == grayscale) { //Drawing 
				m_pFiftyShadersOfGray->removeShader();
				//}


				/*if (m_renderingParams.pParticlesTags != NULL) {
					glBegin(GL_POINTS);
					for (int i = 0; i < m_params.numParticles; i++) {
						if (m_renderingParams.pParticlesTags->at(i) == 0) {
							glColor3f(1.0f, 0.0f, 0.0f);
						}
						else if (m_renderingParams.pParticlesTags->at(i) == 1) {
							glColor3f(0.0f, 0.0f, 1.0f);
						}
						else {
							glColor3f(0.0f, 0.0f, 0.0f);
						}
						glVertex2f(m_pParticlesPosition[i].x, m_pParticlesPosition[i].y*dx);
					}
					glEnd();
				}*/

			}

			if (m_pParticlesVelocity) {
				// draw each component of angular momentum
				if (m_renderingParams.m_drawVelocities) {
					for (int i = 0; i < m_params.numParticles; ++i) {
						Vector2 p(m_pParticlesPosition[i].x, m_pParticlesPosition[i].y);
						Vector2 v(m_pParticlesVelocity[i].x*m_renderingParams.velocityScale, m_pParticlesVelocity[i].y*m_renderingParams.velocityScale);

						RenderingUtils::getInstance()->drawVector(p, p + v, 0.01);
					}
				}
			}
		}
		

		void ParticleSystem2D::tagDisk(Vector2 diskPosition, Scalar diskSize) {

			//First initialize the circle
			if (m_renderingParams.pParticlesTags == NULL) {
				m_renderingParams.pParticlesTags = new vector<int>(m_params.numParticles, 0);
			}
			for (int i = 0; i < m_params.numParticles; i++) {
				Scalar distance = (m_pParticlesPosition[i] - diskPosition).length();
				if (distance < diskSize) {
					m_renderingParams.pParticlesTags->at(i) = 1;
				}
			}

			updateParticlesColors(0, 0);
		}

		void ParticleSystem2D::tagZalezaskDisk(Vector2 diskPosition, Scalar diskSize, Scalar dentSize) {
			//First initialize the circle
			if (m_renderingParams.pParticlesTags == NULL) {
				m_renderingParams.pParticlesTags = new vector<int>(m_params.numParticles, 0);
			}

			tagDisk(diskPosition, diskSize);

			Vector2 boxCenter(diskPosition);
			boxCenter.y -= dentSize*0.5;
			Vector2 boxMin(boxCenter);
			boxMin.x -= dentSize*0.5;
			boxMin.y -= diskSize*0.9;
			Vector2 boxMax(boxCenter);
			boxMax.x += dentSize*0.5;
			boxMax.y += diskSize*0.5;

			for (int i = 0; i < m_params.numParticles; i++) {
				if(m_pParticlesPosition[i].x > boxMin.x && m_pParticlesPosition[i].y > boxMin.y &&
					m_pParticlesPosition[i].x < boxMax.x && m_pParticlesPosition[i].y < boxMax.y) {
					m_renderingParams.pParticlesTags->at(i) = 0;
				}
			}
			updateParticlesColors(0, 0);
		}

	}
	
}