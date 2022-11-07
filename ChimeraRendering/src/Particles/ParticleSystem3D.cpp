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

#include "Particles/ParticleSystem3D.h"
#include "CGAL/PolygonSurface.h"
#include "Physics/PhysicsCore.h"

namespace Chimera {

	namespace Rendering {
		const int ParticleSystem3D::s_maxTrailSize = 8;

		void ParticleSystem3D::initializeParticles() {
			m_numInitializedParticles = 0;
			if (m_renderingParams.visualizeNormalsVelocities) {
				m_particlesVelocitiesVis.resize(m_maxNumberOfParticles);
				m_particlesNormalsVis.resize(m_maxNumberOfParticles);
			}

			m_renderingParams.pParticlesTags = new int[m_maxNumberOfParticles];

			if (m_params.emitters.size() == 0 && m_params.pExternalParticlePositions != NULL) {
				m_pParticlesPosition = m_params.pExternalParticlePositions;
				m_pParticlesColor = new Scalar[m_maxNumberOfParticles * 4 * s_maxTrailSize];
				m_numInitializedParticles = m_params.pExternalParticlePositions->size();
				/** Velocity, vorticity and life time are not used in this configuration */
				m_pParticlesVelocity = NULL;
				m_pParticlesVorticity = NULL;
				m_pParticlesLifeTime = NULL;
			}
			else {
				m_pParticlesPosition = new vector<Vector3>(m_maxNumberOfParticles, Vector3(0, 0, 0));
				m_pParticlesVelocity = new Vector3[m_maxNumberOfParticles];
				m_pParticlesLifeTime = new Scalar[m_maxNumberOfParticles];
				m_pParticlesVorticity = new Scalar[m_maxNumberOfParticles];
				m_pParticlesColor = new Scalar[m_maxNumberOfParticles * 4 * s_maxTrailSize];
				m_params.pResampledParticles = new vector<bool>(m_maxNumberOfParticles);

				for (int i = 0; i < m_maxNumberOfParticles; i++) {
					m_pParticlesPosition->at(i).x = FLT_MIN;
					m_pParticlesPosition->at(i).y = FLT_MIN;
					m_pParticlesPosition->at(i).z = FLT_MIN;
					m_pParticlesVelocity[i] = Vector3(0, 0, 0);
					m_pParticlesLifeTime[i] = FLT_MAX;
					m_pParticlesVorticity[i] = 0;
					m_params.pResampledParticles->at(i) = false;
				}

			}

			for (int i = 0; i < m_params.emitters.size(); i++) {
				if (m_params.emitters[i].emitterType == cubeRandomEmitter) {
					for (int j = 0; j < m_params.emitters[i].initialNumberOfParticles; j++) {
						unsigned int currIndex = m_numInitializedParticles + j;
						m_pParticlesPosition->at(currIndex) = cubeSampling(i);
						m_pParticlesVelocity[currIndex] = Vector3(0, 0, 0);
						m_pParticlesLifeTime[currIndex] = m_params.emitters[i].particlesLife + m_params.emitters[i].particlesLifeVariance*(rand() / (float)RAND_MAX);
						m_pParticlesVorticity[currIndex] = 0;
						m_renderingParams.pParticlesTags[currIndex] = i;
					}

				}
				else  if (m_params.emitters[i].emitterType == sphereRandomEmitter) {
					for (int j = 0; j < m_params.emitters[i].initialNumberOfParticles; j++) {
						unsigned int currIndex = m_numInitializedParticles + j;
						m_pParticlesPosition->at(currIndex) = sphereSampling(i);
						m_pParticlesVelocity[currIndex] = Vector3(0, 0, 0);
						m_pParticlesLifeTime[currIndex] = m_params.emitters[i].particlesLife + m_params.emitters[i].particlesLifeVariance*(rand() / (float)RAND_MAX);
						m_pParticlesVorticity[currIndex] = 0;
						m_renderingParams.pParticlesTags[currIndex] = i;
					}
				}
				m_numInitializedParticles += m_params.emitters[i].initialNumberOfParticles;
				m_params.emitters[i].totalSpawnedParticles = m_params.emitters[i].initialNumberOfParticles;
			}
			Scalar dx = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;

			if (m_params.emitters.size() == 0 && m_params.pExternalParticlePositions != NULL) {
				/** Initializing first particles in trails */
				for (int i = 0; i < m_maxNumberOfParticles; i++) {
					(*m_pParticlesTrails)(i, 0) = m_pParticlesPosition->at(i);
				}
			}
			else {
				/** Initializing first particles in trails */
				for (int i = 0; i < m_maxNumberOfParticles; i++) {
					(*m_pParticlesTrails)(i, 0) = m_pParticlesPosition->at(i);
				}
			}

			//Initializing trails indexes
			m_pParticlesTrailsIndexes = new int[m_maxNumberOfParticles*s_maxTrailSize * 2];

			int currIndex = 0;
			for (int j = 0; j < s_maxTrailSize - 1; j++) {
				for (int i = 0; i < m_maxNumberOfParticles; i++) {
					m_pParticlesTrailsIndexes[currIndex++] = m_pParticlesTrails->getRawPtrIndex(i, j);
					m_pParticlesTrailsIndexes[currIndex++] = m_pParticlesTrails->getRawPtrIndex(i, j + 1);
				}
			}

			if (m_renderingParams.colorScheme == randomColors) {
				initializeRandomParticlesColors();
			}

			if (_CrtCheckMemory()) {
				Logger::getInstance()->get() << "Memory check for heap inconsistency working fine (ONLY WORKS FOR DEBUG MODE). " << endl;
			}
			else {
				Logger::getInstance()->get() << "Memory check for heap inconsistency FAILED (ONLY WORKS FOR DEBUG MODE). " << endl;
			}

		}

		void ParticleSystem3D::initializeRandomParticlesColors() {
			for (int i = 0; i < m_maxNumberOfParticles; i++) {
				m_pParticlesColor[i * 4] = m_pParticlesColor[i * 4 + 1] = m_pParticlesColor[i * 4 + 2] = rand() / ((float)RAND_MAX);
				m_pParticlesColor[i * 4 + 3] = 1.0f;
			}
		}

		void ParticleSystem3D::initializeVBOs() {
			m_pParticlesVBO = new GLuint();
			unsigned int sizeParticlesVBO = m_maxNumberOfParticles * sizeof(Vector3);

			glGenBuffers(1, m_pParticlesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_pParticlesTrails->getRawDataPointer(), GL_DYNAMIC_DRAW);

			m_pParticlesColorVBO = new GLuint;
			unsigned int sizeParticlesColorVBO = m_maxNumberOfParticles * sizeof(float) * 4 * s_maxTrailSize;
			glGenBuffers(1, m_pParticlesColorVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, m_pParticlesColor, GL_DYNAMIC_DRAW);

			/** Initializing particles trails */
			m_pParticleTrailsVBO = new GLuint;
			sizeParticlesVBO = m_maxNumberOfParticles*s_maxTrailSize * sizeof(Vector3);
			glGenBuffers(1, m_pParticleTrailsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_pParticlesTrails->getRawDataPointer(), GL_DYNAMIC_DRAW);

			/** Initializing particles trails indexes */
			m_pParticleTrailsIndexesVBO = new GLuint;
			sizeParticlesVBO = m_maxNumberOfParticles*s_maxTrailSize * sizeof(int) * 2;
			glGenBuffers(1, m_pParticleTrailsIndexesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsIndexesVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_pParticlesTrailsIndexes, GL_STATIC_DRAW);

		}
		void ParticleSystem3D::updateTrails() {
			Scalar dx = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
			if (m_currentTrailSize < s_maxTrailSize - 1)
				m_currentTrailSize++;

			for (int j = m_currentTrailSize; j >= 0; j--) {
				for (int i = 0; i < m_maxNumberOfParticles; i++) {
					if (j == 0) {
						if (m_params.pResampledParticles && m_params.pResampledParticles->at(i)) {
							for (int k = 0; k <= m_currentTrailSize; k++) {
								if (m_params.emitters.size() == 0)
									(*m_pParticlesTrails)(i, k) = m_pParticlesPosition->at(i);
								else
									(*m_pParticlesTrails)(i, k) = m_pParticlesPosition->at(i);
							}
						}
						else {
							if (m_params.emitters.size() == 0)
								(*m_pParticlesTrails)(i, 0) = m_pParticlesPosition->at(i);
							else
								(*m_pParticlesTrails)(i, 0) = m_pParticlesPosition->at(i);
						}
					}
					else {
						(*m_pParticlesTrails)(i, j) = (*m_pParticlesTrails)(i, j - 1);
					}
				}
			}
		}
		void ParticleSystem3D::updateVBOs() {
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
			glBufferData(GL_ARRAY_BUFFER, m_maxNumberOfParticles * sizeof(Vector3), m_pParticlesTrails->getRawDataPointer(), GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
			glBufferData(GL_ARRAY_BUFFER, m_maxNumberOfParticles*s_maxTrailSize * sizeof(Vector3), m_pParticlesTrails->getRawDataPointer(), GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		void ParticleSystem3D::updateParticlesColorsGridSlice() {
			int numTrailsRendered = clamp(m_renderingParams.trailSize, 0, m_currentTrailSize);
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
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, m_pParticlesColor, GL_DYNAMIC_DRAW);
		}

		void ParticleSystem3D::updateEmission(Scalar dt) {
			for (int i = 0; i < m_params.emitters.size(); i++) {
				for (int j = 0; j < m_params.emitters[i].spawnRatio*dt; j++) {
					m_pParticlesVelocity[m_numInitializedParticles] = Vector3(0, 0, 0);
					m_pParticlesLifeTime[m_numInitializedParticles] = m_params.emitters[i].particlesLife + m_params.emitters[i].particlesLifeVariance*(rand() / (float)RAND_MAX);
					m_pParticlesVorticity[m_numInitializedParticles] = 0;
					m_params.pResampledParticles->at(m_numInitializedParticles) = true;
					m_renderingParams.pParticlesTags[m_numInitializedParticles] = i;
					Vector3 particlePos;
					if (m_params.emitters[i].emitterType == cubeRandomEmitter) {
						particlePos = cubeSampling(i);
					}
					else if (m_params.emitters[i].emitterType == sphereRandomEmitter) {
						particlePos = sphereSampling(i);
					}
					for (int k = 0; k < s_maxTrailSize; k++) {
						(*m_pParticlesTrails)(m_numInitializedParticles, k) = particlePos;
					}
					m_pParticlesPosition->at(m_numInitializedParticles++) = particlePos;
				}
			}
		}

		void ParticleSystem3D::updateParticlesTags() {
			Scalar dx = m_pGridData->getScaleFactor(0, 0, 0).x;
			if (m_pCutVoxels) {
				for (int i = 0; i < m_maxNumberOfParticles; i++) {
					dimensions_t currParticleDim(floor(m_pParticlesPosition->at(i).x/dx), floor(m_pParticlesPosition->at(i).y/dx), floor(m_pParticlesPosition->at(i).z/dx));
					if (m_pCutVoxels->isCutVoxel(currParticleDim.x, currParticleDim.y, currParticleDim.z) && currParticleDim == m_renderingParams.selectedVoxelDimension) {
						uint cutVoxelIndex = m_pCutVoxels->getCutVoxelIndex(m_pParticlesPosition->at(i) / dx);
						m_renderingParams.pParticlesTags[i] =  cutVoxelIndex % 5;
					}
				}
			}
			//if (m_pSpecialCells) {
			//	for (int i = 0; i < m_maxNumberOfParticles; i++) {
			//		dimensions_t currParticleDim(floor(m_pParticlesPosition->at(i).x), floor(m_pParticlesPosition->at(i).y), floor(m_pParticlesPosition->at(i).z));
			//		if (m_pSpecialCells->isSpecialCell(currParticleDim.x, currParticleDim.y, currParticleDim.z) && currParticleDim == m_renderingParams.selectedVoxelDimension) {
			//			m_renderingParams.pParticlesTags[i] = m_pSpecialCells->getCutVoxelIndex(convertToVector3D(m_pParticlesPosition->at(i)), m_pNodeVelocityField->pMeshes);
			//			m_renderingParams.pParticlesTags[i] -= m_pSpecialCells->getCutVoxelIndex(currParticleDim.x, currParticleDim.y, currParticleDim.z);
			//		}
			//	}
			//}
			//else if (m_params.pLinearInterpolant) {
			//	for (int i = 0; i < m_maxNumberOfParticles; i++) {
			//		dimensions_t currParticleDim(floor(m_pParticlesPosition->at(i).x), floor(m_pParticlesPosition->at(i).y), floor(m_pParticlesPosition->at(i).z));
			//		Vector3 transformedParticlePos = m_pParticlesPosition->at(i) / m_dx;
			//		int meshIndex = m_params.pLinearInterpolant->getMeshIndex(transformedParticlePos);
			//		if (meshIndex != -1) {
			//			//Trick to find the relative index position relative to the first cut-cell on the regular grid location
			//			m_renderingParams.pParticlesTags[i] = meshIndex - m_params.pLinearInterpolant->getCutCellMap()(currParticleDim);
			//		}
			//	}
			//}

		}

		void ParticleSystem3D::resetParticleSystem() {
			for (int i = 0; i < m_maxNumberOfParticles; i++) {
				m_pParticlesPosition->at(i).x = FLT_MIN;
				m_pParticlesPosition->at(i).y = FLT_MIN;
				m_pParticlesPosition->at(i).z = FLT_MIN;
				m_pParticlesVelocity[i] = Vector3(0, 0, 0);
				m_pParticlesLifeTime[i] = FLT_MAX;
				m_pParticlesVorticity[i] = 0;
			}

			m_numInitializedParticles = 0;
			for (int i = 0; i < m_params.emitters.size(); i++) {
				if (m_params.emitters[i].emitterType == cubeRandomEmitter) {
					for (int j = 0; j < m_params.emitters[i].initialNumberOfParticles; j++) {
						unsigned int currIndex = m_numInitializedParticles + j;
						m_pParticlesPosition->at(currIndex) = cubeSampling(i);
						m_pParticlesVelocity[currIndex] = Vector3(0, 0, 0);
						m_pParticlesLifeTime[currIndex] = m_params.emitters[i].particlesLife + m_params.emitters[i].particlesLifeVariance*(rand() / (float)RAND_MAX);
						m_pParticlesVorticity[currIndex] = 0;
					}

				}
				else  if (m_params.emitters[i].emitterType == sphereRandomEmitter) {
					for (int j = 0; j < m_params.emitters[i].initialNumberOfParticles; j++) {
						unsigned int currIndex = m_numInitializedParticles + j;
						m_pParticlesPosition->at(currIndex) = sphereSampling(i);
						m_pParticlesVelocity[currIndex] = Vector3(0, 0, 0);
						m_pParticlesLifeTime[currIndex] = m_params.emitters[i].particlesLife + m_params.emitters[i].particlesLifeVariance*(rand() / (float)RAND_MAX);
						m_pParticlesVorticity[currIndex] = 0;
					}
				}
				m_numInitializedParticles += m_params.emitters[i].initialNumberOfParticles;
				m_params.emitters[i].totalSpawnedParticles = m_params.emitters[i].initialNumberOfParticles;
			}
		}

		Vector3 ParticleSystem3D::jetShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue) {
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
		void ParticleSystem3D::updateParticlesColors(Scalar minScalarValue, Scalar maxScalarValue) {
			dimensions_t gridDimensions = m_pGridData->getDimensions();
			/*if (m_renderingParams.pParticlesTags != NULL && !m_renderingParams.drawParticlePerSlice) {
				#pragma omp parallel for
				for (int i = 0; i < m_maxNumberOfParticles; i++) {
					if (m_renderingParams.pParticlesTags[i] == 0) {
						m_pParticlesColor[i * 4] = 1.0f;
						m_pParticlesColor[i * 4 + 1] = 0.0f;
						m_pParticlesColor[i * 4 + 2] = 0.0f;
					}
					else if (m_renderingParams.pParticlesTags[i] == 1) {
						m_pParticlesColor[i * 4] = 0.0f;
						m_pParticlesColor[i * 4 + 1] = 1.0f;
						m_pParticlesColor[i * 4 + 2] = 0.0f;
					}
					else if (m_renderingParams.pParticlesTags[i] == 2) {
						m_pParticlesColor[i * 4] = 0.0f;
						m_pParticlesColor[i * 4 + 1] = 0.0f;
						m_pParticlesColor[i * 4 + 2] = 1.0f;
					}
					else if (m_renderingParams.pParticlesTags[i] == 3) {
						m_pParticlesColor[i * 4] = 1.0f;
						m_pParticlesColor[i * 4 + 1] = 1.0f;
						m_pParticlesColor[i * 4 + 2] = 0.0f;
					}
					else {
						m_pParticlesColor[i * 4] = m_pParticlesColor[i * 4 + 1] = m_pParticlesColor[i * 4 + 2] = 0.0f;
					}
				}
			}
			else {*/
				if (m_renderingParams.colorScheme == jet && m_params.pParticlesData->hasScalarBasedAttribute("density")) {
					#pragma omp parallel for
					for (int i = 0; i < m_numInitializedParticles; i++) {
						Scalar scalarFieldValue = 0;
						Vector3 particleGridPosition = m_pParticlesPosition->at(i)/m_pGridData->getGridSpacing();
						Vector3 jetColor = jetShading(i, m_params.pParticlesData->getScalarBasedAttribute("density")[i], minScalarValue, maxScalarValue);
						m_pParticlesColor[i * 4] = jetColor.x;
						m_pParticlesColor[i * 4 + 1] = jetColor.y;
						m_pParticlesColor[i * 4 + 2] = jetColor.z;
						m_pParticlesColor[i * 4 + 3] = 1.0f;
					}

				}
				else if (m_renderingParams.colorScheme == randomColors) {
					//initializeRandomParticlesColors();
				}
			/*}*/
			unsigned int sizeParticlesColorVBO = m_maxNumberOfParticles * sizeof(float) * 4;
			glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, m_pParticlesColor, GL_DYNAMIC_DRAW);
		}

		void ParticleSystem3D::update(Scalar dt) {
			m_elapsedTime += dt;

			updateEmission(dt);

			Scalar dx = m_pGridData->getScaleFactor(0, 0, 0).x;
			dimensions_t gridDimensions = m_pGridData->getDimensions();

			if (m_params.emitters.size() != 0) {
				#pragma omp parallel for
				for (int j = 0; j < m_numInitializedParticles; j++) {
					Vector3 interpVel = m_pVelocityInterpolant->interpolate(m_pParticlesPosition->at(j));
					Vector3 oldPosition = m_pParticlesPosition->at(j);

					m_pParticlesPosition->at(j) += (interpVel)*dt*0.5;
					interpVel = m_pVelocityInterpolant->interpolate(m_pParticlesPosition->at(j));
					
					m_pParticlesVelocity[j] = interpVel;
					m_pParticlesLifeTime[j] -= dt;

					m_pParticlesPosition->at(j) = oldPosition + interpVel*dt;

					/*if (m_pParticlesPosition->at(j).x > m_params.particlesMaxBounds.x ||
						m_pParticlesPosition->at(j).x < m_params.particlesMinBounds.x ||
						m_pParticlesPosition->at(j).y > m_params.particlesMaxBounds.y ||
						m_pParticlesPosition->at(j).y < m_params.particlesMinBounds.y ||
						m_pParticlesPosition->at(j).z > m_params.particlesMaxBounds.z ||
						m_pParticlesPosition->at(j).z < m_params.particlesMinBounds.z ||
						m_pParticlesLifeTime[j] < 0) {
						m_params.pResampledParticles[j] = true;
					}
					else {
						m_params.pResampledParticles[j] = false;
					}*/
				}
			}

			if (m_params.pParticlesData && m_params.pParticlesData->hasScalarBasedAttribute("density")) {
				m_renderingParams.minScalarfieldValue = FLT_MAX;
				m_renderingParams.maxScalarfieldValue = -FLT_MAX;
				const vector<Scalar> &particleDensities = m_params.pParticlesData->getScalarBasedAttribute("density");
				for (int i = 0; i < m_numInitializedParticles; i++) {
					if (particleDensities[i] < m_renderingParams.minScalarfieldValue)
						m_renderingParams.minScalarfieldValue = particleDensities[i];

					if (particleDensities[i] > m_renderingParams.maxScalarfieldValue)
						m_renderingParams.maxScalarfieldValue = particleDensities[i];
				}
			}
			

			updateParticlesColors(m_renderingParams.minScalarfieldValue, m_renderingParams.maxScalarfieldValue);
			updateTrails();
			updateVBOs();
		}

		void ParticleSystem3D::draw() {
			Scalar dx = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
			glPointSize(m_renderingParams.particleSize);

			//updateParticlesColorsGridSlice();
			if (m_renderingParams.m_draw) {
				//glDepthMask(GL_FALSE);
				if (m_renderingParams.drawSelectedVoxelParticles) {

					for (int i = 0; i < m_numInitializedParticles; i++) {
						dimensions_t currParticleDim(floor(m_pParticlesPosition->at(i).x/dx), floor(m_pParticlesPosition->at(i).y/dx), floor(m_pParticlesPosition->at(i).z/dx));
						if (currParticleDim == m_renderingParams.selectedVoxelDimension) {
							if (m_renderingParams.pParticlesTags != NULL) {
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
							}
							glBegin(GL_POINTS);
							glVertex3f(m_pParticlesPosition->at(i).x, m_pParticlesPosition->at(i).y, m_pParticlesPosition->at(i).z);
							glEnd();

							/*if (m_renderingParams.drawVelocities) {
								RenderingUtils::getInstance()->drawVector(m_pParticlesPosition->at(i), m_pParticlesPosition->at(i) + m_particlesVelocitiesVis[i] * 0.01);
							}
							if (m_renderingParams.drawNormals) {
								RenderingUtils::getInstance()->drawVector(m_pParticlesPosition->at(i), m_pParticlesPosition->at(i) + m_particlesNormalsVis[i] * 0.01);
							}*/

							if (m_renderingParams.drawVelocities) {
								if (m_params.pExternalVelocities) {
									Vector3 particleVelcoity = m_params.pExternalVelocities->at(i);
									RenderingUtils::getInstance()->drawVector(m_pParticlesPosition->at(i), m_pParticlesPosition->at(i)
										+ m_params.pExternalVelocities->at(i)*m_renderingParams.velocityScale, 0.01);
								}
							}
						}

					
					}

				}
				else {
					glEnableClientState(GL_VERTEX_ARRAY);
					if (m_renderingParams.colorScheme != singleColor || m_renderingParams.drawParticlePerSlice) {
						glEnableClientState(GL_COLOR_ARRAY);
						glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
						glColorPointer(4, GL_FLOAT, 0, 0);
					}
					else {
						glDisableClientState(GL_COLOR_ARRAY);
						glColor3f(m_renderingParams.particleColor[0] * 0.75, m_renderingParams.particleColor[1] * 0.75, m_renderingParams.particleColor[2] * 0.75); //Drawing particles in a darker tone
					}

					glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
					glVertexPointer(3, GL_FLOAT, 0, 0);

					glEnable(GL_POINT_SMOOTH);
					glDrawArrays(GL_POINTS, 0, m_maxNumberOfParticles);

					if (m_renderingParams.colorScheme != singleColor || m_renderingParams.drawParticlePerSlice) {
						glEnableClientState(GL_COLOR_ARRAY);
						glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
						glColorPointer(4, GL_FLOAT, 0, 0);
					}
					else if (m_renderingParams.colorScheme == singleColor) { //Drawing 
						glColor3f(m_renderingParams.particleColor[0], m_renderingParams.particleColor[1], m_renderingParams.particleColor[2]);
					}


					int numTrailsRendered = clamp(m_renderingParams.trailSize, 0, m_currentTrailSize);
					glBindBuffer(GL_ARRAY_BUFFER, *m_pParticleTrailsVBO);
					glVertexPointer(3, GL_FLOAT, 0, 0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pParticleTrailsIndexesVBO);
					glDrawElements(GL_LINES, m_maxNumberOfParticles*numTrailsRendered * 2, GL_UNSIGNED_INT, 0);
					//glDrawRangeElements(GL_LINES, 0, 5, m_maxNumberOfParticles*numTrailsRendered * 2, GL_UNSIGNED_INT, 0);
					//glDrawElements(GL_LINES, m_maxNumberOfParticles * 2, GL_UNSIGNED_INT, (void*)(l*m_maxNumberOfParticles * 2 * sizeof(int)));


					glBindBuffer(GL_ARRAY_BUFFER, 0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

					glDisableClientState(GL_VERTEX_ARRAY);
					glDisableClientState(GL_COLOR_ARRAY);
					glDisable(GL_POINT_SMOOTH);

					/*if (m_renderingParams.visualizeNormalsVelocities) {
						if (m_renderingParams.drawVelocities) {
							for (int i = 0; i < m_maxNumberOfParticles; i++) {
								RenderingUtils::getInstance()->drawVector(m_pParticlesPosition->at(i), m_pParticlesPosition->at(i) + m_particlesVelocitiesVis[i] * 0.05);
							}
						}
						if (m_renderingParams.drawNormals) {
							for (int i = 0; i < m_maxNumberOfParticles; i++) {
								RenderingUtils::getInstance()->drawVector(m_pParticlesPosition->at(i), m_pParticlesPosition->at(i) + m_particlesNormalsVis[i] * 0.05);
							}
						}

					}*/

					if (m_renderingParams.drawVelocities) {
						if (m_params.pExternalVelocities) {
							for (int i = 0; i < m_numInitializedParticles; i++) {
								RenderingUtils::getInstance()->drawVector(m_pParticlesPosition->at(i), m_pParticlesPosition->at(i)
									+ m_params.pExternalVelocities->at(i)*m_renderingParams.velocityScale, 0.01);
							}
						}
					}
				}
				glDepthMask(GL_TRUE);


				

				
			}
		}


		Vector3 ParticleSystem3D::cubeSampling(int ithEmitter) {
			Vector3 particlePos(m_params.emitters[ithEmitter].emitterSize.x*(rand() / (float)RAND_MAX),
				m_params.emitters[ithEmitter].emitterSize.y*(rand() / (float)RAND_MAX),
				m_params.emitters[ithEmitter].emitterSize.z*(rand() / (float)RAND_MAX));
			particlePos += m_params.emitters[ithEmitter].position;
			return particlePos;
		}

		Vector3 ParticleSystem3D::sphereSampling(int ithEmitter) {
			Vector3 particlePos;
			particlePos.x = 2 * (rand() / (float)RAND_MAX) - 1;
			particlePos.y = 2 * (rand() / (float)RAND_MAX) - 1;
			particlePos.z = 2 * (rand() / (float)RAND_MAX) - 1;
			DoubleScalar randown = (rand() / (float)RAND_MAX);
			randown = pow(randown, 1 / 3.0f);
			particlePos = m_params.emitters[ithEmitter].position + particlePos.normalized()*randown*m_params.emitters[ithEmitter].emitterSize.x;
			return particlePos;
		}
	}
}