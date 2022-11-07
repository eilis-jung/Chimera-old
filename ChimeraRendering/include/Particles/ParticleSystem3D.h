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

#ifndef _CHIMERA_3D_PARTICLE_SYSTEM_3D_
#define _CHIMERA_3D_PARTICLE_SYSTEM_3D_

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraResources.h"
#include "ChimeraCutCells.h"
#include "ChimeraInterpolation.h"
#include "ChimeraParticles.h"

#include "Visualization/ScalarFieldRenderer.h"
#include "RenderingUtils.h"

namespace Chimera {
	using namespace CutCells;
	using namespace Resources;
	using namespace Interpolation;
	using namespace Particles;

	namespace Rendering {
		class ParticleSystem3D {

		public:
			/************************************************************************/
			/* Class definitions                                                    */
			/************************************************************************/
			static const int s_maxTrailSize;

			/************************************************************************/
			/* Internal structs                                                     */
			/************************************************************************/
			/** Emitter type. Sets the way the particle system works. If no emitter is selected, the particles have an
			** external update function source (positions); */
			typedef enum emitterType3D_t {
				cubeUniformEmitter,
				cubeRandomEmitter,
				sphereRandomEmitter,
				noEmitter
			} emitterType3D_t;

			/** Emitter configuration */
			typedef struct emitter_t {
				unsigned int totalSpawnedParticles;
				unsigned int initialNumberOfParticles;
				unsigned int maxNumberOfParticles;
				emitterType3D_t emitterType;
				Scalar spawnRatio;
				Vector3 position, emitterSize;

				Scalar particlesLife;
				Scalar particlesLifeVariance;
				Scalar naturalDamping;

				emitter_t() {
					maxNumberOfParticles = 1 << 16;
					emitterType = noEmitter;
					spawnRatio = 1 << 14;
					particlesLife = FLT_MAX;
					particlesLifeVariance = 0.1f;
					particlesLife = particlesLifeVariance = naturalDamping = 0;
					naturalDamping = 1.0f;
					particlesLife = FLT_MAX;
				}
			};

			typedef struct configParams_t {
				Vector3 particlesMinBounds, particlesMaxBounds;

				/** External positions source */
				vector<Vector3> *pExternalParticlePositions;
				vector<Vector3> *pExternalVelocities;
				vector<bool> *pResampledParticles;

				vector<emitter_t> emitters;

				//LinearInterpolant3D<Vector3> *pLinearInterpolant;
				Interpolant<Vector3, Array3D, Vector3> *pLinearInterpolant;

				collisionDetectionMethod_t collisionDetectionMethod;
				//vector<PolygonSurface *> collisionSurfaces;

				ParticlesData<Vector3> *pParticlesData;

				configParams_t() {
					particlesMinBounds = Vector3(-0.5, -0.5, -0.5);
					particlesMaxBounds = Vector3(1.5, 1.5, 1.5);
					pExternalParticlePositions = NULL;
					pResampledParticles = NULL;
					pLinearInterpolant = NULL;
					collisionDetectionMethod = noCollisionDetection;
					pParticlesData = nullptr;
					pExternalVelocities = nullptr;
				}

				void setExternalParticles(vector<Vector3> *pExtParticles, int numExtParticles, vector<Vector3> *pExtVelocities = NULL, vector<bool> *pExtResampledParticles = NULL) {
					pExternalParticlePositions = pExtParticles;
					pExternalVelocities = pExtVelocities;
					pResampledParticles = pExtResampledParticles;
				}

				void setExternalParticles(ParticlesData<Vector3> *g_pParticlesData) {
					pParticlesData = g_pParticlesData;
				}
			} configParams_t;

			struct renderingParams_t {
				bool m_draw;
				int trailSize;
				Scalar particleSize;
				scalarColorScheme_t colorScheme;
				Scalar particleColor[4];
				int * pParticlesTags;
				Scalar minScalarfieldValue, maxScalarfieldValue;
				bool drawSelectedVoxelParticles;
				dimensions_t selectedVoxelDimension;
				bool visualizeNormalsVelocities;
				bool drawVelocities;
				bool drawNormals;

				bool drawParticlePerSlice;
				int gridSliceDimension;

				Scalar velocityScale;
				renderingParams_t() {
					m_draw = true;
					trailSize = 1;
					particleSize = 2.0f;
					particleColor[0] = particleColor[1] = particleColor[2] = particleColor[3] = 0.0f;
					colorScheme = singleColor;
					pParticlesTags = NULL;
					minScalarfieldValue = 0;
					maxScalarfieldValue = 1.0f;
					drawSelectedVoxelParticles = false;
					visualizeNormalsVelocities = false;
					drawNormals = drawVelocities = false;
					drawParticlePerSlice = false;
					gridSliceDimension = 0;
					velocityScale = 0.01f;
				}
			} m_renderingParams;

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			ParticleSystem3D(configParams_t params, renderingParams_t renderingParams, HexaGrid *pGrid) {
				m_pHexaGrid = pGrid;
				m_pGridData = pGrid->getGridData3D();
				//m_pSpecialCells = NULL;
				m_pCutVoxels = nullptr;
				m_gridOriginPosition = Vector3(0, 0, 0);
				m_gridVelocity = Vector3(0, 0, 0);
				m_gridOrientation = 0;
				m_angularSpeed = m_angularAcceleration = 0;
				m_params = params;
				m_renderingParams = renderingParams;
				m_dx = m_pGridData->getScaleFactor(0, 0, 0).x;
				m_currentTrailSize = 0;
				m_pNodeVelocityField = NULL;
				m_updateParticleTags = true;
				m_elapsedTime = 0;
				m_maxNumberOfParticles = 0;
				if (m_params.emitters.size() == 0 && m_params.pExternalParticlePositions != NULL) {
					m_maxNumberOfParticles = m_params.pExternalParticlePositions->size();
				}
				else {
					for (int i = 0; i < m_params.emitters.size(); i++) {
						m_maxNumberOfParticles += m_params.emitters[i].maxNumberOfParticles;
					}
				}
				m_pParticlesTrails = new Array2D<Vector3>(dimensions_t(m_maxNumberOfParticles, s_maxTrailSize));
				initializeParticles();
				initializeVBOs();
			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			Vector3 jetShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue);
			void resetParticleSystem();
			void updateEmission(Scalar dt);
			void updateParticlesColors(Scalar minScalarValue, Scalar maxScalarValue);
			void update(Scalar dt);
			void draw();
			void updateParticlesTags();

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			FORCE_INLINE void setGridOrigin(const Vector3 &gridOrigin) {
				m_gridOriginPosition = gridOrigin;
			}

			FORCE_INLINE void setGridOrientation(Scalar gridOrientation) {
				m_gridOrientation = gridOrientation;
			}

			FORCE_INLINE void setGridVelocity(const Vector3 &gridVelocity) {
				m_gridVelocity = gridVelocity;
			}

			FORCE_INLINE void setRotationPoint(const Vector3 &rotationPoint) {
				m_rotationPoint = rotationPoint;
			}

			FORCE_INLINE void setAngularSpeed(Scalar angularSpeed) {
				m_angularSpeed = angularSpeed;
			}

			FORCE_INLINE void setAngularAcceleration(Scalar angularAcc) {
				m_angularAcceleration = angularAcc;
			}


			FORCE_INLINE const Vector3 & getGridOrigin() const {
				return m_gridOriginPosition;
			}

			FORCE_INLINE configParams_t & getConfigParams() {
				return m_params;
			}

			FORCE_INLINE renderingParams_t & getRenderingParams() {
				return m_renderingParams;
			}

			FORCE_INLINE void setCutVoxels(CutVoxels3D<Vector3> *pCutVoxels) {
				m_pCutVoxels = pCutVoxels;
			}

			/*FORCE_INLINE void setNodeVelocityField(nodeVelocityField3D_t *pNodeVelocityField) {
				m_pNodeVelocityField = pNodeVelocityField;
			}*/

			FORCE_INLINE void setTriangleMeshMap(Array3D<int> *pCellToTriangleMeshMap) {
				m_pCellToTriangleMeshMap = pCellToTriangleMeshMap;
			}

			vector<Vector3> * getParticlePositionsVectorPtr() {
				return m_pParticlesPosition;
			}
			const Vector3 * const getParticleVelocitiesPtr() const {
				return m_pParticlesVelocity;
			}
			int getNumberOfParticles() const {
				return m_maxNumberOfParticles;
			}

			int getRealNumberOfParticles() const {
				return m_numInitializedParticles;
			}

			void setParticleNormal(int ithParticle, const Vector3 &particleNormal) {
				if (m_renderingParams.visualizeNormalsVelocities) {
					m_particlesNormalsVis[ithParticle] = particleNormal;
				}
			}

			void setParticleVelocity(int ithParticle, const Vector3 &particleVelocity) {
				if (m_renderingParams.visualizeNormalsVelocities) {
					m_particlesVelocitiesVis[ithParticle] = particleVelocity;
				}
			}

			void setVelocityInterpolant(Interpolant<Vector3, Array3D, Vector3> *pVelocityInterpolant) {
				m_pVelocityInterpolant = pVelocityInterpolant;
			}

			Interpolant<Vector3, Array3D, Vector3> * getVelocityInterpolant() {
				return m_pVelocityInterpolant;
			}
		private:

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			/** Particle system config */
			configParams_t m_params;

			unsigned int m_maxNumberOfParticles;

			/** Number of initialized particles */
			unsigned int m_numInitializedParticles;


			/** Particles position vertex buffer object. */
			GLuint *m_pParticlesVBO;
			/** Color VBO*/
			GLuint *m_pParticlesColorVBO;
			/** Trails positions and indexes VBOs. */
			GLuint *m_pParticleTrailsVBO;
			GLuint *m_pParticleTrailsIndexesVBO;

			/** CPU space particles position */
			vector<Vector3> *m_pParticlesPosition;
			/** Particles normals & velocities */
			vector<Vector3> m_particlesNormalsVis;
			vector<Vector3> m_particlesVelocitiesVis;
			/** CPU space trails */
			Array2D<Vector3> *m_pParticlesTrails;
			int * m_pParticlesTrailsIndexes;
			/** CPU space particles velocity */
			Vector3 *m_pParticlesVelocity;
			Scalar *m_pParticlesVorticity;
			Scalar *m_pParticlesLifeTime;
			/** CPU space particles color */
			Scalar *m_pParticlesColor;

			int m_currentTrailSize;

			bool m_updateParticleTags;

			/** Grid data*/
			HexaGrid *m_pHexaGrid;
			GridData3D *m_pGridData;
			Scalar m_elapsedTime;
			Scalar m_dx;

			/** Grid origin's */
			Vector3 m_gridOriginPosition;
			Vector3 m_rotationPoint;

			/** Grid velocity */
			Vector3 m_gridVelocity;
			/** Grid orientation along k axis */
			Scalar m_gridOrientation;
			Scalar m_angularSpeed;
			Scalar m_angularAcceleration;

			/** Special cells data*/
			CutVoxels3D<Vector3> *m_pCutVoxels;
			Array2D<Vector3> *m_pNodeVelocityField;
			Interpolant<Vector3, Array3D, Vector3> *m_pVelocityInterpolant;
			Array3D<int> *m_pCellToTriangleMeshMap;

			/************************************************************************/
			/* Initialization                                                       */
			/************************************************************************/
			void initializeParticles();
			void initializeRandomParticlesColors();
			void initializeVBOs();

			/** Updating */
			void updateTrails();
			void updateVBOs();
			void updateParticlesColorsGridSlice();

			/*Emission*/
			Vector3 cubeSampling(int ithEmitter);
			Vector3 sphereSampling(int ithEmitter);

		};
	}

	
}

#endif