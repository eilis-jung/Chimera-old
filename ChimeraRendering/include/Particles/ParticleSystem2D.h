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

#ifndef _CHIMERA_2D_PARTICLE_SYSTEM_
#define _CHIMERA_2D_PARTICLE_SYSTEM_

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
		class ParticleSystem2D {

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
			typedef enum emitterType2D_t {
				squareUniformEmitter,
				squareRandomEmitter,
				circleRandomEmitter,
				noEmitter
			} emitterType2D_t;

			typedef struct configParams_t {
				int numParticles;
				int initialNumParticles;

				Scalar particlesLife;
				Scalar particlesLifeVariance;
				Scalar naturalDamping;

				/** Emitter configuration */
				emitterType2D_t emitterType;
				Scalar emitterSpawnRatio;
				Vector2 emitterPosition, emitterSize;
				Vector2 particlesMinBounds, particlesMaxBounds;

				/** External positions source */
				vector<Vector2> *pExternalParticlePositions;
				vector<Vector2> *pExternalVelocities;
				vector<bool> *pResampledParticles;
				ParticlesData<Vector2> *pParticlesData;

				configParams_t() {
					emitterSpawnRatio = 1 << 14; //Per second
					numParticles = 1 << 14; //16K?
					initialNumParticles = 1 << 14;
					particlesLife = FLT_MAX;
					particlesLifeVariance = 0.1f;
					emitterType = squareRandomEmitter;
					emitterPosition = Vector2(-0.5, -0.5);
					emitterSize = Vector2(2, 2);
					particlesMinBounds = Vector2(-FLT_MAX, -FLT_MAX);
					particlesMaxBounds = Vector2(FLT_MAX, FLT_MAX);
					naturalDamping = 0.99;
					pExternalParticlePositions = NULL;
					pExternalVelocities = NULL;
					pResampledParticles = NULL;
					pParticlesData = nullptr;
				}

				void setExternalParticles(vector<Vector2> *pExtParticles, int numExtParticles, vector<Vector2> *pExtVelocities = NULL, vector<bool> *pExtResampledParticles = NULL) {
					emitterType = noEmitter;
					pExternalParticlePositions = pExtParticles;
					pExternalVelocities = pExtVelocities;
					numParticles = initialNumParticles = numExtParticles;
					pResampledParticles = pExtResampledParticles;
				}

				void setExternalParticles(ParticlesData<Vector2> *g_pParticlesData) {
					emitterType = noEmitter;
					pParticlesData = g_pParticlesData;
				}
			} configParams_t;

			struct renderingParams_t {
				bool m_draw;
				bool m_drawParticles;
				bool m_drawVelocities;
				bool m_clipParticlesTrails;
				int trailSize;
				Scalar particleSize;
				scalarColorScheme_t colorScheme;
				Scalar particleColor[3];
				vector<int> *pParticlesTags;
				Scalar minScalarfieldValue, maxScalarfieldValue;
				Scalar velocityScale;
				renderingParams_t() {
					m_draw = true;
					m_drawParticles = true;
					m_drawVelocities = false;
					m_clipParticlesTrails = false;
					trailSize = 1;
					particleSize = 1.0f;
					particleColor[0] = particleColor[1] = particleColor[2] = 0.0f;
					colorScheme = singleColor;
					pParticlesTags = NULL;
					minScalarfieldValue = 0;
					maxScalarfieldValue = 1.0f;
					velocityScale = 0.01;
				}
			} m_renderingParams;

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			ParticleSystem2D(configParams_t params, QuadGrid *pGrid) : m_particlesTrails(dimensions_t(params.numParticles, s_maxTrailSize)) {
				m_pQuadGrid = pGrid;
				m_pGridData = pGrid->getGridData2D();
				m_pCutCells2D = NULL;
				m_gridOriginPosition = Vector2(0, 0);
				m_gridVelocity = Vector2(0, 0);
				m_gridOrientation = 0;
				m_angularSpeed = m_angularAcceleration = 0;
				m_params = params;
				m_dx = m_pGridData->getScaleFactor(0, 0).x;
				m_currentTrailSize = 0;
				m_pVelocityInterpolant = nullptr;
				initializeParticles();
				initializeVBOs();
				initializeShaders();
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

			
			/** Tagging functions */
			void tagDisk(Vector2 diskPosition, Scalar diskSize);
			/** Tags particles with Zalezask disk  */
			void tagZalezaskDisk(Vector2 diskPosition, Scalar diskSize, Scalar dentSize);
			


			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			FORCE_INLINE void setGridOrigin(const Vector2 &gridOrigin) {
				m_gridOriginPosition = gridOrigin;
			}

			FORCE_INLINE void setGridOrientation(Scalar gridOrientation) {
				m_gridOrientation = gridOrientation;
			}

			FORCE_INLINE void setGridVelocity(const Vector2 &gridVelocity) {
				m_gridVelocity = gridVelocity;
			}

			FORCE_INLINE void setRotationPoint(const Vector2 &rotationPoint) {
				m_rotationPoint = rotationPoint;
			}

			FORCE_INLINE void setAngularSpeed(Scalar angularSpeed) {
				m_angularSpeed = angularSpeed;
			}

			FORCE_INLINE void setAngularAcceleration(Scalar angularAcc) {
				m_angularAcceleration = angularAcc;
			}

			FORCE_INLINE const Vector2 & getLocalAxisX() const {
				return m_localAxisX;
			}

			FORCE_INLINE const Vector2 & getLocalAxisY() const {
				return m_localAxisY;
			}

			FORCE_INLINE const Vector2 & getGridOrigin() const {
				return m_gridOriginPosition;
			}

			FORCE_INLINE configParams_t & getConfigParams() {
				return m_params;
			}

			FORCE_INLINE renderingParams_t & getRenderingParams() {
				return m_renderingParams;
			}

			FORCE_INLINE void setCutCells2D(CutCells2D<Vector2> *pCutCells) {
				m_pCutCells2D = pCutCells;
			}
			FORCE_INLINE void setNodeVelocityField(Array2D<Vector2> *pNodeVelocityField) {
				m_pNodeVelocityField = pNodeVelocityField;
			}

			FORCE_INLINE void setVelocityInterpolant(Interpolant<Vector2, Array2D, Vector2> *pVelocityInterpolant) {
				m_pVelocityInterpolant = pVelocityInterpolant;
			}
		private:

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			/** Particle system config */
			configParams_t m_params;

			/** Number of initialized particles */
			int numInitializedParticles;

			/** Particles position vertex buffer object. */
			GLuint *m_pParticlesVBO;
			/** Color VBO*/
			GLuint *m_pParticlesColorVBO;
			/** Trails positions and indexes VBOs. */
			GLuint *m_pParticleTrailsVBO;
			GLuint *m_pParticleTrailsIndexesVBO;

			/** CPU space particles position */
			Vector2 *m_pParticlesPosition;
			Vector2 *m_pParticlesVelocity;
			/** CPU space trails */
			Array2D<Vector2> m_particlesTrails;
			int * m_pParticlesTrailsIndexes;
			/** CPU space particles velocity */
			Scalar *m_pParticlesVorticity;
			Scalar *m_pParticlesLifeTime;
			/** CPU space particles color */
			Vector3 *m_pParticlesColor;

			shared_ptr<GLSLShader> m_pFiftyShadersOfGray;
			GLuint m_trailSizeLoc;
			GLuint m_maxNumberOfParticlesLoc;
			GLfloat m_modelViewMatrix[16];

			int m_currentTrailSize;

			Vector2 m_localAxisX;
			Vector2 m_localAxisY;

			/** Grid data*/
			QuadGrid *m_pQuadGrid;
			GridData2D *m_pGridData;
			Scalar m_dx;

			/** Grid origin's */
			Vector2 m_gridOriginPosition;
			Vector2 m_rotationPoint;

			/** Grid velocity */
			Vector2 m_gridVelocity;
			/** Grid orientation along k axis */
			Scalar m_gridOrientation;
			Scalar m_angularSpeed;
			Scalar m_angularAcceleration;

			/** Special cells data*/
			CutCells2D<Vector2> *m_pCutCells2D;
			Array2D<Vector2> *m_pNodeVelocityField;

			Interpolant<Vector2, Array2D, Vector2> *m_pVelocityInterpolant;

			/************************************************************************/
			/* Initialization                                                       */
			/************************************************************************/
			void initializeParticles();
			void initializeVBOs();
			void initializeShaders();


			/************************************************************************/
			/* Private functionalities		                                        */
			/************************************************************************/
			FORCE_INLINE Vector2 calculateRotationalVector(const Vector2 &particlePos, Scalar dt) {
				Vector2 centripetalVelocity = particlePos - Vector2(0.5, 0.5);
				Vector2 rotationalVelocity = centripetalVelocity.perpendicular();
				rotationalVelocity = rotationalVelocity*m_angularSpeed;
				return rotationalVelocity;
			}

			void updateTrails();
			void updateVBOs();
			void updateLocalAxis();
			void drawLocalAxis();


		};
	}
}

#endif