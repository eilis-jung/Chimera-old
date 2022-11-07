//#ifndef _CHIMERA_3D_PARTICLE_SYSTEM_
//#define _CHIMERA_3D_PARTICLE_SYSTEM_
//
///************************************************************************/
///* Core                                                                 */
///************************************************************************/
//#include "ChimeraCore.h"
//
///************************************************************************/
///* Math                                                                 */
///************************************************************************/
//#include "ChimeraMath.h"
//
///************************************************************************/
///* Data                                                                 */
///************************************************************************/
//#include "ChimeraData.h"
//
///************************************************************************/
///* Rendering                                                            */
///************************************************************************/
//#include "ChimeraRendering.h"
//
//namespace Chimera {
//
//	class ParticleSystem3D {
//
//	public:
//		/************************************************************************/
//		/* Internal structs                                                     */
//		/************************************************************************/
//		typedef enum EmitterType2D_t {
//			squareUniformEmitter,
//			squareRandomEmitter,
//			circleRandomEmitter
//		} EmitterType2D_t;
//
//		//Oscillation
//		typedef enum EmitterAnimation_t {
//			oscillateZ,
//			oscillateYZ,
//			none,
//		} EmitterAnimation_T;
//
//		typedef struct configParams_t {
//			int numParticles;
//			int initialNumParticles;
//
//			Scalar particlesLife;
//			Scalar particlesLifeVariance;
//			Scalar naturalDamping;
//
//			/** Emitter configuration */
//			EmitterType2D_t EmitterType;
//			EmitterAnimation_T EmitterAnimation;
//			Scalar oscillationSpeed; //Radians per sec
//			Scalar oscillationSize;
//			Scalar EmitterSpawnRatio;
//			Vector3 EmitterPosition, EmitterSize;
//			Vector3 particlesMinBounds, particlesMaxBounds;
//
//			configParams_t() {
//				EmitterSpawnRatio = 1 << 14; //Per second
//				numParticles = 1 << 16; //16K?
//				initialNumParticles = 1 << 14; 
//				particlesLife = 3.0f;
//				particlesLifeVariance = 2.0f;
//				EmitterType = squareRandomEmitter;
//				EmitterPosition = Vector3(-1.5, -1.5, -1.5);
//				EmitterSize = Vector3(4, 4, 4);
//				particlesMinBounds = Vector3(-1.5, -1.5, -1.5);
//				particlesMaxBounds = Vector3(2.5, 2.5, 2.5);
//				naturalDamping = 0.99;
//				EmitterAnimation = oscillateZ;
//				oscillationSpeed = 10;
//				oscillationSize = 0.1;
//			}
//		} configParams_t;
//
//		/************************************************************************/
//		/* ctors                                                                */
//		/************************************************************************/
//		ParticleSystem3D(configParams_t params, HexaGrid *pGrid) {
//			m_pHexaGrid = pGrid;
//			m_oscillationY = m_oscillationZ = 0;
//			GridData3D *pGridData = pGrid->getGridData3D();
//			m_velocityFieldDimensions = pGridData->getDimensions();
//			m_gridOriginPosition = pGrid->getGridOrigin();
//			m_gridVelocity = Vector3(0, 0, 0);
//			m_gridOrientation = 0;
//			m_angularSpeed = m_angularAcceleration = 0;
//			m_params = params;
//			m_dx = pGridData->getScaleFactor(0, 0, 0).x;
//			initializeParticles();
//			initializeVBOs();
//		}
//
//		/************************************************************************/
//		/* Functionalities                                                      */
//		/************************************************************************/
//		Vector3 jetShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue);
//		void resetParticleSystem();
//		void updateEmission(Scalar dt);
//		void updateParticlesColors(Scalar minScalarValue, Scalar maxScalarValue);
//		void update(Scalar dt);
//		void draw();
//
//		/************************************************************************/
//		/* Access functions                                                     */
//		/************************************************************************/
//		FORCE_INLINE void setGridOrigin(const Vector3 &gridOrigin) {
//			m_gridOriginPosition = gridOrigin;
//		}
//
//		FORCE_INLINE void setGridOrientation(Scalar gridOrientation) {
//			m_gridOrientation = gridOrientation;
//		}
//
//		FORCE_INLINE void setGridVelocity(const Vector3 &gridVelocity) {
//			m_gridVelocity = gridVelocity;
//		}
//
//		FORCE_INLINE void setRotationPoint(const Vector3 &rotationPoint) {
//			m_rotationPoint = rotationPoint;
//		}
//
//
//
//		FORCE_INLINE void setAngularSpeed(Scalar angularSpeed) {
//			m_angularSpeed = angularSpeed;
//		}
//
//		FORCE_INLINE void setAngularAcceleration(Scalar angularAcc) {
//			m_angularAcceleration = angularAcc;
//		}
//
//		FORCE_INLINE const Vector3 & getLocalAxisX() const {
//			return m_localAxisX;
//		}
//
//		FORCE_INLINE const Vector3 & getLocalAxisY() const {
//			return m_localAxisY;
//		}
//
//		FORCE_INLINE const Vector3 & getGridOrigin() const {
//			return m_gridOriginPosition;
//		}
//
//	private:
//		/************************************************************************/
//		/* Class members                                                        */
//		/************************************************************************/
//		/** Particle system config */
//		configParams_t m_params;
//
//		/** Number of initialized particles */
//		int numInitializedParticles;
//
//		/** Particles position vertex buffer object. */
//		GLuint *m_pParticlesVBO;
//		GLuint *m_pParticlesColorVBO;
//
//		/** CPU space particles position */
//		Vector3 *m_pParticlesPosition;
//		/** CPU space particles velocity */
//		Vector3 *m_pParticlesVelocity;
//		Scalar *m_pParticlesVorticity;
//		Scalar *m_pParticlesLifeTime;
//		/** CPU space particles color */
//		Vector3 *m_pParticlesColor;
//
//		Vector3 m_localAxisX;
//		Vector3 m_localAxisY;
//
//		HexaGrid *m_pHexaGrid;
//
//		/** Fluid velocity field external ptr*/
//		Vector3 *m_pVelocityField;
//		dimensions_t m_velocityFieldDimensions;
//		Scalar m_dx;
//
//		/** Grid origin's */
//		Vector3 m_gridOriginPosition;
//		Vector3 m_rotationPoint;
//
//		/** Grid velocity */
//		Vector3 m_gridVelocity;
//		/** Grid orientation along k axis */
//		Scalar m_gridOrientation;
//		Scalar m_angularSpeed;
//		Scalar m_angularAcceleration;
//
//		Vector3 m_EmitterInitialPosition;
//		Scalar m_oscillationY;
//		Scalar m_oscillationZ;
//
//		/************************************************************************/
//		/* Initialization                                                       */
//		/************************************************************************/
//		void initializeParticles();
//		void initializeVBOs();
//
//
//		/************************************************************************/
//		/* Private functionalities		                                        */
//		/************************************************************************/
//		void updateLocalAxis();
//		void drawLocalAxis();
//		void applyTurbulence(Vector3 &velocity, int i);
//
//
//	};
//}
//
//#endif