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

#ifndef _CHIMERA_PRECOMPUTED_ANIMATION_2D_H_
#define _CHIMERA_PRECOMPUTED_ANIMATION_2D_H_
#pragma  once


#include "ChimeraRendering.h"
#include "ChimeraSolvers.h"
#include "ChimeraSolids.h"

#include "Applications/Application2D.h"

namespace Chimera {

	typedef struct scalarFieldAnimation_t {
		vector<Array2D<Scalar>> m_scalarFieldBuffer;

		Scalar timeElapsed;
		int totalFrames;
	};

	typedef struct velocityFieldAnimation_t {
		vector<Array2D<Vector2>>  m_velocityBuffer;

		Scalar timeElapsed;
		int totalFrames;
	};

	typedef struct positionAnimation_t {
		Vector2 *pPositionAnimationBuffer;

		Scalar timeElapsed;
		int totalFrames;
	};
	
	typedef struct thinObjectAnimation_t {
		vector<Vector2> *m_pThinObjectAnimationBuffer;
		
		int thinObjectPointsSize;
		Scalar timeElapsed;
		int totalFrames;
	};

	typedef struct doubleGyreConfig_t {
		DoubleScalar epsilon;
		DoubleScalar velocityMagnitude;
		DoubleScalar oscillationFrequency;
	} doubleGyreConfig_t;

	
	typedef enum scalarFieldType_t {
		densityField,
		pressureField,
		vorticityField,
		divergenceField
	} scalarFieldType_t;

	typedef struct fineGridConfig_t {
		int numSubdivisions;
		scalarFieldType_t scalarFieldType;
	};


	
	class PrecomputedAnimation2D : public Application2D {

	public:
		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		PrecomputedAnimation2D(int argc, char** argv, TiXmlElement *pChimeraConfig);


		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void draw();
		void update();


		/************************************************************************/
		/* Call-backs                                                           */
		/************************************************************************/
		void keyboardCallback(unsigned char key, int x, int y);
		void keyboardUpCallback(unsigned char key, int x, int y);
		void specialKeyboardCallback(int key, int x, int y);
		void specialKeyboardUpCallback(int key, int x, int y);

	private:

		//Grid variables 
		QuadGrid *m_pQuadGrid;

		//ThinObject 
		Line<Vector2> *m_pThinObjectLine;

		//Velocity Interpolant
		Interpolant<Vector2, Array2D, Vector2> *m_pVelocityInterpolant;

		//Precomputed loadable fields
		scalarFieldAnimation_t * m_pDensityFieldAnimation; 
		scalarFieldAnimation_t * m_pPressureFieldAnimation;
		velocityFieldAnimation_t *m_pVelocityFieldAnimation;
		positionAnimation_t * m_pPositionAnimation;
		thinObjectAnimation_t * m_pThinObjectAnimation;
		doubleGyreConfig_t *m_pDoubleGyreConfig;
		fineGridConfig_t *m_pFineGridConfig;

		vector<FlowSolver<Vector2, Array2D>::rotationalVelocity_t> m_rotationalVels;

		int m_currentTimeStep;
		
		//Configurable animation parameters
		int m_frameRate;
		int m_numFrames;
		Scalar m_timeStepSize;

		Timer m_updateTimer;
		Scalar m_elapsedTime;
		Scalar m_animationVelocity;
		/************************************************************************/
		/* Loading functions                                                    */
		/************************************************************************/
		void loadGridFile();
		void loadPhysics();
		void setupCamera(TiXmlElement *pCameraNode);
		void loadAnimations(TiXmlElement *pAnimationsNode);


		//Loads a collection of velocity fields up to frame numberFrames
		scalarFieldAnimation_t * loadScalarFieldCollection(const string &scalarFieldFile, scalarFieldType_t scalarFieldType, int numberFrames);
		//Loads a collection of velocity fields up to frame numberFrames
		velocityFieldAnimation_t * loadVelocityFieldCollection(const string &velocityFieldFile, int numberFrames);
		
		//Loads a single frame of type <VarType (Scalar, Vector2) into Array2D 
		template <class VarType>
		void loadFrame(const string &frameFile, Array2D<VarType> &values);
		
		positionAnimation_t * loadPositionAnimation(const string &positionAnimationFile);
		thinObjectAnimation_t * loadThinObjectAnimation(const string &thinObjectAnimationFile);

		void updateFineGridDivergence(); 
		void updateDivergence();

		/************************************************************************/
		/* Private functionalities                                              */
		/************************************************************************/

		doubleGyreConfig_t * loadDoubleGyreConfig(TiXmlElement *pDoubleGyreNode);

		fineGridConfig_t * loadFineGridScalar(TiXmlElement *pFineGridNode);

		void updateParticleSystem(Scalar dt);
		void updateRotationalVelocities();
		void updateDoubleGyreVelocities();
		DoubleScalar calculateGyreVelocity(const Vector2 &position, velocityComponent_t velocityComponent);

		FORCE_INLINE int getAnimationBufferIndex(int i, int j, int timeFrame, const dimensions_t &gridDimensions) {
			return timeFrame*+gridDimensions.x*+gridDimensions.y		+
					j*+gridDimensions.x								+
					i;
		}

		FORCE_INLINE int getAnimationBufferIndex(int i, int t, int bufferSize) {
			return t*bufferSize + i;
		}


		
	};
}
#endif