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

#ifndef _CHIMERA_PRECOMPUTED_ANIMATION_3D_H_
#define _CHIMERA_PRECOMPUTED_ANIMATION_3D_H_
#pragma  once

#include "ChimeraRendering.h"
#include "ChimeraSolvers.h"
#include "ChimeraSolids.h"
#include "ChimeraIO.h"

#include "Applications/Application3D.h"

namespace Chimera {

	class PrecomputedAnimation3D : public Application3D {

	public:
		typedef struct scalarFieldAnimation_t {
			Scalar *pScalarFieldBuffer;

			Scalar timeElapsed;
			int totalFrames;
		};

		typedef struct velocityFieldAnimation_t {
			vector<Array3D<Vector3>> velocityBuffers;

			//vector<nodeVelocityField3D_t> nodeVelocityFields;

			Scalar timeElapsed;
			int totalFrames;
		};

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		PrecomputedAnimation3D(int argc, char** argv, TiXmlElement *pChimeraConfig);

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void draw();
		void update();
		void updateVorticity();


		/************************************************************************/
		/* Call-backs                                                           */
		/************************************************************************/
		void keyboardCallback(unsigned char key, int x, int y);
		void keyboardUpCallback(unsigned char key, int x, int y);
		void specialKeyboardCallback(int key, int x, int y);
		void specialKeyboardUpCallback(int key, int x, int y);

	private:

		//Grid variables 
		HexaGrid *m_pHexaGrid;

		//Particle System
		ParticleSystem3D *m_pParticleSystem;

		//Precomputed loadable fields
		scalarFieldAnimation_t * m_pDensityFieldAnimation; 
		scalarFieldAnimation_t * m_pPressureFieldAnimation;
		velocityFieldAnimation_t m_velocityFieldAnimation;

		collisionDetectionMethod_t m_collisionDetectionMethod;

		Interpolant<Vector3, Array3D, Vector3> *m_pVelocityInterpolant;
		NParticleExporter * m_pMayaCacheParticlesExporter;
		VelFieldExporter * m_pMayaCacheVelFieldExporter;
		AmiraExporter * m_pAmiraExporter;
		
		bool m_nodeBasedVelocities;
		bool m_useMayaCache;
		bool m_exportObjFiles;
		bool m_loadPerFrame;
		string m_baseFileName;
		string m_velocityFieldName;


		int m_currentTimeStep;

		//vector<PolygonSurface *> m_pPolySurfaces;

		Timer m_updateTimer;
		Scalar m_elapsedTime;
		/************************************************************************/
		/* Loading functions                                                    */
		/************************************************************************/
		void loadGridFile();
		void loadPhysics();
		void setupCamera(TiXmlElement *pCameraNode);
		void loadAnimations(TiXmlElement *pAnimationsNode);

		scalarFieldAnimation_t * loadScalarField(const string &scalarFieldFile);
		void loadVelocityField(const string &velocityFieldFile);
		void loadIthFrame(int ithFrame, const string &meshName);
		vector<vector<Vector3>> loadNodeVelocityField(const string &nodeVelocityField);
		vector<PolygonalMesh<Vector3>> *loadMesh(const string &polygonMesh);

		void syncNodeBasedVelocities();
		/************************************************************************/
		/* Private functionalities                                              */
		/************************************************************************/
		void updateParticleSystem(Scalar dt);
		void staggeredToNodeCentered();

		void initializeMayaCache();
		void logToMayaCache();
		void exportAnimationToObj();
		void exportAllMeshesToObj();
		

	};
}
#endif