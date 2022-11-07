#ifndef _CHIMERA_PRECOMPUTED_RENDERING_3D_H_
#define _CHIMERA_PRECOMPUTED_RENDERING_3D_H_
#pragma  once

#include "Applications/Application3D.h"

#include "Physics/RegularGridSolver3D.h"
#include "Physics/CurvilinearGridSolver3D.h"

namespace Chimera {

	typedef struct scalarFieldAnimation_t {
		Scalar *pScalarFieldBuffer;

		Scalar timeElapsed;
		int totalFrames;
	};

	typedef struct velocityFieldAnimation_t {
		Vector3 *pVelocityFieldBuffer;

		Scalar timeElapsed;
		int totalFrames;
	};



	class PrecomputedRendering : public Application3D {

	public:
		typedef struct precomputedVariables_t {
			Scalar *precomputedPressures;
			Scalar *precomputedVorticities;
			Scalar *precomputedVelocities;
			
			int totalLoadedFrames;
			int currentFrame;
			Scalar animationTimeElapsed;
		} precomputedVariables_t;

		

		enum drawingPlane {
			drawVorticity,
			drawPressure,
			drawVelocityMagnitude,
			drawDensity,
			none
		} drawingPlane;

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		PrecomputedRendering(int argc, char** argv, TiXmlElement *pChimeraConfig);


		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		virtual void draw();
		virtual void update();

		void keyboardCallback(unsigned char key, int x, int y) {
			Application3D::keyboardCallback(key, x, y);
			switch(key) {
				case '+':
					m_ithPlane++;
				break;

				case '-':
					m_ithPlane--;
				break;

				case 't':
					m_drawPlane = m_drawPlane + 1;
					if(m_drawPlane > 2) //None
						m_drawPlane = 0;
				break;

				case 'p':
					m_playSimulation = true;
				break;

				case 'r':
					m_pPhysicsCore->resetTimers();
					m_pPrecomputedVariables->currentFrame = 0;
				break;
			}

			if(m_pHexaGrid != NULL && m_drawPlane != none) {
				m_ithPlane = roundClamp(m_ithPlane, 0, m_pHexaGrid->getDimensions().z);
				//m_pRenderer->renderIthPlane(m_pHexaGrid, m_slicingPlane, m_ithPlane);
			}
				
		}

	private:

		//Current drawing plane vars
		int m_drawPlane;
		int m_ithPlane;
		HexaGrid::gridPlanes_t m_slicingPlane;

		//Precomputed loaded variables
		precomputedVariables_t *m_pPrecomputedVariables;
		
		//Particle Emitter
		Vector3 g_EmitterInitialPos;

		//Car vars
		bool m_updateCar;
		Scalar m_carOrientation;

		//Grid
		HexaGrid *m_pHexaGrid;

		//Misc
		int m_simulationResetTimes;
		bool m_playSimulation;
		Timer m_systemElapsedTime;
		shared_ptr<PhongShader> m_pPhongShading;



		/************************************************************************/
		/* Private functionalities                                              */
		/************************************************************************/
		void setupLocalCoordinateSystem();
		
		/************************************************************************/
		/* Loading functions                                                    */
		/************************************************************************/
		void loadGrid();
		void loadPhysics();
		void loadGridPlanes();
		Scalar * loadPrecomputedScalarVariable(const string &precomputedVarFile);

	};

}

#endif
