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

#ifndef _CHIMERA_APPLICATION_2D_H_
#define _CHIMERA_APPLICATION_2D_H_
#pragma  once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraResources.h"
#include "ChimeraRendering.h"
#include "ChimeraIO.h"
#include "Rendering/GLRenderer2D.h"

namespace Chimera {

	using namespace Core;
	/** Base class for different applications:
	 **		Realtime simulation;
	 **		Offline simulation;
	 **		Fetching of boundary conditions;
	 **		Precomputed animation rendering;
	 **		*/
	class Application2D {

	public:

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		Application2D(int argc, char** argv, TiXmlElement *pChimeraConfig) {
			m_pMainNode = pChimeraConfig;
			m_configFilename = pChimeraConfig->GetDocument()->Value();
			m_pRenderer = NULL;
			m_pPhysicsCore = NULL;
			m_pMainSimCfg = NULL;
			m_pFlowSolverParams = NULL;
			m_logTimeStep = -1;
			m_pPhysicsParams = new PhysicsCore<Vector2>::params_t();
			
			
			//m_pDataLogger = NULL;
		}

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		virtual void draw() = 0;
		virtual void update() = 0;


		/************************************************************************/
		/* Callbacks                                                            */
		/************************************************************************/
		virtual void mouseCallback(int button, int state, int x, int y) {
			m_pRenderer->mouseCallback(button, state, x, y);
		} 

		virtual void motionCallback(int x, int y) {
			m_pRenderer->motionCallback(x, y);
		}

		virtual void keyboardCallback(unsigned char key, int x, int y) {
			m_pRenderer->keyboardCallback(key, x, y);
		}

		virtual void keyboardUpCallback(unsigned char key, int x, int y) {

		}

		virtual void specialKeyboardCallback(int key, int x, int y) {
			m_pRenderer->keyboardCallback(key, x, y);
		}

		virtual void specialKeyboardUpCallback(int key, int x, int y) {
			m_pRenderer->keyboardCallback(key, x, y);
		}

		virtual void reshapeCallback(int width, int height) {
			m_pRenderer->reshapeCallback(width, height);
		}

		virtual void exitCallback() {

		}

		/************************************************************************/
		/* Access functions                                                     */
		/************************************************************************/
		GLRenderer2D * getRenderer() const {
			return m_pRenderer;
		}

		PhysicsCore<Vector2> * getPhysicsCore() const {
			return m_pPhysicsCore;
		}

	protected:

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void initializeGL(int argc, char **argv) {
			glutInit(&argc, argv);
			int screenWidth = glutGet(GLUT_SCREEN_WIDTH); 
			int screenHeight = glutGet(GLUT_SCREEN_HEIGHT); 
			glutInitWindowSize(screenWidth, screenHeight);
			glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
			glutCreateWindow("Fluids Renderer");
			GLenum err = glewInit();
			if (GLEW_OK != err) {
				Logger::get() << "GLEW initialization error! " << endl;
				exit(1);
			}

			/**GLUT and GLEW initialization */
			const char* GLVersion = (const char*)glGetString(GL_VERSION); 
			Logger::get() << "OpenGL version: " << GLVersion << endl; 
		}

		/************************************************************************/
		/* General loaders                                                      */
		/************************************************************************/
		plataform_t loadPlataform(TiXmlElement *pNode);
		void loadSimulationParams();
		void loadConvectionMethodParams(TiXmlElement *pConvectionNode);
		void loadParticleBasedParams(TiXmlElement *pParticleBasedNode);
		void loadPoissonSolverParams(TiXmlElement *pSolverParamsNode);
		void loadProjectionMethodParams(TiXmlElement *pProjectionNode);
		void loadFarFieldParams(TiXmlElement *pFarFieldNode);
		void loadIntegrationMethodParams(TiXmlElement *pFarFieldNode);		
		void loadSolidWallConditions(TiXmlElement *pSolidWallNode);
		
		void loadParticleSystem(TiXmlElement *pParticleSystemNode, QuadGrid *pQuadGrid = NULL);
		void loadDensityField(TiXmlElement *pDensityNode, QuadGrid *pGrid);
		vector<FlowSolver<Vector2, Array2D>::rotationalVelocity_t> loadRotationalVelocityField(TiXmlElement *pRotationalVelocityNode);

		/** Class based loaders */
		QuadGrid * loadGrid(TiXmlElement *pGridNode);

		template <class valueType>
		Interpolant<valueType, Array2D, Vector2> * loadInterpolant(TiXmlElement *pInterpolantType, const Array2D<valueType> &values, Scalar gridDx) {
			TiXmlElement *pTempNode = NULL;
			if (pTempNode = pInterpolantType->FirstChildElement("Type")) {
				string interpolantType = pTempNode->GetText();
				transform(interpolantType.begin(), interpolantType.end(), interpolantType.begin(), tolower);
				if (interpolantType == "bilinearstaggered") {
					return new BilinearStaggeredInterpolant2D<valueType>(values, gridDx);
				} else if (interpolantType == "cubicstreamfunction") {
					return new CubicStreamfunctionInterpolant2D<valueType>(values, gridDx);
				} else if (interpolantType == "subdivisionstreamfunction") {
					return new SubdivisionStreamfunctionInterpolant2D<valueType>(values, gridDx, gridDx/4);
				} else if (interpolantType == "bilinearstreamfunction") {
					return new BilinearStreamfunctionInterpolant2D<valueType>(values, gridDx);
				}
			}
			return nullptr;
		}

		/************************************************************************/
		/* Specific loaders                                                     */
		/************************************************************************/
		void loadMultigridParams(TiXmlElement *pMultigridNode);
		void loadLoggingParams(TiXmlElement *pLoggingNode);

		/************************************************************************/
		/* Class members                                                        */
		/************************************************************************/
		TiXmlElement *m_pMainNode;
		PhysicsCore<Vector2>::params_t *m_pPhysicsParams;

		//Main simulation configuration: must be initilized by the application
		SimulationConfig<Vector2, Array2D> * m_pMainSimCfg;

		/** Facilitators */
		PhysicsCore<Vector2> *m_pPhysicsCore;
		GLRenderer2D *m_pRenderer;
		string m_configFilename;
		FlowSolverParameters *m_pFlowSolverParams;
		int m_logTimeStep;
		DataExporter<Vector2, Array2D>::configParams_t m_dataExporterParams;
		ParticleSystem2D *m_pParticleSystem;
	};


}

#endif