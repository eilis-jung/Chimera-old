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

#ifndef _CHIMERA_REALTIME_SIMULATION_2D_APP_H_
#define _CHIMERA_REALTIME_SIMULATION_2D_APP_H_
#pragma  once

#include "ChimeraRendering.h"
#include "ChimeraSolvers.h"

#include "Applications/Application2D.h"


namespace Chimera {


	using namespace Rendering;
	class RealtimeSimulation2D : public Application2D {

	public:
		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		RealtimeSimulation2D(int argc, char** argv, TiXmlElement *pChimeraConfig);

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

		#pragma region ClassMembers
		//Boundary conditions vector
		vector<BoundaryCondition<Vector2> *> m_boundaryConditions;

		//Line Mesh params
		vector<LineMesh<Vector2>::params_t *> m_lineMeshParams;
		vector<RigidObject2D *> m_rigidObjects;

		//Liquid objects outer representation mesh
		vector<LineMesh<Vector2> *> m_liquidObjects;

		//Animated objects that are going to be updated by the physical simulation
		vector<PhysicalObject<Vector2> *> m_pPhysObjects;

		//Rotational and translational constants
		Scalar m_maxAngularSpeed;
		Scalar m_maxTranslationalSpeed;
		Scalar m_maxTranslationalAcceleration;
		bool m_circularTranslation;

		vector<FlowSolver<Vector2, Array2D>::forcingFunction_t> m_forcingFunctions;

		//Flow solver
		FlowSolver<Vector2, Array2D> *m_pFlowSolver;
		//Density and temperature markers;
		vector<FlowSolver<Vector2, Array2D>::scalarFieldMarker_t> m_densityMarkers;

		//Grid variables 
		QuadGrid *m_pQuadGrid;

		//DataExporter
		DataExporter<Vector2, Array2D> *m_pDataExporter;

		//Liquid Representation
		LiquidRepresentation2D *m_pLiquidRepresentation;
		
		#pragma endregion ClassMembers

		/************************************************************************/
		/* Loading functions                                                    */
		/************************************************************************/
		void loadObjects(TiXmlElement *pObjectsNode);
		void loadThinObjects();
		void loadRigidObjects();
		void loadLiquidObjects();
		LineMesh<Vector2>::params_t* loadLineMeshNode(TiXmlElement *pThinObjectNode);
		RigidObject2D * loadRigidObject(TiXmlElement *pRigidObjectNode, uint lineParamID);
		vector<LineMesh<Vector2>::params_t *> loadMultiLinesMeshNode(TiXmlElement *pLineNode);
		void loadGridFile();
		void loadSolver();
		void loadPhysics();
		void loadDensityField(TiXmlElement *pDensityFieldNode);
		void loadTemperatureField(TiXmlElement *pTemperatureFieldNode);
		void loadGridObject(TiXmlElement *pDensityFieldNode);
		void loadLine(TiXmlElement *pLineNode);
		void loadWindForce(TiXmlElement *pLineNode);
		void loadSolidCells(TiXmlElement *pObjectsNode);
		void setupCamera(TiXmlElement *pCameraNode);
		void loadVelocityImpulses(TiXmlElement *pVelocityImpulsesNode);
		
		
		/************************************************************************/
		/* Private functionalities                                              */
		/************************************************************************/
		void updateParticleSystem(Scalar dt);
		vector<Vector2> createGearGeometry(TiXmlElement *pGeometryElement, int numSubdivisions, const Vector2 initialPosition);
	};
}
#endif