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

#ifndef __CHIMERA_FLOW_SOLVER_LOADER_H_
#define __CHIMERA_FLOW_SOLVER_LOADER_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraCutCells.h"
#include "ChimeraResources.h"
#include "ChimeraPoisson.h"
#include "ChimeraSolvers.h"
#include "ChimeraRendering.h"
#include "ChimeraAdvection.h"

namespace Chimera {
	
	namespace Loaders {

		using namespace Resources;
		using namespace Solvers;
		using namespace Advection;

		/** Main XML loader Singleton. */
		template <class VectorType, template <class> class ArrayType> 
		class FlowSolverLoader : public Singleton<FlowSolverLoader<VectorType, ArrayType>> {
		public:
			FlowSolverLoader() {
			}

			#pragma region LoadingFunctions
			typename FlowSolver<VectorType, ArrayType>::params_t * loadSimulationParams(TiXmlElement *pSimulationNode);

			typename PhysicsCore<VectorType>::params_t * loadPhysicsCoreParams(TiXmlElement *pSimulationNode);
			#pragma endregion

		protected:

			#pragma region InternalLoadingFunctions
			PoissonSolver::params_t * loadPoissonSolverParams(TiXmlElement *pSolverParamsNode);

			AdvectionBase::baseParams_t * loadAdvectionParams(TiXmlElement *pAdvectionNode);


			#pragma endregion
		};
	}
}

//
//void RealtimeSimulation3D::loadSimulationConfig() {
//	if (m_pMainNode->FirstChildElement("SimulationConfig")) {
//		TiXmlElement *pSimulationConfig = m_pMainNode->FirstChildElement("SimulationConfig");
//		if (pSimulationConfig->FirstChildElement("TotalTime")) {
//			pSimulationConfig->FirstChildElement("TotalTime")->QueryFloatAttribute("value", &m_pPhysicsParams->totalSimulationTime);
//		}
//		if (pSimulationConfig->FirstChildElement("TotalTime")) {
//			pSimulationConfig->FirstChildElement("TotalTime")->QueryFloatAttribute("value", &m_pPhysicsParams->totalSimulationTime);
//		}
//
//		/** Logging */
//		if (pSimulationConfig->FirstChildElement("Logging")) {
//			loadLoggingParams(pSimulationConfig->FirstChildElement("Logging"));
//		}
//
//		/** Impulses */
//		if (pSimulationConfig->FirstChildElement("VelocityImpulse")) {
//			loadVelocityImpulses(pSimulationConfig->FirstChildElement("VelocityImpulse"));
//		}
//
//		if (pSimulationConfig->FirstChildElement("TorusVelocityField")) {
//			vector<FlowSolver<Vector3, Array3D>::torusVelocity_t> torusVelocities = loadTorusVelocityField(pSimulationConfig->FirstChildElement("TorusVelocityField"));
//			for (int i = 0; i < torusVelocities.size(); i++) {
//				m_pMainSimCfg->getFlowSolver()->addTorusVelocity(torusVelocities[i]);
//			}
//		}
//		if (pSimulationConfig->FirstChildElement("InternalVelocityField")) {
//			loadInternalVelocityField(pSimulationConfig->FirstChildElement("InternalVelocityField"));
//		}
//		if (pSimulationConfig->FirstChildElement("CutFaceVelocity")) {
//			loadCutFaceVelocity(pSimulationConfig->FirstChildElement("CutFaceVelocity"));
//		}
//	}
//}
//
//void RealtimeSimulation3D::loadVelocityImpulses(TiXmlElement *pImpulsesNode) {
//	while (pImpulsesNode) {
//		FlowSolver<Vector3, Array3D>::velocityImpulse_t velocityImpulse;
//		if (pImpulsesNode->FirstChildElement("Position")) {
//			pImpulsesNode->FirstChildElement("Position")->QueryFloatAttribute("x", &velocityImpulse.position.x);
//			pImpulsesNode->FirstChildElement("Position")->QueryFloatAttribute("y", &velocityImpulse.position.y);
//			pImpulsesNode->FirstChildElement("Position")->QueryFloatAttribute("z", &velocityImpulse.position.z);
//		}
//		if (pImpulsesNode->FirstChildElement("Velocity")) {
//			pImpulsesNode->FirstChildElement("Velocity")->QueryFloatAttribute("x", &velocityImpulse.velocity.x);
//			pImpulsesNode->FirstChildElement("Velocity")->QueryFloatAttribute("y", &velocityImpulse.velocity.y);
//			pImpulsesNode->FirstChildElement("Velocity")->QueryFloatAttribute("z", &velocityImpulse.velocity.z);
//		}
//		m_pMainSimCfg->getFlowSolver()->addVelocityImpulse(velocityImpulse);
//
//		pImpulsesNode = pImpulsesNode->NextSiblingElement();
//	}
//}
//
//vector<FlowSolver<Vector3, Array3D>::torusVelocity_t> RealtimeSimulation3D::loadTorusVelocityField(TiXmlElement *pTorusNode) {
//	vector<FlowSolver<Vector3, Array3D>::torusVelocity_t> torusVelocities;
//	while (pTorusNode) {
//
//		FlowSolver<Vector3, Array3D>::torusVelocity_t torusVelocity;
//		if (pTorusNode->FirstChildElement("position")) {
//			pTorusNode->FirstChildElement("position")->QueryFloatAttribute("x", &torusVelocity.position.x);
//			pTorusNode->FirstChildElement("position")->QueryFloatAttribute("y", &torusVelocity.position.y);
//			pTorusNode->FirstChildElement("position")->QueryFloatAttribute("z", &torusVelocity.position.z);
//		}
//		if (pTorusNode->FirstChildElement("Radius")) {
//			torusVelocity.radius = atof(pTorusNode->FirstChildElement("Radius")->GetText());
//		}
//		if (pTorusNode->FirstChildElement("SectionRadius")) {
//			torusVelocity.sectionRadius = atof(pTorusNode->FirstChildElement("SectionRadius")->GetText());
//		}
//		if (pTorusNode->FirstChildElement("Orientation")) {
//			torusVelocity.orientation = atof(pTorusNode->FirstChildElement("Orientation")->GetText());
//		}
//		if (pTorusNode->FirstChildElement("Strength")) {
//			torusVelocity.strength = atof(pTorusNode->FirstChildElement("Strength")->GetText());
//		}
//
//		if (pTorusNode->FirstChildElement("UpDirection")) {
//			pTorusNode->FirstChildElement("UpDirection")->QueryFloatAttribute("x", &torusVelocity.upDirection.x);
//			pTorusNode->FirstChildElement("UpDirection")->QueryFloatAttribute("y", &torusVelocity.upDirection.y);
//			pTorusNode->FirstChildElement("UpDirection")->QueryFloatAttribute("z", &torusVelocity.upDirection.z);
//		}
//		else {
//			torusVelocity.upDirection = Vector3(0, 1, 0);
//		}
//
//		torusVelocities.push_back(torusVelocity);
//
//		//m_pMainSimCfg->getFlowSolver()->addTorusVelocity(torusVelocity);
//
//		pTorusNode = pTorusNode->NextSiblingElement("TorusVelocityField");
//	}
//	return torusVelocities;
//}
//
//void RealtimeSimulation3D::loadInternalVelocityField(TiXmlElement *pInternalVelocityNode) {
//	while (pInternalVelocityNode) {
//		FlowSolver<Vector3, Array3D>::internalVelocity_t internalVelocity;
//		if (pInternalVelocityNode->FirstChildElement("position")) {
//			pInternalVelocityNode->FirstChildElement("position")->QueryFloatAttribute("x", &internalVelocity.position.x);
//			pInternalVelocityNode->FirstChildElement("position")->QueryFloatAttribute("y", &internalVelocity.position.y);
//			pInternalVelocityNode->FirstChildElement("position")->QueryFloatAttribute("z", &internalVelocity.position.z);
//		}
//		else {
//			return;
//		}
//		if (pInternalVelocityNode->FirstChildElement("Radius")) {
//			internalVelocity.radius = atof(pInternalVelocityNode->FirstChildElement("Radius")->GetText());
//		}
//
//		if (pInternalVelocityNode->FirstChildElement("Velocity")) {
//			pInternalVelocityNode->FirstChildElement("Velocity")->QueryFloatAttribute("x", &internalVelocity.velocity.x);
//			pInternalVelocityNode->FirstChildElement("Velocity")->QueryFloatAttribute("y", &internalVelocity.velocity.y);
//			pInternalVelocityNode->FirstChildElement("Velocity")->QueryFloatAttribute("z", &internalVelocity.velocity.z);
//		}
//
//		m_pMainSimCfg->getFlowSolver()->addInternalVelocity(internalVelocity);
//
//		pInternalVelocityNode = pInternalVelocityNode->NextSiblingElement("InternalVelocityField");
//	}
//}
//
//void RealtimeSimulation3D::loadCutFaceVelocity(TiXmlElement *pCutFaceVelocityNode) {
//	while (pCutFaceVelocityNode) {
//		FlowSolver<Vector3, Array3D>::cutFaceVelocity_t cutFaceVelocity;
//		if (pCutFaceVelocityNode->FirstChildElement("Velocity")) {
//			pCutFaceVelocityNode->FirstChildElement("Velocity")->QueryFloatAttribute("x", &cutFaceVelocity.velocity.x);
//			pCutFaceVelocityNode->FirstChildElement("Velocity")->QueryFloatAttribute("y", &cutFaceVelocity.velocity.y);
//			pCutFaceVelocityNode->FirstChildElement("Velocity")->QueryFloatAttribute("z", &cutFaceVelocity.velocity.z);
//		}
//		else if (pCutFaceVelocityNode->FirstChildElement("TorusVelocityField")) {
//			cutFaceVelocity.torusVel = loadTorusVelocityField(pCutFaceVelocityNode->FirstChildElement("TorusVelocityField")).front();
//		}
//
//		if (pCutFaceVelocityNode->FirstChildElement("FaceLocation")) {
//			string faceLocationStr = pCutFaceVelocityNode->FirstChildElement("FaceLocation")->GetText();
//			transform(faceLocationStr.begin(), faceLocationStr.end(), faceLocationStr.begin(), ::tolower);
//			/*if (faceLocationStr == "bottomface") {
//			cutFaceVelocity.faceLocation = bottomFace;
//			}
//			else if (faceLocationStr == "leftface") {
//			cutFaceVelocity.faceLocation = leftFace;
//			}
//			else if (faceLocationStr == "backface") {
//			cutFaceVelocity.faceLocation = backFace;
//			}*/
//		}
//		if (pCutFaceVelocityNode->FirstChildElement("FaceID")) {
//			cutFaceVelocity.faceID = atoi(pCutFaceVelocityNode->FirstChildElement("FaceID")->GetText());
//		}
//		m_pMainSimCfg->getFlowSolver()->addCutFaceVelocity(cutFaceVelocity);
//
//		pCutFaceVelocityNode = pCutFaceVelocityNode->NextSiblingElement("CutFaceVelocity");
//	}
//}

#endif