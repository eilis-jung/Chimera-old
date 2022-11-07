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

#ifndef __CHIMERA_XML_PARAMS_LOADER_H__
#define __CHIMERA_XML_PARAMS_LOADER_H__
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
#include "ChimeraIO.h"

namespace Chimera {

	using namespace Resources;
	using namespace Solvers;
	namespace Loaders {
		
		/** Main XML loader Singleton. */
		class XMLParamsLoader: public Singleton<XMLParamsLoader> {

		public:

			#pragma region Constructors
			/** Initializes internal params with the configuration node */
			XMLParamsLoader() { };
			#pragma endregion

			#pragma region LoadingFunctions
			/** Simple loading functions */
			bool loadTrueOrFalse(TiXmlElement *pNode);
			plataform_t loadPlatform(TiXmlElement *pPlatformNode);
			gridArrangement_t loadGridArrangement(TiXmlElement *pGridArrangementNode);
			kernelTypes_t loadKernel(TiXmlElement *pKernelNode);
			particlesSampling_t loadSampler(TiXmlElement *pSamplerNode);
			collisionDetectionMethod_t loadCollisionDetectionMethod(TiXmlElement *pCollisionDetectionMethod);
			interpolationMethod_t loadInterpolation(TiXmlElement *pInterpolationNode);
			integrationMethod_t loadIntegrationMethodParams(TiXmlElement *pIntegrationNode);
			solidBoundaryType_t loadSolidWallConditions(TiXmlElement *pSolidWallNode);

			/** Loads a vector from a node by querying the attributes "x", "y" and "z". As default throws an exception for 
				invalid configurations.*/
			template<class VectorType>
			FORCE_INLINE VectorType loadVectorFromNode(TiXmlElement *pNode, bool throwException = true) {
				VectorType velocity;
				if (pNode->QueryFloatAttribute("x", &velocity.x) == TIXML_NO_ATTRIBUTE && throwException) {
					throw exception("loadVectorFromNode: invalid vector configuration");
				}
				if (pNode->QueryFloatAttribute("y", &velocity.y) == TIXML_NO_ATTRIBUTE && throwException) {
					throw exception("loadVectorFromNode: invalid vector configuration");
				}
				//Trick to verify if it is a vector3, do not use often, bad practice :O
				if (!isVector2<VectorType>::value) {
					if (pNode->QueryFloatAttribute("z", &velocity[2]) == TIXML_NO_ATTRIBUTE && throwException) {
						throw exception("loadVectorFromNode: invalid vector configuration");
					}
				}
				return velocity;
			}

			FORCE_INLINE Scalar loadScalarFromNode(TiXmlElement *pNode) {
				return atof(pNode->GetText());
			}

			template<class VectorType, template <class> class ArrayType>
			typename DataExporter<VectorType, ArrayType>::configParams_t * loadLoggingParams(TiXmlElement *pLoggingNode);

			Multigrid::solverParams_t * loadMultigridParams(TiXmlElement *pMultigridNode);

			GLRenderer2D::params_t * loadRendererParams2D(TiXmlElement *pRenderingNode);
			GLRenderer3D::params_t * loadRendererParams3D(TiXmlElement *pRenderingNode);

			#pragma endregion
		};
	}
}


//void RealtimeSimulation3D::loadPhysics() {
//	/** Physics initialization - configure by XML*/
//	m_pPhysicsParams->timestep = 0.01;
//
//	//Unbounded simulation
//	m_pPhysicsParams->totalSimulationTime = -1;
//
//	loadSimulationConfig();
//	m_pDataLogger = new DataExporter<Vector3, Array3D>(m_dataExporterParams, m_pHexaGrid->getDimensions());
//
//
//
//	m_pPhysicsCore = PhysicsCore<Vector3>::getInstance();
//	m_pPhysicsCore->initialize(*m_pPhysicsParams);
//	m_pPhysicsParams = m_pPhysicsCore->getParams();
//}
#endif