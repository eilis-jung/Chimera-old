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

#include "ApplicationBase.h"

namespace Chimera {

	namespace Applications {
	
		template <class VectorT, template <class> class ArrayType>
		ApplicationBase<VectorT, ArrayType>::ApplicationBase(int argc, char** argv, TiXmlElement *pChimeraConfig) {
			m_pMainNode = pChimeraConfig;
			m_configFilename = pChimeraConfig->GetDocument()->Value();
			m_pPhysicsCore = NULL;
			m_pFlowSolver = NULL;
			m_pFlowSolverParams = NULL;

			m_pDataLoggerParams = nullptr;
			m_pDataLogger = nullptr;

			m_pPhysicsCoreParams = nullptr;

			try {
				/** Load simulation config */
				TiXmlElement *pSimulationConfig = m_pMainNode->FirstChildElement("SimulationConfig");
				if (pSimulationConfig != nullptr) {
					m_pPhysicsCoreParams = FlowSolverLoader<VectorT, ArrayType>::getInstance()->loadPhysicsCoreParams(pSimulationConfig);

					/** Load logging parameters*/
					TiXmlElement *pLoggingNode = pSimulationConfig->FirstChildElement("Logging");
					if (pLoggingNode) {
						m_pDataLoggerParams = XMLParamsLoader::getInstance()->loadLoggingParams<VectorT, ArrayType>(pLoggingNode);
					}
				}
				else {
					throw(exception("SimulationConfig node not found!"));
				}

				/** Load solver params */
				TiXmlElement *pSolverNode = m_pMainNode->FirstChildElement("FlowSolverConfig");
				if (pSolverNode != nullptr) {
					m_pFlowSolverParams = FlowSolverLoader<VectorT, ArrayType>::getInstance()->loadSimulationParams(pSolverNode);
				}
				else {
					throw(exception("FlowSolverConfig node not found!"));
				}
			}
			catch (exception e) {
				exitProgram(e.what());
			}
		}

		template class ApplicationBase<Vector3, Array3D>;
		template class ApplicationBase<Vector2, Array2D>;
	}
}