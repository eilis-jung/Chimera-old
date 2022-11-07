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

#ifndef __CHIMERA_FORCING_FUNCTIONS_LOADER_H_
#define __CHIMERA_FORCING_FUNCTIONS_LOADER_H_
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

		
		class ForcingFunctionsLoader : public Singleton<ForcingFunctionsLoader> {
		public:

			#pragma region LoadingFunctions
			template <class VectorT, template <class> class ArrayType>
			typename FlowSolver<VectorT, ArrayType>::hotSmokeSource_t * loadHotSmokeSource(TiXmlElement *pHotSmokeSourceNode);

			template <class VectorT, template <class> class ArrayType>
			vector<typename FlowSolver<VectorT, ArrayType>::hotSmokeSource_t *> loadHotSmokeSources(TiXmlElement *pForcingFunctionsNode) {
				vector<typename FlowSolver<VectorT, ArrayType>::hotSmokeSource_t *> forcingFunctions;
				TiXmlElement *pHotSmokeNode = pForcingFunctionsNode->FirstChildElement("HotSmokeSource");
				while (pHotSmokeNode != NULL) {
					forcingFunctions.push_back(loadHotSmokeSource<VectorT, ArrayType>(pHotSmokeNode));
					pHotSmokeNode = pHotSmokeNode->NextSiblingElement("HotSmokeSource");
				}
				return forcingFunctions;
			}
			#pragma endregion
		};
	}
}

#endif
