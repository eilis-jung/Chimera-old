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

#ifndef _CHIMERA_APPLICATION_BASE_H_
#define _CHIMERA_APPLICATION_BASE_H_
#pragma  once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraResources.h"
#include "ChimeraRendering.h"
#include "ChimeraSolvers.h"
#include "ChimeraIO.h"
#include "ChimeraLoaders.h"

namespace Chimera {

	using namespace Loaders;

	namespace Applications {

		/** Base class for different applications:
		**		Realtime simulation;
		**		Offline simulation;
		**		Fetching of boundary conditions;
		**		Precomputed animation rendering;
		**		*/

		template <class VectorT, template <class> class ArrayType>
		class ApplicationBase {

		public:

			#pragma region Constructors
			ApplicationBase(int argc, char** argv, TiXmlElement *pChimeraConfig);
			#pragma endregion

			#pragma region Functionalities
			virtual void draw() = 0;
			virtual void update() = 0;
			#pragma endregion


			#pragma region CallbackFunctions
			virtual void mouseCallback(int button, int state, int x, int y) = 0;

			virtual void motionCallback(int x, int y) = 0;

			virtual void keyboardCallback(unsigned char key, int x, int y) = 0;

			virtual void keyboardUpCallback(unsigned char key, int x, int y) = 0;

			virtual void specialKeyboardCallback(int key, int x, int y) = 0;

			virtual void specialKeyboardUpCallback(int key, int x, int y) = 0;

			virtual void reshapeCallback(int width, int height) = 0;

			virtual void exitCallback() = 0;
			#pragma endregion

			#pragma region AccessFunctions
			PhysicsCore<VectorT> * getPhysicsCore() const {
				return m_pPhysicsCore;
			}
			#pragma endregion

		protected:

			#pragma region ClassMembers
			TiXmlElement *m_pMainNode;
			
			//Flow solver params
			typename FlowSolver<VectorT, ArrayType>::params_t *m_pFlowSolverParams;
			//Flow solver
			FlowSolver<VectorT, ArrayType> *m_pFlowSolver;

			//Data Logger params
			typename DataExporter<VectorT, ArrayType>::configParams_t *m_pDataLoggerParams;
			/** Data Logger */
			DataExporter<VectorT, ArrayType> *m_pDataLogger;

			//Boundary conditions
			vector<BoundaryCondition<VectorT> *> m_boundaryConditions;

			/** Facilitators */
			typename PhysicsCore<VectorT>::params_t *m_pPhysicsCoreParams;
			PhysicsCore<VectorT> *m_pPhysicsCore;

			string m_configFilename;
			#pragma endregion
		};
	}
	
	


}

#endif