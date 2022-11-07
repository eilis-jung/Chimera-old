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

#include "Physics/PhysicsCore.h"

namespace Chimera {

	namespace Core {

		#pragma region Constructors
		template <class VectorT>
		PhysicsCore<VectorT>::PhysicsCore() {
			m_runSimulation = false;
			m_stepSimulation = false;
			m_timeElapsed = 0;
			m_simulationPercent = 10;

			m_totalSteps = 0;
			m_CFLNumber = 0.0;
		}
		#pragma endregion
		
		#pragma region Functionalities
		template <class VectorT>
		void PhysicsCore<VectorT>::update() {
			if (m_runSimulation) {
				for (int i = 0; i < m_pObjects.size(); i++) {
					m_pObjects[i]->update(m_params.timestep);
				}
				m_timeElapsed += m_params.timestep;
				m_totalSteps++;
				if (m_params.totalSimulationTime != -1 && m_params.totalSimulationTime - m_timeElapsed <= 0) {
					m_runSimulation = false;
					cout << "Simulation ended!" << endl;
				}
			}
			else if (m_stepSimulation) {
				m_stepSimulation = false;
				for (int i = 0; i < m_pObjects.size(); i++) {
					m_pObjects[i]->update(m_params.timestep);
				}
				m_timeElapsed += m_params.timestep;
				m_totalSteps++;
			}
		}
		#pragma endregion

		template PhysicsCore<Vector2>;
		template PhysicsCore<Vector3>;
	}

}
