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

#ifndef __CHIMERA_PHYSICS_CORE__
#define __CHIMERA_PHYSICS_CORE__

#pragma once

#include "Math/Scalar.h"
#include "Data/ChimeraStructures.h"
#include "Utils/Singleton.h"
#include "Physics/PhysicalObject.h"

using namespace std;
namespace Chimera {

	namespace Core {

		template <class VectorT>
		class PhysicsCore : public Singleton<PhysicsCore<VectorT>> {

		public:
			#pragma region External Structures
			typedef struct params_t {

				/** Total simulation time - if the simulation is unconstrained by time, this is set to -1*/
				Scalar timestep;
				Scalar totalSimulationTime;

				params_t() {
					timestep = 0.01;
					totalSimulationTime = -1;
				}
			} params_t;
			#pragma endregion
			
			#pragma region Constructors
			PhysicsCore();

			void initialize(const params_t &physicsParams) {
				m_params = physicsParams;
			}
			#pragma endregion

			#pragma region AntTweakBarVariables
			params_t m_params;
			Scalar m_timeElapsed;
			Scalar m_CFLNumber;
			int m_totalSteps;
			#pragma endregion

			#pragma region AccessFunctions
			bool isRunningSimulation() const {
				return m_runSimulation;
			}

			bool isSteppingSimulation() const {
				return m_stepSimulation;
			}
			
			inline void addObject(PhysicalObject<VectorT> *pObject) {
				m_pObjects.push_back(pObject);
			}

			params_t * getParams() const {
				return ((params_t*) &m_params);
			}

			Scalar getElapsedTime() const {
				return m_timeElapsed;
			}
			#pragma endregion

			#pragma region Functionalities
			void update();

			void runSimulation(bool run) {
				m_runSimulation = run;
			}

			void stepSimulation(bool step = true) {
				m_stepSimulation = step;
			}


			void resetTimers() {
				m_timeElapsed = 0;
			}
			#pragma endregion

		private:
			
			#pragma region PrivateFunctionalities
			inline bool updateSimulationPercent() {
				bool updatePercent = 100 * m_timeElapsed / m_params.totalSimulationTime >= m_simulationPercent;
				if (updatePercent) {
					m_simulationPercent = 100 * m_timeElapsed / m_params.totalSimulationTime;
					m_simulationPercent = (m_simulationPercent / 10) * 10;
					m_simulationPercent += 10;
				}
				return updatePercent;
			}

			inline  bool updateSimulationPercent(int currIndex, int totalIndexes) {
				bool updatePercent = 100 * (currIndex / (Scalar)totalIndexes) >= m_simulationPercent;
				if (updatePercent) {
					m_simulationPercent = 100 * (currIndex / (Scalar)totalIndexes);
					m_simulationPercent = (m_simulationPercent / 10) * 10;
					m_simulationPercent += 10;
				}
				return updatePercent;
			}
			#pragma endregion

			#pragma region ClassMembers
			/** Objects to be updated */
			vector<PhysicalObject<VectorT> *> m_pObjects;

			/** Simulation control*/
			bool m_runSimulation;
			bool m_stepSimulation;

			/** Misc */
			int m_simulationPercent;
			#pragma endregion
		};
	}
}

#endif