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

#ifndef _RENDERING_SIMULATION_STATS_WINDOW_H
#define _RENDERING_SIMULATION_STATS_WINDOW_H

#pragma  once

#include "Windows/BaseWindow.h"
#include "ChimeraSolvers.h"

namespace Chimera {
	using namespace Solvers;
	namespace Windows {

		template <class VectorT, template <class> class ArrayType>
		class SimulationStatsWindow : public BaseWindow {

			FlowSolver<VectorT, ArrayType> *m_pFlowSolver;
			Scalar m_lastSimulationTime;
			Scalar m_cutCellsGenerationTime;
			Scalar m_totalAdvectionTime;
			Scalar m_averagedAdvectionTime;
			Scalar m_totalProjectionTime;
			Scalar m_averagedProjectionTime;
			Scalar m_totalStepTime;
			Scalar m_averageStepTime;
			Scalar m_flowSolverResidual;
			Scalar m_totalKineticEnergy;
			
		public:
			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			SimulationStatsWindow(FlowSolver<VectorT, ArrayType> *pFlowSolver) : BaseWindow(Vector2(16, 576), Vector2(300, 310), "Simulation Stats") {
				m_pFlowSolver = pFlowSolver;
				m_lastSimulationTime = 0;
				m_totalStepTime = m_totalAdvectionTime = m_totalProjectionTime = 0;
				m_averagedAdvectionTime = m_averagedProjectionTime = m_averageStepTime = 0;
				m_cutCellsGenerationTime = 0;
				m_totalKineticEnergy = 0;

				TwAddVarRO(m_pBaseBar, "timeElapsed", TW_TYPE_FLOAT, &PhysicsCore<VectorT>::getInstance()->m_timeElapsed, "label='Time elapsed' group='General'");
				TwDefine("'Simulation Stats' text='light' refresh=0.05");
				TwAddVarRO(m_pBaseBar, "linearResidual", TW_TYPE_FLOAT, &m_flowSolverResidual, "label='Linear solver residual' group='Pressure Solver'");
				
				TwAddVarRO(m_pBaseBar, "avgSimulationTimeTime", TW_TYPE_FLOAT, &m_averageStepTime, "label='Average Simulation Time' group='Performance'");
				TwAddVarRO(m_pBaseBar, "avgAdvectionTime", TW_TYPE_FLOAT, &m_averagedAdvectionTime, "label='Average Advection Time' group='Performance'");
				TwAddVarRO(m_pBaseBar, "avgProjectionTime", TW_TYPE_FLOAT, &m_averagedProjectionTime, "label='Average Projection Time' group='Performance'");

				TwAddVarRO(m_pBaseBar, "totalKineticEnergy", TW_TYPE_FLOAT, &m_totalKineticEnergy, "label='Total Kinetic Energy' group='Other'");
			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			void update() {
				if(m_lastSimulationTime != PhysicsCore<VectorT>::getInstance()->getElapsedTime()) {
					m_lastSimulationTime = PhysicsCore<VectorT>::getInstance()->getElapsedTime();
					m_totalAdvectionTime += m_pFlowSolver->m_advectionTime;
					m_averagedAdvectionTime = m_totalAdvectionTime/(m_pFlowSolver->getTotalIterations() + 1);

					m_totalProjectionTime += m_pFlowSolver->m_projectionTime;
					m_averagedProjectionTime = m_totalProjectionTime/(m_pFlowSolver->getTotalIterations() + 1);

					m_totalStepTime += m_pFlowSolver->m_totalSimulationTime;
					m_averageStepTime = m_totalStepTime/(m_pFlowSolver->getTotalIterations() + 1);
					//m_flowSolverResidual = m_pFlowSolver->m_maxResidual;

					m_totalKineticEnergy = m_pFlowSolver->getTotalKineticEnergy();
				}
			}

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			void setResidual(Scalar residual) {
				m_flowSolverResidual = residual;
			}

			Scalar getResidual() const {
				return m_flowSolverResidual;
			}

			
			void setFlowSolver(FlowSolver<VectorT, ArrayType> *pFlowSolver);

		};
	}
}

#endif