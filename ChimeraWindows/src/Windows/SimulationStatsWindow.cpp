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

#include "Windows/SimulationStatsWindow.h"
#include "Physics/PhysicsCore.h"

namespace Chimera {
	namespace Windows {
		template<>
		void Chimera::Windows::SimulationStatsWindow<Vector2, Array2D>::setFlowSolver(FlowSolver<Vector2, Array2D>* pFlowSolver)
		{
			m_pFlowSolver = pFlowSolver;

			TwAddVarRO(m_pBaseBar, "linearIterations", TW_TYPE_INT32, &m_pFlowSolver->m_linearSolverIterations, "label='Linear solver iterations' group='Solver'");

			TwAddVarRO(m_pBaseBar, "totalSimulationTime", TW_TYPE_FLOAT, &m_pFlowSolver->m_totalSimulationTime, "label='Total simulation time' group='Performance'");
			TwAddVarRO(m_pBaseBar, "advectionTime", TW_TYPE_FLOAT, &m_pFlowSolver->m_advectionTime, "label='Advection Time' group='Performance'");
			TwAddVarRO(m_pBaseBar, "projectionTime", TW_TYPE_FLOAT, &m_pFlowSolver->m_solvePressureTime, "label='Projection Time' group='Performance'");
			if (dynamic_cast<CutCellSolver2D*>(pFlowSolver)) {
				TwAddVarRO(m_pBaseBar, "cutCellGenerationTime", TW_TYPE_FLOAT, &m_pFlowSolver->m_cutCellGenerationTime, "label='Cut cell generation time' group='Performance'");
			}

			TwAddVarRO(m_pBaseBar, "totalDivergence", TW_TYPE_FLOAT, &m_pFlowSolver->m_totalDivergent, "label='Total Divergence' group='General'");
			TwAddVarRO(m_pBaseBar, "meanDivergence", TW_TYPE_FLOAT, &m_pFlowSolver->m_meanDivergent, "label='Mean Divergence' group='General'");

			
		}

		template<>
		void Chimera::Windows::SimulationStatsWindow<Vector3, Array3D>::setFlowSolver(FlowSolver<Vector3, Array3D>* pFlowSolver)
		{
			m_pFlowSolver = pFlowSolver;

			TwAddVarRO(m_pBaseBar, "linearIterations", TW_TYPE_INT32, &m_pFlowSolver->m_linearSolverIterations, "label='Linear solver iterations' group='Solver'");

			TwAddVarRO(m_pBaseBar, "totalSimulationTime", TW_TYPE_FLOAT, &m_pFlowSolver->m_totalSimulationTime, "label='Total simulation time' group='Performance'");
			TwAddVarRO(m_pBaseBar, "advectionTime", TW_TYPE_FLOAT, &m_pFlowSolver->m_advectionTime, "label='Advection Time' group='Performance'");
			TwAddVarRO(m_pBaseBar, "projectionTime", TW_TYPE_FLOAT, &m_pFlowSolver->m_solvePressureTime, "label='Projection Time' group='Performance'");
			//if (dynamic_cast<CtuCellSolver*>(pFlowSolver)) {

			//}
			//TwAddVarRO(m_pBaseBar, "cutCellGenerationTime", TW_TYPE_FLOAT, &m_pFlowSolver->m_cutCellGenerationTime, "label='Cut cell generation time' group='Performance'");
		}
	}
}