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

#ifndef _RENDERING_SIMULATION_CONTROL_WINDOW_H
#define _RENDERING_SIMULATION_CONTROL_WINDOW_H

#pragma  once

#include "Windows/BaseWindow.h"
#include "ChimeraSolvers.h"

namespace Chimera {
	using namespace Solvers;
	namespace Windows {

		template <class VectorT>
		class SimulationControlWindow : public BaseWindow {
			static void TW_CALL playSimulation(void *clientData) {
				PhysicsCore<VectorT>::getInstance()->runSimulation(!PhysicsCore<VectorT>::getInstance()->isRunningSimulation());
			}
			
			static void TW_CALL stepSimulation(void *clientData) {
				PhysicsCore<VectorT>::getInstance()->stepSimulation();
			}

		public:
			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			template <template <class> class ArrayType>
			SimulationControlWindow(FlowSolver<VectorT, ArrayType> *pFlowSolver) : BaseWindow(Vector2(16, 376), Vector2(300, 260), "Simulation Control") {
				/** Simulation Control*/
				TwAddButton(m_pBaseBar, "playButton", &SimulationControlWindow<VectorT>::playSimulation, NULL, "label='Play/Stop Simulation' group='Simulation Control'" );
				TwAddButton(m_pBaseBar, "stepButton", &SimulationControlWindow<VectorT>::stepSimulation, NULL, "label='Step Simulation' group='Simulation Control'" );
				TwAddVarRW(m_pBaseBar, "Timestep", TW_TYPE_FLOAT, &PhysicsCore<VectorT>::getInstance()->m_params.timestep, "label='Timestep' group='Simulation Control' step=0.01");
				TwAddVarRW(m_pBaseBar, "CFL", TW_TYPE_FLOAT, &PhysicsCore<VectorT>::getInstance()->m_CFLNumber, "label='CFL' group='Simulation Control'");

				/** Simulation Parameters*/
				TwEnumVal advectionCategoryEV[] = { LagrangianAdvection,  {"Grid-Based Advection"},
													EulerianAdvection,  {"Particle-Based Advection"} };
				TwType advectionCategory = TwDefineEnum("advectionCategoryEV", advectionCategoryEV, 2);
				TwAddVarRO(m_pBaseBar, "advectionCategory", advectionCategory, voidCast<advectionCategory_t>(pFlowSolver->getParams().pAdvectionParams->advectionCategory), "label='Advection Category' group='Simulation Parameters'");
				if (pFlowSolver->getParams().pAdvectionParams->advectionCategory == EulerianAdvection) {
					TwEnumVal advectionEV[] = {		{ SemiLagrangian, "Semi Lagrangian" },
													{ MacCormack, "Modified Mac-Cormack" },
													{ USCIP, "USCIP" } };
					TwType advectionType = TwDefineEnum("AdvectionType", advectionEV, 3);
					TwAddVarRO(m_pBaseBar, "advectionType", advectionType, voidCast<gridBasedAdvectionMethod_t>(pFlowSolver->getParams().pAdvectionParams->gridBasedAdvectionMethod), "label='Advection Method' group='Simulation Parameters'");
				}
				else if(pFlowSolver->getParams().pAdvectionParams->advectionCategory == LagrangianAdvection) {
					TwEnumVal advectionEV[] = { { FLIP, "FLIP" },
												{ APIC, "APIC" },
												{ RPIC, "RPIC" } };
					TwType advectionType = TwDefineEnum("AdvectionType", advectionEV, 3);
					TwAddVarRO(m_pBaseBar, "advectionType", advectionType, voidCast<particleBasedAdvectionMethod_t>(pFlowSolver->getParams().pAdvectionParams->particleBasedAdvectionMethod), "label='Advection Method' group='Simulation Parameters'");
				}
				
				/** Solver specific parameters */				
				TwEnumVal pressureEV[] = { {MultigridMethod, "Multigrid"}, {GPU_CG, "Conjugate Gradient (GPU)"}, {CPU_CG, "Conjugate Gradient (CPU)"}, {GaussSeidelMethod, "Gauss Seidel"} };
				TwType pressureType = TwDefineEnum("PressureType", pressureEV, 3);
				TwAddVarRO(m_pBaseBar, "pressureSolverMethod", pressureType,  voidCast<pressureMethod_t>(pFlowSolver->getParams().pPoissonSolverParams->solverMethod),
					"label='Pressure Solver Method' group='Simulation Parameters'");

				string solverName;
				if(pFlowSolver->getParams().pPoissonSolverParams->solverMethod == GPU_CG || pFlowSolver->getParams().pPoissonSolverParams->solverMethod == CPU_CG) {
					/*solverName = "Conjugate Gradient (GPU)";
					TwEnumVal preconditionerEV[] =	{	{ConjugateGradient::Diagonal, "Diagonal"},
														{ConjugateGradient::AINV, "AINV"},
														{ConjugateGradient::SmoothedAggregation, "Smoothed aggregation"},
														{ConjugateGradient::NoPreconditioner, "No Preconditioner"}
													};
					TwType preconditionerType = TwDefineEnum("PreconditionerType", preconditionerEV, 4);
					ConjugateGradient::solverParams_t *pSolverParams = (ConjugateGradient::solverParams_t *) pFlowSolver->getParams().getPressureSolverParams().getSpecificSolverParams();

					string barCmd = "label='Preconditioner' group='" + solverName + "'";
					TwAddVarRO(m_pBaseBar, "preconditionerType", preconditionerType, voidCast<ConjugateGradient::LSPreconditioner>(pSolverParams->preconditioner),
											barCmd.c_str());
					barCmd = "label='Tolerance' group='" + solverName + "'";
					TwAddVarRO(m_pBaseBar, "tolerance", TW_TYPE_FLOAT, &pSolverParams->tolerance, barCmd.c_str());
					barCmd = "label='Max iterations' group='" + solverName + "'";
					TwAddVarRO(m_pBaseBar, "maxIterations", TW_TYPE_INT32, &pSolverParams->maxIterations, barCmd.c_str());

					string defName = "'" + m_windowName + "'/Grid group=Visualization";
					barCmd = "'" + m_windowName + "'" + "/'" + solverName + "' group='Simulation Parameters'";
					TwDefine(barCmd.c_str());*/
					

				} else if(pFlowSolver->getParams().pPoissonSolverParams->solverMethod == MultigridMethod) {
					solverName = "Multigrid (CPU)";
				} else if(pFlowSolver->getParams().pPoissonSolverParams->solverMethod == GaussSeidelMethod) {
					solverName = "Gauss-Seidel (CPU)";
					GaussSeidel::solverParams_t *pSolverParams = (GaussSeidel::solverParams_t *) pFlowSolver->getParams().pPoissonSolverParams;

					string barCmd = "label='Tolerance' group='" + solverName + "'";
					TwAddVarRO(m_pBaseBar, "tolerance", TW_TYPE_FLOAT, &pSolverParams->tolerance, barCmd.c_str());
					barCmd = "label='Max iterations' group='" + solverName + "'";
					TwAddVarRO(m_pBaseBar, "maxIterations", TW_TYPE_INT32, &pSolverParams->maxIterations, barCmd.c_str());

					string defName = "'" + m_windowName + "'/Grid group=Visualization";
					barCmd = "'" + m_windowName + "'" + "/'" + solverName + "' group='Simulation Parameters'";
					TwDefine(barCmd.c_str());
				}
				
				

				//TwDefine("Simulation/'Debug Options' opened=false "); // fold the group Debug Options

			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			void update() {

			}



		};
	}
}

#endif