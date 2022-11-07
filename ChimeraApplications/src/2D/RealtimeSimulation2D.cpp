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

#include "2D/RealtimeSimulation2D.h"

namespace Chimera {
	
	namespace Applications {
		
		#pragma region Constructors
		template <class VectorT>
		RealtimeSimulation2D<VectorT>::RealtimeSimulation2D(int argc, char** argv, TiXmlElement *pChimeraConfig) : ApplicationBase(argc, argv, pChimeraConfig) {
			m_pQuadGrid = nullptr;
			try {
				/** Load grid */
				TiXmlElement *pGridNode = m_pMainNode->FirstChildElement("Grid");
				if (pGridNode) {
					m_pQuadGrid = GridLoader::getInstance()->loadQuadGrid(pGridNode);
				}

				/** Load boundary conditions */
				TiXmlElement *pBoundaryConditionsNode = m_pMainNode->FirstChildElement("boundaryConditionsFile");
				if (pBoundaryConditionsNode && m_pQuadGrid) {
					string boundaryConditionsFile = pBoundaryConditionsNode->GetText();
					m_boundaryConditions = BoundaryConditionFactory::getInstance()->loadBCs<Vector2>(boundaryConditionsFile, m_pQuadGrid->getDimensions());
				}

				/** Load meshes */
				TiXmlElement *pObjectsNode = m_pMainNode->FirstChildElement("Objects");
				if (pObjectsNode && m_pQuadGrid) {
					m_rigidObjects = SolidsLoader::getInstance()->loadRigidObjects2D<VectorT>(pObjectsNode, m_pQuadGrid->getDimensions(), m_pQuadGrid->getGridData2D()->getGridSpacing());
					for (int i = 0; i < m_rigidObjects.size(); i++) {
						m_meshes.push_back(m_rigidObjects[i]->getLineMesh());
					}
				}

			}
			catch (exception e) {
				exitProgram(e.what());
			}

			/** Initialization phase: */
			{
				m_pPhysicsCore = PhysicsCore<VectorT>::getInstance();
				m_pPhysicsCore->initialize(*m_pPhysicsCoreParams);
			}

			/** FlowSolver Initialization */
			switch (m_pFlowSolverParams->solverType)
			{
				case finiteDifferenceMethod:
					m_pFlowSolver = new RegularGridSolver2D(*m_pFlowSolverParams, m_pQuadGrid, m_boundaryConditions);
				break;

				case cutCellMethod:
					m_pFlowSolver = new CutCellSolver2D(*m_pFlowSolverParams, m_pQuadGrid, m_boundaryConditions, m_rigidObjects);
				break;

				case cutCellSOMethod:
					m_pFlowSolver = new CutCellSolverSO2D(*m_pFlowSolverParams, m_pQuadGrid, m_boundaryConditions, m_rigidObjects);
				break;
				/*case ghostLiquids:
					m_pFlowSolver = new GhostLiquidSolver2D(*m_pFlowSolverParams, m_pQuadGrid, m_boundaryConditions, m_liquidObjects);
				break;*/

				default:
					throw("Solver type not supported!");
				break;
			}

			/** Rendering initialization */
			{
				m_pRenderer = GLRenderer2D::getInstance();
				m_pRenderer->setFlowSolver(m_pFlowSolver);
				m_pRenderer->initialize(1280, 800);
				m_pRenderer->getSimulationStatsWindow()->setFlowSolver(m_pFlowSolver);
				m_pRenderer->addGridVisualizationWindow(m_pQuadGrid);
				m_pRenderer->setLineMeshes(m_lineMeshes);

				if (CutCellSolver2D *pCutCellSolver = dynamic_cast<CutCellSolver2D *>(m_pFlowSolver)) {
					if (pCutCellSolver->getCutCells() != NULL) {
						m_pRenderer->addSpecialCellsRenderer(pCutCellSolver->getCutCells(),
																pCutCellSolver->getVelocityInterpolant(),
																pCutCellSolver->getNodalVelocityField(), m_pQuadGrid,
																m_pRenderer->getGridVisualizationWindow());
					}
				}
			}

			/** Data Logger */
			{
				if (m_pDataLoggerParams && m_pQuadGrid) {
					m_pDataLogger = new DataExporter<VectorT, Array2D>(*m_pDataLoggerParams, m_pQuadGrid->getDimensions());
					m_pDataLogger->setFlowSolver(m_pFlowSolver);
				}
			}

			/** Particles rendering, in case of particle-based advection schemes */
			if (m_pFlowSolverParams->pAdvectionParams->advectionCategory == LagrangianAdvection) {
				ParticlesRenderer<Vector2, Array2D>::renderingParams_t renderingParams;
				ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pFlowSolver->getAdvectionClass());
				ParticlesRenderer<Vector2, Array2D> *pParticlesRenderer = new ParticlesRenderer<Vector2, Array2D>(pParticleBasedAdv->getParticlesData(), renderingParams, m_pQuadGrid->getGridData2D()->getGridSpacing());
				m_pRenderer->setParticlesRenderer(pParticlesRenderer);
			}

			/** Setting up CutVoxelSolver shenanigans */
			CutCellSolver2D *pCutCellSolver = dynamic_cast<CutCellSolver2D *>(m_pFlowSolver);
			if (pCutCellSolver) {
				MeanValueInterpolant2D<Vector2> *pVelInterp = dynamic_cast<MeanValueInterpolant2D<Vector2> *>(pCutCellSolver->getVelocityInterpolant());
				m_pRenderer->setCutCells(pCutCellSolver->getCutCells());
				m_pRenderer->setCutCellsSolver(pCutCellSolver);
			}

			Vector3 currCamPosition;
			currCamPosition.x = m_pQuadGrid->getGridCentroid().x;
			currCamPosition.y = m_pQuadGrid->getGridCentroid().y;
			Scalar scaleFactor = 2.5;
			if (currCamPosition.y > currCamPosition.x) {
				//Scale factor on the y-dimension has to consider widescreen resolution ratio
				currCamPosition.z = currCamPosition.y*scaleFactor*1.7777;
			}
			else {
				currCamPosition.z = currCamPosition.x*scaleFactor;
			}
			//currCamPosition.z = m_pRenderer->getCamera()->getPosition().z;
			m_pRenderer->getCamera()->setPosition(currCamPosition);
			
			m_pPhysicsCore->addObject(m_pFlowSolver);
		}

		template <class VectorT>
		void RealtimeSimulation2D<VectorT>::update() {
			Scalar dt = m_pPhysicsCore->getParams()->timestep;
			if (m_pPhysicsCore->isRunningSimulation() || m_pPhysicsCore->isSteppingSimulation()) {
				if(m_pDataLogger)
					m_pDataLogger->log(PhysicsCore<Vector2>::getInstance()->getElapsedTime());
				m_pPhysicsCore->update();
				m_pFlowSolver->updatePostProjectionDivergence();
				m_pRenderer->update(dt);
				m_pRenderer->getSimulationStatsWindow()->setResidual(m_pFlowSolver->getPoissonSolver()->getResidual());
			}
		}

		template <class VectorT>
		void RealtimeSimulation2D<VectorT>::draw() {
			m_pRenderer->renderLoop();
		}

		template class RealtimeSimulation2D<Vector2>;
	}

}