#include "3D/RealtimeSimulation3D.h"

namespace Chimera {

	namespace Applications {
		
		#pragma region Constructors
		template <class VectorT>
		RealtimeSimulation3D<VectorT>::RealtimeSimulation3D(int argc, char** argv, TiXmlElement *m_pMainNode) : ApplicationBase(argc, argv, m_pMainNode) {
			m_pHexaGrid = nullptr;
			
			try {
				/** Load grid */
				TiXmlElement *pGridNode = m_pMainNode->FirstChildElement("Grid");
				if(pGridNode) {
					m_pHexaGrid = GridLoader::getInstance()->loadHexaGrid(pGridNode);
				}
				
				/** Load boundary conditions */
				TiXmlElement *pBoundaryConditionsNode = m_pMainNode->FirstChildElement("boundaryConditionsFile");
				if(pBoundaryConditionsNode && m_pHexaGrid) {
					string boundaryConditionsFile = pBoundaryConditionsNode->GetText();
					m_boundaryConditions = BoundaryConditionFactory::getInstance()->loadBCs<Vector3>(boundaryConditionsFile, m_pHexaGrid->getDimensions());
				}

				/** Load meshes */
				TiXmlElement *pObjectsNode = m_pMainNode->FirstChildElement("Objects");
				if (pObjectsNode && m_pHexaGrid) {
					m_polyMeshes = MeshLoader::getInstance()->loadPolyMeshes<VectorT>(pObjectsNode, m_pHexaGrid->getDimensions(), m_pHexaGrid->getGridData3D()->getGridSpacing());
					for (int i = 0; i < m_polyMeshes.size(); i++) {
						m_meshes.push_back(m_polyMeshes[i]);
					}
				}

			} catch (exception e) {
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
					m_pFlowSolver = new RegularGridSolver3D(*m_pFlowSolverParams, m_pHexaGrid, m_boundaryConditions);
				break;
				
				case cutCellMethod:
					m_pFlowSolver = new CutVoxelSolver3D(*m_pFlowSolverParams, m_pHexaGrid, m_boundaryConditions, m_polyMeshes);
				break;
				
				default:
					throw("Solver type not supported!");
				break;
			}
			
			/** Rendering initialization */
			{
				m_pRenderer = GLRenderer3D::getInstance();
				m_pRenderer->setFlowSolver(m_pFlowSolver);
				m_pRenderer->initialize(1280, 800);
				m_pRenderer->getSimulationStatsWindow()->setFlowSolver(m_pFlowSolver);
				m_pRenderer->addGridVisualizationWindow(m_pHexaGrid);
				m_pRenderer->addMeshRenderer(m_meshes);	
			}

			/** Data Logger */
			{
				if (m_pDataLoggerParams && m_pHexaGrid) {
					m_pDataLogger = new DataExporter<VectorT, Array3D>(*m_pDataLoggerParams, m_pHexaGrid->getDimensions());
					m_pDataLogger->setFlowSolver(m_pFlowSolver);
				}
			}

			/** Particles rendering, in case of particle-based advection schemes */
			if (m_pFlowSolverParams->pAdvectionParams->advectionCategory == LagrangianAdvection) {
				ParticlesRenderer<Vector3, Array3D>::renderingParams_t renderingParams;
				ParticleBasedAdvection<Vector3, Array3D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector3, Array3D> *>(m_pFlowSolver->getAdvectionClass());
				ParticlesRenderer<Vector3, Array3D> *pParticlesRenderer = new ParticlesRenderer<Vector3, Array3D>(pParticleBasedAdv->getParticlesData(), renderingParams, m_pHexaGrid->getGridData3D()->getGridSpacing());
				m_pRenderer->setParticlesRenderer(pParticlesRenderer);
			}

			/** Setting up CutVoxelSolver shenanigans */
			CutVoxelSolver3D *pCutVoxelSolver = dynamic_cast<CutVoxelSolver3D *>(m_pFlowSolver);
			if (pCutVoxelSolver) {
				MeanValueInterpolant3D<Vector3> *pVelInterp = dynamic_cast<MeanValueInterpolant3D<Vector3> *>(pCutVoxelSolver->getVelocityInterpolant());
				m_pRenderer->setCutVoxels(pCutVoxelSolver->getCutVoxels(), pVelInterp);
				/*if (m_pRenderer->getParticleSystem()) {
					m_pRenderer->getParticleSystem()->setCutVoxels(pCutVoxelSolver->getCutVoxels());
				}*/
			}

			/** Setup camera */
			/*if (m_pMainNode->FirstChildElement("Camera")) {
				loadCamera(m_pMainNode->FirstChildElement("Camera"));
			}*/

			m_pRenderer->getCamera()->setRotationAroundGridMode(m_pHexaGrid->getGridCentroid());
			m_pPhysicsCore->addObject(m_pFlowSolver);
		}
		
		
		#pragma region Functionalities
		template <class VectorT>
		void RealtimeSimulation3D<VectorT>::draw() {
			m_pRenderer->renderLoop();
		}

		template <class VectorT>
		void RealtimeSimulation3D<VectorT>::update() {
			Scalar dt = m_pPhysicsCore->getParams()->timestep;

			static bool firstTimeUpdate = true;
			bool updateRenderer = false;
			if (m_pPhysicsCore->isRunningSimulation() || m_pPhysicsCore->isSteppingSimulation()) {
				updateRenderer = true;
			}

			m_pPhysicsCore->update();

			if (updateRenderer) {
				if (m_pRenderer->getParticlesRenderer() != nullptr) {
					m_pRenderer->getParticlesRenderer()->update(dt);
				}
				m_pRenderer->update(dt);
				if(m_pDataLogger)
					m_pDataLogger->log(m_pPhysicsCore->getElapsedTime());
			}
		}

		template class RealtimeSimulation3D<Vector3>;
	}
}