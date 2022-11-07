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

#include "Applications/RealtimeSimulation2D.h"

namespace Chimera {

	RealtimeSimulation2D::RealtimeSimulation2D(int argc, char** argv, TiXmlElement *pChimeraConfig) : Application2D(argc, argv, pChimeraConfig) {
		/************************************************************************/
		/* TO-be initialized vars                                               */
		/************************************************************************/
		/** Rendering initialization */
		m_pRenderer = (GLRenderer2D::getInstance());
		m_pFlowSolver = NULL;
		m_circularTranslation = false;
		
		QuadGrid *pQuadGrid = NULL;
		m_pLiquidRepresentation = NULL;

		//Rotational and translational constants
		m_maxAngularSpeed = DegreeToRad(120.0f);
		m_maxTranslationalSpeed = 1.0f;
		m_maxTranslationalAcceleration = 2.0f;

		initializeGL(argc, argv);
		loadSimulationParams();
		loadThinObjects();
		loadGridFile();
		if (m_pMainNode->FirstChildElement("Objects")) {
			loadObjects(m_pMainNode->FirstChildElement("Objects"));
		}
		loadRigidObjects();
		
		loadSolver();
		loadLiquidObjects();
		/*if (GhostLiquidSolver *pGhostLiquid = dynamic_cast<GhostLiquidSolver *>(m_pMainSimCfg->getFlowSolver())) {
			pGhostLiquid->setLiquidRepresentation(m_pLiquidRepresentation);
		}*/

		loadPhysics();

		/** Rendering and windows initialization */
		{
			m_pRenderer->addSimulationConfig(m_pMainSimCfg);
			m_pRenderer->initialize(1280, 800);
			m_pRenderer->getSimulationStatsWindow()->setFlowSolver(m_pMainSimCfg->getFlowSolver());
			m_pRenderer->addGridVisualizationWindow(m_pQuadGrid);

			if (CutCellSolver2D *pCutCellSolver = dynamic_cast<CutCellSolver2D *>(m_pMainSimCfg->getFlowSolver())) {
				if (pCutCellSolver->getCutCells() != NULL) {
					m_pRenderer->addSpecialCellsRenderer(pCutCellSolver->getCutCells(),
														 pCutCellSolver->getVelocityInterpolant(),
														 pCutCellSolver->getNodalVelocityField(), m_pQuadGrid,
														 m_pRenderer->getGridVisualizationWindow());

				}
			}
			GridData2D *pGridData2D = m_pQuadGrid->getGridData2D();
			m_pRenderer->getGridRenderer(0)->getScalarFieldRenderer().setFineGridScalarValues2D(pGridData2D->getFineGridScalarFieldArrayPtr(), pGridData2D->getFineGridScalarFieldDx());
			//m_pRenderer->setThinObjectVector(m_thinObjects);
			if (m_pLiquidRepresentation) {
				m_pRenderer->setLiquidsVector(m_pLiquidRepresentation->getLineMeshes());
			}
		}

		setupCamera(m_pMainNode->FirstChildElement("Camera"));

		if(m_pMainNode->FirstChildElement("CircularTranslation")) {
			m_circularTranslation = true;
		}

		ParticleSystem2D::configParams_t params;
		auto pFlowSolver = m_pMainSimCfg->getFlowSolver();
		if (pFlowSolver->getAdvectionClass()->getType() == CPU_ParticleBasedAdvection) {
			ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(pFlowSolver->getAdvectionClass());
			params.setExternalParticles(pParticleBasedAdv->getParticlesPositionsVectorPtr(),
										pParticleBasedAdv->getParticlesPosition().size(),
										pParticleBasedAdv->getParticlesVelocitiesVectorPtr(),
										pParticleBasedAdv->getResampledParticlesVecPtr());
			params.setExternalParticles(pParticleBasedAdv->getParticlesData());
		}
		
		ParticleSystem2D *pParticleSystem = new ParticleSystem2D(params, m_pQuadGrid);
		m_pRenderer->setParticleSystem(pParticleSystem);

		if (CutCellSolver2D *pCutCellSolver = dynamic_cast<CutCellSolver2D *>(m_pMainSimCfg->getFlowSolver())) {
			pParticleSystem->setCutCells2D(pCutCellSolver->getCutCells());
			m_pRenderer->setLineMeshes(pCutCellSolver->getLineMeshes());
			m_pRenderer->setCutCells(pCutCellSolver->getCutCells());
			m_pRenderer->setCutCellsSolver(pCutCellSolver);
		}
		else if (TurbulenceSolver2D *pTurbulenceSolver = dynamic_cast<TurbulenceSolver2D *>(m_pMainSimCfg->getFlowSolver())) {
			m_pRenderer->getGridRenderer(0)->getScalarFieldRenderer().setFineGridScalarValues2D(pTurbulenceSolver->getStreamfunctionPtr(), pTurbulenceSolver->getStreamfunctionDx());
			m_pRenderer->getGridRenderer(0)->getScalarFieldRenderer().m_drawFineGridCells = true;
			m_pRenderer->getCellVisualizationWindow()->setStreamfunctionGrid(pTurbulenceSolver->getStreamfunctionInterpolant(), pTurbulenceSolver->getStreamfunctionPtr(), pTurbulenceSolver->getStreamfunctionDx());
			m_pRenderer->getGridRenderer(0)->getVectorFieldRenderer().setFineGridVelocities(pTurbulenceSolver->getFineGridVelocities(), pTurbulenceSolver->getStreamfunctionDx());
		}
		

		/** Adding density markers to flow solvers */
		for (int i = 0; i < m_densityMarkers.size(); i++) {
			m_pMainSimCfg->getFlowSolver()->addScalarFieldMarker(m_densityMarkers[i]);
		}

		m_pPhysicsCore->addObject(m_pMainSimCfg->getFlowSolver());

		m_pDataExporter = new DataExporter<Vector2, Array2D>(m_dataExporterParams, m_pQuadGrid->getDimensions());
		m_pDataExporter->addSimulationConfig(m_pMainSimCfg);
	}

	/************************************************************************/
	/* Callbacks                                                            */
	/************************************************************************/
	void RealtimeSimulation2D::keyboardCallback(unsigned char key, int x, int y) {
		m_pRenderer->keyboardCallback(key, x, y);
		switch(key) {
			case 'p': case 'P':
				m_pPhysicsCore->runSimulation(!m_pPhysicsCore->isRunningSimulation());
			break;

			case 'q': case 'Q':
				m_pMainSimCfg->setAngularAcceleration(DegreeToRad(100.0f));
			break;

			case 'e': case 'E':
				m_pMainSimCfg->setAngularAcceleration(-DegreeToRad(100.0f));
			break;

			case 'r': case 'R':
				m_pMainSimCfg->setAngularAcceleration(DegreeToRad(0.0f));
			break;
		}
	}

	void RealtimeSimulation2D::keyboardUpCallback(unsigned char key, int x, int y) {
		switch(key) {
			case 'e': case 'E': case 'q': case 'Q':
				m_pMainSimCfg->setAngularAcceleration(DegreeToRad(0.0f));
			break;
		}
	}

	void RealtimeSimulation2D::specialKeyboardCallback(int key, int x, int y) {
		//Inflow speed is contrary to real moving speed
		Vector2 inflowAcceleration = m_pMainSimCfg->getInflowAcceleration();

		switch(key) {
			case GLUT_KEY_UP:
				inflowAcceleration.y = -m_maxTranslationalAcceleration;
				break;
			case GLUT_KEY_DOWN:
				inflowAcceleration.y = m_maxTranslationalAcceleration;
				break;
			case GLUT_KEY_LEFT:
				inflowAcceleration.x = m_maxTranslationalAcceleration;
				break;
			case GLUT_KEY_RIGHT:
				inflowAcceleration.x = -m_maxTranslationalAcceleration;
				break;
		}
		m_pMainSimCfg->setInflowAcceleration(inflowAcceleration);
	}
	
	void RealtimeSimulation2D::specialKeyboardUpCallback(int key, int x, int y) {
		Vector2 inflowAcceleration = m_pMainSimCfg->getInflowAcceleration();
		switch(key) {
			case GLUT_KEY_UP: case GLUT_KEY_DOWN:
				inflowAcceleration.y = 0;
			break;
			case GLUT_KEY_LEFT: case GLUT_KEY_RIGHT:
				inflowAcceleration.x = 0;
			break;
		}
		m_pMainSimCfg->setInflowAcceleration(inflowAcceleration);
	}


	/************************************************************************/
	/* Loading functions                                                    */
	/************************************************************************/
	void RealtimeSimulation2D::loadGridFile() {
		string tempValue;
		dimensions_t tempDimensions;
		try { /** Load grids and boundary conditions: try-catchs for invalid file exceptions */
			TiXmlElement *pTempNode;
			SimulationConfig<Vector2, Array2D> *pSimCfg = new SimulationConfig<Vector2, Array2D>();
			if((pTempNode = m_pMainNode->FirstChildElement("GridFile")) == NULL) {
				m_pQuadGrid = loadGrid(m_pMainNode->FirstChildElement("Grid"));
			} else {
				tempValue = m_pMainNode->FirstChildElement("GridFile")->GetText();
				shared_ptr<TiXmlAttribute> pGridAttribs(m_pMainNode->FirstChildElement("GridFile")->FirstAttribute());
				if(pGridAttribs) {
					if(string(pGridAttribs->Name()) == string("periodic")) {
						string bcType(pGridAttribs->Value());
						if(bcType == "true") {
							pSimCfg->setPeriodicGrid(true);
						}
					}
				}
				m_pQuadGrid = new QuadGrid(tempValue, pSimCfg->isPeriodicGrid());
			}
						
			pSimCfg->setGrid(m_pQuadGrid);
			dimensions_t tempDimensions = pSimCfg->getGrid()->getDimensions();
			/** Load boundary for background grid*/
			string bcFilename = m_pMainNode->FirstChildElement("boundaryConditionsFile")->GetText();
			pSimCfg->setBoundaryConditions(BoundaryConditionFactory::getInstance()->loadBCs<Vector2>(bcFilename, tempDimensions));
			shared_ptr<TiXmlAttribute> pBCAttribs(m_pMainNode->FirstChildElement("boundaryConditionsFile")->FirstAttribute());
			if(pBCAttribs) {
				string bcType(pBCAttribs->Value());
				if(bcType == "farField") {
					pSimCfg->setFarfieldBoundaries(true);
				}
			}
			m_pMainSimCfg = pSimCfg;
		} catch(exception e) {
			exitProgram(e.what());
		}
	}

	void RealtimeSimulation2D::loadSolver() {
		TiXmlElement *pTempNode = m_pMainNode->FirstChildElement("FlowSolverConfig");
		if(pTempNode && pTempNode->FirstChildElement("FlowSolverType") != NULL) {
			string solverType = pTempNode->FirstChildElement("FlowSolverType")->GetText();
			if(solverType == "Regular") {
				m_pFlowSolverParams->setDiscretizationMethod(finiteDifferenceMethod);
			} else if(solverType == "NonRegular") {
				m_pFlowSolverParams->setDiscretizationMethod(finiteVolumeMethod);
			} else if(solverType == "CutCell") {
				m_pFlowSolverParams->setDiscretizationMethod(cutCellMethod);
			} else if (solverType == "Raycast") {
				m_pFlowSolverParams->setDiscretizationMethod(raycastMethod);
			}
			else if (solverType == "StreamfunctionTurbulence") {
				m_pFlowSolverParams->setDiscretizationMethod(streamfunctionTurbulenceMethod);
			}
 		}

		//bool useFLIP = false;
		//if(m_pFlowSolverParams->getConvectionMethod() == CPU_ParticleBasedAdvection) {
		//	useFLIP = true;
		//}

		if(m_pFlowSolverParams->getDiscretizationMethod() == finiteDifferenceMethod) {
			m_pMainSimCfg->setNonRegularSolver(false);
			m_pMainSimCfg->setFlowSolver(new RegularGridSolver2D(*m_pFlowSolverParams, m_pMainSimCfg->getGrid(), *m_pMainSimCfg->getBoundaryConditions()));
		} else if(m_pFlowSolverParams->getDiscretizationMethod() == finiteVolumeMethod) {
			m_pMainSimCfg->setNonRegularSolver(true);
			//m_pMainSimCfg->setFlowSolver(new CurvilinearGridSolver2D(*m_pFlowSolverParams,m_pMainSimCfg->getGrid(), *m_pMainSimCfg->getBoundaryConditions()));	
		} else if(m_pFlowSolverParams->getDiscretizationMethod() == cutCellMethod) {
			m_pMainSimCfg->setNonRegularSolver(false);
			m_pMainSimCfg->setFlowSolver(new CutCellSolver2D(*m_pFlowSolverParams, m_pMainSimCfg->getGrid(), *m_pMainSimCfg->getBoundaryConditions(), m_rigidObjects));
		}
		else if (m_pFlowSolverParams->getDiscretizationMethod() == raycastMethod) {
			m_pMainSimCfg->setNonRegularSolver(false);
			//m_pMainSimCfg->setFlowSolver(new RaycastSolver2D(*m_pFlowSolverParams, m_pMainSimCfg->getGrid(), *m_pMainSimCfg->getBoundaryConditions(), m_thinObjects));
		}
		else if (m_pFlowSolverParams->getDiscretizationMethod() == streamfunctionTurbulenceMethod) {
			m_pMainSimCfg->setNonRegularSolver(false);
			m_pMainSimCfg->setFlowSolver(new TurbulenceSolver2D(*m_pFlowSolverParams, 2, m_pMainSimCfg->getGrid(), *m_pMainSimCfg->getBoundaryConditions()));
		}
		else if (m_pFlowSolverParams->getDiscretizationMethod() == sharpLiquids) {
			m_pMainSimCfg->setNonRegularSolver(false);
			//m_pMainSimCfg->setFlowSolver(new SharpLiquidSolver2D(*m_pFlowSolverParams, m_pMainSimCfg->getGrid(), *m_pMainSimCfg->getBoundaryConditions(), m_liquidObjects));
		}
		/*else if (m_pFlowSolverParams->getDiscretizationMethod() == ghostLiquids) {
			m_pMainSimCfg->setNonRegularSolver(false);
			m_pMainSimCfg->setFlowSolver(new GhostLiquidSolver(*m_pFlowSolverParams, m_pMainSimCfg->getGrid(), *m_pMainSimCfg->getBoundaryConditions(), NULL));
		}*/
		
		for (int i = 0; i < m_forcingFunctions.size(); i++) {
			m_pMainSimCfg->getFlowSolver()->addForcingFunction(m_forcingFunctions[i]);
		}
	}

	void RealtimeSimulation2D::loadPhysics() {
		/** Physics initialization - configure by XML*/
		m_pPhysicsParams->timestep = 1/128.0f;

		//Unbounded simulation
		m_pPhysicsParams->totalSimulationTime = -1;

		if(m_pMainNode->FirstChildElement("SimulationConfig")) {
			TiXmlElement *pSimulationConfig = m_pMainNode->FirstChildElement("SimulationConfig");
			if(pSimulationConfig->FirstChildElement("TotalTime")) {
				pSimulationConfig->FirstChildElement("TotalTime")->QueryFloatAttribute("value", &m_pPhysicsParams->totalSimulationTime);
			}

			if (pSimulationConfig->FirstChildElement("VelocityImpulse")) {
				loadVelocityImpulses(pSimulationConfig->FirstChildElement("VelocityImpulse"));
			}

			if (pSimulationConfig->FirstChildElement("RotationalVelocityField")) {
				
				vector<FlowSolver<Vector2, Array2D>::rotationalVelocity_t> rotVels = loadRotationalVelocityField(pSimulationConfig->FirstChildElement("RotationalVelocityField"));
				for (int i = 0; i < rotVels.size(); i++) {
					m_pMainSimCfg->getFlowSolver()->addRotationalVelocity(rotVels[i]);
				}
				
			}

			/** Logging */
			if(pSimulationConfig->FirstChildElement("Logging")) {
				loadLoggingParams(pSimulationConfig->FirstChildElement("Logging"));
			}

			/** Rotational frame */
			TiXmlElement *pRotationNode = NULL;
			if((pRotationNode = pSimulationConfig->FirstChildElement("FrameRotation")) != NULL) {
				Scalar angularSpeed, angularAcceleration;
				pRotationNode->QueryFloatAttribute("initialSpeed", &angularSpeed);
				pRotationNode->QueryFloatAttribute("initialAcceleration", &angularAcceleration);
				TiXmlElement pRotationNodePoint = NULL;
				if((pRotationNode = pRotationNode->FirstChildElement("Rotation point")) != NULL) {
					Vector2 rotationPoint;
					pRotationNode->QueryFloatAttribute("px", &rotationPoint.x);
					pRotationNode->QueryFloatAttribute("py", &rotationPoint.y);
					m_pMainSimCfg->setRotationPoint(rotationPoint);
				}
				m_pMainSimCfg->setAngularSpeed(angularSpeed);
				m_pMainSimCfg->setAngularAcceleration(angularAcceleration);
			}
		}
		m_pPhysicsCore = PhysicsCore<Vector2>::getInstance();
		m_pPhysicsCore->initialize(*m_pPhysicsParams);
		m_pPhysicsParams = m_pPhysicsCore->getParams();
	}

	void RealtimeSimulation2D::loadObjects(TiXmlElement *pObjectsNode) {
		TiXmlElement *pTempNode;
		pTempNode = pObjectsNode->FirstChildElement();
		while(pTempNode != NULL) {
			if(string(pTempNode->Value()) == "Sphere") {
				TiXmlElement *pSphereNode = pTempNode;
				TiXmlAttribute *pSphereAttrbs = pSphereNode->FirstAttribute();

				Scalar r1 = pSphereAttrbs->DoubleValue(); //Radius
				Vector2 spherePosition;
				pSphereAttrbs = pSphereAttrbs->Next(); //px
				spherePosition.x = pSphereAttrbs->DoubleValue();
				pSphereAttrbs = pSphereAttrbs->Next(); //py
				spherePosition.y = pSphereAttrbs->DoubleValue();

				//Normal information is not used in 2-D circles, hence Vector2(0, 0)
				Circle<Vector2> *pCircle = new Circle<Vector2>(spherePosition, Vector2(0, 0), r1);
				
				Vector2 circleVelocity;
				pSphereNode->QueryFloatAttribute("vx", &circleVelocity.x);
				pSphereNode->QueryFloatAttribute("vy", &circleVelocity.y);
				pCircle->setVelocity(circleVelocity);

				m_pPhysObjects.push_back(pCircle);
				if(m_pFlowSolverParams->getDiscretizationMethod() == finiteDifferenceMethod) {
					
					RegularGridSolver2D * pSolver = (RegularGridSolver2D *) m_pMainSimCfg->getFlowSolver();
					pSolver->updateSolidCircle(circleVelocity, spherePosition, r1);

					//m_pRenderer->addObject(pCircle);
				} else if(m_pFlowSolverParams->getDiscretizationMethod() == cutCellMethod) {
					vector<Vector2> circlePoints2D;
					for(int i = 0; i < pCircle->getCirclePoints().size(); i++) {
						Vector3 tempVector3 = pCircle->getCirclePoints()[i];
						Vector2 tempVector2(tempVector3.x, tempVector3.y);
						circlePoints2D.push_back(tempVector2);
					}
					
					m_pQuadGrid->loadSolidCircle(spherePosition, r1);
					/*CutCellSolver2D * pSolver = (CutCellSolver2D *)m_pMainSimCfg->getFlowSolver();
					pSolver->initializeBoundaryLevelSet(circlePoints2D);
					pSolver->updatePoissonSolidWalls();*/
				} else {
					m_pRenderer->addObject(pCircle);
				}

			} else if(string(pTempNode->Value()) == "Rectangle") {
				TiXmlElement *pRecNode = pTempNode;
				Vector2 recPosition, recSize;
				pRecNode->QueryFloatAttribute("px", &recPosition.x);
				pRecNode->QueryFloatAttribute("py", &recPosition.y);
				pRecNode->QueryFloatAttribute("sx", &recSize.x);
				pRecNode->QueryFloatAttribute("sy", &recSize.y);

				if(m_pFlowSolverParams->getDiscretizationMethod() == finiteDifferenceMethod  || 
					m_pFlowSolverParams->getDiscretizationMethod() == cutCellMethod) {
					RegularGridSolver2D * pSolver = (RegularGridSolver2D *) m_pMainSimCfg->getFlowSolver();
					pSolver->updateSolidRectangle(Vector2(0, 0), recPosition, recSize);
				}


				//Todo: add rectangle primitive to renderer

			} else if(string(pTempNode->Value()) == "ParticleSystem") {
				loadParticleSystem(pTempNode);
			} else if(string(pTempNode->Value()) == "DensityField") {
				loadDensityField(pTempNode);
			} else if (string(pTempNode->Value()) == "TemperatureField") {
				loadTemperatureField(pTempNode);
			} else if (string(pTempNode->Value()) == "GridObject") {
				loadGridObject(pTempNode);
			} else if (string(pTempNode->Value()) == "Line") {
				loadLine(pTempNode);
			} else if (string(pTempNode->Value()) == "WindForce") {
				loadWindForce(pTempNode);
			}
 			pTempNode = pTempNode->NextSiblingElement();
		} 
	}

	void RealtimeSimulation2D::loadThinObjects() {
		try { /** Try-catch for invalid file exceptions */
			TiXmlElement *pThinObjectNode = m_pMainNode->FirstChildElement("ThinObject");

			while(pThinObjectNode) {
				LineMesh<Vector2>::params_t *pLineMeshParams = loadLineMeshNode(pThinObjectNode);
				m_lineMeshParams.push_back(pLineMeshParams);
				pThinObjectNode = pThinObjectNode->NextSiblingElement("ThinObject");
			}
		} catch(exception e) {
			exitProgram(e.what());
		}

		try { /** Try-catch for invalid file exceptions */
			TiXmlElement *pMultiThinObjectNode = m_pMainNode->FirstChildElement("MultiThinObjects");
			while (pMultiThinObjectNode) {
				vector<LineMesh<Vector2>::params_t *> lineMeshMultiParams = loadMultiLinesMeshNode(pMultiThinObjectNode);
				for (int i = 0; i < lineMeshMultiParams.size(); i++) {
					m_lineMeshParams.push_back(lineMeshMultiParams[i]);
				}
				pMultiThinObjectNode = pMultiThinObjectNode->NextSiblingElement("ThinObject");
			}
		}
		catch (exception e) {
			exitProgram(e.what());
		}
	}

	void RealtimeSimulation2D::loadRigidObjects() {
		TiXmlElement *pThinObjectNode = m_pMainNode->FirstChildElement("ThinObject");
		for (int i = 0; i < m_lineMeshParams.size(); i++) {
			m_rigidObjects.push_back(loadRigidObject(pThinObjectNode, i));
			pThinObjectNode = pThinObjectNode->NextSiblingElement("ThinObject");
		}
		
	}

	void RealtimeSimulation2D::loadLiquidObjects() {
		LiquidRepresentation2D::params_t liquidParams;
		try { /** Try-catch for invalid file exceptions */
			TiXmlElement *pLiquidObjectNode = m_pMainNode->FirstChildElement("LiquidObject");

			while (pLiquidObjectNode) {
				LineMesh<Vector2>::params_t *pLineMeshParams = loadLineMeshNode(pLiquidObjectNode);
				//liquidParams.initialLineMeshes.push_back(new LineMesh<Vector2>(lineMeshParams));
				pLiquidObjectNode = pLiquidObjectNode->NextSiblingElement("LiquidObject");
			}
		}
		catch (exception e) {
			exitProgram(e.what());
		}
		if (liquidParams.initialLineMeshes.size() > 0) {
			
			liquidParams.pGridData = m_pQuadGrid->getGridData2D();
			liquidParams.levelSetGridSubdivisions = 2;
			/*{
				ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAvection = m_pMainSimCfg->getFlowSolver()->getParticleBasedAdvection();
				pParticleBasedAvection->getParticlesData()->addIntBasedAttribute("liquid");
				pParticleBasedAvection->getParticlesData()->getIntegerBasedAttribute("liquid").resize(pParticleBasedAvection->getParticlesData()->getPositions().size());
				liquidParams.pParticlesTags = &pParticleBasedAvection->getParticlesData()->getIntegerBasedAttribute("liquid");
				liquidParams.pParticlesPositions = &pParticleBasedAvection->getParticlesData()->getPositions();
				m_pLiquidRepresentation = new LiquidRepresentation2D(liquidParams);
			}*/
		}
		
	}

	LineMesh<Vector2>::params_t * RealtimeSimulation2D::loadLineMeshNode(TiXmlElement *pThinObjectNode) {
		LineMesh<Vector2>::params_t * pLineMeshParams = new LineMesh<Vector2>::params_t();
		if(pThinObjectNode->FirstChildElement("position")) {
			pThinObjectNode->FirstChildElement("position")->QueryFloatAttribute("x", &pLineMeshParams->position.x);
			pThinObjectNode->FirstChildElement("position")->QueryFloatAttribute("y", &pLineMeshParams->position.y);
		}
		TiXmlElement *pGeometryNode = pThinObjectNode->FirstChildElement("Geometry"); 
		if (pGeometryNode->FirstChildElement("ExtrudeAlongNormalsSize")) {
			Scalar extrudeSize = atof(pGeometryNode->FirstChildElement("ExtrudeAlongNormalsSize")->GetText());
			pLineMeshParams->extrudeAlongNormalWidth = extrudeSize;
		}

		if (pGeometryNode->FirstChildElement("File")) {
			string lineStr = "Geometry/2D/";
			lineStr += pGeometryNode->FirstChildElement("File")->GetText();
			Line<Vector2> *pLine = new Line<Vector2>(pLineMeshParams->position, lineStr);
			pLineMeshParams->initialPoints = pLine->getPoints();

			if (pGeometryNode->FirstChildElement("ClosedMesh")) {
				string closedMeshStr = pGeometryNode->FirstChildElement("ClosedMesh")->GetText();
			}
 		} else {
			Scalar lengthSize;
			if(pGeometryNode->FirstChildElement("lengthSize")) {
				lengthSize = atof(pGeometryNode->FirstChildElement("lengthSize")->GetText());
			}
			int numSubdivis;
			if(pGeometryNode->FirstChildElement("numSubdivisions")) {
				numSubdivis = atoi(pGeometryNode->FirstChildElement("numSubdivisions")->GetText());
			}

			if(pGeometryNode->FirstChildElement("SinFunction")) {
				Scalar amplitude, frequency;
				pGeometryNode->FirstChildElement("SinFunction")->QueryFloatAttribute("amplitude", &amplitude);
				pGeometryNode->FirstChildElement("SinFunction")->QueryFloatAttribute("frequency", &frequency);

				Scalar dx = lengthSize /(numSubdivis - 1);
				for(int i = 0; i < numSubdivis; i++) { 
					Vector2 thinObjectPoint;
					thinObjectPoint.x = dx*i - lengthSize*0.5;
					thinObjectPoint.y = sin(dx*i*PI*frequency)*amplitude;
					pLineMeshParams->initialPoints.push_back(thinObjectPoint + pLineMeshParams->position);
				} 
			} else if(pGeometryNode->FirstChildElement("VerticalLine"))  {
				Scalar dx = lengthSize /(numSubdivis - 1);
				for(int i = 0; i < numSubdivis; i++) { 
					Vector2 thinObjectPoint;
					thinObjectPoint.x = 0;
					thinObjectPoint.y = dx*i - lengthSize*0.5;
					pLineMeshParams->initialPoints.push_back(thinObjectPoint + pLineMeshParams->position);
				} 
			} else if(pGeometryNode->FirstChildElement("HorizontalLine")){
				Scalar dx = lengthSize/(numSubdivis - 1);
				for(int i = 0; i < numSubdivis; i++) { 
					Vector2 thinObjectPoint;
					thinObjectPoint.x = dx*i - lengthSize*0.5;
					thinObjectPoint.y = 0;
					pLineMeshParams->initialPoints.push_back(thinObjectPoint + pLineMeshParams->position);
				}
			} else if (pGeometryNode->FirstChildElement("CircularLine")) {
				Scalar radius = 0.0f;
				if (pGeometryNode->FirstChildElement("CircularLine")->FirstChildElement("Radius")) {
					radius = atof(pGeometryNode->FirstChildElement("CircularLine")->FirstChildElement("Radius")->GetText());
				}
				Scalar dx = 2.0f / (numSubdivis - 1);
				for (int i = 0; i < numSubdivis; i++) {
					Vector2 thinObjectPoint;
					thinObjectPoint.x = cos(dx*i*PI)*radius;
					thinObjectPoint.y = sin(dx*i*PI)*radius;
					pLineMeshParams->initialPoints.push_back(thinObjectPoint + pLineMeshParams->position);
				}
				if (pLineMeshParams->initialPoints.back() != pLineMeshParams->initialPoints.front()) {
					pLineMeshParams->initialPoints.push_back(pLineMeshParams->initialPoints.front());
				}
			}
			else if (pGeometryNode->FirstChildElement("GearLine")) {
				pLineMeshParams->initialPoints = createGearGeometry(pGeometryNode->FirstChildElement("GearLine"), numSubdivis, pLineMeshParams->position);
			}
			else if (pGeometryNode->FirstChildElement("RectangularLine")) {
				Vector2 rectangleSize(1,1);
				
				if (pGeometryNode->FirstChildElement("RectangularLine")->FirstChildElement("size")) {
					pGeometryNode->FirstChildElement("RectangularLine")->FirstChildElement("size")->QueryFloatAttribute("x", &rectangleSize.x);
					pGeometryNode->FirstChildElement("RectangularLine")->FirstChildElement("size")->QueryFloatAttribute("y", &rectangleSize.y);
				}

				//Position is the centroid of the rectangularLine
				pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(-rectangleSize.x, -rectangleSize.y)*0.5);
				pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(rectangleSize.x, -rectangleSize.y)*0.5);
				pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(rectangleSize.x, rectangleSize.y)*0.5);
				pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(-rectangleSize.x, rectangleSize.y)*0.5);
				pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(-rectangleSize.x, -rectangleSize.y)*0.5);
			}
		}
		pLineMeshParams->updateCentroid();

		/*TiXmlElement *pVelocityNode = pThinObjectNode->FirstChildElement("VelocityFunction"); 
		if(pVelocityNode) {
			if(pVelocityNode->FirstChildElement("SinFunction")) {
				pVelocityNode->FirstChildElement("SinFunction")->QueryFloatAttribute("amplitude", &lineMeshParams.velocityAmplitude);
				pVelocityNode->FirstChildElement("SinFunction")->QueryFloatAttribute("frequency", &lineMeshParams.velocityFrequency);
				lineMeshParams.velocityFunction = LineMesh<Vector2>::sinFunction;
			} else if(pVelocityNode->FirstChildElement("CosineFunction")) {
				pVelocityNode->FirstChildElement("CosineFunction")->QueryFloatAttribute("amplitude", &lineMeshParams.velocityAmplitude);
				pVelocityNode->FirstChildElement("CosineFunction")->QueryFloatAttribute("frequency", &lineMeshParams.velocityFrequency);
				lineMeshParams.velocityFunction = LineMesh<Vector2>::cosineFunction;
			} else if(pVelocityNode->FirstChildElement("UniformFunction")) {
				pVelocityNode->FirstChildElement("UniformFunction")->QueryFloatAttribute("amplitude", &lineMeshParams.velocityAmplitude);
				lineMeshParams.velocityFunction = LineMesh<Vector2>::uniformVelocity;
			} else if (pVelocityNode->FirstChildElement("Path")) {
				string lineStr = "Geometry/2D/";
				lineStr += pVelocityNode->FirstChildElement("Path")->FirstChildElement("File")->GetText();
				Line<Vector2> *pLine = new Line<Vector2>(Vector2(0, 0), lineStr);
				lineMeshParams.velocityFunction = LineMesh<Vector2>::pathAnimation;
				lineMeshParams.pathMesh = pLine->getPoints();
				pVelocityNode->FirstChildElement("Path")->QueryFloatAttribute("amplitude", &lineMeshParams.velocityAmplitude);
				if (pVelocityNode->FirstChildElement("Path")->FirstChildElement("position")) {
					Vector2 position;
					pVelocityNode->FirstChildElement("Path")->FirstChildElement("position")->QueryFloatAttribute("x", &position.x);
					pVelocityNode->FirstChildElement("Path")->FirstChildElement("position")->QueryFloatAttribute("y", &position.y);
					for (int i = 0; i < lineMeshParams.pathMesh.size(); i++) {
						lineMeshParams.pathMesh[i] += position;
					}
				}
				if (lineMeshParams.initialPoints.size() > 0) {
					Vector2 pathAlign = lineMeshParams.pointsCentroid - lineMeshParams.pathMesh[0];
					for (int i = 0; i < lineMeshParams.initialPoints.size(); i++) {
						lineMeshParams.initialPoints[i] -= pathAlign;
					}
					lineMeshParams.updateCentroid();
				}
			}
			if(pVelocityNode->FirstChildElement("Direction")) {
				pVelocityNode->FirstChildElement("Direction")->QueryFloatAttribute("x", &lineMeshParams.velocityDirection.x);
				pVelocityNode->FirstChildElement("Direction")->QueryFloatAttribute("y", &lineMeshParams.velocityDirection.y);
			}
		}
		TiXmlElement *pRotationNode = pThinObjectNode->FirstChildElement("RotationFunction");
		if (pRotationNode) {
			if(pRotationNode->FirstChildElement("AngularSpeed")) {
				lineMeshParams.initialAngularVelocity = atof(pRotationNode->FirstChildElement("AngularSpeed")->GetText());
			} 
			if (pRotationNode->FirstChildElement("AngularAcceleration")) {
				lineMeshParams.angularAcceleration = atof(pRotationNode->FirstChildElement("AngularAcceleration")->GetText());
			}
		}*/
		return pLineMeshParams;
	}

	RigidObject2D * RealtimeSimulation2D::loadRigidObject(TiXmlElement *pRigidObject, uint lineMeshID) {
		const dimensions_t &gridDimensions = m_pQuadGrid->getDimensions();
		Scalar gridSpacing = m_pQuadGrid->getGridData2D()->getGridSpacing();
		LineMesh<Vector2> * pLineMesh = new LineMesh<Vector2>(*m_lineMeshParams[lineMeshID], gridDimensions, gridSpacing);

		PhysicalObject<Vector2>::positionUpdate_t positionUpdate;
		PhysicalObject<Vector2>::rotationUpdate_t rotationUpdate;
		couplingType_t couplingType = oneWayCouplingSolidToFluid;
		
		TiXmlElement *pVelocityNode = pRigidObject->FirstChildElement("VelocityFunction");
		if(pVelocityNode) {
			if(pVelocityNode->FirstChildElement("SinFunction")) {
				pVelocityNode->FirstChildElement("SinFunction")->QueryFloatAttribute("amplitude", &positionUpdate.amplitude);
				pVelocityNode->FirstChildElement("SinFunction")->QueryFloatAttribute("frequency", &positionUpdate.frequency);
				positionUpdate.positionUpdateType = positionUpdateType_t::sinFunction;
			} else if(pVelocityNode->FirstChildElement("CosineFunction")) {
				pVelocityNode->FirstChildElement("CosineFunction")->QueryFloatAttribute("amplitude", &positionUpdate.amplitude);
				pVelocityNode->FirstChildElement("CosineFunction")->QueryFloatAttribute("frequency", &positionUpdate.frequency);
				positionUpdate.positionUpdateType = positionUpdateType_t::cosineFunction;
			} else if(pVelocityNode->FirstChildElement("UniformFunction")) {
				pVelocityNode->FirstChildElement("UniformFunction")->QueryFloatAttribute("amplitude", &positionUpdate.amplitude);
				positionUpdate.positionUpdateType = positionUpdateType_t::uniformFunction;
			} else if (pVelocityNode->FirstChildElement("Path")) {
				string lineStr = "Geometry/2D/";
				lineStr += pVelocityNode->FirstChildElement("Path")->FirstChildElement("File")->GetText();
				Line<Vector2> *pLine = new Line<Vector2>(Vector2(0, 0), lineStr);
				positionUpdate.positionUpdateType = positionUpdateType_t::pathAnimation;
				positionUpdate.pathMesh = pLine->getPoints();
				pVelocityNode->FirstChildElement("Path")->QueryFloatAttribute("amplitude", &positionUpdate.amplitude);
				if (pVelocityNode->FirstChildElement("Path")->FirstChildElement("position")) {
					Vector2 position;
					pVelocityNode->FirstChildElement("Path")->FirstChildElement("position")->QueryFloatAttribute("x", &position.x);
					pVelocityNode->FirstChildElement("Path")->FirstChildElement("position")->QueryFloatAttribute("y", &position.y);
					for (int i = 0; i < positionUpdate.pathMesh.size(); i++) {
						positionUpdate.pathMesh[i] += position;
					}
				}
				/*if (positionUpdate.initialPoints.size() > 0) {
					Vector2 pathAlign = positionUpdate.pointsCentroid - positionUpdate.pathMesh[0];
					for (int i = 0; i < positionUpdate.initialPoints.size(); i++) {
						positionUpdate.initialPoints[i] -= pathAlign;
					}
					positionUpdate.updateCentroid();
				}*/
			}
			if(pVelocityNode->FirstChildElement("Direction")) {
				pVelocityNode->FirstChildElement("Direction")->QueryFloatAttribute("x", &positionUpdate.direction.x);
				pVelocityNode->FirstChildElement("Direction")->QueryFloatAttribute("y", &positionUpdate.direction.y);
			}
		}
		TiXmlElement *pRotationNode = pRigidObject->FirstChildElement("RotationFunction");
		rotationUpdate.m_rotationType = noRotation;
		if (pRotationNode) {
			rotationUpdate.m_rotationType = constantRotation;
			if (pRotationNode->FirstChildElement("InitialAngle")) {
				rotationUpdate.initialRotation = DegreeToRad(atof(pRotationNode->FirstChildElement("InitialAngle")->GetText()));
			}
			if (pRotationNode->FirstChildElement("Speed")) {
				rotationUpdate.speed = atof(pRotationNode->FirstChildElement("Speed")->GetText());
			}
			if (pRotationNode->FirstChildElement("Acceleration")) {
				rotationUpdate.acceleration = atof(pRotationNode->FirstChildElement("Acceleration")->GetText());
			}
		}
		return new RigidObject2D(pLineMesh, positionUpdate, rotationUpdate, couplingType);;
	}

	vector<LineMesh<Vector2>::params_t *> RealtimeSimulation2D::loadMultiLinesMeshNode(TiXmlElement *pThinObjectNode) {
		vector<LineMesh<Vector2>::params_t *> m_multiParams;
		Vector2 initialPosition;
		if (pThinObjectNode->FirstChildElement("position")) {
			pThinObjectNode->FirstChildElement("position")->QueryFloatAttribute("x", &initialPosition.x);
			pThinObjectNode->FirstChildElement("position")->QueryFloatAttribute("y", &initialPosition.y);
		}
		TiXmlElement *pGeometryNode = pThinObjectNode->FirstChildElement("Geometry");
		if (pGeometryNode->FirstChildElement("File")) {
			string lineStr = "Geometry/2D/";
			lineStr += pGeometryNode->FirstChildElement("File")->GetText();
			shared_ptr<ifstream> fileStream(new ifstream(lineStr.c_str()));
			if (fileStream->fail())
				throw("File not found: " + lineStr);

			while (!fileStream->eof()) {
				LineMesh<Vector2>::params_t *pLineMeshParams = new LineMesh<Vector2>::params_t();
				pLineMeshParams->position = initialPosition;
				vector<Vector2> linePoints;
				Scalar temp;
				Vector2 currPoint;
				int numPoints;
				(*fileStream) >> numPoints;
				for (int i = 0; i < numPoints; i++) {
					(*fileStream) >> currPoint.x;
					(*fileStream) >> currPoint.y;
					(*fileStream) >> temp;
					linePoints.push_back(currPoint + initialPosition);
				}
				pLineMeshParams->initialPoints = linePoints;
				pLineMeshParams->updateCentroid();
				//Line<Vector2> *pLine = new Line<Vector2>(Vector2(0, 0), linePoints);
				//m_pRenderer->addObject(pLine);
				m_multiParams.push_back(pLineMeshParams);
			}
		}
		return m_multiParams;
	}


	void RealtimeSimulation2D::loadDensityField(TiXmlElement *pDensityFieldNode) {
		if (pDensityFieldNode->FirstChildElement("Rectangle")) {
			TiXmlElement *pRecNode = pDensityFieldNode->FirstChildElement("Rectangle");
			Vector2 recPosition, recSize;
			pRecNode->QueryFloatAttribute("px", &recPosition.x);
			pRecNode->QueryFloatAttribute("py", &recPosition.y);
			pRecNode->QueryFloatAttribute("sx", &recSize.x);
			pRecNode->QueryFloatAttribute("sy", &recSize.y);

			FlowSolver<Vector2, Array2D>::scalarFieldMarker_t scalarFieldMarker;
			scalarFieldMarker.position = recPosition;
			scalarFieldMarker.size = recSize;
			scalarFieldMarker.value = 1.0f;
			m_densityMarkers.push_back(scalarFieldMarker);

			int lowerBoundX, lowerBoundY, upperBoundX, upperBoundY;
			if (m_pFlowSolverParams->getDiscretizationMethod() == finiteVolumeMethod) {
				lowerBoundX = recPosition.x;
				lowerBoundY = recPosition.y;
				upperBoundX = lowerBoundX + recSize.x;
				upperBoundY = lowerBoundY + recSize.y;
			}
			else {
				Scalar dx = m_pQuadGrid->getGridData2D()->getScaleFactor(0, 0).x;
				lowerBoundX = floor(recPosition.x / dx);
				lowerBoundY = floor(recPosition.y / dx);
				upperBoundX = lowerBoundX + floor(recSize.x / dx);
				upperBoundY = lowerBoundY + floor(recSize.y / dx);
			}

			for (int i = lowerBoundX; i < upperBoundX; i++) {
				for (int j = lowerBoundY; j < upperBoundY; j++) {
					m_pQuadGrid->getGridData2D()->getDensityBuffer().setValueBothBuffers(1, i, j);
				}
			}
		}
	}

	void RealtimeSimulation2D::loadTemperatureField(TiXmlElement *pTemperatureNode) {
		if (pTemperatureNode->FirstChildElement("Rectangle")) {
			TiXmlElement *pRecNode = pTemperatureNode->FirstChildElement("Rectangle");
			Vector2 recPosition, recSize;
			pRecNode->QueryFloatAttribute("px", &recPosition.x);
			pRecNode->QueryFloatAttribute("py", &recPosition.y);
			pRecNode->QueryFloatAttribute("sx", &recSize.x);
			pRecNode->QueryFloatAttribute("sy", &recSize.y);


			Scalar dx = m_pQuadGrid->getGridData2D()->getScaleFactor(0, 0).x;
			int lowerBoundX, lowerBoundY, upperBoundX, upperBoundY;
			lowerBoundX = floor(recPosition.x / dx);
			lowerBoundY = floor(recPosition.y / dx);
			upperBoundX = lowerBoundX + floor(recSize.x / dx);
			upperBoundY = lowerBoundY + floor(recSize.y / dx);


			for (int i = lowerBoundX; i < upperBoundX; i++) {
				for (int j = lowerBoundY; j < upperBoundY; j++) {
					m_pQuadGrid->getGridData2D()->getTemperatureBuffer().setValueBothBuffers(1, i, j);
				}
			}
		}
		else if (pTemperatureNode->FirstChildElement("Circle")) {
			TiXmlElement *pPositionNode = pTemperatureNode->FirstChildElement("position");
			Vector2 circlePosition;
			pPositionNode->QueryFloatAttribute("px", &circlePosition.x);
			pPositionNode->QueryFloatAttribute("py", &circlePosition.y);

			TiXmlElement *pCircleNode = pTemperatureNode->FirstChildElement("Circle");
			Scalar radius = 1.0f;
			if (pCircleNode->FirstChildElement("Radius")) {
				radius = atof(pCircleNode->FirstChildElement("Radius")->GetText());
			}

			bool constantValue = true;
			TiXmlElement *pValueNode = pTemperatureNode->FirstChildElement("Value");
			string valueType = "constant";
			//pValueNode->QueryStringAttribute("type", &valueType);
			Scalar temperatureValue;
			if (valueType == "falloff") {
				constantValue = false;
			}
			temperatureValue = atof(pValueNode->GetText());

			Scalar dx = m_pQuadGrid->getGridData2D()->getScaleFactor(0, 0).x;
			int lowerBoundX, lowerBoundY, upperBoundX, upperBoundY;
			lowerBoundX = max(0.0f, (circlePosition.x - radius) / dx);
			lowerBoundY = max(0.0f, (circlePosition.y - radius) / dx);

			cout << circlePosition.x;

			upperBoundX = min(m_pQuadGrid->getDimensions().x - 1 + 0.0f, (circlePosition.x + radius) / dx);
			upperBoundY = min(m_pQuadGrid->getDimensions().y - 1 + 0.0f, (circlePosition.y + radius) / dx);
			for (int i = lowerBoundX; i < upperBoundX; i++) {
				for (int j = lowerBoundY; j < upperBoundY; j++) {
					Vector2 cellPosition((i + 0.5)*dx, (j + 0.5)*dx);
					if ((cellPosition - circlePosition).length() <= radius) {
						if (constantValue) 
							m_pQuadGrid->getGridData2D()->getTemperatureBuffer().setValueBothBuffers(temperatureValue, i, j); 
						else 
							m_pQuadGrid->getGridData2D()->getTemperatureBuffer().setValueBothBuffers(temperatureValue, i, j);
					}
					
				}
			}
		} 		
	}

	void RealtimeSimulation2D::loadGridObject(TiXmlElement *pGridObjectNode) {
		if (pGridObjectNode->FirstChildElement("GridFile")) {
			string gridName = pGridObjectNode->FirstChildElement("GridFile")->GetText();
			shared_ptr<TiXmlAttribute> pGridAttribs(pGridObjectNode->FirstChildElement("GridFile")->FirstAttribute());
			bool isPeriodic = false;
			if (pGridAttribs) {
				if (string(pGridAttribs->Name()) == string("periodic")) {
					string bcType(pGridAttribs->Value());
					isPeriodic = bcType == "true";
				}
			}
			QuadGrid *tempGrid = new QuadGrid(gridName, isPeriodic);
			vector<Vector2> objectPoints;
			Vector2 objectCentroid;
			for (int i = 0; i <= tempGrid->getDimensions().x; i++) {
				objectPoints.push_back(tempGrid->getGridData2D()->getPoint(i, 0));
				objectCentroid += tempGrid->getGridData2D()->getPoint(i, 0);
			}

			if (m_pFlowSolverParams->getDiscretizationMethod() == cutCellMethod) {
				/*((CutCellSolver2D *)m_pMainSimCfg->getFlowSolver())->initializeBoundaryLevelSet(objectPoints);
				((CutCellSolver2D *)m_pMainSimCfg->getFlowSolver())->updatePoissonSolidWalls();*/
			}
			else if (m_pFlowSolverParams->getDiscretizationMethod() == finiteDifferenceMethod) {
				((RegularGridSolver2D *)m_pMainSimCfg->getFlowSolver())->updateObject(Vector2(0, 0), objectPoints);
			}

			//GLRenderer2D::getInstance()->addObject(new Polygon2D(Vector2(0, 0), objectPoints));

			delete tempGrid;
		}
	}

	void RealtimeSimulation2D::loadLine(TiXmlElement *pLineNode) {
		Vector2 linePosition;
		if (pLineNode->FirstChildElement("position")) {
			pLineNode->FirstChildElement("position")->QueryFloatAttribute("x", &linePosition.x);
			pLineNode->FirstChildElement("position")->QueryFloatAttribute("y", &linePosition.y);
		}
		TiXmlElement *pGeometryNode = pLineNode->FirstChildElement("Geometry");
		if (pGeometryNode->FirstChildElement("File")) {
			string lineStr = "Geometry/2D/";
			lineStr += pGeometryNode->FirstChildElement("File")->GetText();
			Line<Vector2> *pLine = new Line<Vector2>(linePosition, lineStr);
			m_pRenderer->addObject(pLine);
			if (pGeometryNode->FirstChildElement("ClosedMesh")) {
				string closedMeshStr = pGeometryNode->FirstChildElement("ClosedMesh")->GetText();
				if (closedMeshStr == "true") {
					
				}
			}
		}
		else if (pGeometryNode->FirstChildElement("CircularLine")) {
			int numSubdivisions;
			if (pGeometryNode->FirstChildElement("numSubdivisions")) {
				numSubdivisions = atoi(pGeometryNode->FirstChildElement("numSubdivisions")->GetText());
			}

			vector<Vector2> linePoints;
			Scalar radius = 0.0f;
			if (pGeometryNode->FirstChildElement("CircularLine")->FirstChildElement("Radius")) {
				radius = atof(pGeometryNode->FirstChildElement("CircularLine")->FirstChildElement("Radius")->GetText());
			}
			Scalar dx = 2.0f / (numSubdivisions - 1);
			for (int i = 0; i < numSubdivisions; i++) {
				Vector2 thinObjectPoint;
				thinObjectPoint.x = cos(dx*i*PI)*radius;
				thinObjectPoint.y = sin(dx*i*PI)*radius;
				linePoints.push_back(thinObjectPoint + linePosition);
			}
			Line<Vector2> *pLine = new Line<Vector2>(Vector2(0, 0), linePoints);
			m_pRenderer->addObject(pLine);
		} else if (pGeometryNode->FirstChildElement("GearLine")) {
			int numSubdivisions;
			if (pGeometryNode->FirstChildElement("numSubdivisions")) {
				numSubdivisions = atoi(pGeometryNode->FirstChildElement("numSubdivisions")->GetText());
			}

			vector<Vector2> linePoints = createGearGeometry(pGeometryNode->FirstChildElement("GearLine"), numSubdivisions, linePosition);
			Line<Vector2> *pLine = new Line<Vector2>(Vector2(0, 0), linePoints);
			m_pRenderer->addObject(pLine);
		}
	}

	void RealtimeSimulation2D::loadWindForce(TiXmlElement *pWindNode) {
		Vector2 windPosition, windSize, windStrength;
		FlowSolver<Vector2, Array2D>::forcingFunction_t forcingFunction;
		if (pWindNode->FirstChildElement("position")) {
			pWindNode->FirstChildElement("position")->QueryFloatAttribute("px", &forcingFunction.position.x);
			pWindNode->FirstChildElement("position")->QueryFloatAttribute("py", &forcingFunction.position.y);
		}
		if (pWindNode->FirstChildElement("size")) {
			pWindNode->FirstChildElement("size")->QueryFloatAttribute("x", &forcingFunction.size.x);
			pWindNode->FirstChildElement("size")->QueryFloatAttribute("y", &forcingFunction.size.y);
		}
		if (pWindNode->FirstChildElement("strength")) {
			pWindNode->FirstChildElement("strength")->QueryFloatAttribute("x", &forcingFunction.strength.x);
			pWindNode->FirstChildElement("strength")->QueryFloatAttribute("y", &forcingFunction.strength.y);
		}

		m_forcingFunctions.push_back(forcingFunction);
	}
	void RealtimeSimulation2D::setupCamera(TiXmlElement *pCameraNode) {
		Vector3 camPosition;
		camPosition.z = 4.0f;

		//Fixing up camera position
		Scalar xSize = m_pQuadGrid->getBoundingBox().upperBounds.x - m_pQuadGrid->getBoundingBox().lowerBounds.y;
		Scalar ySize = m_pQuadGrid->getBoundingBox().upperBounds.y - m_pQuadGrid->getBoundingBox().upperBounds.y;
		Scalar biggerSize = xSize > ySize ? xSize : ySize;
		Vector3 cameraPosition(m_pQuadGrid->getGridCentroid().x, m_pQuadGrid->getGridCentroid().y, (xSize + ySize)*0.42);
		/*cameraPosition.x = 2.655;
		cameraPosition.y = 1;
		cameraPosition.z = 0.45;*/
		cameraPosition.z = 4;
		m_pRenderer->getCamera()->setPosition(cameraPosition);

		if(pCameraNode && pCameraNode->FirstChildElement("Mode")) {
			if(string(pCameraNode->FirstChildElement("Mode")->GetText()) == "Follow")
				m_pRenderer->getCamera()->followGrid(m_pQuadGrid, camPosition);
		}

	}

	void RealtimeSimulation2D::loadVelocityImpulses(TiXmlElement *pVelocityImpulsesNode) {
		while (pVelocityImpulsesNode) {
			FlowSolver<Vector2, Array2D>::velocityImpulse_t velocityImpulse;
			if (pVelocityImpulsesNode->FirstChildElement("Position")) {
				pVelocityImpulsesNode->FirstChildElement("Position")->QueryFloatAttribute("x", &velocityImpulse.position.x);
				pVelocityImpulsesNode->FirstChildElement("Position")->QueryFloatAttribute("y", &velocityImpulse.position.y);
			}
			if (pVelocityImpulsesNode->FirstChildElement("Velocity")) {
				pVelocityImpulsesNode->FirstChildElement("Velocity")->QueryFloatAttribute("x", &velocityImpulse.velocity.x);
				pVelocityImpulsesNode->FirstChildElement("Velocity")->QueryFloatAttribute("y", &velocityImpulse.velocity.y);
			}
			m_pMainSimCfg->getFlowSolver()->addVelocityImpulse(velocityImpulse);
			pVelocityImpulsesNode = pVelocityImpulsesNode->NextSiblingElement();
		}
	}

	/************************************************************************/
	/* Private functionalities												*/
	/************************************************************************/
	void RealtimeSimulation2D::updateParticleSystem(Scalar dt) {
		/*
		m_pParticleSystem->setGridOrientation(m_pRenderer->getSystemRotationAngle());
		m_pParticleSystem->setGridVelocity(m_pMainSimCfg->getTranslationalVelocity());
		m_pParticleSystem->setGridOrigin(m_pQuadGrid->getPosition());
		m_pParticleSystem->setAngularSpeed(m_pMainSimCfg->getAngularSpeed());
		m_pParticleSystem->setAngularAcceleration(m_pMainSimCfg->getAngularAcceleration());

		m_pParticleSystem->update(dt);*/
	}

	vector<Vector2> RealtimeSimulation2D::createGearGeometry(TiXmlElement *pGearElement, int numSubdivisions, const Vector2 initialPosition) {
		vector<Vector2> linePoints;
		Scalar innerRadius = 0.0f;
		int numberOfDents = 0;
		Scalar dentSmoothing = 0.75;
		Scalar dentSize = 0.07;
		Scalar dentAngleCorrection = 0;
		if (pGearElement->FirstChildElement("Radius")) {
			innerRadius = atof(pGearElement->FirstChildElement("Radius")->GetText());
		}	

		if (pGearElement->FirstChildElement("DentSize")) {
			dentSize = atof(pGearElement->FirstChildElement("DentSize")->GetText());
		}

		if (pGearElement->FirstChildElement("AngleCorrection")) {
			dentAngleCorrection = DegreeToRad(atof(pGearElement->FirstChildElement("AngleCorrection")->GetText()));
		}

		if (pGearElement->FirstChildElement("NumberOfDents")) {
			numberOfDents = atoi(pGearElement->FirstChildElement("NumberOfDents")->GetText());
		}
		
		//Adjusting number of subdivisions
		int numberOfSubdivisPointPerDent = max((int) floor(numSubdivisions / numberOfDents), 2);
		numSubdivisions = numberOfSubdivisPointPerDent*numberOfDents;

		Scalar angleDx = DegreeToRad(180.0f/ numSubdivisions);
		Scalar angleBiggerDx = DegreeToRad(180.0f / numberOfDents);

		Scalar angleTemp = (DegreeToRad(180.f) - angleBiggerDx) / 2;
		DoubleScalar angleCorrection = max(DegreeToRad(90.f) - angleTemp, 0.f);
		angleCorrection += dentAngleCorrection;
		
		for (int i = 0; i < numberOfDents; i++) {
			//Circular part
			for (int j = 0; j < numberOfSubdivisPointPerDent; j++) {
				Vector2 circlePoint;
				circlePoint.x = cos(i*angleBiggerDx * 2 + angleDx*j);
				circlePoint.y = sin(i*angleBiggerDx * 2 + angleDx*j);
				if (j == 0) { //First, create gear dent
					Vector2 gearPoint = circlePoint*dentSize;
					gearPoint.rotate(-angleCorrection);
					linePoints.push_back(circlePoint*innerRadius + gearPoint + initialPosition);
					linePoints.push_back(circlePoint*innerRadius + initialPosition);
				}
				else if (j == numberOfSubdivisPointPerDent - 1) { //Last, create gear dent
					linePoints.push_back(circlePoint*innerRadius + initialPosition);
					Vector2 gearPoint = circlePoint*dentSize;
					gearPoint.rotate(angleCorrection);
					linePoints.push_back(circlePoint*innerRadius + gearPoint + initialPosition);
				}
				else {
					linePoints.push_back(circlePoint*innerRadius + initialPosition);
				}
			}
		}
		linePoints.push_back(linePoints.front());

		
		return linePoints;
	}

	/************************************************************************/
	/* Functionalities                                                      */
	/************************************************************************/
	void RealtimeSimulation2D::update() {
		Scalar dt = m_pPhysicsCore->getParams()->timestep;
		if(m_pPhysicsCore->isRunningSimulation() || m_pPhysicsCore->isSteppingSimulation()) {
			/** Update physical objects position*/
			for(int i = 0; i < m_pPhysObjects.size(); i++) {
				Vector2 spherePosition(m_pPhysObjects[i]->getPosition().x, m_pPhysObjects[i]->getPosition().y);
			
				if(m_circularTranslation) {
					Vector2 circularAxis = spherePosition - m_pMainSimCfg->getGrid()->getGridCentroid();
					Vector2 circularVelocity = circularAxis.perpendicular().normalize();
					m_pPhysObjects[i]->setVelocity(circularVelocity);
				}

				m_pPhysObjects[i]->setPosition(m_pPhysObjects[i]->getPosition() + m_pPhysObjects[i]->getVelocity()*dt); //Simple forward Euler integration
				
				Vector2 sphereVelocity(m_pPhysObjects[i]->getVelocity().x, m_pPhysObjects[i]->getVelocity().y);
				if(m_pFlowSolverParams->getDiscretizationMethod() == finiteDifferenceMethod  || 
					m_pFlowSolverParams->getDiscretizationMethod() == cutCellMethod) {
					m_pQuadGrid->loadSolidCircle(spherePosition, 0.125);
					RegularGridSolver2D * pSolver = (RegularGridSolver2D *) m_pMainSimCfg->getFlowSolver();
					pSolver->updateSolidCircle(spherePosition, sphereVelocity, 0.125);
					pSolver->updatePoissonSolidWalls();
				} 
			}
			
			m_pDataExporter->log(PhysicsCore<Vector2>::getInstance()->getElapsedTime());
			m_pPhysicsCore->update();
			m_pMainSimCfg->getFlowSolver()->updatePostProjectionDivergence();
			m_pRenderer->update();
			m_pRenderer->getSimulationStatsWindow()->setResidual(m_pMainSimCfg->getFlowSolver()->getPoissonSolver()->getResidual());
			
			if (m_pLiquidRepresentation) {
				m_pRenderer->setLiquidsVector(m_pLiquidRepresentation->getLineMeshes());
			}
			
		} 

		
		
	}
	void RealtimeSimulation2D::draw() {
		m_pRenderer->renderLoop();
	}
}