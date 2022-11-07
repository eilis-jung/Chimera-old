#include "Applications/PrecomputedAnimation3D.h"
#include "MayaCache/MayaNCache.h"

namespace Chimera {

	PrecomputedAnimation3D::PrecomputedAnimation3D(int argc, char** argv, TiXmlElement *pChimeraConfig) : Application3D(argc, argv, pChimeraConfig) {
		/************************************************************************/
		/* TO-be initialized vars                                               */
		/************************************************************************/
		//m_pParticleSystem = NULL;
		m_pDensityFieldAnimation = NULL;
		m_pPressureFieldAnimation = NULL;
		m_pParticleSystem = NULL;
		m_elapsedTime = 0.0f;
		m_currentTimeStep = 0;
		m_loadPerFrame = false;
		m_useMayaCache = false;
		m_exportObjFiles = false;
		m_pVelocityInterpolant = NULL;
		m_pMayaCacheParticlesExporter = nullptr;
		m_pAmiraExporter = nullptr;
		m_velocityFieldAnimation.totalFrames = 0;
		
		//Rotational and translational constants

		initializeGL(argc, argv);
		loadRenderingParams();
		loadGridFile();
		loadPhysics();

		m_pSceneLoader = new SceneLoader(m_pHexaGrid, m_pRenderer);
		if (m_pMainNode->FirstChildElement("Objects"))
			m_pSceneLoader->loadScene(m_pMainNode->FirstChildElement("Objects"), false);

		/** Rendering and windows initialization */
		{
			/** Rendering initialization */
			m_pRenderer = (GLRenderer3D::getInstance());
			m_pRenderer->initialize(1280, 800);
			if (m_pHexaGrid && m_initializeGridVisualization)
				m_pRenderer->addGridVisualizationWindow(m_pHexaGrid);
		}
	
		if(m_pMainNode->FirstChildElement("Camera")) {
			setupCamera(m_pMainNode->FirstChildElement("Camera"));
		}
		m_pRenderer->getCamera()->setRotationAroundGridMode(m_pHexaGrid->getGridCentroid());

		if(TwGetLastError())
			Logger::getInstance()->log(string(TwGetLastError()), Log_HighPriority);


		if(m_pMainNode->FirstChildElement("Animations")) {
			loadAnimations(m_pMainNode->FirstChildElement("Animations"));
		}

		Logger::getInstance()->get() << "Syncing Node Base Velocities" << endl;

		//syncNodeBasedVelocities();


		Logger::getInstance()->get() << "Velocities synced" << endl;

		if(m_pMainNode->FirstChildElement("Objects")) {
			m_pSceneLoader = new SceneLoader(m_pHexaGrid, m_pRenderer);
			m_pSceneLoader->loadScene(m_pMainNode->FirstChildElement("Objects") ,false);
			m_pParticleSystem = m_pSceneLoader->getParticleSystem();
			m_pRenderer->setParticleSystem(m_pParticleSystem);
		}
		
		if (m_pParticleSystem) {
			m_pMayaCacheParticlesExporter = new NParticleExporter(m_pParticleSystem, 45000, 250);
		}

		if (m_pMainNode->FirstChildElement("MayaFluidExporter"))
		{
			string path;
			TiXmlElement* maya = m_pMainNode->FirstChildElement("MayaFluidExporter");
			if (maya->FirstChildElement("Path")) {
				path = maya->FirstChildElement("Path")->GetText();
			}
			else
			{
				Logger::getInstance()->get() << "Invalid maya exporter path" << endl;
				exit(-1);
			}
			if (m_pHexaGrid)
			{
				m_pMayaCacheVelFieldExporter = new VelFieldExporter(m_pHexaGrid->getGridData3D(), 45000, 250, path);
			}
		}
		if (m_pMainNode->FirstChildElement("AmiraExporter"))
		{
			string path;
			TiXmlElement* amira = m_pMainNode->FirstChildElement("AmiraExporter");
			if (amira->FirstChildElement("Path")) {
				path = amira->FirstChildElement("Path")->GetText();
			}
			else
			{
				Logger::getInstance()->get() << "Invalid maya exporter path" << endl;
				exit(-1);
			}
			if (m_pHexaGrid)
			{
				double bounds[6] = { 0.0,4,0.0,8,0.0,4 };
				int dim[3] = { 15,31,15 };
				m_pAmiraExporter = new AmiraExporter(m_pHexaGrid->getGridData3D(), path, dim, bounds);
			}
		}

		Logger::getInstance()->get() << "Particle System Initialized" << endl;
		if(m_useMayaCache)
			initializeMayaCache();

		if (_CrtCheckMemory()) {
			Logger::getInstance()->get() << "Memory check for heap inconsistency working fine (ONLY WORKS FOR DEBUG MODE). " << endl;
		}
		else {
			Logger::getInstance()->get() << "Memory check for heap inconsistency FAILED (ONLY WORKS FOR DEBUG MODE). " << endl;
		}

		/*for (int i = 0; i < m_pSceneLoader->getPolygonSurfaces().size(); i++) {
			m_pRenderer->addObject(m_pSceneLoader->getPolygonSurfaces()[i]);
		}
*/

		/*if (m_velocityFieldAnimation.nodeVelocityFields.size() > 0) {
			m_pRenderer->addMeshRenderer(&m_velocityFieldAnimation.nodeVelocityFields.front(), m_pRenderer->getGridVisualizationWindow());
		}
		*/

		/*m_pPolySurfaces = m_pSceneLoader->getPolygonSurfaces();*/
		//First frame has to be exported
		if (m_exportObjFiles)
			exportAnimationToObj();

		if (m_useMayaCache)
			logToMayaCache();


		exportAllMeshesToObj();

		/*if (m_loadPerFrame)
			m_pParticleSystem->setNodeVelocityField(&m_velocityFieldAnimation.nodeVelocityFields[0]);*/

		if (m_pParticleSystem) {
			if (m_collisionDetectionMethod != noCollisionDetection) {
				m_pParticleSystem->getConfigParams().collisionDetectionMethod = m_collisionDetectionMethod;
				//m_pParticleSystem->getConfigParams().collisionSurfaces = m_pPolySurfaces;
			}
		}

		m_pVelocityInterpolant = new BilinearStaggeredInterpolant3D<Vector3>(m_velocityFieldAnimation.velocityBuffers[0], m_pHexaGrid->getGridData3D()->getGridSpacing());
		if (m_pParticleSystem)
			m_pParticleSystem->setVelocityInterpolant(m_pVelocityInterpolant);
		/*if (m_nodeBasedVelocities) {
			m_pRenderer->getGridRenderer(0)->getVectorFieldRenderer().setNodeBasedVelocities(&m_velocityFieldAnimation.nodeVelocityFields[0]);
		}*/
	}

	/************************************************************************/
	/* Callbacks                                                            */
	/************************************************************************/
	void PrecomputedAnimation3D::keyboardCallback(unsigned char key, int x, int y) {
		m_pRenderer->keyboardCallback(key, x, y);
		switch(key) {
			case 'p': case 'P':
				m_pPhysicsCore->runSimulation(!m_pPhysicsCore->isRunningSimulation());
			break;
		}
	}

	void PrecomputedAnimation3D::keyboardUpCallback(unsigned char key, int x, int y) {
		
	}

	void PrecomputedAnimation3D::specialKeyboardCallback(int key, int x, int y) {
		
	}
	
	void PrecomputedAnimation3D::specialKeyboardUpCallback(int key, int x, int y) {

	}


	/************************************************************************/
	/* Loading functions                                                    */
	/************************************************************************/
	void PrecomputedAnimation3D::loadGridFile() {
		string tempValue;
		dimensions_t tempDimensions;
		try {	/** Load grids and boundary conditions: try-catchs for invalid file exceptions */
			SimulationConfig<Vector3, Array3D> *pSimCfg = new SimulationConfig<Vector3, Array3D>();

			if(m_pMainNode->FirstChildElement("GridFile") == NULL) {
				m_pHexaGrid = loadGrid(m_pMainNode->FirstChildElement("Grid"));
			} else {
				/** Load grid */
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

				m_pHexaGrid = new HexaGrid(tempValue, pSimCfg->isPeriodicGrid());
			}
			tempDimensions = m_pHexaGrid->getDimensions();
			pSimCfg->setGrid(m_pHexaGrid);

			m_pMainSimCfg = pSimCfg;
		} catch(exception e) {
			Logger::get() << e.what() << endl;
			exit(1);
		}
	}

	void PrecomputedAnimation3D::loadPhysics() {
		/** Physics initialization - configure by XML*/
		m_pPhysicsParams->timestep = 0.01f;

		//Unbounded simulation
		m_pPhysicsParams->totalSimulationTime = -1;

		m_pPhysicsCore = PhysicsCore<Vector3>::getInstance();
		m_pPhysicsCore->initialize(*m_pPhysicsParams);
		m_pPhysicsParams = m_pPhysicsCore->getParams();
	}



	void PrecomputedAnimation3D::setupCamera(TiXmlElement *pCameraNode) {

		Vector3 camPosition;
		if(pCameraNode->FirstChildElement("Position")) {
			pCameraNode->FirstChildElement("Position")->QueryFloatAttribute("px", &camPosition.x);
			pCameraNode->FirstChildElement("Position")->QueryFloatAttribute("py", &camPosition.y);
			pCameraNode->FirstChildElement("Position")->QueryFloatAttribute("pz", &camPosition.z);
			GLRenderer3D::getInstance()->getCamera()->setPosition(camPosition);
		}
		if(pCameraNode->FirstChildElement("Direction")) {
			Vector3 camDirection;
			pCameraNode->FirstChildElement("Direction")->QueryFloatAttribute("px", &camDirection.x);
			pCameraNode->FirstChildElement("Direction")->QueryFloatAttribute("py", &camDirection.y);
			pCameraNode->FirstChildElement("Direction")->QueryFloatAttribute("pz", &camDirection.z);
			GLRenderer3D::getInstance()->getCamera()->setDirection(camDirection);
			GLRenderer3D::getInstance()->getCamera()->setFixedDirection(true);
		}

	}

	void PrecomputedAnimation3D::loadAnimations(TiXmlElement *pAnimationsNode) {
		int numFrames = 0;
		if (pAnimationsNode->FirstChildElement("NumFrames")) {
			numFrames = atoi(pAnimationsNode->FirstChildElement("NumFrames")->GetText());
		}
		else {
			Logger::getInstance()->get() << "Invalid number of frames" << endl;
			exit(-1);
		}

		if (pAnimationsNode->FirstChildElement("ExportObjFiles")) {
			string exportObj = pAnimationsNode->FirstChildElement("ExportObjFiles")->GetText();
			if (exportObj == "true") {
				m_exportObjFiles = true;
			}
		}

		if (pAnimationsNode->FirstChildElement("LoadPerFrame")) {
			string exportObj = pAnimationsNode->FirstChildElement("LoadPerFrame")->GetText();
			if (exportObj == "true") {
				m_loadPerFrame = true;
			}
		}

		if (pAnimationsNode->FirstChildElement("DensityFile")) {
			string densityFile = pAnimationsNode->FirstChildElement("DensityFile")->GetText();
			m_pDensityFieldAnimation = loadScalarField("Flow logs/Density/" + densityFile + ".log");
		}
		if (pAnimationsNode->FirstChildElement("PressureFile")) {
			string pressureFile = pAnimationsNode->FirstChildElement("PressureFile")->GetText();
			m_pPressureFieldAnimation = loadScalarField("Flow logs/Pressure/" + pressureFile + ".log");
		}
		if (pAnimationsNode->FirstChildElement("VelocityFile")) {
			m_velocityFieldName = pAnimationsNode->FirstChildElement("VelocityFile")->GetText();
			m_velocityFieldAnimation.totalFrames = numFrames;
			if (!m_loadPerFrame) {
				for (int i = 0; i < numFrames; i++) {
					loadVelocityField("Flow logs/3D/Velocity/" + m_velocityFieldName + intToStr(i) + ".log");
				}
			}
			else {
				loadVelocityField("Flow logs/3D/Velocity/" + m_velocityFieldName + intToStr(0) + ".log");
			}
			
		}
		if (pAnimationsNode->FirstChildElement("PolygonMeshes")){
			Logger::getInstance()->get() << "Loading polygon meshes" << endl;
			m_nodeBasedVelocities = true;
			TiXmlElement *pPolygonMeshesNode = pAnimationsNode->FirstChildElement("PolygonMeshes");
			if (pPolygonMeshesNode->FirstChildElement("BaseFilename")) {
				if (!m_loadPerFrame) {
					for (int i = 0; i < numFrames; i++) {
						loadIthFrame(i, pPolygonMeshesNode->FirstChildElement("BaseFilename")->GetText());
					}
				}
				else {
					m_baseFileName = pPolygonMeshesNode->FirstChildElement("BaseFilename")->GetText();
					loadIthFrame(0, m_baseFileName);	
				}
			}

			/*if (pPolygonMeshesNode->FirstChildElement("LinearInterpolationMethod")) {
				LinearInterpolant3D<Vector3>::params_t linearInterpParams = loadLinearInterpolationParams(pPolygonMeshesNode->FirstChildElement("LinearInterpolationMethod"));
				linearInterpParams.pNodeVelocityField = &m_velocityFieldAnimation.nodeVelocityFields[0];
				linearInterpParams.gridSpacing = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
				linearInterpParams.transformParticles = true;
				if (linearInterpParams.interpolationMethod == sbcInterpolation) {
					m_pLinearInterpolant = new SphericalBarycentricInterpolant<Vector3>(linearInterpParams);
				}
				else if (linearInterpParams.interpolationMethod == mvcInterpolation) {
					m_pLinearInterpolant = new MeanValueInterpolant3D<Vector3>(linearInterpParams);
				}
				
			}
			else {
				throw ("PrecomputedAnimation3D: LinearInterpolationMethod Node not found");
			}*/
		} else {
			m_nodeBasedVelocities = false;
		}

		if (pAnimationsNode->FirstChildElement("CollisionDetectionMethod")) {
			m_collisionDetectionMethod = loadCollisionDetectionMethod(pAnimationsNode->FirstChildElement("CollisionDetectionMethod"));
		}
		else {
			m_collisionDetectionMethod = noCollisionDetection;
		}
	}

	void PrecomputedAnimation3D::syncNodeBasedVelocities() {
		/*if (m_loadPerFrame) {
			for (int i = 0; i < m_pHexaGrid->getDimensions().x; i++) {
				for (int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
					for (int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
						Vector3 nodeVelocity = m_velocityFieldAnimation.velocityBuffers.back()(i, j, k);
						(*m_velocityFieldAnimation.nodeVelocityFields.back().pGridNodesVelocities)(i, j, k) = nodeVelocity;
					}
				}
			}
		}
		else {
			for (int t = 0; t < m_velocityFieldAnimation.totalFrames; t++) {
				for (int i = 0; i < m_pHexaGrid->getDimensions().x; i++) {
					for (int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
						for (int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
							Vector3 nodeVelocity = m_velocityFieldAnimation.velocityBuffers[t](i, j, k);
							(*m_velocityFieldAnimation.nodeVelocityFields[t].pGridNodesVelocities)(i, j, k) = nodeVelocity;
						}
					}
				}
			}
		}*/
	}
 
	PrecomputedAnimation3D::scalarFieldAnimation_t * PrecomputedAnimation3D::loadScalarField(const string &scalarFieldFile) {
		auto_ptr<ifstream> precomputedStream(new ifstream(scalarFieldFile.c_str(), ifstream::binary));

		scalarFieldAnimation_t *pScalarFieldAnimation = new scalarFieldAnimation_t();

		dimensions_t gridDimensions;

		precomputedStream->read(reinterpret_cast<char *>(&gridDimensions.x), sizeof(Scalar)*3); //xyz
		precomputedStream->read(reinterpret_cast<char *>(&pScalarFieldAnimation->timeElapsed), sizeof(Scalar));
		precomputedStream->read(reinterpret_cast<char *>(&pScalarFieldAnimation->totalFrames), sizeof(int));

		Logger::get() << "Loading scalar field file:" << scalarFieldFile << endl;
		Logger::get() << "Grid dimensions: [" << gridDimensions.x << "x" << gridDimensions.y << "x" << gridDimensions.z << "]" << endl;
		Logger::get() << "Total number of frames: " << pScalarFieldAnimation->totalFrames << endl;

		if(m_pHexaGrid->getDimensions().x != gridDimensions.x || m_pHexaGrid->getDimensions().y != gridDimensions.y || 
			m_pHexaGrid->getDimensions().z != gridDimensions.z) {
			Logger::get() << "Scalar field and loaded grid dimensions doesn't match." << endl;
			exit(-1);
		}

		pScalarFieldAnimation->pScalarFieldBuffer = new Scalar[gridDimensions.x*gridDimensions.y*gridDimensions.z*pScalarFieldAnimation->totalFrames];

		Scalar tempScalar;
		for(int t = 0; t < pScalarFieldAnimation->totalFrames; t++) {
			for(int i = 0; i < m_pHexaGrid->getDimensions().x; i++) {
				for (int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
					for (int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
						precomputedStream->read(reinterpret_cast<char *>(&tempScalar), sizeof(Scalar));
						pScalarFieldAnimation->pScalarFieldBuffer[getBufferIndex(i, j, k, t, gridDimensions)] = tempScalar;
					}
				}
			}
		}

		Logger::get() << "Scalar field file sucessfully loaded." << endl;

		return pScalarFieldAnimation;
	}

	void PrecomputedAnimation3D::loadVelocityField(const string &velocityFieldFile) {
		Logger::get() << "Loading velocity field file: " << velocityFieldFile << endl;

		auto_ptr<ifstream> precomputedStream(new ifstream(velocityFieldFile.c_str(), ifstream::binary));
		dimensions_t gridDimensions;
		if (precomputedStream->fail()) {
			Logger::getInstance()->get() << "File not found: " + velocityFieldFile << endl;
 		}
		
		if (!m_loadPerFrame) {
			m_velocityFieldAnimation.velocityBuffers.push_back(Array3D<Vector3>(m_pHexaGrid->getDimensions()));
		}
		else if (m_velocityFieldAnimation.velocityBuffers.size() == 0) {
			m_velocityFieldAnimation.velocityBuffers.push_back(Array3D<Vector3>(m_pHexaGrid->getDimensions()));
		}

		Vector3 tempVec;
		for (int i = 0; i < m_pHexaGrid->getDimensions().x; i++) {
			for (int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
				for (int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
					precomputedStream->read(reinterpret_cast<char *>(&tempVec), sizeof(Scalar)* 3);
					m_velocityFieldAnimation.velocityBuffers.back()(i, j, k) = tempVec;
				}
			}
		}
		
	}

	void PrecomputedAnimation3D::loadIthFrame(int ithFrame, const string &meshName) {
		string basePolygonMeshFile = "Flow Logs/Thin Objects Meshes/";
		basePolygonMeshFile += meshName;
		Logger::getInstance()->get() << "Loading mesh: " << ithFrame << endl;
		/*if (!m_loadPerFrame) {
			nodeVelocityField3D_t nodeVelocityField(loadMesh(basePolygonMeshFile + intToStr(ithFrame) + ".log"), m_pHexaGrid->getDimensions());
			nodeVelocityField.nodesVelocities = loadNodeVelocityField(basePolygonMeshFile + "NV" + intToStr(ithFrame) + ".log");
			m_velocityFieldAnimation.nodeVelocityFields.push_back(nodeVelocityField);
		} else {
			if (m_velocityFieldAnimation.nodeVelocityFields.size() == 0) {
				nodeVelocityField3D_t nodeVelocityField(loadMesh(basePolygonMeshFile + intToStr(ithFrame) + ".log"), m_pHexaGrid->getDimensions());
				nodeVelocityField.nodesVelocities = loadNodeVelocityField(basePolygonMeshFile + "NV" + intToStr(ithFrame) + ".log");
				m_velocityFieldAnimation.nodeVelocityFields.push_back(nodeVelocityField);
			}
			else {
				delete m_velocityFieldAnimation.nodeVelocityFields.back().pMeshes;
				m_velocityFieldAnimation.nodeVelocityFields.back().pMeshes = loadMesh(basePolygonMeshFile + intToStr(ithFrame) + ".log"), m_pHexaGrid->getDimensions();
				m_velocityFieldAnimation.nodeVelocityFields.back().nodesVelocities = loadNodeVelocityField(basePolygonMeshFile + "NV" + intToStr(ithFrame) + ".log");
			}	
		}*/
			
	}

	vector<vector<Vector3>> PrecomputedAnimation3D::loadNodeVelocityField(const string &nodeVelocityField) {
		vector<vector<Vector3>> meshNodeVelocities;
		auto_ptr<ifstream> precomputedStream(new ifstream(nodeVelocityField.c_str(), ifstream::binary));

		int numberOfTriangleMeshes;
		precomputedStream->read(reinterpret_cast<char *>(&numberOfTriangleMeshes), sizeof(int)); //xyz
		for(int i = 0; i < numberOfTriangleMeshes; i++) {
			int totalNumberOfVertices;
			precomputedStream->read(reinterpret_cast<char *>(&totalNumberOfVertices), sizeof(int)); //xyz
			vector<Vector3> currNodeVelocities;
			for(int j = 0; j < totalNumberOfVertices; j++) {
				Vector3 currVelocity;
				precomputedStream->read(reinterpret_cast<char*>(&currVelocity), sizeof(Scalar)*3);
				currNodeVelocities.push_back(currVelocity);
			}
			meshNodeVelocities.push_back(currNodeVelocities);
		}

		return meshNodeVelocities;
	}

	vector<PolygonalMesh<Vector3>> * PrecomputedAnimation3D::loadMesh(const string &polygonMesh) {
		//vector<Mesh<Vector3D>> *pMeshVec = new vector<Mesh<Vector3D>>();
		//auto_ptr<ifstream> precomputedStream(new ifstream(polygonMesh.c_str(), ifstream::binary));

		//if (precomputedStream->fail()) {
		//	Logger::getInstance()->get() << "Error loading polygon mesh" << endl;
		//	exit(1);
		//}

		//int numberOfPolygonMeshes;
		//precomputedStream->read(reinterpret_cast<char *>(&numberOfPolygonMeshes), sizeof(int)); //xyz
		//for (int i = 0; i < numberOfPolygonMeshes; i++) {
		//	bool hasTriangleMesh;
		//	precomputedStream->read(reinterpret_cast<char *>(&hasTriangleMesh), sizeof(char)); //xyz

		//	int totalNumberOfVertices;
		//	precomputedStream->read(reinterpret_cast<char *>(&totalNumberOfVertices), sizeof(int)); //xyz
		//	vector<Vector3D> currMeshPoints;
		//	for (int j = 0; j < totalNumberOfVertices; j++) {
		//		Vector3D currPoint;
		//		precomputedStream->read(reinterpret_cast<char*>(&currPoint), sizeof(DoubleScalar)* 3);
		//		currMeshPoints.push_back(currPoint);
		//	}

		//	vector<Mesh<Vector3D>::meshPolygon_t> currPolygonsVec;
		//	int totalNumberOfPolygons;
		//	precomputedStream->read(reinterpret_cast<char *>(&totalNumberOfPolygons), sizeof(int)); //xyz
		//	for (int j = 0; j < totalNumberOfPolygons; j++) {
		//		Mesh<Vector3D>::meshPolygon_t currPolygon;
		//		
		//		/**Normal */
		//		precomputedStream->read(reinterpret_cast<char*>(&currPolygon.normal), sizeof(DoubleScalar)* 3);

		//		/**Centroid */
		//		precomputedStream->read(reinterpret_cast<char*>(&currPolygon.centroid), sizeof(DoubleScalar)* 3);

		//		/** Polygon type && number of edges*/
		//		precomputedStream->read(reinterpret_cast<char *>(&currPolygon.polygonType), sizeof(int)); //xyz
		//		int numberOfEdges;
		//		precomputedStream->read(reinterpret_cast<char *>(&numberOfEdges), sizeof(int)); //xyz

		//		/** Loading edges */
		//		for (int k = 0; k < numberOfEdges; k++) {
		//			unsigned int currIndex;
		//			precomputedStream->read(reinterpret_cast<char*>(&currIndex), sizeof(unsigned int));
		//			currPolygon.edges.push_back(pair<unsigned int, unsigned int>(currIndex, currIndex));
		//		}
		//		/** Fixing edges second index */
		//		for (int k = 0; k < numberOfEdges; k++) {
		//			int nextK = roundClamp<int>(k + 1, 0, numberOfEdges);
		//			currPolygon.edges[k].second = currPolygon.edges[nextK].first;
		//		}
		//		currPolygonsVec.push_back(currPolygon);
		//	}
		//	Scalar dx = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
		//	pMeshVec->push_back(Mesh<Vector3D>(currMeshPoints, currPolygonsVec, hasTriangleMesh, dx));
		//}
		//return pMeshVec;
		return NULL;
	}

	/************************************************************************/
	/* Functionalities                                                      */
	/************************************************************************/
	void PrecomputedAnimation3D::update() {
		//m_pPhysicsCore->getParams()->timestep = 0.03;
		Scalar dt = m_pPhysicsCore->getParams()->timestep;
		dt = 1 / 30.0f;
		m_pPhysicsCore->getParams()->timestep = dt;
		if(m_pPhysicsCore->isRunningSimulation()) {
			if(m_updateTimer.started()) {
				m_updateTimer.stop();
			} 

			m_updateTimer.start();

			/*if(m_nodeBasedVelocities) {
				if(m_pParticleSystem && !m_loadPerFrame) {
					m_pParticleSystem->setNodeVelocityField(&m_velocityFieldAnimation.nodeVelocityFields[m_currentTimeStep]);
				}			
			}*/

			

			if(m_pDensityFieldAnimation) {
				for(int i = 0; i < m_pHexaGrid->getDimensions().x; i++) { 
					for(int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
						for(int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
							m_pHexaGrid->getGridData3D()->getDensityBuffer().setValueBothBuffers(
								m_pDensityFieldAnimation->pScalarFieldBuffer[getBufferIndex(i, j, k, m_currentTimeStep, m_pHexaGrid->getDimensions())], i, j, k);
						}
					}
				}
				
			}
			if(m_pPressureFieldAnimation) {
				for(int i = 0; i < m_pHexaGrid->getDimensions().x; i++) { 
					for(int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
						for(int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
							m_pHexaGrid->getGridData3D()->setPressure(
								m_pPressureFieldAnimation->pScalarFieldBuffer[getBufferIndex(i, j, k, m_currentTimeStep, m_pHexaGrid->getDimensions())], i, j, k);
						}
					}
				}
				
			}
			if(m_velocityFieldAnimation.totalFrames > 0) {
				for(int i = 0; i < m_pHexaGrid->getDimensions().x; i++) { 
					for(int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
						for(int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
							if (m_loadPerFrame) {
								m_pHexaGrid->getGridData3D()->setVelocity(m_velocityFieldAnimation.velocityBuffers[0](i, j, k), i, j, k);
							}
							else {
								m_pHexaGrid->getGridData3D()->setVelocity(m_velocityFieldAnimation.velocityBuffers[m_currentTimeStep](i, j, k), i, j, k);
							}
						}
					}
				}
				updateVorticity();
			}

			if (m_pParticleSystem) {
				updateParticleSystem(dt);
			}

			/*for (int i = 0; i < m_pPolySurfaces.size(); i++) {
				m_pPolySurfaces[i]->update(dt);
			}*/

			Scalar temp = m_currentTimeStep*dt;
			if (m_elapsedTime >= m_currentTimeStep*(1 / 30.0f)) {
				m_currentTimeStep++;
				temp = m_currentTimeStep*dt;
				cout << "Updated " << m_currentTimeStep << " " << temp << endl;
				if (m_currentTimeStep >= m_velocityFieldAnimation.totalFrames) {
					cout << "Reseting " << m_currentTimeStep << " " << temp << endl;
					m_currentTimeStep = 0;
					m_elapsedTime = 0;
					m_useMayaCache = false;
					// close the maya ncache file and exit
					closeMayaNCacheFile();
					if (m_pParticleSystem) {
						m_pParticleSystem->resetParticleSystem();
					}
					m_exportObjFiles = false;
				}
				if (m_loadPerFrame) {
					loadIthFrame(m_currentTimeStep, m_baseFileName);
					loadVelocityField("Flow logs/3D/Velocity/" + m_velocityFieldName + intToStr(m_currentTimeStep) + ".log");
					syncNodeBasedVelocities();
					//m_pLinearInterpolant->rebuildInternalCacheStructures(&m_velocityFieldAnimation.nodeVelocityFields[0]);
					/*if (m_pRenderer->getMeshRenderer())
						m_pRenderer->getMeshRenderer()->setNodeVelocityField(&m_velocityFieldAnimation.nodeVelocityFields[0]);*/
				}
				else {
					if (m_pVelocityInterpolant != NULL)
						delete m_pVelocityInterpolant;
					m_pVelocityInterpolant = new BilinearStaggeredInterpolant3D<Vector3>(m_velocityFieldAnimation.velocityBuffers[m_currentTimeStep], m_pHexaGrid->getGridData3D()->getGridSpacing());
					
					if (m_pParticleSystem)
						m_pParticleSystem->setVelocityInterpolant(m_pVelocityInterpolant);
					/*m_pLinearInterpolant->rebuildInternalCacheStructures(&m_velocityFieldAnimation.nodeVelocityFields[m_currentTimeStep]);
					if (m_pRenderer->getMeshRenderer())
						m_pRenderer->getMeshRenderer()->setNodeVelocityField(&m_velocityFieldAnimation.nodeVelocityFields[m_currentTimeStep]);*/
				}

				if (m_useMayaCache)
					logToMayaCache();

				if (m_pMayaCacheParticlesExporter)
					m_pMayaCacheParticlesExporter->dumpFrame();

				if (m_pMayaCacheVelFieldExporter)
					m_pMayaCacheVelFieldExporter->dumpFrame();

				if (m_exportObjFiles)
					exportAnimationToObj();
			}

			m_elapsedTime += dt;
		}
		

		m_pPhysicsCore->update();
		
	}
	void PrecomputedAnimation3D::updateVorticity() {
		dimensions_t gridDimensions = m_pHexaGrid->getDimensions();
		GridData3D *pGridData = m_pHexaGrid->getGridData3D();
		for(int i = 0; i < gridDimensions.x - 1; i++) {
			for(int j = 0; j < gridDimensions.y - 1; j++) {
				for(int k = 0; k < gridDimensions.z - 1; k++) {
					Vector3 vorticityVec;

					vorticityVec.x = (pGridData->getVelocity(i, j + 1, k).z - pGridData->getVelocity(i, j, k).z)
										/ pGridData->getScaleFactor(i, j, k).y;
					vorticityVec.x += (pGridData->getVelocity(i, j, k + 1).y - pGridData->getVelocity(i, j, k).y)
										/ pGridData->getScaleFactor(i, j, k).z;

					vorticityVec.y = (pGridData->getVelocity(i, j, k + 1).x - pGridData->getVelocity(i, j, k).x)
										/ pGridData->getScaleFactor(i, j, k).z;
					vorticityVec.y += (pGridData->getVelocity(i + 1, j, k).z - pGridData->getVelocity(i, j, k).z)
										/ pGridData->getScaleFactor(i, j, k).x;

					vorticityVec.z = (pGridData->getVelocity(i + 1, j, k).y - pGridData->getVelocity(i, j, k).y)
										/ pGridData->getScaleFactor(i, j, k).x;
					vorticityVec.z += (pGridData->getVelocity(i, j + 1, k).x - pGridData->getVelocity(i, j, k).x)
										/ pGridData->getScaleFactor(i, j, k).y;

					pGridData->setVorticity(vorticityVec.length(), i, j, k);
				}
			}
		}
	}

	void PrecomputedAnimation3D::draw() {
		m_pRenderer->renderLoop();
	}

	/************************************************************************/
	/* Private functionalities												*/
	/************************************************************************/
	void PrecomputedAnimation3D::updateParticleSystem(Scalar dt) {
		if(m_pParticleSystem) {
			m_pParticleSystem->update(dt);
		}	
	}

	void PrecomputedAnimation3D::staggeredToNodeCentered() {
		/*if (m_pCellToTriangleMeshMap) {
			for (int i = 0; i < m_velocityFieldAnimation.totalFrames; i++) {
				*m_velocityFieldAnimation.m_nodeVelocityFields[i].pGridNodesVelocities = *m_velocityFieldAnimation.pVelocityFieldBuffer[i];
			}
		}*/
	}

	void PrecomputedAnimation3D::initializeMayaCache() {
		// Initializing channels options, simulation parameters and saving method type
		CACHEFORMAT cachingMethod = ONEFILEPERFRAME; //ONEFILE; 
		Scalar dt = 1/30;
		int fps = round(1/dt);
		char *extras[4];	// extra parameters list
		int nExtras = 4;
		extras[0]=	"saving path";						// path to maya file where simulation was done
		extras[1]=	"maya 2014 x64";					// maya version
		extras[2]=	"100cells";							// owner
		extras[3]=	"NCache Info for nParticleShape1";	// 

		double start, end;
		start = 0;
		end = m_velocityFieldAnimation.timeElapsed;

		//init("nParticleShape1", "D://temp//test dll//cache1//ONEFILE", cachingMethod, fps, start, end, extras, nExtras);
		init("nParticleShape1", "Flow Logs/Particle Cache/NPARTICLESHAPE1", cachingMethod, m_pParticleSystem->getNumberOfParticles(), fps, start, end, extras, nExtras);
		
		enableChannel(IDCHANNEL, ENABLED);
		enableChannel(COUNTCHANNEL, ENABLED);
		//enableChannel(BIRTHTIMECHANNEL, ENABLED);
		enableChannel(POSITIONCHANNEL, ENABLED);
		//enableChannel(LIFESPANPPCHANNEL, ENABLED);
		//enableChannel(FINALLIFESPANPPCHANNEL, ENABLED);
		//enableChannel(VELOCITYCHANNEL, ENABLED);
		//enableChannel(RGBPPCHANNEL, ENABLED);
	}

	void PrecomputedAnimation3D::logToMayaCache() {
		int particleNum = m_pParticleSystem->getNumberOfParticles();
		/** Initializing ID channel */
		double *idChannel = new double[particleNum];
		//double *birthtime = new double[particleNum];
		//double *lifespanPP = new double[particleNum];
		//double *finalLifespanPP = new double[particleNum];
		double count = (double) particleNum;
		//float *color = new float[particleNum*3];
		float *position = new float[particleNum*3];
		//float *velocity = new float[particleNum*3];

		vector<Vector3> *pParticles = m_pParticleSystem->getParticlePositionsVectorPtr();
		const Vector3 *pVelocities = m_pParticleSystem->getParticleVelocitiesPtr();

		for(int i = 0; i < particleNum; i++) {
			if (m_pParticleSystem->getRenderingParams().pParticlesTags) {
				idChannel[i] = m_pParticleSystem->getRenderingParams().pParticlesTags[i];
			}
			else {
				idChannel[i] = 0;
			}
			

			//color[i * 3] = 1.0f;
			//color[i * 3 + 1] = 0.0f;
			//color[i * 3 + 2] = 0.0f;

			Vector3 currPosition = pParticles->at(i);
			position[i*3] = currPosition.x;
			position[i*3 + 1] = currPosition.y;
			position[i*3 + 2] = currPosition.z;

			//Vector3 currVelocity = pVelocities[i];
			//velocity[i*3] = currVelocity.x;
			//velocity[i*3 + 1] = currVelocity.y;
			//velocity[i*3 + 2] = currVelocity.z;
		}

		assignChannelValues(COUNTCHANNEL, &count);
		assignChannelValues(IDCHANNEL, idChannel);
		assignChannelValues(POSITIONCHANNEL, position);
		//assignChannelValues(VELOCITYCHANNEL, velocity);
		//assignChannelValues(BIRTHTIMECHANNEL,birthtime);
		//assignChannelValues(LIFESPANPPCHANNEL,lifespanPP);
		//assignChannelValues(FINALLIFESPANPPCHANNEL,finalLifespanPP);
		//assignChannelValues(RGBPPCHANNEL,color);

		mayaCache();

		delete idChannel;
		//delete birthtime;
		//delete lifespanPP;
		//delete finalLifespanPP;
		//delete color;
		delete position;
		//delete velocity;
	}


	void PrecomputedAnimation3D::exportAnimationToObj() {
		/*Logger::getInstance()->get() << "Exporting obj meshes" << endl;
		for (int i = 0; i < m_pPolySurfaces.size(); i++) {
			string objFilename("Flow Logs/Objects Animation/" + m_pPolySurfaces[i]->getName() + intToStr(i) + "f" + intToStr(m_currentTimeStep) + ".obj");
			Logger::getInstance()->get() << "Exporting " << objFilename << endl;
			auto_ptr<ofstream> fileStream(new ofstream(objFilename.c_str()));
			(*fileStream) << "# Vertices" << endl;
			for (int j = 0; j < m_pPolySurfaces[i]->getVertices().size(); j++) {
				(*fileStream)	<< "v ";
				(*fileStream)	<< m_pPolySurfaces[i]->getVertices()[j].x << " " 
								<< m_pPolySurfaces[i]->getVertices()[j].y << " " 
								<< m_pPolySurfaces[i]->getVertices()[j].z << endl;
			}

			(*fileStream) << "# Normals" << endl;
			for (int j = 0; j < m_pPolySurfaces[i]->getNormals().size(); j++) {
				(*fileStream)	<< "vn ";
				(*fileStream)	<< m_pPolySurfaces[i]->getNormals()[j].x << " " 
								<< m_pPolySurfaces[i]->getNormals()[j].y << " " 
								<< m_pPolySurfaces[i]->getNormals()[j].z << endl;
			}

			(*fileStream) << "# Polygonal faces" << endl;
			for (int j = 0; j < m_pPolySurfaces[i]->getFaces().size(); j++) {
				(*fileStream)	<< "f ";
				simpleFace_t currFace = m_pPolySurfaces[i]->getFaces()[j];
				for (int k = 0; k < currFace.edges.size(); k++) {
					(*fileStream) << currFace.edges[k].first + 1 << " ";
				}	
				(*fileStream) << endl;
			}
			fileStream->close();
		}*/
	}

	void PrecomputedAnimation3D::exportAllMeshesToObj() {
		/*Logger::getInstance()->get() << "Exporting all meshes to an obj file" << endl;
		vector<Mesh<Vector3D>> *pCurrMeshes = m_velocityFieldAnimation.nodeVelocityFields.front().pMeshes;
		string objFilename("Flow Logs/Thin Objects Meshes/currAllMesh.obj");
		auto_ptr<ofstream> fileStream(new ofstream(objFilename.c_str()));
		Scalar dx = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;

		(*fileStream) << "# Vertices" << endl;
		for (int i = 0; i < pCurrMeshes->size(); i++) {
			for (int j = 0; j < pCurrMeshes->at(i).getPoints().size(); j++) {
				(*fileStream)	<< "v ";
				(*fileStream)	<< pCurrMeshes->at(i).getPoints()[j].x << " "
								<< pCurrMeshes->at(i).getPoints()[j].y << " "
								<< pCurrMeshes->at(i).getPoints()[j].z << endl;
			
			}
		}*/

		/*(*fileStream) << "# Normals" << endl;
		for (int i = 0; i < pCurrMeshes->size(); i++) {
			for (int j = 0; j < pCurrMeshes->at(i).getPointsNormals().size(); j++) {
				(*fileStream)	<< "vn ";
				(*fileStream)	<< pCurrMeshes->at(i).getPointsNormals()[j].x << " "
								<< pCurrMeshes->at(i).getPointsNormals()[j].y << " "
								<< pCurrMeshes->at(i).getPointsNormals()[j].z << endl;

			}
		}*/

		//(*fileStream) << "# Polygonal faces" << endl;
		//int currFacePadding = 0;
		//for (int i = 0; i < pCurrMeshes->size(); i++) {
		//	for (int j = 0; j < pCurrMeshes->at(i).getMeshPolygons().size(); j++) {
		//		if (pCurrMeshes->at(i).hasTriangleMesh()) {
		//			bool exportToMesh = false;
		//			Mesh<Vector3D>::meshPolygon_t currFace = pCurrMeshes->at(i).getMeshPolygons()[j];
		//			for (int k = 0; k < currFace.edges.size(); k++) {
		//				if (!isOnGridPoint(pCurrMeshes->at(i).getPoints()[currFace.edges[k].first], dx) && currFace.polygonType != Mesh<Vector3D>::geometryPolygon) {
		//					exportToMesh = true;
		//					break;
		//				}
		//			}
		//			//exportToMesh = true;
		//			if (exportToMesh) {
		//				(*fileStream) << "f ";
		//				for (int k = 0; k < currFace.edges.size(); k++) {
		//					(*fileStream) << currFace.edges[k].first + 1 + currFacePadding << " ";
		//				}
		//				(*fileStream) << endl;
		//			}
		//		}
		//	}
		//	currFacePadding += pCurrMeshes->at(i).getPoints().size();
		//}
		//fileStream->close();
	}
}