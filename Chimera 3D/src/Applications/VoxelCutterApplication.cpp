#include "Applications/VoxelCutterApplication.h"

namespace Chimera {
	#pragma region Constructors
	VoxelCutterApplication::VoxelCutterApplication(int argc, char** argv, TiXmlElement *pChimeraConfig) : Application3D(argc, argv, pChimeraConfig) {
		m_pPolygonSurface = NULL;
		m_pCuttingPlane = NULL;
		m_pHexaGrid = NULL;
		m_rotatePlane = true;
		initializeGL(argc, argv);

		loadGridFile();

		/** Rendering and windows initialization */
		{
			/** Rendering initialization */
			m_pRenderer = (GLRenderer3D::getInstance());
			m_pRenderer->initialize(1280, 800);
		}

		if (m_pMainNode->FirstChildElement("Camera")) {
			setupCamera(m_pMainNode->FirstChildElement("Camera"));
		}

		if (TwGetLastError())
			Logger::getInstance()->log(string(TwGetLastError()), Log_HighPriority);


		if (m_pMainNode->FirstChildElement("Objects")) {
			m_pSceneLoader = new SceneLoader(NULL, m_pRenderer);
			m_pSceneLoader->loadScene(m_pMainNode->FirstChildElement("Objects"), false);
		}

		for (int i = 0; i < m_pSceneLoader->getPolygonSurfaces().size(); i++) {
			m_pRenderer->addObject(m_pSceneLoader->getPolygonSurfaces()[i]);
		}

		m_pRenderer->getCamera()->setRotationAroundGridMode(Vector3(0, 0, 0));
		
		m_pTriangleRefs = new Array3D<vector<int>>(m_pHexaGrid->getDimensions());

		vector<simpleFace_t> polygonFaces = m_pPolygonSurface->getFaces();
		vector<Vector3D> polygonVertices = m_pPolygonSurface->getVertices();
		Scalar dx = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;

		for (int i = 0; i < polygonFaces.size(); i++){
			for (int j = 0; j < polygonFaces[i].edges.size(); j++) {
				Vector3D currVertex = polygonVertices[polygonFaces[i].edges[j].first];
				dimensions_t pointLocation(currVertex.x / dx, currVertex.y / dx, currVertex.z /dx);
				(*m_pTriangleRefs)(pointLocation).push_back(i);
			}
		}

		for (int i = 0; i < m_pHexaGrid->getDimensions().x; i++) {
			for (int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
				for (int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
					vector<int> *pTriangleRefVec = &(*m_pTriangleRefs)(i, j, k);
					if (pTriangleRefVec->size() > 0) {
						sort(pTriangleRefVec->begin(), pTriangleRefVec->end());
					}

					for (int l = 0; l < pTriangleRefVec->size() - 1; l) {
						int numVerticesInsideVoxel = 0;
						while (pTriangleRefVec->at(l) == pTriangleRefVec->at(l + 1)) {
							if (l + 1 < pTriangleRefVec->size() - 1) {
								l++;
								numVerticesInsideVoxel++;
							} else{
								break;
							}
						}

						//Break the triangle here
						if (numVerticesInsideVoxel < 3) {
							
						}
					}
				}
			}
		}
	}

	void VoxelCutterApplication::loadGridFile() {
		string tempValue;
		dimensions_t tempDimensions;
		try {	/** Load grids and boundary conditions: try-catchs for invalid file exceptions */
			SimulationConfig<Vector3> *pSimCfg = new SimulationConfig<Vector3>();

			if (m_pMainNode->FirstChildElement("GridFile") == NULL) {
				m_pHexaGrid = loadGrid(m_pMainNode->FirstChildElement("Grid"));
			}
			else {
				/** Load grid */
				tempValue = m_pMainNode->FirstChildElement("GridFile")->GetText();

				shared_ptr<TiXmlAttribute> pGridAttribs(m_pMainNode->FirstChildElement("GridFile")->FirstAttribute());
				if (pGridAttribs) {
					if (string(pGridAttribs->Name()) == string("periodic")) {
						string bcType(pGridAttribs->Value());
						if (bcType == "true") {
							pSimCfg->setPeriodicGrid(true);
						}
					}
				}

				m_pHexaGrid = new HexaGrid(tempValue, pSimCfg->isPeriodicGrid());
			}
			tempDimensions = m_pHexaGrid->getDimensions();
			pSimCfg->setGrid(m_pHexaGrid);

			m_pMainSimCfg = pSimCfg;
		}
		catch (exception e) {
			Logger::get() << e.what() << endl;
			exit(1);
		}
	}
	#pragma endregion
	#pragma region Functionalities
	void VoxelCutterApplication::update() {

	}

	void VoxelCutterApplication::draw() {
		m_pRenderer->renderLoop();
	}
	#pragma endregion


	#pragma region Callbacks
	void VoxelCutterApplication::keyboardCallback(unsigned char key, int x, int y) {
		m_pRenderer->keyboardCallback(key, x, y);
		switch (key) {
			case 'p':
			case 'P':
				m_rotatePlane = true;
			break;
		}
	}

	void VoxelCutterApplication::keyboardUpCallback(unsigned char key, int x, int y) {
		switch (key) {
			case 'p':
			case 'P':
			m_rotatePlane = false;
			break;
		}
	}

	void VoxelCutterApplication::specialKeyboardCallback(int key, int x, int y) {

	}

	void VoxelCutterApplication::specialKeyboardUpCallback(int key, int x, int y) {

	}
	void VoxelCutterApplication::motionCallback(int x, int y) {
		Application3D::motionCallback(x, y);
		
	}
	#pragma endregion

	#pragma region LoadingFunctions
	void VoxelCutterApplication::setupCamera(TiXmlElement *pCameraNode) {

		Vector3 camPosition;
		if (pCameraNode->FirstChildElement("Position")) {
			pCameraNode->FirstChildElement("Position")->QueryFloatAttribute("px", &camPosition.x);
			pCameraNode->FirstChildElement("Position")->QueryFloatAttribute("py", &camPosition.y);
			pCameraNode->FirstChildElement("Position")->QueryFloatAttribute("pz", &camPosition.z);
			GLRenderer3D::getInstance()->getCamera()->setPosition(camPosition);
		}
		if (pCameraNode->FirstChildElement("Direction")) {
			Vector3 camDirection;
			pCameraNode->FirstChildElement("Direction")->QueryFloatAttribute("px", &camDirection.x);
			pCameraNode->FirstChildElement("Direction")->QueryFloatAttribute("py", &camDirection.y);
			pCameraNode->FirstChildElement("Direction")->QueryFloatAttribute("pz", &camDirection.z);
			GLRenderer3D::getInstance()->getCamera()->setDirection(camDirection);
			GLRenderer3D::getInstance()->getCamera()->setFixedDirection(true);
		}

	}
	#pragma endregion
}