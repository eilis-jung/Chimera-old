#include "Applications/MeshCutterApplication.h"

namespace Chimera {
	#pragma region Constructors
	MeshCutterApplication::MeshCutterApplication(int argc, char** argv, TiXmlElement *pChimeraConfig) : Application3D(argc, argv, pChimeraConfig) {
		//m_pPolygonSurface = NULL;
		//m_pCuttingPlane = NULL;
		m_rotatePlane = true;
		m_pGrid = nullptr;
		initializeGL(argc, argv);

		m_selectedFace = 0;

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
			
			if (m_pMainNode->FirstChildElement("Objects")->FirstChildElement("Grid")) {
				m_pGrid = loadGrid(m_pMainNode->FirstChildElement("Objects")->FirstChildElement("Grid"));
			}
			m_pSceneLoader->setHexaGrid(m_pGrid);
			m_pSceneLoader->loadScene(m_pMainNode->FirstChildElement("Objects"), false);
		}

		m_pRenderer->addMeshRenderer(m_pSceneLoader->getMeshes());
		m_pRenderer->getMeshRenderer()->setDrawingVertices(true);

		/** Setting up camera rotation */
		PolygonalMesh<Vector3> *pPolygonMesh = dynamic_cast<PolygonalMesh<Vector3> *>(m_pSceneLoader->getMeshes()[0]);
		m_pRenderer->getCamera()->setRotationAroundGridMode(pPolygonMesh->getCentroid());
		
		/** Initializing Cut Voxels */
		vector<PolygonalMesh<Vector3> *> polygonMeshes;
		polygonMeshes.push_back(pPolygonMesh);
		CutVoxels3D<Vector3> *pCutVoxels3D;
		Core::Timer cutVoxelsTime;
		cutVoxelsTime.start();
		if (m_pGrid)
			pCutVoxels3D = new CutCells::CutVoxels3D<Vector3>(polygonMeshes, m_pGrid->getGridData3D()->getGridSpacing(), m_pGrid->getGridData3D()->getDimensions());
		else
			throw(exception("MeshCutterApplication: grid node and grid not found"));
		cutVoxelsTime.stop();
		cout << "Time for cut-cells generation: " << cutVoxelsTime.secondsElapsed() << endl;

		m_pRenderer->setCutVoxels(pCutVoxels3D);
	}
	#pragma endregion
	#pragma region Functionalities
	void MeshCutterApplication::update() {
		m_selectedFace = clamp<uint>(m_selectedFace, 0, m_totalNumberOfFaces - 1);
		uint tempSelectedFace = m_selectedFace;
		for (int i = 0; i < m_perCutCellSize.size(); i++) {
			if (tempSelectedFace >= m_perCutCellSize[i]) {
				tempSelectedFace -= m_perCutCellSize[i];
				m_pRenderer->getPolygonMeshRenderers()[i]->setSelectedCutCell(-1);
			}
			else {
				m_pRenderer->getPolygonMeshRenderers()[i]->setSelectedCutCell(tempSelectedFace);
				break;
			}
		}
	}

	void MeshCutterApplication::draw() {
		m_pRenderer->renderLoop();
	}
	#pragma endregion


	#pragma region Callbacks
	void MeshCutterApplication::keyboardCallback(unsigned char key, int x, int y) {
		m_pRenderer->keyboardCallback(key, x, y);
		switch (key) {
			case 'r':
			case 'R':
				m_pRenderer->getLineMeshRenderers()[0]->setDrawingVertices(!m_pRenderer->getLineMeshRenderers()[0]->isDrawingVertices());
			break;

			case 'w':
			case 'W':
				m_pRenderer->getLineMeshRenderers()[1]->setDrawingVertices(!m_pRenderer->getLineMeshRenderers()[1]->isDrawingVertices());
			break;

			case 'e':
			case 'E':
				m_pRenderer->getLineMeshRenderers()[2]->setDrawingVertices(!m_pRenderer->getLineMeshRenderers()[2]->isDrawingVertices());
			break;

			case '1':
				m_pRenderer->getLineMeshRenderers()[0]->setDrawing(!m_pRenderer->getLineMeshRenderers()[0]->isDrawing());
			break;

			case '2':
				m_pRenderer->getLineMeshRenderers()[1]->setDrawing(!m_pRenderer->getLineMeshRenderers()[1]->isDrawing());
			break;

			case '3':
				m_pRenderer->getLineMeshRenderers()[2]->setDrawing(!m_pRenderer->getLineMeshRenderers()[2]->isDrawing());
			break;

			case 'v':
			case 'V':
				m_pRenderer->getMeshRenderer()->setDrawing(!m_pRenderer->getMeshRenderer()->isDrawing());
			break;

			case '+':
				m_selectedFace += 1;
			break;
			case '-':
				m_selectedFace -= 1;
			break;

			case 'o':
			case 'O':
				for (int i = 0; i < m_pRenderer->getPolygonMeshRenderers().size(); i++) {
					m_pRenderer->getPolygonMeshRenderers()[i]->setDrawingCutCells(!m_pRenderer->getPolygonMeshRenderers()[i]->isDrawingCutCells());
				}
			break;

			case 'p':
			case 'P':
				m_pRenderer->getMeshRenderer()->setDrawing(!m_pRenderer->getMeshRenderer()->isDrawing());
			break;
		}
	}

	void MeshCutterApplication::keyboardUpCallback(unsigned char key, int x, int y) {
		switch (key) {
			case 'p':
			case 'P':
			m_rotatePlane = false;
			break;
		}
	}

	void MeshCutterApplication::specialKeyboardCallback(int key, int x, int y) {

	}

	void MeshCutterApplication::specialKeyboardUpCallback(int key, int x, int y) {

	}
	void MeshCutterApplication::motionCallback(int x, int y) {
		Application3D::motionCallback(x, y);
		
	}
	#pragma endregion

	#pragma region LoadingFunctions
	void MeshCutterApplication::setupCamera(TiXmlElement *pCameraNode) {

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