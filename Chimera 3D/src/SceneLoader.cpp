#include "SceneLoader.h"


namespace Chimera {

	SceneLoader::SceneLoader(HexaGrid *pHexaGrid, GLRenderer3D *pRenderer) {
		m_pRenderer = pRenderer;
 		m_pHexaGrid = pHexaGrid;
		m_initializePhysics = true;
		m_pParticleSystem = NULL;
	}

	/*PhysicalObject<Vector3>::velocityFunction_t SceneLoader::loadVelocityFunction(TiXmlElement *pVelocityNode) {
		PhysicalObject<Vector3>::velocityFunction_t velocityFunction;
		if (pVelocityNode->FirstChildElement("SinFunction")) {
			pVelocityNode->FirstChildElement("SinFunction")->QueryFloatAttribute("amplitude", &velocityFunction.amplitude);
			pVelocityNode->FirstChildElement("SinFunction")->QueryFloatAttribute("frequency", &velocityFunction.frequency);
			velocityFunction.velocityFunctionType = velocityFunctionType_t::sine;
		}
		else if (pVelocityNode->FirstChildElement("CosineFunction")) {
			pVelocityNode->FirstChildElement("CosineFunction")->QueryFloatAttribute("amplitude", &velocityFunction.amplitude);
			pVelocityNode->FirstChildElement("CosineFunction")->QueryFloatAttribute("frequency", &velocityFunction.frequency);
			velocityFunction.velocityFunctionType = velocityFunctionType_t::cosine;
		}
		else if (pVelocityNode->FirstChildElement("UniformFunction")) {
			pVelocityNode->FirstChildElement("UniformFunction")->QueryFloatAttribute("amplitude", &velocityFunction.amplitude);
			velocityFunction.velocityFunctionType = velocityFunctionType_t::uniform;
		}
		if (pVelocityNode->FirstChildElement("Direction")) {
			pVelocityNode->FirstChildElement("Direction")->QueryFloatAttribute("x", &velocityFunction.direction.x);
			pVelocityNode->FirstChildElement("Direction")->QueryFloatAttribute("y", &velocityFunction.direction.y);
			pVelocityNode->FirstChildElement("Direction")->QueryFloatAttribute("z", &velocityFunction.direction.z);
		}
		if (pVelocityNode->FirstChildElement("StartingTime")) {
			velocityFunction.startingTime = atof(pVelocityNode->FirstChildElement("StartingTime")->GetText());
		}
		if (pVelocityNode->FirstChildElement("EndingTime")) {
			velocityFunction.endingTime = atof(pVelocityNode->FirstChildElement("EndingTime")->GetText());
		}
		velocityFunction.absoluteValuesOnly = false;
		if (pVelocityNode->FirstChildElement("AbsoluteValues")) {
			string absoluteValueStr = pVelocityNode->FirstChildElement("AbsoluteValues")->GetText();
			if (absoluteValueStr == "true") {
				velocityFunction.absoluteValuesOnly = true;
			}
		}
		return velocityFunction;
	}

	PhysicalObject<Vector3>::rotationFunction_t SceneLoader::loadRotationFunction(TiXmlElement *pRotationNode) {
		PhysicalObject<Vector3>::rotationFunction_t rotationFunction;
		
		if (pRotationNode->FirstChildElement("InitialAngle")) {
			rotationFunction.initialRotation = atof(pRotationNode->FirstChildElement("InitialAngle")->GetText());
			rotationFunction.initialRotation = DEG2RAD*rotationFunction.initialRotation;
		}
		if (pRotationNode->FirstChildElement("Axis")) {
			pRotationNode->FirstChildElement("Axis")->QueryFloatAttribute("x", &rotationFunction.axis.x);
			pRotationNode->FirstChildElement("Axis")->QueryFloatAttribute("y", &rotationFunction.axis.y);
			pRotationNode->FirstChildElement("Axis")->QueryFloatAttribute("z", &rotationFunction.axis.z);
		}
		if (pRotationNode->FirstChildElement("Speed")) {
			rotationFunction.speed = atof(pRotationNode->FirstChildElement("Speed")->GetText());
		}
		if (pRotationNode->FirstChildElement("Acceleration")) {
			rotationFunction.acceleration = atof(pRotationNode->FirstChildElement("Acceleration")->GetText());
		}
		if (pRotationNode->FirstChildElement("StartingTime")) {
			rotationFunction.startingTime = atof(pRotationNode->FirstChildElement("StartingTime")->GetText());
		}
		if (pRotationNode->FirstChildElement("EndingTime")) {
			rotationFunction.endingTime = atof(pRotationNode->FirstChildElement("EndingTime")->GetText());
		}
		return rotationFunction;
	}*/

	/*void SceneLoader::loadPlane(TiXmlElement *pPlaneNode) {
		Vector3 planePosition;
		pPlaneNode->FirstChildElement("Position")->QueryFloatAttribute("x", &planePosition.x);
		pPlaneNode->FirstChildElement("Position")->QueryFloatAttribute("y", &planePosition.y);
		pPlaneNode->FirstChildElement("Position")->QueryFloatAttribute("z", &planePosition.z);
		Vector3 upVec;
		pPlaneNode->FirstChildElement("UpVector")->QueryFloatAttribute("x", &upVec.x);
		pPlaneNode->FirstChildElement("UpVector")->QueryFloatAttribute("y", &upVec.y);
		pPlaneNode->FirstChildElement("UpVector")->QueryFloatAttribute("z", &upVec.z);
		Vector2 planeSize;
		pPlaneNode->FirstChildElement("PlaneSize")->QueryFloatAttribute("x", &planeSize.x);
		pPlaneNode->FirstChildElement("PlaneSize")->QueryFloatAttribute("y", &planeSize.y);
		Scalar planeTiling = atof(pPlaneNode->FirstChildElement("Scale")->GetText());
		Plane *pPlane = new Plane(planePosition, upVec, planeSize, planeTiling);

		velocityFunction_t &rVelocityFunction = pPlane->getVelocityFunction();
		TiXmlElement *pVelocityNode = pPlaneNode->FirstChildElement("VelocityFunction"); 
		if (pVelocityNode) {
			pPlane->setVelocityFunction(loadVelocityFunction(pVelocityNode));
		}
		m_planes.push_back(pPlane);
	}*/

	void SceneLoader::loadDensityField(TiXmlElement *pDensityFieldNode) {
		Scalar dx = m_pHexaGrid->getGridData3D()->getGridSpacing();
		if(pDensityFieldNode->FirstChildElement("Cube")) {
			TiXmlElement *pRecNode = pDensityFieldNode->FirstChildElement("Cube");
			Vector3 recPosition, recSize;
			pRecNode->QueryFloatAttribute("px", &recPosition.x);
			pRecNode->QueryFloatAttribute("py", &recPosition.y);
			pRecNode->QueryFloatAttribute("pz", &recPosition.z);
			pRecNode->QueryFloatAttribute("sx", &recSize.x);
			pRecNode->QueryFloatAttribute("sy", &recSize.y);
			pRecNode->QueryFloatAttribute("sz", &recSize.z);

			Vector3 lowerBound, upperBound;
			lowerBound = recPosition/ dx;
			upperBound = lowerBound + recSize/ dx;
			
			for(int i = lowerBound.x; i < upperBound.x; i++) {
				for(int j = lowerBound.y; j < upperBound.y; j++) {
					for(int k = lowerBound.z; k < upperBound.z; k++) {
						m_pHexaGrid->getGridData3D()->getDensityBuffer().setValueBothBuffers(1, i, j, k);
					}
				}
			}	
			FlowSolver<Vector3, Array3D>::scalarFieldMarker_t scalarFieldMarker;
			scalarFieldMarker.position = recPosition;
			scalarFieldMarker.size = recSize;
			scalarFieldMarker.value = 1;
			m_scalarFieldMarkers.push_back(scalarFieldMarker);
		}
	}


	ParticleSystem3D::emitter_t SceneLoader::loadEmitter(TiXmlElement *pEmitterNode) {
		ParticleSystem3D::emitter_t emitter;
		if (pEmitterNode->FirstChildElement("Rectangle")) {
			pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("px", &emitter.position.x);
			pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("py", &emitter.position.y);
			pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("pz", &emitter.position.z);
			pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("sx", &emitter.emitterSize.x);
			pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("sy", &emitter.emitterSize.y);
			pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("sz", &emitter.emitterSize.z);
			emitter.emitterType = ParticleSystem3D::cubeRandomEmitter;
		}
		else if (pEmitterNode->FirstChildElement("Sphere")) {
			pEmitterNode->FirstChildElement("Sphere")->QueryFloatAttribute("px", &emitter.position.x);
			pEmitterNode->FirstChildElement("Sphere")->QueryFloatAttribute("py", &emitter.position.y);
			pEmitterNode->FirstChildElement("Sphere")->QueryFloatAttribute("pz", &emitter.position.z);
			pEmitterNode->FirstChildElement("Sphere")->QueryFloatAttribute("radius", &emitter.emitterSize.x);
			emitter.emitterType = ParticleSystem3D::sphereRandomEmitter;
		}
		if (pEmitterNode->FirstChildElement("MaxAmount")) {
			emitter.maxNumberOfParticles = atof(pEmitterNode->FirstChildElement("MaxAmount")->GetText());
		}
		if (pEmitterNode->FirstChildElement("InitialAmount")) {
			emitter.initialNumberOfParticles = atof(pEmitterNode->FirstChildElement("InitialAmount")->GetText());
		}
		if (pEmitterNode->FirstChildElement("SpawnRatio")) {
			emitter.spawnRatio = atof(pEmitterNode->FirstChildElement("SpawnRatio")->GetText());
		}
		if (pEmitterNode->FirstChildElement("LifeTime")) {
			emitter.particlesLife = atof(pEmitterNode->FirstChildElement("LifeTime")->GetText());
		}
		if (pEmitterNode->FirstChildElement("LifeVariance")) {
			emitter.particlesLifeVariance = atof(pEmitterNode->FirstChildElement("LifeVariance")->GetText());
		}
		emitter.totalSpawnedParticles = 0;
		return emitter;
	}
	

	void SceneLoader::loadParticleSystem(TiXmlElement *pParticleSystemNode) {
		ParticleSystem3D::configParams_t m_particleSystemParams;
		TiXmlElement *pEmitterNode = pParticleSystemNode->FirstChildElement("Emitter");
		while (pEmitterNode) {
			m_particleSystemParams.emitters.push_back(loadEmitter(pEmitterNode));
			pEmitterNode = pEmitterNode->NextSiblingElement("Emitter");
		}
		if(pParticleSystemNode->FirstChildElement("MinBounds")) {
			pParticleSystemNode->FirstChildElement("MinBounds")->QueryFloatAttribute("px", &m_particleSystemParams.particlesMinBounds.x);
			pParticleSystemNode->FirstChildElement("MinBounds")->QueryFloatAttribute("py", &m_particleSystemParams.particlesMinBounds.y);
			pParticleSystemNode->FirstChildElement("MinBounds")->QueryFloatAttribute("pz", &m_particleSystemParams.particlesMinBounds.z);
		}
		if(pParticleSystemNode->FirstChildElement("MaxBounds")) {
			pParticleSystemNode->FirstChildElement("MaxBounds")->QueryFloatAttribute("px", &m_particleSystemParams.particlesMaxBounds.x);
			pParticleSystemNode->FirstChildElement("MaxBounds")->QueryFloatAttribute("py", &m_particleSystemParams.particlesMaxBounds.y);
			pParticleSystemNode->FirstChildElement("MaxBounds")->QueryFloatAttribute("pz", &m_particleSystemParams.particlesMaxBounds.z);
		}

		ParticleSystem3D::renderingParams_t renderingParams;
		renderingParams.colorScheme = singleColor;
		if(pParticleSystemNode->FirstChildElement("RenderingProperties")) {
			TiXmlElement *pRenderingProperties = pParticleSystemNode->FirstChildElement("RenderingProperties");
			if(pRenderingProperties->FirstChildElement("ColorScheme")) {
				string coloringScheme = pRenderingProperties->FirstChildElement("ColorScheme")->GetText();
				if(coloringScheme == "jet") {
					renderingParams.colorScheme = jet;
				}
				else if (coloringScheme == "grayscale") {
					renderingParams.colorScheme = grayscale;
				}
				else if (coloringScheme == "random") {
					renderingParams.colorScheme = randomColors;
				}
			} else {
				renderingParams.colorScheme = singleColor;
			}

			if (pRenderingProperties->FirstChildElement("VisualizeVectors")) {
				string visualizeVectors = pRenderingProperties->FirstChildElement("VisualizeVectors")->GetText();
				if (visualizeVectors == "true") {
					renderingParams.visualizeNormalsVelocities = true;
				}
			}
		}

		m_pParticleSystem = new ParticleSystem3D(m_particleSystemParams, renderingParams, m_pHexaGrid);
		

		//Just once
		m_pParticleSystem->setGridOrigin(m_pHexaGrid->getGridOrigin());
	}

	void SceneLoader::loadMesh(TiXmlElement *pCgalObjMesh) {
		if (pCgalObjMesh->FirstChildElement("Filename")) {
			string cgalObjMeshFile = "Geometry/3D/";
			string cgalObjShortFile = pCgalObjMesh->FirstChildElement("Filename")->GetText();
			string objectName;
			size_t tempFound = cgalObjShortFile.rfind("/");
			if (tempFound != string::npos) {
				string tempStr = cgalObjShortFile.substr(tempFound, cgalObjShortFile.length() - tempFound);
				objectName = tempStr.substr(1, tempStr.rfind(".") - 1);
			}
			else {
				objectName = cgalObjShortFile.substr(0, cgalObjShortFile.rfind(".") - 1);
			}
			cgalObjMeshFile += cgalObjShortFile;
			Vector3 scale(1, 1, 1);
			Vector3 position(0, 0, 0);
			if (pCgalObjMesh->FirstChildElement("Scale")) {
				pCgalObjMesh->FirstChildElement("Scale")->QueryFloatAttribute("x", &scale.x);
				pCgalObjMesh->FirstChildElement("Scale")->QueryFloatAttribute("y", &scale.y);
				pCgalObjMesh->FirstChildElement("Scale")->QueryFloatAttribute("z", &scale.z);
			}
			if (pCgalObjMesh->FirstChildElement("position")) {
				pCgalObjMesh->FirstChildElement("position")->QueryFloatAttribute("x", &position.x);
				pCgalObjMesh->FirstChildElement("position")->QueryFloatAttribute("y", &position.y);
				pCgalObjMesh->FirstChildElement("position")->QueryFloatAttribute("z", &position.z);
			}
			objectName += intToStr(m_meshes.size());
			if (m_pHexaGrid) {
				Scalar dx = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
				m_meshes.push_back(new PolygonalMesh<Vector3>(position, cgalObjMeshFile, m_pHexaGrid->getDimensions(), dx));
				m_meshes.back()->setName(objectName);
			}
			else {
				Scalar dx = 1;
				m_meshes.push_back(new PolygonalMesh<Vector3>(position, cgalObjMeshFile, dimensions_t(8, 8, 8), 1));
				m_meshes.back()->setName(objectName);
			}

			

			TiXmlElement *pVelocityNode = pCgalObjMesh->FirstChildElement("VelocityFunction");
			/*if (pVelocityNode) {
				m_meshes.back()->setVelocityFunction(loadVelocityFunction(pVelocityNode));
			}*/

			/*TiXmlElement *pRotationNode = pCgalObjMesh->FirstChildElement("RotationFunction");
			if (pRotationNode) {
				m_pPolygonSurfaces.back()->setRotationFunction(loadRotationFunction(pRotationNode));
				m_pPolygonSurfaces.back()->updateInitialRotation();
			}*/
	
		}
	}

	void SceneLoader::loadScene(TiXmlElement *pObjectsNode, bool initiliazePhysics) {
		TiXmlElement *pTempNode;
		m_initializePhysics = initiliazePhysics;
		pTempNode = pObjectsNode->FirstChildElement();
		while(pTempNode != NULL) {
			if(string(pTempNode->Value()) == "Plane") {
				//loadPlane(pTempNode);
			} else if(string(pTempNode->Value()) == "DensityField") {
				//loadDensityField(pTempNode);
			} else if(string(pTempNode->Value()) == "ParticleSystem") {
				loadParticleSystem(pTempNode);
			}
			else if (string(pTempNode->Value()) == "Mesh") {
				loadMesh(pTempNode);
			}
			pTempNode = pTempNode->NextSiblingElement();
		}
	}
}
