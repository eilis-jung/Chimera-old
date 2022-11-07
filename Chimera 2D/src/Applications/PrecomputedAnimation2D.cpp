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

#include "Applications/PrecomputedAnimation2D.h"


namespace Chimera {

	PrecomputedAnimation2D::PrecomputedAnimation2D(int argc, char** argv, TiXmlElement *pChimeraConfig) : Application2D(argc, argv, pChimeraConfig) {
		/************************************************************************/
		/* TO-be initialized vars                                               */
		/************************************************************************/
		m_pParticleSystem = NULL;
		m_pDensityFieldAnimation = NULL;
		m_pPressureFieldAnimation = NULL;
		m_pThinObjectAnimation = NULL;
		m_pVelocityFieldAnimation = NULL;
		m_pDoubleGyreConfig = NULL;
		m_pFineGridConfig = NULL;
		m_elapsedTime = 0.0f;
		m_currentTimeStep = 0;
		m_animationVelocity = 1.0f;

		//Rotational and translational constants

		initializeGL(argc, argv);
		loadGridFile();
		loadPhysics();

		/** Rendering and windows initialization */
		{
			/** Rendering initialization */
			m_pRenderer = (GLRenderer2D::getInstance());
			m_pRenderer->initialize(1280, 800);
			m_pRenderer->addGridVisualizationWindow(m_pQuadGrid);
		}
	
		if(m_pMainNode->FirstChildElement("Camera")) {
			setupCamera(m_pMainNode->FirstChildElement("Camera"));
		}

		if(TwGetLastError())
			Logger::getInstance()->log(string(TwGetLastError()), Log_HighPriority);


		if(m_pMainNode->FirstChildElement("Animations")) {
			loadAnimations(m_pMainNode->FirstChildElement("Animations"));
		}

		if(m_pMainNode->FirstChildElement("ParticleSystem")) {
			loadParticleSystem(m_pMainNode->FirstChildElement("ParticleSystem"), m_pQuadGrid);
		}

		if (m_pParticleSystem) {
			m_pParticleSystem->tagZalezaskDisk(Vector2(1, 1), 0.75, 0.2);
			//m_pParticleSystem->tagDisk(Vector2(0.75, 0.5), 0.25);
			//m_pParticleSystem->tagDisk(Vector2(1.25, 0.5), 0.25);
		}

		if (m_pMainNode->FirstChildElement("VelocityInterpolant")) {
			m_pVelocityInterpolant = loadInterpolant(m_pMainNode->FirstChildElement("VelocityInterpolant"), m_pQuadGrid->getGridData2D()->getVelocityArray(), m_pQuadGrid->getGridData2D()->getGridSpacing());
		}

		if (m_pMainNode->FirstChildElement("FineGridData")) {
			m_pFineGridConfig = loadFineGridScalar(m_pMainNode->FirstChildElement("FineGridData"));
			m_pQuadGrid->getGridData2D()->initializeFineGridScalarField(m_pFineGridConfig->numSubdivisions);
			m_pRenderer->getGridRenderer(0)->getScalarFieldRenderer().setFineGridScalarValues2D(m_pQuadGrid->getGridData2D()->getFineGridScalarFieldArrayPtr(), 
																								m_pQuadGrid->getGridData2D()->getFineGridScalarFieldDx());
		}

		if (m_pParticleSystem) {
			m_pParticleSystem->setVelocityInterpolant(m_pVelocityInterpolant);
		}
	}

	/************************************************************************/
	/* Callbacks                                                            */
	/************************************************************************/
	void PrecomputedAnimation2D::keyboardCallback(unsigned char key, int x, int y) {
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

			case '+':
				m_animationVelocity += 0.05f;
				m_animationVelocity = clamp(m_animationVelocity, 0.0f, 1.0f);
			break;

			case '-':
				m_animationVelocity -= 0.05f;
				m_animationVelocity = clamp(m_animationVelocity, 0.0f, 1.0f);
			break;
		}
	}

	void PrecomputedAnimation2D::keyboardUpCallback(unsigned char key, int x, int y) {
		switch(key) {
			case 'e': case 'E': case 'q': case 'Q':
				m_pMainSimCfg->setAngularAcceleration(DegreeToRad(0.0f));
			break;
		}
	}

	void PrecomputedAnimation2D::specialKeyboardCallback(int key, int x, int y) {
		
	}
	
	void PrecomputedAnimation2D::specialKeyboardUpCallback(int key, int x, int y) {

	}


	/************************************************************************/
	/* Loading functions                                                    */
	/************************************************************************/
	void PrecomputedAnimation2D::loadGridFile() {
		string tempValue;
		dimensions_t tempDimensions;
		try { /** Load grids and boundary conditions: try-catchs for invalid file exceptions */
			
			 /** Load grid */
			if ((m_pMainNode->FirstChildElement("GridFile")) == NULL) {
				m_pQuadGrid = loadGrid(m_pMainNode->FirstChildElement("Grid"));
			}
		} catch(exception e) {
			exitProgram(e.what());
		}
	}

	void PrecomputedAnimation2D::loadPhysics() {
		/** Physics initialization - configure by XML*/
		m_pPhysicsParams->timestep = 0.01f;
		//m_pPhysicsParams->frameRate = 30;

		//Unbounded simulation
		m_pPhysicsParams->totalSimulationTime = -1;

		m_pPhysicsCore = PhysicsCore<Vector2>::getInstance();
		m_pPhysicsCore->initialize(*m_pPhysicsParams);
		m_pPhysicsParams = m_pPhysicsCore->getParams();
	}


	void PrecomputedAnimation2D::setupCamera(TiXmlElement *pCameraNode) {
		Vector3 camPosition;
		camPosition.z = 4.0f;
		if(pCameraNode->FirstChildElement("Mode")) {
			if(string(pCameraNode->FirstChildElement("Mode")->GetText()) == "Follow")
				m_pRenderer->getCamera()->followGrid(m_pQuadGrid, camPosition);
		}

	}

	/************************************************************************/
	/* Private functionalities												*/
	/************************************************************************/
	doubleGyreConfig_t * PrecomputedAnimation2D::loadDoubleGyreConfig(TiXmlElement * pDoubleGyreNode) {
		doubleGyreConfig_t *pDoubleGyreConfig = new doubleGyreConfig_t();

		if (pDoubleGyreNode->FirstChildElement("Epsilon")) {
			pDoubleGyreConfig->epsilon = atof(pDoubleGyreNode->FirstChildElement("Epsilon")->GetText());
		}

		if (pDoubleGyreNode->FirstChildElement("VelocityMagnitude")) {
			pDoubleGyreConfig->velocityMagnitude = atof(pDoubleGyreNode->FirstChildElement("VelocityMagnitude")->GetText());
		}
		
		if (pDoubleGyreNode->FirstChildElement("OscillationFrequency")) {
			pDoubleGyreConfig->oscillationFrequency = atof(pDoubleGyreNode->FirstChildElement("OscillationFrequency")->GetText());
		}

		return pDoubleGyreConfig;
	}
	fineGridConfig_t * PrecomputedAnimation2D::loadFineGridScalar(TiXmlElement *pFineGridNode) {
		fineGridConfig_t *pFineGridConfig = new fineGridConfig_t();
		if (pFineGridNode->FirstChildElement("Type")) {
			string scalarType = pFineGridNode->FirstChildElement("Type")->GetText();
			transform(scalarType.begin(), scalarType.end(), scalarType.begin(), tolower);
			if (scalarType == "divergence") {
				pFineGridConfig->scalarFieldType = divergenceField;
			}
		}
		if (pFineGridNode->FirstChildElement("Subdivisions")) {
			pFineGridConfig->numSubdivisions = atoi(pFineGridNode->FirstChildElement("Subdivisions")->GetText());
		}
		return pFineGridConfig;
	}

	void PrecomputedAnimation2D::updateParticleSystem(Scalar dt) {

		m_pParticleSystem->setGridOrigin(m_pQuadGrid->getPosition());
		m_pParticleSystem->update(dt);
	}
	void PrecomputedAnimation2D::updateRotationalVelocities() {
		Scalar dx = m_pQuadGrid->getGridData2D()->getGridSpacing();
		GridData2D *pGridData = m_pQuadGrid->getGridData2D();
		for (int k = 0; k < m_rotationalVels.size(); k++) {
	
			for (int i = 0; i < m_pQuadGrid->getDimensions().x; i++) {
				for (int j = 0; j < m_pQuadGrid->getDimensions().y; j++) {
					Vector2 velocity;
					Vector2 cellCenter(i*dx, (j + 0.5)*dx); //Staggered
					Vector2 radiusVec = cellCenter - m_rotationalVels[k].center;
					Scalar radius = radiusVec.length();
					if (radius > m_rotationalVels[k].minRadius && radius < m_rotationalVels[k].maxRadius) {
						if (m_rotationalVels[k].orientation) {
							velocity.x = -radiusVec.perpendicular().x*m_rotationalVels[k].strenght;
						}
						else {
							velocity.x = radiusVec.perpendicular().x*m_rotationalVels[k].strenght;
						}
						velocity.y = pGridData->getVelocity(i, j).y;
						pGridData->setVelocity(velocity, i, j);
						pGridData->setAuxiliaryVelocity(velocity, i, j);
					}
					cellCenter = Vector2((i + 0.5)*dx, j*dx);
					radiusVec = cellCenter - m_rotationalVels[k].center;
					radius = radiusVec.length();
					if (radius > m_rotationalVels[k].minRadius && radius < m_rotationalVels[k].maxRadius) {
						if (m_rotationalVels[k].orientation) {
							velocity.y = -radiusVec.perpendicular().y*m_rotationalVels[k].strenght;
						}
						else {
							velocity.y = radiusVec.perpendicular().y*m_rotationalVels[k].strenght;
						}
						velocity.x = pGridData->getVelocity(i, j).x;
						pGridData->setVelocity(velocity, i, j);
						pGridData->setAuxiliaryVelocity(velocity, i, j);
					}

					//m_pQuadGrid->getGridData2D()->getDensityBuffer().setValueBothBuffers(m_pDensityFieldAnimation->m_scalarFieldBuffer[m_currentTimeStep](i, j), i, j);
				}
			}
		}
	}

	void PrecomputedAnimation2D::updateDoubleGyreVelocities() {
		
		Scalar dx = m_pQuadGrid->getGridData2D()->getGridSpacing();
		for (int i = 0; i < m_pQuadGrid->getDimensions().x; i++) {
			for (int j = 0; j < m_pQuadGrid->getDimensions().y; j++) {
				Vector2 xPosition(i*dx, (j + 0.5)*dx);
				Vector2 velocity;
				velocity.x = calculateGyreVelocity(xPosition, xComponent);
				
				Vector2 yPosition((i + 0.5)*dx, j*dx);
				velocity.y = calculateGyreVelocity(yPosition, yComponent);

				m_pQuadGrid->getGridData2D()->setVelocity(velocity, i, j);
			}
		}
	}

	DoubleScalar PrecomputedAnimation2D::calculateGyreVelocity(const Vector2 &position, velocityComponent_t velocityComponent) {
		Scalar t = m_pPhysicsCore->getInstance()->getElapsedTime();
		DoubleScalar eps = m_pDoubleGyreConfig->epsilon;
		DoubleScalar w = m_pDoubleGyreConfig->oscillationFrequency;
		DoubleScalar vm = m_pDoubleGyreConfig->velocityMagnitude;
		DoubleScalar a = eps*sin(w*t);
		DoubleScalar b = 1 - 2 * eps*sin(w*t);
		DoubleScalar f = a*position.x*position.x + b*position.x;
		DoubleScalar dfdx = 2 * a*position.x + b;
		if (velocityComponent == xComponent) {
			return -PI*vm*sin(PI*f)*cos(PI*position.y);
		}
		else if (velocityComponent == yComponent) {
			return PI*vm*cos(PI*f)*sin(PI*position.y)*dfdx;
		}
		return 0;
	}

	void PrecomputedAnimation2D::loadAnimations(TiXmlElement *pAnimationsNode) {
		if (pAnimationsNode->FirstChildElement("Timestep")) {
			m_timeStepSize = atof(pAnimationsNode->FirstChildElement("Timestep")->GetText());
			m_pPhysicsParams->timestep = m_timeStepSize;
		}
		if (pAnimationsNode->FirstChildElement("Framerate")) {
			m_frameRate = atoi(pAnimationsNode->FirstChildElement("Framerate")->GetText());
		}
		if (pAnimationsNode->FirstChildElement("NumFrames")) {
			m_numFrames = atoi(pAnimationsNode->FirstChildElement("NumFrames")->GetText());
		}
		if(pAnimationsNode->FirstChildElement("DensityFile")) {
			string densityFile = pAnimationsNode->FirstChildElement("DensityFile")->GetText();
			m_pDensityFieldAnimation = loadScalarFieldCollection(densityFile, densityField,m_numFrames);
		} 
		if(pAnimationsNode->FirstChildElement("PressureFile")) {
			string pressureFile = pAnimationsNode->FirstChildElement("PressureFile")->GetText();
			m_pPressureFieldAnimation = loadScalarFieldCollection(pressureFile, pressureField, m_numFrames);
		}
		if(pAnimationsNode->FirstChildElement("VelocityFile")) {
			string velocityFile = pAnimationsNode->FirstChildElement("VelocityFile")->GetText();
			m_pVelocityFieldAnimation = loadVelocityFieldCollection(velocityFile, m_numFrames);
		}
		if(pAnimationsNode->FirstChildElement("PositionFile")) {
			string positionFile = pAnimationsNode->FirstChildElement("PositionFile")->GetText();
		}
		if(pAnimationsNode->FirstChildElement("ThinObjectFile")) {
			string thinObjectFile = pAnimationsNode->FirstChildElement("ThinObjectFile")->GetText();
			m_pThinObjectAnimation = loadThinObjectAnimation("Flow Logs/2D/ThinObject/" + thinObjectFile + ".log");
				m_pThinObjectLine = new Line<Vector2>(Vector2(0, 0), vector<Vector2>(m_pThinObjectAnimation->thinObjectPointsSize));
			for(int i = 0; i < m_pThinObjectAnimation->thinObjectPointsSize; i++) {
				m_pThinObjectLine->getPoints()[i] = m_pThinObjectAnimation->m_pThinObjectAnimationBuffer->at(getAnimationBufferIndex(i, 0,  m_pThinObjectAnimation->thinObjectPointsSize));
			}
			m_pRenderer->addObject(m_pThinObjectLine);
		}
		if (pAnimationsNode->FirstChildElement("RotationalVelocityField")) {
			m_rotationalVels = loadRotationalVelocityField(pAnimationsNode->FirstChildElement("RotationalVelocityField"));
		}
		if (pAnimationsNode->FirstChildElement("DoubleGyre")) {
			m_pDoubleGyreConfig = loadDoubleGyreConfig(pAnimationsNode->FirstChildElement("DoubleGyre"));
		}
	}

	scalarFieldAnimation_t * PrecomputedAnimation2D::loadScalarFieldCollection(const string &scalarFieldFile, scalarFieldType_t scalarFieldType, int numberFrames) {
		scalarFieldAnimation_t * pScalarFieldAnimation = new scalarFieldAnimation_t();

		Logger::get() << "Loading scalar field collection:" << scalarFieldFile << endl;
		Logger::get() << "Total number of frames: " << numberFrames << endl;

		for (int i = 0; i < numberFrames; i++) {
			Array2D<Scalar> *pScalarArray = new Array2D<Scalar>(m_pQuadGrid->getDimensions());
			switch (scalarFieldType)  {
				case Chimera::densityField:
					loadFrame<Scalar>("Flow Logs/2D/Density/" + scalarFieldFile + intToStr(i) + ".log", *pScalarArray);
				break;
				
				case Chimera::pressureField:
					loadFrame<Scalar>("Flow Logs/2D/Pressure/" + scalarFieldFile + intToStr(i) + ".log", *pScalarArray);
				break;
				
				case Chimera::vorticityField:
					loadFrame<Scalar>("Flow Logs/2D/Vorticity/" + scalarFieldFile + intToStr(i) + ".log", *pScalarArray);
				break;
				
				default:
				break;
			}
			
			pScalarFieldAnimation->m_scalarFieldBuffer.push_back(*pScalarArray);
		}

		pScalarFieldAnimation->totalFrames = numberFrames;
		pScalarFieldAnimation->timeElapsed = numberFrames*m_frameRate;

		Logger::get() << "Scalar field file successfully loaded." << endl;
		return pScalarFieldAnimation;
	}


	velocityFieldAnimation_t * PrecomputedAnimation2D::loadVelocityFieldCollection(const string &velocityFieldFile, int numberFrames) {
		velocityFieldAnimation_t *pVelocityFieldAnimation = new velocityFieldAnimation_t();
		
		Logger::get() << "Loading velocity field collection:" << velocityFieldFile << endl;
		Logger::get() << "Total number of frames: " << numberFrames << endl;

		for (int i = 0; i < numberFrames; i++) {
			Array2D<Vector2> *pVelArray = new Array2D<Vector2>(m_pQuadGrid->getDimensions());
			loadFrame<Vector2>("Flow Logs/2D/Velocity/" + velocityFieldFile + intToStr(i) + ".log", *pVelArray);
			pVelocityFieldAnimation->m_velocityBuffer.push_back(*pVelArray);
		}
		
		pVelocityFieldAnimation->totalFrames = numberFrames;
		pVelocityFieldAnimation->timeElapsed = numberFrames*m_frameRate;

		Logger::get() << "Scalar field file successfully loaded." << endl;
		return pVelocityFieldAnimation;
	}

	template <class VarType>
	void PrecomputedAnimation2D::loadFrame(const string &frameFile, Array2D<VarType> &values) {
		auto_ptr<ifstream> precomputedStream(new ifstream(frameFile.c_str(), ifstream::binary));

		if (!precomputedStream->is_open()) {
			Logger::get() << "Unable to load velocity field file: " << frameFile << endl;
			exit(-1);
		}

		dimensions_t gridDimensions;
		Scalar timeElapsed, totalFrames;

		precomputedStream->read(reinterpret_cast<char *>(&gridDimensions.x), sizeof(int)*3); //xyz
		precomputedStream->read(reinterpret_cast<char *>(&timeElapsed), sizeof(Scalar));
		precomputedStream->read(reinterpret_cast<char *>(&totalFrames), sizeof(int));


		if(m_pQuadGrid->getDimensions().x != gridDimensions.x || m_pQuadGrid->getDimensions().y != gridDimensions.y) {
			Logger::get() << "Velocity field and loaded grid dimensions doesn't match." << endl;
			exit(-1);
		}

		VarType tempVar;
		for (int i = 0; i < gridDimensions.x; i++) {
			for (int j = 0; j < gridDimensions.y; j++) {
				precomputedStream->read(reinterpret_cast<char *>(&tempVar), sizeof(VarType));
				values(i, j) = tempVar;
				//pVelocityFieldAnimation->pVelocityFieldBuffer[getAnimationBufferIndex(i, j, t, gridDimensions)] = tempVec;
			}
		}
	}

	thinObjectAnimation_t * PrecomputedAnimation2D::loadThinObjectAnimation(const string &thinObjectAnimationFile) {
		auto_ptr<ifstream> precomputedStream(new ifstream(thinObjectAnimationFile.c_str(), ifstream::binary));

		thinObjectAnimation_t *pThinObjectAnimation = new thinObjectAnimation_t();

		precomputedStream->read(reinterpret_cast<char *>(&pThinObjectAnimation->thinObjectPointsSize), sizeof(int)); //xyz

		precomputedStream->read(reinterpret_cast<char *>(&pThinObjectAnimation->timeElapsed), sizeof(Scalar));
		precomputedStream->read(reinterpret_cast<char *>(&pThinObjectAnimation->totalFrames), sizeof(int));

		Logger::get() << "Loading thinObject animation file:" << thinObjectAnimationFile << endl;
		Logger::get() << "ThinObject total number of points: " << pThinObjectAnimation->thinObjectPointsSize << endl;
		Logger::get() << "Total number of frames: " << pThinObjectAnimation->totalFrames << endl;

		pThinObjectAnimation->m_pThinObjectAnimationBuffer = new vector<Vector2>(pThinObjectAnimation->thinObjectPointsSize*pThinObjectAnimation->totalFrames);
		
		Vector2 tempVec;
		for(int t = 0; t < pThinObjectAnimation->totalFrames; t++) {
			for(int i = 0; i < pThinObjectAnimation->thinObjectPointsSize; i++) {
				precomputedStream->read(reinterpret_cast<char *>(&tempVec), sizeof(Scalar)*2);
				(*pThinObjectAnimation->m_pThinObjectAnimationBuffer)[getAnimationBufferIndex(i, t, pThinObjectAnimation->thinObjectPointsSize)] = tempVec;
			}
		}
		return pThinObjectAnimation;
	}

	void PrecomputedAnimation2D::updateFineGridDivergence() {
		Scalar dx = m_pQuadGrid->getGridData2D()->getGridSpacing();
		Scalar fineDx = m_pQuadGrid->getGridData2D()->getFineGridScalarFieldDx();
		int numSubCells = dx / fineDx;
		dimensions_t fineGridDimensions(m_pQuadGrid->getGridData2D()->getDimensions().x*numSubCells, m_pQuadGrid->getGridData2D()->getDimensions().y*numSubCells);

		for (int i = numSubCells * 1; i < fineGridDimensions.x - numSubCells * 1; i++) {
			for (int j = numSubCells * 1; j < fineGridDimensions.y - numSubCells * 1; j++) {
				Vector2 centroid((i + 0.5)*fineDx, (j + 0.5)*fineDx);
				Vector2 centroidCoarse(centroid.x / dx, centroid.y / dx);
				dimensions_t coarseGridIndex(floor(centroidCoarse.x), floor(centroidCoarse.y));

				Scalar deltaError = 1e-5;

				Scalar divDx = fineDx*0.5 - deltaError;

				Vector2 velTop = m_pVelocityInterpolant->interpolate(Vector2(centroid.x, centroid.y + divDx));
				Vector2 velBottom = m_pVelocityInterpolant->interpolate(Vector2(centroid.x, centroid.y - divDx));
				Vector2 velLeft = m_pVelocityInterpolant->interpolate(Vector2(centroid.x + divDx, centroid.y));
				Vector2 velRight = m_pVelocityInterpolant->interpolate(Vector2(centroid.x - divDx, centroid.y));

				Scalar currDiv = (velTop.y - velBottom.y) / (2 * divDx) + (velLeft.x - velRight.x) / (2 * divDx);
				m_pQuadGrid->getGridData2D()->setFineGridScalarValue((currDiv), i, j);
			}
		}
	}

	void PrecomputedAnimation2D::updateDivergence() {
		GridData2D *pGridData = m_pQuadGrid->getGridData2D();
		for (uint i = 1; i < m_pQuadGrid->getDimensions().x - 1; i++) {
			for (uint j = 1; j < m_pQuadGrid->getDimensions().y - 1; j++) {
				Scalar dx = (pGridData->getVelocity(i + 1, j).x - pGridData->getVelocity(i, j).x) / pGridData->getGridSpacing();
				Scalar dy = (pGridData->getVelocity(i, j + 1).y - pGridData->getVelocity(i, j).y) / pGridData->getGridSpacing();
				pGridData->setDivergent(dx + dy, i, j);
			}
		}
		
	}

	/************************************************************************/
	/* Functionalities                                                      */
	/************************************************************************/
	void PrecomputedAnimation2D::update() {
		Scalar updateFramesDt = 1 / ((Scalar)m_frameRate);
		bool updateTimestep = false;
		if(m_pPhysicsCore->isRunningSimulation()) {
			m_elapsedTime += m_timeStepSize;
			//m_elapsedTime += m_updateTimer.secondsElapsed()*m_animationVelocity;
			if(m_elapsedTime > m_currentTimeStep*(1/((Scalar)m_frameRate))) {
				m_currentTimeStep++;
				updateTimestep = true;
			}
			m_updateTimer.start();

			CubicStreamfunctionInterpolant2D<Vector2> *pCInterpolant = dynamic_cast<CubicStreamfunctionInterpolant2D<Vector2> *>(m_pVelocityInterpolant);
			if (pCInterpolant)
				pCInterpolant->computeStreamfunctions();
			else {
				BilinearStreamfunctionInterpolant2D<Vector2> *pSInterpolant = dynamic_cast<BilinearStreamfunctionInterpolant2D<Vector2> *>(m_pVelocityInterpolant);
				if (pSInterpolant)
					pSInterpolant->computeStreamfunctions();
			}
			if (m_pFineGridConfig) {
				updateFineGridDivergence();
			}
			updateDivergence();




			if (m_rotationalVels.size() > 0) {
				updateRotationalVelocities();
			}
			if (m_pDoubleGyreConfig != nullptr) {
				updateDoubleGyreVelocities();
			}
			if(m_pDensityFieldAnimation) {
				if(m_currentTimeStep >= m_pDensityFieldAnimation->totalFrames) {
					m_currentTimeStep = 0;
					m_elapsedTime = 0;
					//m_pParticleSystem->resetParticleSystem();
				}
				if(updateTimestep) {
					for(int i = 0; i < m_pQuadGrid->getDimensions().x; i++) { 
						for(int j = 0; j < m_pQuadGrid->getDimensions().y; j++) {
							m_pQuadGrid->getGridData2D()->getDensityBuffer().setValueBothBuffers(m_pDensityFieldAnimation->m_scalarFieldBuffer[m_currentTimeStep](i, j), i, j);
						}
					}
				}
			}
			if(m_pPressureFieldAnimation) {
				if(m_currentTimeStep >= m_pPressureFieldAnimation->totalFrames) {
					m_currentTimeStep = 0;
					m_elapsedTime = 0;
					//m_pParticleSystem->resetParticleSystem();
				}
				if(updateTimestep) {
					for(int i = 0; i < m_pQuadGrid->getDimensions().x; i++) { 
						for(int j = 0; j < m_pQuadGrid->getDimensions().y; j++) {
							m_pQuadGrid->getGridData2D()->setPressure(m_pPressureFieldAnimation->m_scalarFieldBuffer[m_currentTimeStep](i, j), i, j);
						}
					}
				}
			}
			if(m_pVelocityFieldAnimation) {
				if(m_currentTimeStep >= m_pVelocityFieldAnimation->totalFrames) {
					m_currentTimeStep = 0;
					m_elapsedTime = 0;
					//m_pParticleSystem->resetParticleSystem();
				}
				if(updateTimestep) {
					for(int i = 0; i < m_pQuadGrid->getDimensions().x; i++) { 
						for(int j = 0; j < m_pQuadGrid->getDimensions().y; j++) {
							m_pQuadGrid->getGridData2D()->setVelocity(
								m_pVelocityFieldAnimation->m_velocityBuffer[m_currentTimeStep](i,j), i, j);
						}
					}
				}
			}

			if(m_pThinObjectAnimation) {
				if(updateTimestep) {
					for(int i = 0; i < m_pThinObjectAnimation->thinObjectPointsSize; i++) {
						m_pThinObjectLine->getPoints()[i] = m_pThinObjectAnimation->m_pThinObjectAnimationBuffer->at(getAnimationBufferIndex(i, m_currentTimeStep, m_pThinObjectAnimation->thinObjectPointsSize));
					}
				}
			}
			m_pRenderer->update();

			if(m_pParticleSystem && updateTimestep) {
				//updateParticleSystem(m_timeStepSize);
			}
			
		} else {
			m_updateTimer.stop();
			m_updateTimer.reset();
		}
		
		m_pPhysicsCore->update();
		
	}
	void PrecomputedAnimation2D::draw() {
		m_pRenderer->renderLoop();
	}
}