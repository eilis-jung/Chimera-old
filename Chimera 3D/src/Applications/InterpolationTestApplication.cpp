#include "Applications/InterpolationTestApplication.h"

namespace Chimera {

	#pragma region Constructors
	InterpolationTestApplication::InterpolationTestApplication(int argc, char** argv, TiXmlElement *pChimeraConfig) : Application3D(argc, argv, pChimeraConfig) {
		m_pPolygonSurface = NULL;
		initializeGL(argc, argv);

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
		
		m_pPolygonSurface = m_pSceneLoader->getPolygonSurfaces().front();

		if (pChimeraConfig->FirstChildElement("InterpolationMethod")) {
			string interpolationType = pChimeraConfig->FirstChildElement("InterpolationMethod")->GetText();
			if (interpolationType == "sbc")
				m_interpolationMethod = sbcInterpolation;
			else
				m_interpolationMethod = mvcInterpolation;
		}
		int numberOfInterpolationPoints = 0;
		if (pChimeraConfig->FirstChildElement("NumberOfInterpolationPoints")) {
			string numberOfInterpolationPointsStr = pChimeraConfig->FirstChildElement("NumberOfInterpolationPoints")->GetText();
			numberOfInterpolationPoints = atoi(numberOfInterpolationPointsStr.c_str());
		}
		else {
			numberOfInterpolationPoints = 100;
		}

		if (pChimeraConfig->FirstChildElement("NodeValuesMethod")) {
			string nodeValuesMethod = pChimeraConfig->FirstChildElement("NodeValuesMethod")->GetText();
			if (nodeValuesMethod == "random")
				m_nodeValuesMethod  = randomValues;
			else if (nodeValuesMethod == "signedDistance")
				m_nodeValuesMethod = signedDistanceValues;
		}

		bool checkInterpolationPointsInsideMesh = false;
		if (pChimeraConfig->FirstChildElement("InterpolationPointsInsideMesh")) {
			string pointsInsideMesh = pChimeraConfig->FirstChildElement("InterpolationPointsInsideMesh")->GetText();
			if (pointsInsideMesh == "true")
				checkInterpolationPointsInsideMesh = true;
		}

		/** Initializing polygon interpolation values */
		for (int i = 0; i < m_pPolygonSurface->getVertices().size(); i++) {
			if (m_nodeValuesMethod == randomValues) {
				m_polygonInterpolationValues.push_back(rand() / ((float)RAND_MAX));
			}
			else if(m_nodeValuesMethod == signedDistanceValues) {
				m_polygonInterpolationValues.push_back(m_pPolygonSurface->getVertices()[i].length()*1.75 - 0.65);
			}
			else { //Initialize all with zeros
				m_polygonInterpolationValues.push_back(0);
			}
	
			Sphere *pSphere = new Sphere(convertToVector3F(m_pPolygonSurface->getVertices()[i]), 0.025);
			pSphere->setColor(m_polygonInterpolationValues.back(), m_polygonInterpolationValues.back(), m_polygonInterpolationValues.back(), 1.0f);
			m_pRenderer->addObject(pSphere, false);
		}

		int insideMeshMaxTries = 30;
		/** Initializing interpolation values */
		for (int i = 0; i < numberOfInterpolationPoints; i++) {
			if (checkInterpolationPointsInsideMesh) {
				for (int j = 0; j < insideMeshMaxTries; j++) {
					Vector3 pointPosition(rand() / ((float)RAND_MAX) - 0.5, rand() / ((float)RAND_MAX) - 0.5, rand() / ((float)RAND_MAX) - 0.5);
					if (m_pPolygonSurface->isInside(convertToVector3D(pointPosition))) {
						m_interpolationPoints.push_back(pointPosition);
						m_interpolationValues.push_back(trilinearInterpolation(m_interpolationPoints.back(), m_interpolationMethod));
						break;
					}
				}
			} else {
				Vector3 pointPosition(rand() / ((float)RAND_MAX) - 0.5, rand() / ((float)RAND_MAX) - 0.5, rand() / ((float)RAND_MAX) - 0.5);
				m_interpolationPoints.push_back(pointPosition);
				m_interpolationValues.push_back(trilinearInterpolation(m_interpolationPoints.back(), m_interpolationMethod));
			}
			
		}

		m_pRenderer->getCamera()->setRotationAroundGridMode(Vector3(0, 0, 0));
		
		for (int i = 0; i < m_pSceneLoader->getPolygonSurfaces().size(); i++) {
			m_pRenderer->addObject(m_pSceneLoader->getPolygonSurfaces()[i]);
		}


		Sphere *pSphere = new Sphere(Vector3(0, 0, 0), 1);
		pSphere->setColor(0.0f, 0.0f, 0.0f, 0.25f);
		m_pRenderer->addObject(pSphere);
	}

	
	#pragma endregion
	#pragma region Functionalities
	void InterpolationTestApplication::update() {

	}

	void InterpolationTestApplication::draw() {
		m_pRenderer->renderLoop(false);
		
		drawInterpolationValues();

		glutSwapBuffers();
		glutPostRedisplay();
	}
	void InterpolationTestApplication::drawInterpolationValues() {
		glPushMatrix();
		//glTranslatef(-1, -1, -1);
		glPointSize(7.0f);
		glBegin(GL_POINTS);
		for (int i = 0; i < m_interpolationPoints.size(); i++) {
			glColor3f(m_interpolationValues[i], m_interpolationValues[i], m_interpolationValues[i]);
			glVertex3f(m_interpolationPoints[i].x, m_interpolationPoints[i].y, m_interpolationPoints[i].z);
		}
		glEnd();
		glPopMatrix();
	}
 	#pragma endregion
	Scalar InterpolationTestApplication::trilinearInterpolation(const Vector3 &position, LinearInterpolationMethod_t interpolationMethod) {
		vector<Scalar> weights;
		if (m_interpolationMethod == sbcInterpolation) {
			sphericalBarycentricCoordinatesWeights(position, weights);
		}
		else {
			meanValueCoordinatesWeights(position, weights);
		}

		Scalar interpolatedValue = 0;
		for (int i = 0; i < m_polygonInterpolationValues.size(); i++) {
			interpolatedValue += m_polygonInterpolationValues[i] * weights[i];
		}

		return interpolatedValue;
	}

	void InterpolationTestApplication::meanValueCoordinatesWeights(const Vector3 &position, vector<Scalar> &weights) {
			const vector<Vector3D> &nodalPoints = m_pPolygonSurface->getVertices();
			int npts = nodalPoints.size();
			weights.resize(npts);
			Vector3D transformedPosition = convertToVector3D(position);

			// arrays storing point-to-vertex vectors and distances
			std::vector<double> dist(npts, 0);
			std::vector<Vector3D> uVec(npts);
			static const double eps = 0.0001;

			//Check for special cases: if the point lies on a cut-face or cut-edge
			for (int i = 0; i < m_pPolygonSurface->getFaces().size(); i++) {
				int pid0 = m_pPolygonSurface->getFaces()[i].edges[0].first;
				int pid1 = m_pPolygonSurface->getFaces()[i].edges[1].first;
				int pid2 = m_pPolygonSurface->getFaces()[i].edges[2].first;

				Vector3D u0(uVec[pid0]);
				Vector3D u1(uVec[pid1]);
				Vector3D u2(uVec[pid2]);

				if (isOnEdge(u0, u1, transformedPosition, eps)) {
					DoubleScalar alfa = linearInterpolationWeight(transformedPosition, u0, u1);
					weights[pid0] = 1 - alfa;
					weights[pid1] = alfa;
					return;
				}

				if (isOnEdge(u1, u2, transformedPosition, eps)) {
					DoubleScalar alfa = linearInterpolationWeight(transformedPosition, u1, u2);
					weights[pid1] = 1 - alfa;
					weights[pid2] = alfa;
					return;
				}

				if (isOnEdge(u2, u0, transformedPosition, eps)) {
					DoubleScalar alfa = linearInterpolationWeight(transformedPosition, u2, u0);
					weights[pid2] = 1 - alfa;
					weights[pid0] = alfa;
					return;
				}
			}

			for (int i = 0; i < nodalPoints.size(); i++) {
				uVec[i] = (nodalPoints[i] - transformedPosition);
				dist[i] = (uVec[i].length());
				if (dist[i] < eps){
					weights[i] = 1.0;
					return;
				}
				uVec[i] /= dist[i];
			}

			for (int i = 0; i < m_pPolygonSurface->getFaces().size(); i++) {
				int pid0 = m_pPolygonSurface->getFaces()[i].edges[0].first;
				int pid1 = m_pPolygonSurface->getFaces()[i].edges[1].first;
				int pid2 = m_pPolygonSurface->getFaces()[i].edges[2].first;

				Vector3D u0(uVec[pid0]);
				Vector3D u1(uVec[pid1]);
				Vector3D u2(uVec[pid2]);

				// edge lengths
				DoubleScalar l0 = (u1 - u2).length();
				DoubleScalar l1 = (u2 - u0).length();
				DoubleScalar l2 = (u0 - u1).length();

				// angles
				DoubleScalar theta0 = 2.0*asin(l0 / 2.0);
				DoubleScalar theta1 = 2.0*asin(l1 / 2.0);
				DoubleScalar theta2 = 2.0*asin(l2 / 2.0);
				DoubleScalar halfSum = (theta0 + theta1 + theta2) / 2.0;

				// special case when the point lies on the triangle
				if (PI - halfSum < eps) {
					weights.clear();
					weights.resize(npts, 0.0); // clear all

					weights[pid0] = sin(theta0) * dist[pid1] * dist[pid2];
					weights[pid1] = sin(theta1) * dist[pid2] * dist[pid0];
					weights[pid2] = sin(theta2) * dist[pid0] * dist[pid1];

					Scalar sumWeight = weights[pid0] + weights[pid1] + weights[pid2];

					weights[pid0] /= sumWeight;
					weights[pid1] /= sumWeight;
					weights[pid2] /= sumWeight;

					return;
				}

				// coefficient
				DoubleScalar sinHalfSum = sin(halfSum);
				DoubleScalar sinHalfSumSubTheta0 = sin(halfSum - theta0);
				DoubleScalar sinHalfSumSubTheta1 = sin(halfSum - theta1);
				DoubleScalar sinHalfSumSubTheta2 = sin(halfSum - theta2);
				DoubleScalar sinTheta0 = sin(theta0), sinTheta1 = sin(theta1), sinTheta2 = sin(theta2);

				DoubleScalar c0 = 2 * sinHalfSum * sinHalfSumSubTheta0 / sinTheta1 / sinTheta2 - 1;
				DoubleScalar c1 = 2 * sinHalfSum * sinHalfSumSubTheta1 / sinTheta2 / sinTheta0 - 1;
				DoubleScalar c2 = 2 * sinHalfSum * sinHalfSumSubTheta2 / sinTheta0 / sinTheta1 - 1;

				if (fabs(c0) > 1) c0 = c0 > 0 ? 1 : -1;
				if (fabs(c1) > 1) c1 = c1 > 0 ? 1 : -1;
				if (fabs(c2) > 1) c2 = c2 > 0 ? 1 : -1;

				// sign
				Matrix3x3 matDet(convertToVector3F(u0), convertToVector3F(u1), convertToVector3F(u2));
				Scalar det = matDet.determinant();
				//Scalar det = abs(matDet.determinant());
				/*if(det != matDet.determinant()) {
				Logger::getInstance()->get() << "Error" << endl;
				}*/

				// skip when less than eps
				if (abs(det) < eps){
					i++; continue;
				}

				DoubleScalar detSign = det > 0 ? 1 : -1;
				DoubleScalar sign0 = detSign * sqrt(1 - c0*c0);
				DoubleScalar sign1 = detSign * sqrt(1 - c1*c1);
				DoubleScalar sign2 = detSign * sqrt(1 - c2*c2);

				// if 'x' lies on the plane of current triangle but outside it, ignore the current triangle
				if (abs(sign0) < eps || abs(sign1) < eps || abs(sign2) < eps)
				{
					i++; continue;
				}

				// weight 
				weights[pid0] += (theta0 - c1*theta2 - c2*theta1) / (dist[pid0] * sinTheta1*sign2);
				weights[pid1] += (theta1 - c2*theta0 - c0*theta2) / (dist[pid1] * sinTheta2*sign0);
				weights[pid2] += (theta2 - c0*theta1 - c1*theta0) / (dist[pid2] * sinTheta0*sign1);
			}

			// normalize weight
			DoubleScalar sumWeight = 0.0;
			for (int pid = 0; pid < npts; ++pid)	sumWeight += weights[pid];
			if (!sumWeight) printf("WARNING: zero weights.\n");
			for (int pid = 0; pid < npts; ++pid)	weights[pid] /= sumWeight;
	}

	void InterpolationTestApplication::sphericalBarycentricCoordinatesWeights(const Vector3 &position, vector<Scalar> &weights) {
		vector<Vector3> vfs(m_pPolygonSurface->getFaces().size());
		vector<vector<Scalar>> perFaceLambdas;
		const vector<Vector3D> &vertices = m_pPolygonSurface->getVertices();
		weights.resize(vertices.size());
		Vector3D transformedPosition = convertToVector3D(position);
		
		//Calculate vf for each face
		for (int i = 0; i < m_pPolygonSurface->getFaces().size(); i++) {
			Vector3 vf;
			
			/** First compute VFs per face and the total denominator */
			for (int j = 0; j < m_pPolygonSurface->getFaces()[i].edges.size(); j++) {
				Vector3D currV = vertices[m_pPolygonSurface->getFaces()[i].edges[j].first] - transformedPosition;
				Vector3D nextV = vertices[m_pPolygonSurface->getFaces()[i].edges[j].second] - transformedPosition;
				currV.normalize();
				nextV.normalize();

				Vector3D currNormal = currV.cross(nextV).normalized();
				DoubleScalar tetaAngle = currV.angle(nextV);
				vf += convertToVector3F(currNormal) * 0.5 * tetaAngle;
			}
			vfs[i] = vf;

			DoubleScalar denominator = 0;

			vector<Scalar> currFaceLambdas;
			for (int j = 0; j < m_pPolygonSurface->getFaces()[i].edges.size(); j++) {
				int prevJ = roundClamp<int>(j - 1, 0, m_pPolygonSurface->getFaces()[i].edges.size());

				Vector3D prevV = vertices[m_pPolygonSurface->getFaces()[i].edges[prevJ].first] - transformedPosition;
				Vector3D currV = vertices[m_pPolygonSurface->getFaces()[i].edges[j].first] - transformedPosition;
				Vector3D nextV = vertices[m_pPolygonSurface->getFaces()[i].edges[j].second] - transformedPosition;
				DoubleScalar lambda = vfs[i].length() / currV.length();
				prevV.normalize();
				currV.normalize();
				nextV.normalize();

				Vector3D v = convertToVector3D(vfs[i].normalized());
				DoubleScalar tetaAngle = v.angle(currV);

				/*prevV = gnomonicProjection(prevV, v, v);
				currV = gnomonicProjection(currV, v, v);
				nextV = gnomonicProjection(nextV, v, v);

				currV -= v;
				prevV -= v;
				nextV -= v;*/

				prevV = v.cross(prevV).normalized();
				currV = v.cross(currV).normalized();
				nextV = v.cross(nextV).normalized();
				
				DoubleScalar prevAlfaAngle = prevV.angle(currV);
				prevAlfaAngle = angle3D(prevV, currV);
				DoubleScalar alfaAngle = currV.angle(nextV);
				alfaAngle = angle3D(currV, nextV);
				
				lambda *= (tan(prevAlfaAngle*0.5) + tan(alfaAngle*0.5)) / sin(tetaAngle);
				
				denominator += (tan(prevAlfaAngle*0.5) + tan(alfaAngle*0.5))*tan(1.57079632679489661923 - tetaAngle);
				currFaceLambdas.push_back(lambda);
			}
			
			for (int j = 0; j < currFaceLambdas.size(); j++) {
				currFaceLambdas[j] /= denominator;
			}

			DoubleScalar weightsSum = 0;
			for (int j = 0; j < m_pPolygonSurface->getFaces()[i].edges.size(); j++) {
				weightsSum += currFaceLambdas[j];
			}

			if (weightsSum < 0.0f) {
				for (int j = 0; j < m_pPolygonSurface->getFaces()[i].edges.size(); j++) {
					currFaceLambdas[j] = 0;
				}
			}

			Vector3 otherVerification;
			for (int j = 0; j < m_pPolygonSurface->getFaces()[i].edges.size(); j++) {
				otherVerification += convertToVector3F(vertices[m_pPolygonSurface->getFaces()[i].edges[j].first] - transformedPosition)*currFaceLambdas[j];
			}

			otherVerification.normalize();
			vf.normalize();

			perFaceLambdas.push_back(currFaceLambdas);
		}
	
		Vector3 sumVF;
		for (int i = 0; i < vfs.size(); i++) {
			sumVF += vfs[i];
		}

		Scalar allWeightsSum = 0;
		for (int i = 0; i < m_pPolygonSurface->getFaces().size(); i++) {
			for (int j = 0; j < m_pPolygonSurface->getFaces()[i].edges.size(); j++) {
				weights[m_pPolygonSurface->getFaces()[i].edges[j].first] += perFaceLambdas[i][j];
			}
		}
		for (int i = 0; i < weights.size(); i++) {
			allWeightsSum += weights[i];
		}
		for (int i = 0; i < weights.size(); i++) {
			weights[i] /= allWeightsSum;
		}


		DoubleScalar verification = 0;
		for (int i = 0; i < weights.size(); i++) {
			verification += Vector3(weights[i], weights[i], weights[i]).dot(convertToVector3F(vertices[i]) - position);
		}
		
	}

	#pragma region Callbacks
	void InterpolationTestApplication::keyboardCallback(unsigned char key, int x, int y) {
		m_pRenderer->keyboardCallback(key, x, y);
		switch (key) {

		}
	}

	void InterpolationTestApplication::keyboardUpCallback(unsigned char key, int x, int y) {
		switch (key) {
			
		}
	}

	void InterpolationTestApplication::specialKeyboardCallback(int key, int x, int y) {

	}

	void InterpolationTestApplication::specialKeyboardUpCallback(int key, int x, int y) {

	}
	void InterpolationTestApplication::motionCallback(int x, int y) {
		Application3D::motionCallback(x, y);
		
	}
	#pragma endregion

	#pragma region LoadingFunctions
	void InterpolationTestApplication::setupCamera(TiXmlElement *pCameraNode) {

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

	#pragma region PrivateFunctionalities
	Vector3D InterpolationTestApplication::gnomonicProjection(const Vector3D &vec, const Vector3D &spherePoint, const Vector3D &normal) {
		Vector3D rayDirection = vec.normalized();
		Vector3D intersectedPoint;
		if (!rayPlaneIntersect(Vector3D(0, 0, 0), rayDirection, spherePoint, normal, intersectedPoint)) {
			throw exception("Invalid gnomonicProjection");
		}
		return intersectedPoint;
	}
	#pragma endregion
}