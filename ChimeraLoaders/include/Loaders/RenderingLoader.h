#pragma once

void RealtimeSimulation3D::loadCamera(TiXmlElement *pCameraNode) {
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