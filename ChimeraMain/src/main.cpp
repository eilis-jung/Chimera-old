#include "stdio.h"

#include "ChimeraApplications.h"


Chimera::Applications::ApplicationBase<Chimera::Core::Vector2, Chimera::Core::Array2D> *pApplication2D;
Chimera::Applications::ApplicationBase<Chimera::Core::Vector3, Chimera::Core::Array3D> *pApplication3D;

#pragma region Callbacks 3D
static void MouseCallback3D(int button, int state, int x, int y) {
	pApplication3D->mouseCallback(button, state, x, y);
}

static void MotionCallback3D(int x, int y) {
	pApplication3D->motionCallback(x, y);
}

static void KeyboardCallback3D(unsigned char key, int x, int y) {
	pApplication3D->keyboardCallback(key, x, y);
}
static void SpecialKeyboardCallback3D(int key, int x, int y) {
	pApplication3D->specialKeyboardCallback(key, x, y);
}

static void SpecialKeyboardUpCallback3D(int key, int x, int y) {
	pApplication3D->specialKeyboardUpCallback(key, x, y);
	
}

static void ExitCallback3D() {
	pApplication3D->exitCallback();
}

static void RenderCallback3D() {
	pApplication3D->draw();
	pApplication3D->update();
}

static void ReshapeCallback3D(int width, int height) {
	pApplication3D->reshapeCallback(width, height);
}
#pragma endregion

#pragma region Callbacks 2D
static void MouseCallback2D(int button, int state, int x, int y) {
	pApplication2D->mouseCallback(button, state, x, y);
}

static void MotionCallback2D(int x, int y) {
	pApplication2D->motionCallback(x, y);
}


static void KeyboardCallback2D(unsigned char key, int x, int y) {
	pApplication2D->keyboardCallback(key, x, y);
}
static void SpecialKeyboardCallback2D(int key, int x, int y) {
	pApplication2D->specialKeyboardCallback(key, x, y);
}

static void SpecialKeyboardUpCallback2D(int key, int x, int y) {
	pApplication2D->specialKeyboardUpCallback(key, x, y);

}

static void ExitCallback2D() {
	pApplication2D->exitCallback();
}

static void ReshapeCallback2D(int width, int height) {
	pApplication2D->reshapeCallback(width, height);
}

static void RenderCallback2D() {
	pApplication2D->draw();
	pApplication2D->update();
}
#pragma endregion

static void IdleCallback() {
	glutPostRedisplay();
}

int main(int argc, char** argv) {
	/** To-be initialized within initialization functions */
	string chimeraFile;
	shared_ptr<Chimera::Resources::XMLDoc> pMainConfigFile;

	bool is2D = false;

	try {
		//Static configuration file - do not change
		shared_ptr<Chimera::Resources::XMLDoc> pTempConfigFile = Chimera::Resources::ResourceManager::getInstance()->loadXMLDocument("Configuration/ChimeraCFG.xml");
		chimeraFile = pTempConfigFile->FirstChildElement("ChimeraFile")->GetText();
		if (chimeraFile.find("2D")) {
			is2D = true;
		} else if(chimeraFile.find("3D")) {
			is2D = false;
		}
		else {
			throw("Invalid folder setup: configuration files should be under Configuration/2D/ or Configuration/3D/");
		}
		pMainConfigFile = Chimera::Resources::ResourceManager::getInstance()->loadXMLDocument(chimeraFile);
	}
	catch (exception e) {
		Chimera::Core::exitProgram(e.what());
	}

	TiXmlElement *pChimeraConfig = pMainConfigFile->FirstChildElement("ChimeraConfig");
	string tempValue;

	try {
		if (is2D) {
			tempValue = pChimeraConfig->FirstChildElement("SimulationType")->GetText();

			if (tempValue == "meshCutter") {
				//pApplication = new MeshCutterApplication(argc, argv, pChimeraConfig);
			}
			else if (tempValue == "realTimeSimulation") {
				pApplication2D = new Chimera::Applications::RealtimeSimulation2D<Chimera::Core::Vector2>(argc, argv, pChimeraConfig);
			}
			else if (tempValue == "precomputedAnimation") {
				//pApplication = new PrecomputedAnimation3D(argc, argv, pChimeraConfig);
			}
		}
		else {
			tempValue = pChimeraConfig->FirstChildElement("SimulationType")->GetText();

			if (tempValue == "meshCutter") {
				//pApplication = new MeshCutterApplication(argc, argv, pChimeraConfig);
			}
			else if (tempValue == "realTimeSimulation") {
				pApplication3D = new Chimera::Applications::RealtimeSimulation3D<Chimera::Core::Vector3>(argc, argv, pChimeraConfig);
			}
			else if (tempValue == "precomputedAnimation") {
				//pApplication = new PrecomputedAnimation3D(argc, argv, pChimeraConfig);
			}
		}
	} catch (exception e) {
		Chimera::Core::exitProgram(e.what());
	}

	/** OpenGL callbacks */
	glutPassiveMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT); // same as MouseMotion
	glutSpecialFunc((GLUTspecialfun)TwEventSpecialGLUT);

	if (!is2D) {
		glutMouseFunc(MouseCallback3D);
		glutMotionFunc(MotionCallback3D);
		glutKeyboardFunc(KeyboardCallback3D);
		glutSpecialFunc(SpecialKeyboardCallback3D);
		glutSpecialUpFunc(SpecialKeyboardUpCallback3D);
		glutReshapeFunc(ReshapeCallback3D);
		atexit(ExitCallback3D);
		glutDisplayFunc(RenderCallback3D);
	}
	else {
		glutMouseFunc(MouseCallback2D);
		glutMotionFunc(MotionCallback2D);
		glutKeyboardFunc(KeyboardCallback2D);
		glutSpecialFunc(SpecialKeyboardCallback2D);
		glutSpecialUpFunc(SpecialKeyboardUpCallback2D);
		glutReshapeFunc(ReshapeCallback2D);
		atexit(ExitCallback2D);
		glutDisplayFunc(RenderCallback2D);
	} 
	
	glutIdleFunc(IdleCallback);
	glutMainLoop();

	return 0;
}