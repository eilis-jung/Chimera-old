#include "stdio.h"

/** Applications */
#include "Applications/Application3D.h"
#include "Applications/PrecomputedAnimation3D.h"
#include "Applications/RealtimeSimulation3D.h"
#include "Applications/MeshCutterApplication.h"
//#include "Applications/InterpolationTestApplication.h"


using namespace Chimera;

Application3D *pApplication;

static void MouseCallback(int button, int state, int x, int y) {
	pApplication->mouseCallback(button, state, x, y);
}

static void MotionCallback(int x, int y) {
	pApplication->motionCallback(x, y);
}


static void KeyboardCallback(unsigned char key, int x, int y) {
	pApplication->keyboardCallback(key, x, y);
}
static void SpecialKeyboardCallback(int key, int x, int y) {
	pApplication->specialKeyboardCallback(key, x, y);
}

static void SpecialKeyboardUpCallback(int key, int x, int y) {
	pApplication->specialKeyboardUpCallback(key, x, y);
	
}

static void ExitCallback() {
	pApplication->exitCallback();
}


static void IdleCallback() {
	glutPostRedisplay();
}

static void ReshapeCallback(int width, int height) {
	pApplication->reshapeCallback(width, height);
}


static void RenderCallback() {
	pApplication->draw();
	pApplication->update();
}


int main(int argc, char** argv) {
	/** To-be initialized within initialization functions */
	string chimeraFile;

	shared_ptr<XMLDoc> pMainConfigFile;
	{
		//Static configuration file - do not change
		shared_ptr<XMLDoc> pTempConfigFile = ResourceManager::getInstance()->loadXMLDocument("Configuration/3D/ChimeraCFG.xml");
		chimeraFile = pTempConfigFile->FirstChildElement("ChimeraFile")->GetText();
		pMainConfigFile = ResourceManager::getInstance()->loadXMLDocument(chimeraFile);
	}

	TiXmlElement *pChimeraConfig = pMainConfigFile->FirstChildElement("ChimeraConfig");
	string tempValue;
	/** Main try-catch node: XMLParseErrorExceptions*/
	try {
		tempValue = pChimeraConfig->FirstChildElement("SimulationType")->GetText(); 
		
		if (tempValue == "meshCutter") {
			pApplication = new MeshCutterApplication(argc, argv, pChimeraConfig);
		} else if(tempValue == "realTimeSimulation") {
			pApplication = new RealtimeSimulation3D(argc, argv, pChimeraConfig);
		} else if(tempValue == "precomputedAnimation") {
			pApplication = new PrecomputedAnimation3D(argc, argv, pChimeraConfig);
		}

		//if(tempValue == "realTimeSimulation") { /** Simulation type */
		//	pApplication = new RealtimeSimulation3D(argc, argv, pChimeraConfig);
		//} else if(tempValue == "precomputedAnimation") {
		//	pApplication = new PrecomputedAnimation3D(argc, argv, pChimeraConfig);
		//} else if(tempValue == "offlineSimulation")  {
		//	pApplication = new RealtimeSimulation3D(argc, argv, pChimeraConfig);
		//} else if(tempValue == "meshCutter") {
		//	pApplication = new MeshCutterApplication(argc, argv, pChimeraConfig);
		//}
		//else if (tempValue == "interpolationTest") {
		//	pApplication = new InterpolationTestApplication(argc, argv, pChimeraConfig);
		//}
	} catch(exception e){
		Logger::get() << e.what() << endl;
		return (1);
	}

	/** OpenGL callbacks */
	glutPassiveMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT); // same as MouseMotion
	glutSpecialFunc((GLUTspecialfun)TwEventSpecialGLUT);

	glutMouseFunc(MouseCallback);
	glutMotionFunc(MotionCallback);
	glutKeyboardFunc(KeyboardCallback);
	glutSpecialFunc(SpecialKeyboardCallback);
	glutSpecialUpFunc(SpecialKeyboardUpCallback);
	glutIdleFunc(IdleCallback);
	glutReshapeFunc(ReshapeCallback);
	atexit(ExitCallback);
	glutDisplayFunc(RenderCallback);
	
	glutMainLoop();
}