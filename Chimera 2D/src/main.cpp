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

/************************************************************************/
/* Chimera 2D                                                           */
/************************************************************************/
#include "Applications/Application2D.h"
#include "Applications/RealtimeSimulation2D.h"
#include "Applications/PrecomputedAnimation2D.h"

Chimera::Application2D *g_pApplication;

static void IdleCallback() {
	glutPostRedisplay();
}

static void ReshapeCallback(int width, int height) {
	g_pApplication->getRenderer()->reshapeCallback(width, height);
}

static void RenderCallback() {	
	g_pApplication->update();
	g_pApplication->draw();
}

static void MouseCallback(int button, int state, int x, int y) {
	g_pApplication->mouseCallback(button, state, x, y);
}

static void MotionCallback(int x, int y) {
	g_pApplication->motionCallback(x, y);
}


static void KeyboardCallback(unsigned char key, int x, int y) {
	g_pApplication->keyboardCallback(key, x, y);
}

static void KeyboardUpCallback(unsigned char key, int x, int y) {
	g_pApplication->keyboardUpCallback(key, x, y);
}

static void SpecialKeyboardCallback(int key, int x, int y) {
	g_pApplication->specialKeyboardCallback(key, x, y);
}


static void SpecialKeyboardUpCallback(int key, int x, int y) {
	g_pApplication->specialKeyboardUpCallback(key, x, y);
}


int main(int argc, char** argv) {
	
	shared_ptr<Chimera::Rendering::XMLDoc> pMainConfigFile;
	string chimeraFile;
	try {
		//Static configuration file - do not change
		shared_ptr<Chimera::Rendering::XMLDoc> pTempConfigFile = Chimera::Rendering::ResourceManager::getInstance()->loadXMLDocument("Configuration/2D/ChimeraCFG.xml");
		chimeraFile = pTempConfigFile->FirstChildElement("ChimeraFile")->GetText();
		pMainConfigFile = Chimera::Rendering::ResourceManager::getInstance()->loadXMLDocument(chimeraFile);
	} catch (exception e) {
		Chimera::Core::exitProgram(e.what());
	}

	TiXmlElement *pChimeraConfig = pMainConfigFile->FirstChildElement("ChimeraConfig");
	string tempValue;


	if(string(pChimeraConfig->FirstChildElement("SimulationType")->GetText()) == "realTimeSimulation") {
		g_pApplication = new Chimera::RealtimeSimulation2D(argc, argv, pChimeraConfig);
	} else if (string(pChimeraConfig->FirstChildElement("SimulationType")->GetText()) == "precomputedAnimation") {
		g_pApplication = new Chimera::PrecomputedAnimation2D(argc, argv, pChimeraConfig);
	}

	/** OpenGL callbacks */
	glutPassiveMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT); // same as MouseMotion
	glutSpecialFunc((GLUTspecialfun)TwEventSpecialGLUT);

	glutMouseFunc(MouseCallback);
	glutMotionFunc(MotionCallback);
	glutKeyboardFunc(KeyboardCallback);
	glutKeyboardUpFunc(KeyboardUpCallback);
	glutIdleFunc(IdleCallback);
	glutDisplayFunc(RenderCallback);
	glutReshapeFunc(ReshapeCallback);
	glutSpecialFunc(SpecialKeyboardCallback);
	glutSpecialUpFunc(SpecialKeyboardUpCallback);

	glutMainLoop();

	return (0);

}

