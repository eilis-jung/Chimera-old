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

#ifndef _CHIMERA_MESH_CUTTER_H_
#define _CHIMERA_MESH_CUTTER_H_
#pragma  once

/************************************************************************/
/* Core                                                                 */
/************************************************************************/
#include "ChimeraCore.h"

/************************************************************************/
/* Math                                                                 */
/************************************************************************/
#include "ChimeraMath.h"

/************************************************************************/
/* Data                                                                 */
/************************************************************************/
#include "ChimeraData.h"

/************************************************************************/
/* Rendering                                                            */
/************************************************************************/
#include "ChimeraRendering.h"

/************************************************************************/
/* CGAL Wrapper                                                         */
/************************************************************************/
#include "ChimeraCGALWrapper.h"

/************************************************************************/
/* Chimera 3D                                                           */
/************************************************************************/
#include "Applications/Application3D.h"
#include "Rendering/GLRenderer3D.h"

namespace Chimera {

	class VoxelCutterApplication : public Application3D {

	public:

		#pragma region Constructors
		VoxelCutterApplication(int argc, char** argv, TiXmlElement *pChimeraConfig);
		#pragma endregion Constructors
		
		#pragma region Functionalities
		void draw();
		void update();
		#pragma endregion

		void loadGridFile();

		#pragma region Callbacks
		void keyboardCallback(unsigned char key, int x, int y);
		void keyboardUpCallback(unsigned char key, int x, int y);
		void specialKeyboardCallback(int key, int x, int y);
		void specialKeyboardUpCallback(int key, int x, int y);
		void motionCallback(int x, int y);
		#pragma endregion

	private:
		#pragma region ClassMembers
		//Grid vars
		HexaGrid *m_pHexaGrid;
		Timer m_updateTimer;
		Scalar m_elapsedTime;
		PolygonSurface *m_pPolygonSurface;
		Plane *m_pCuttingPlane;
		bool m_rotatePlane;
		Array3D<vector<int>> *m_pTriangleRefs;
		#pragma endregion

		#pragma region LoadingFunctions
		void setupCamera(TiXmlElement *pCameraNode);
		#pragma endregion

	};
}
#endif