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

#ifndef _CHIMERA_INTERPOLATION_TEST_H_
#define _CHIMERA_INTERPOLATION_TEST_H_
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

	class InterpolationTestApplication : public Application3D {

	public:
		#pragma region InternalStructures
		typedef enum nodeValuesMethod_t {
			randomValues,
			signedDistanceValues
		};
		#pragma endregion 
		
		#pragma region Constructors
		InterpolationTestApplication(int argc, char** argv, TiXmlElement *pChimeraConfig);
		#pragma endregion Constructors
		
		#pragma region Functionalities
		void draw();
		void drawInterpolationValues();
		void update();
		#pragma endregion
		
		void loadGridFile();

		Scalar trilinearInterpolation(const Vector3 &position, LinearInterpolationMethod_t interpolationMethod);
		void meanValueCoordinatesWeights(const Vector3 &position, vector<Scalar> &weights);
		void sphericalBarycentricCoordinatesWeights(const Vector3 &position, vector<Scalar> &weights);

		#pragma region Callbacks
		void keyboardCallback(unsigned char key, int x, int y);
		void keyboardUpCallback(unsigned char key, int x, int y);
		void specialKeyboardCallback(int key, int x, int y);
		void specialKeyboardUpCallback(int key, int x, int y);
		void motionCallback(int x, int y);
		#pragma endregion

	private:
		#pragma region ClassMembers
		PolygonSurface *m_pPolygonSurface;
		/** This vector has size equal to the number of vertices of the polygon surface */
		vector<Scalar> m_polygonInterpolationValues;

		LinearInterpolationMethod_t m_interpolationMethod;
		nodeValuesMethod_t m_nodeValuesMethod;

		vector<Vector3> m_interpolationPoints;
		vector<Scalar> m_interpolationValues;
		#pragma endregion

		#pragma region LoadingFunctions
		void setupCamera(TiXmlElement *pCameraNode);
		#pragma endregion

		#pragma region PrivateFunctionalities
		Vector3D gnomonicProjection(const Vector3D &vec, const Vector3D &spherePoint, const Vector3D &normal);
		#pragma endregion

	};
}
#endif