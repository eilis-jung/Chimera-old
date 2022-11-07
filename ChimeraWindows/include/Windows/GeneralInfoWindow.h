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

#ifndef _RENDERING_GENERAL_INFO_WINDOW_H
#define _RENDERING_GENERAL_INFO_WINDOW_H

#pragma  once

#include "ChimeraCore.h"

#include "Windows/BaseWindow.h"

//Rendering cross ref
#include "Visualization/QuadGridRenderer.h"
#include "Primitives/Camera.h"

namespace Chimera {
	using namespace Rendering;
	namespace Windows {

		class GeneralInfoWindow : public BaseWindow {

			Vector2 m_mousePosition;
			Camera *m_pCamera;

			Vector2 m_vectorInterpolation;
			bool m_drawVectorInterpolation;

		public:
			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			GeneralInfoWindow(Camera *pCamera);

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			void update();

			void setVectorInterpolationPosition(const Vector2 &vectorPosition) {
				m_vectorInterpolation = vectorPosition;
			}

			bool drawVectorInterpolation() const {
				return m_drawVectorInterpolation;
			}

			const Vector2 & getInterpolatedVector() const {
				return m_vectorInterpolation;
			}
		};
	}
}

#endif