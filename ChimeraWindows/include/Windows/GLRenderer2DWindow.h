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

#ifndef _RENDERING_GL_RENDERER_2D_WINDOW_H
#define _RENDERING_GL_RENDERER_2D_WINDOW_H

#pragma  once

#include "Windows/BaseWindow.h"

namespace Chimera {
	namespace Windows {

		class GLRenderer2DWindow : public BaseWindow {

		public:
			typedef struct params_t {
				/** Drawing meshes */
				bool *m_pDrawLiquidMeshes;
				bool *m_pDrawObjectsMeshes;

				/** Camera control*/
				bool *m_pCameraFollowObject;
				int *m_pObjectIDToFollow;

				params_t() {
					m_pDrawObjectsMeshes = m_pDrawObjectsMeshes = m_pCameraFollowObject = NULL;
					m_pObjectIDToFollow = NULL;
				}
			} params_t;

			#pragma region Constructors
			GLRenderer2DWindow(const params_t &windowsParams);
			#pragma endregion

			#pragma region AccessFunctions
			
			#pragma endregion

			#pragma region Functionalities
			virtual void update();
			#pragma endregion

		private:
			#pragma region ClassMembers
			params_t m_params;
			#pragma endregion

		};
	}
}

#endif
#pragma once
