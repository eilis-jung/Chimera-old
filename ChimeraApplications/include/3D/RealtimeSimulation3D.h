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

#ifndef _CHIMERA_REALTIME_SIMULATION_APP_H_
#define _CHIMERA_REALTIME_SIMULATION_APP_H_
#pragma  once

#include "ApplicationBase.h"

namespace Chimera {

	namespace Applications {

		template <class VectorT>
		class RealtimeSimulation3D : public ApplicationBase<VectorT, Array3D> {

		public:
			#pragma region Constructors
			RealtimeSimulation3D(int argc, char** argv, TiXmlElement *pChimeraConfig);
			#pragma endregion

			#pragma region Functionalities		
			void draw();
			void update();
			#pragma endregion

			#pragma region Callbacks
			FORCE_INLINE virtual void mouseCallback(int button, int state, int x, int y) {
				m_pRenderer->mouseCallback(button, state, x, y);
			}

			FORCE_INLINE virtual void motionCallback(int x, int y) {
				m_pRenderer->motionCallback(x, y);
			}

			FORCE_INLINE virtual void keyboardCallback(unsigned char key, int x, int y) {
				m_pRenderer->keyboardCallback(key, x, y);
			}

			FORCE_INLINE virtual void keyboardUpCallback(unsigned char key, int x, int y) {

			}

			FORCE_INLINE virtual void specialKeyboardCallback(int key, int x, int y) {
				m_pRenderer->keyboardCallback(key, x, y);
			}

			FORCE_INLINE virtual void specialKeyboardUpCallback(int key, int x, int y) {
				m_pRenderer->keyboardCallback(key, x, y);
			}

			FORCE_INLINE virtual void reshapeCallback(int width, int height) {
				m_pRenderer->reshapeCallback(width, height);
			}

			FORCE_INLINE virtual void exitCallback() {

			}
			#pragma endregion

		private:	
			#pragma region ClassMembers
			/** Grid */
			HexaGrid *m_pHexaGrid;

			/** Renderer*/
			GLRenderer3D *m_pRenderer;
	
			/** Polygonal meshes */
			vector<Mesh<VectorT, Face> *> m_meshes;

			/** Polygonal meshes */
			vector<PolygonalMesh<VectorT> *> m_polyMeshes;
			#pragma endregion
		};
	}
}
#endif