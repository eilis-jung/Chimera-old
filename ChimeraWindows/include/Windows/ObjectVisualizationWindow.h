////  Copyright (c) 2013, Vinicius Costa Azevedo
////	All rights reserved.
////
////	Redistribution and use in source and binary forms, with or without
////	modification, are permitted provided that the following conditions are met: 
////
////1. Redistributions of source code must retain the above copyright notice, this
////	list of conditions and the following disclaimer. 
////	2. Redistributions in binary form must reproduce the above copyright notice,
////	this list of conditions and the following disclaimer in the documentation
////	and/or other materials provided with the distribution. 
////
////	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
////	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
////	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
////	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
////	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
////	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
////LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
////	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
////	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
////	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////
////	The views and conclusions contained in the software and documentation are those
////	of the authors and should not be interpreted as representing official policies, 
////	either expressed or implied, of the FreeBSD Project.
//
//#ifndef _RENDERING_OBJECT_VISUALIZATION_WINDOW_H
//#define _RENDERING_OBJECT_VISUALIZATION_WINDOW_H
//
//#pragma  once
//
///************************************************************************/
///* Data                                                                 */
///************************************************************************/
//#include "ChimeraData.h"
//
///************************************************************************/
///* Rendering                                                            */
///************************************************************************/
//#include "Windows/BaseWindow.h"
//#include "Primitives/Object3D.h"
//
//namespace Chimera {
//	namespace Windows {
//
//		class ObjectVisualizationWindow : public BaseWindow {
//
//
//		private:
//			#pragma region ClassMembers
//			Object3D *m_pObject;
//
//			/** Grid */
//			bool m_drawPoints;
//			bool m_drawWireframe;
//			bool m_drawShaded;
//			bool m_drawNormals;
//			#pragma endregion
//
//		public:
//			#pragma region Constructors
//			ObjectVisualizationWindow(Object3D *pObject);
//			#pragma endregion
//
//			#pragma region AccessFunctions
//			FORCE_INLINE bool drawPoints() const {
//				return m_drawPoints;
//			}
//
//			FORCE_INLINE bool drawWireframe() const {
//				return m_drawWireframe;
//			}
//
//			FORCE_INLINE bool drawShaded() const {
//				return m_drawShaded;
//			}
//			#pragma endregion
//
//			#pragma region Functionalities
//			virtual void update();
//			#pragma endregion
//		};
//	}
//}
//
//#endif