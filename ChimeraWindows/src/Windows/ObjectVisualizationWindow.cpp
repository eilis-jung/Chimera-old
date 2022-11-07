//#include "Windows/ObjectVisualizationWindow.h"
//
//namespace Chimera {
//	namespace Windows {
//		#pragma region Constructors
//		ObjectVisualizationWindow::ObjectVisualizationWindow(Object3D *pObject) : BaseWindow(Vector2(16, 16), Vector2(300, 150), pObject->getName()),
//			m_pObject(pObject) {
//			m_drawPoints = false;
//			m_drawShaded = true;
//			m_drawWireframe = true;
//			m_drawNormals = false;
//
//			TwAddVarRW(m_pBaseBar, "drawPoints", TW_TYPE_BOOL8, &m_pObject->getDrawPoints(), " label='Draw Points'");
//			TwAddVarRW(m_pBaseBar, "drawShaded", TW_TYPE_BOOL8, &m_pObject->getDrawShaded(), " label='Draw Shaded'");
//			TwAddVarRW(m_pBaseBar, "drawWireframe", TW_TYPE_BOOL8, &m_pObject->getDrawWireframe(), " label='Draw Wireframe'");
//			TwAddVarRW(m_pBaseBar, "drawNormals", TW_TYPE_BOOL8, &m_pObject->getDrawNormals(), " label='Draw Normals'");
//			TwAddVarRW(m_pBaseBar, "drawVertexNormals", TW_TYPE_BOOL8, &m_pObject->getDrawVertexNormals(), " label='Draw Vertex Normals'");
//			TwAddVarRW(m_pBaseBar, "drawEdgeNormals", TW_TYPE_BOOL8, &m_pObject->getDrawEdgeNormals(), " label='Draw Edge Normals'");
//			
//
//			TwAddVarRW(m_pBaseBar, "particleColor", TW_TYPE_COLOR4F, (void *)m_pObject->getColor(), "label='Color'");
//		}
//		#pragma endregion
//
//		#pragma region Functionalities
//		void ObjectVisualizationWindow::update() {
//
//		}
//		#pragma endregion
//	}
//}