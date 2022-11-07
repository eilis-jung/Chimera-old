#include "Windows/GLRenderer2DWindow.h"

namespace Chimera {
	namespace Windows {

		GLRenderer2DWindow::GLRenderer2DWindow(const params_t &windowsParams) : 
			BaseWindow(Vector2(1588, 675), Vector2(300, 90), "Renderer General Options"), m_params(windowsParams) {

			if(windowsParams.m_pDrawObjectsMeshes) {
				TwAddVarRW(m_pBaseBar, "drawSolids", TW_TYPE_BOOL8, windowsParams.m_pDrawObjectsMeshes, "label='Draw Solid Objects' group='Drawing'");
			}
			if (windowsParams.m_pDrawLiquidMeshes) {
				TwAddVarRW(m_pBaseBar, "drawLiquids", TW_TYPE_BOOL8, windowsParams.m_pDrawLiquidMeshes, "label='Draw Liquids' group='Drawing'");
			}
			if (windowsParams.m_pCameraFollowObject) {
				TwAddVarRW(m_pBaseBar, "followObject", TW_TYPE_BOOL8, windowsParams.m_pCameraFollowObject, "label='Follow Object' group='Camera Control'");
			}
			if (windowsParams.m_pObjectIDToFollow) {
				//Ble
			}	
		}

		void GLRenderer2DWindow::update()
		{
		}


	}
}