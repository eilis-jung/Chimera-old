#include "Windows/GeneralInfoWindow.h"

namespace Chimera {
	namespace Windows {
		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		GeneralInfoWindow::GeneralInfoWindow(Camera *pCamera) 
			: BaseWindow(Vector2(1588, 675), Vector2(300, 130), "General Info") {
			m_pCamera = pCamera;
			m_drawVectorInterpolation = true;

			TwDefine("'General Info' text='light' refresh=0.05");
			TwAddVarRO(m_pBaseBar, "mousePosition_x", TW_TYPE_FLOAT, &m_mousePosition.x, "label='Mouse position x' group='General'");
			TwAddVarRO(m_pBaseBar, "mousePosition_y", TW_TYPE_FLOAT, &m_mousePosition.y, "label='Mouse position y' group='General'");
			TwAddVarRO(m_pBaseBar, "drawVectorInterpolation", TW_TYPE_BOOL8, &m_drawVectorInterpolation, "label='Draw Vector interpolation' group='General'");
			TwAddVarRO(m_pBaseBar, "vectorInterp_x", TW_TYPE_FLOAT, &m_vectorInterpolation.x, "label='Interpolated Vector x' group='General'");
			TwAddVarRO(m_pBaseBar, "vectorInterp_y", TW_TYPE_FLOAT, &m_vectorInterpolation.y, "label='Interpolated Vector y' group='General'");
		}

		/**************************************s**********************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void GeneralInfoWindow::update() {
			m_mousePosition = m_pCamera->getWorldMousePosition();
		}

	}
}