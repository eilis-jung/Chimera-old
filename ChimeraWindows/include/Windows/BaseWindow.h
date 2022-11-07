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

#ifndef _RENDERING_BASE_WINDOW_H
#define _RENDERING_BASE_WINDOW_H

#pragma  once
#include "ChimeraCore.h"
#include "AntTweakBar.h"


using namespace std;
namespace Chimera {

	using namespace Core;

	namespace Windows {


		class BaseWindow {

		public:

			/** Visualization types */
			typedef enum scalarVisualization_t {
				drawNoScalarField,
				drawVorticity,
				drawLevelSet,
				drawPressure,
				drawDivergent,
				drawVelocityMagnitude,
				drawDensityField,
				drawTemperature,
				drawStreamfunction,
				drawFineGridScalars,
				drawKineticEnergy,
				drawKineticEnergyChange
			} scalarVisualization_t;

			typedef enum vectorVisualization_t {
				drawNoVectorField,
				drawStaggeredVelocity,
				drawVelocity,
				drawNodeVelocity,
				drawAuxiliaryVelocity,
				drawGradients
			} vectorVisualization_t;

			BaseWindow(const Vector2 &position, const Vector2 size, const string &windowName) 
				: m_position(position), m_size(size), m_windowName(windowName) {
				m_pBaseBar = TwNewBar(windowName.c_str());
				string commandString;
				commandString = "'" + windowName + "'" + " ";
				commandString += " position='" + scalarToStr(m_position.x) + " " + scalarToStr(m_position.y) + "'";
				commandString += " size='" + scalarToStr(m_size.x) + " " + scalarToStr(m_size.y) + "'";
				commandString += " color='" + intToStr(m_windowColor.getRed()) + " " + intToStr(m_windowColor.getGreen()) 
														+ " " + intToStr(m_windowColor.getBlue()) + "'";
				commandString += " alpha='" + scalarToStr(m_windowAlpha) + "'";
				TwDefine(commandString.c_str());
				//"'Simulation Status' position='16 616' size='300 140' color='20 15 30' alpha=220"
			}

			/** Initialize base window with previous TwBaseBar configuration */
			BaseWindow(BaseWindow *pBaseWindow) { 
				m_pBaseBar = pBaseWindow->getBaseTwBar();
				m_position = pBaseWindow->getWindowPosition();
				m_size = pBaseWindow->getWindowSize();
				m_windowName = pBaseWindow->getWindowName();
			}
			
			virtual ~BaseWindow() {
				TwDeleteBar(m_pBaseBar);
			}

			virtual void update() = 0;

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			void setWindowName(const string &windowName) {
				m_windowName = windowName;
			}

			const string & getWindowName() const {
				return m_windowName;
			}

			void setWindowPosition(const Vector2 &windowPosition) {
				m_position = windowPosition;

				string commandString;
				commandString = "'" + m_windowName + "'" + " ";
				commandString += " position='" + scalarToStr(m_position.x) + " " + scalarToStr(m_position.y) + "'";
				
				TwDefine(commandString.c_str());
			}

			const Vector2 & getWindowPosition() const {
				return m_position;
			}

			void setWindowSize(const Vector2 &windowSize) {
				m_size = windowSize;

				string commandString;
				commandString = "'" + m_windowName + "'" + " ";
				commandString += " size='" + scalarToStr(m_size.x) + " " + scalarToStr(m_size.y) + "'";

				TwDefine(commandString.c_str());
			}

			const Vector2 & getWindowSize() const {
				return m_size;
			}

			TwBar * getBaseTwBar() const {
				return m_pBaseBar;
			}



		protected:	
			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			/** Windows unique ID name */
			string m_windowName;

			/**Position */
			Vector2 m_position;
			/** Size */
			Vector2 m_size;

			/**Ant-tweak bar */
			TwBar *m_pBaseBar;

			/************************************************************************/
			/* Static members                                                       */
			/************************************************************************/
			static Color m_windowColor;
			static Scalar m_windowAlpha;
		};

		
	}
}


#endif
