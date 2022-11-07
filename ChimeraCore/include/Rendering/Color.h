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


#ifndef __CHIMERA_COLOR_H__
#define __CHIMERA_COLOR_H__

#pragma once

#include "Config/ChimeraConfig.h"
#include "GL/glew.h"
#include "GL/glut.h"


namespace Chimera {

	namespace Core {

		class Color {

		public:
			//Coloring vars
			unsigned char red;
			unsigned char green;
			unsigned char blue;
			unsigned char alpha;

			/************************************************************************/
			/* Static definitions                                                   */
			/************************************************************************/
			const static Color RED;
			const static Color BLUE;
			const static Color GREEN;
			const static Color PINK;
			const static Color YELLOW;
			const static Color CYAN;
			const static Color BLACK;
			const static Color WHITE;
			const static Color GRAY;

			/************************************************************************/
			/* ctors and dtors					                                    */
			/************************************************************************/
			//Default constructor
			Color();

			//Copy constructor
			Color(const Color &rhs);

			//Assignment constructor
			Color & operator=(const Color & rhs);

			//Initialization constructor
			explicit Color(unsigned char r,  unsigned char g, unsigned char b, unsigned char a = 0);

			//Destructor
			~Color();

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			FORCE_INLINE float getRed() const { return red/255.0f; }
			FORCE_INLINE float getBlue() const { return blue/255.0f; }
			FORCE_INLINE float getGreen() const { return green/255.0f; }

			FORCE_INLINE void setColors(unsigned char r, unsigned char g, unsigned char b) {
				red = r; green = g; blue = b;
			}

			FORCE_INLINE bool operator==(const Color &rhs) {
				if(red == rhs.red && green == rhs.green && blue == rhs.blue)
					return true;
				return false;
			}



		};
	}
}

#endif

