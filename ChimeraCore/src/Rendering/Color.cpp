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

#include "Rendering/Color.h"


namespace Chimera {
	namespace Core {
		/************************************************************************/
		/* Static definitions                                                   */
		/************************************************************************/
		//Primary
		const Color Color::RED(0xFF, 0, 0);
		const Color Color::GREEN(0, 0xFF, 0);
		const Color Color::BLUE(0, 0, 0xFF);
		//Secondary
		const Color Color::PINK(0xFF, 0, 0xFF);
		const Color Color::YELLOW(0xFF, 0xFF, 0);
		const Color Color::CYAN(0, 0xFF, 0xFF);
		//B&W
		const Color Color::BLACK(0, 0, 0);
		const Color Color::WHITE(0xFF, 0xFF, 0xFF);
		const Color Color::GRAY(124, 124, 124);

		/************************************************************************/
		/* ctors and dtors                                                      */
		/************************************************************************/
		Color::Color() {
			red = green = blue = 0x00;
		}
		Color::Color(const Color &rhs) {
			red = rhs.red;
			green = rhs.green;
			blue = rhs.blue;
			alpha = rhs.alpha;
		}

		Color & Color::operator=(const Color &rhs) {
			red = rhs.red;
			green = rhs.green;
			blue = rhs.blue;
			alpha = rhs.alpha;
			return(*this);
		} 


		Color::Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
			red = r;
			green = g;
			blue = b;
			alpha = a;
		}

		Color::~Color() {
			//Nothing to do
		}

	}
}
