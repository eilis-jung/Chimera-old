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


#include "SDL/SDL_image.h"
#include "Resources/Image.h"

namespace Chimera {

	namespace Resources {

		//
		Image::Image(const string &strFilePath)  : Resource (strFilePath) {
			//	load the image file
			m_pSurface = IMG_Load(strFilePath.c_str());

			if (NULL == m_pSurface)
				throw exception(SDL_GetError());

			testInvariant();
		}


		//
		bool Image::isTransparent() const
		{
			return false;
			//return (0 != m_pSurface->format->Amask) || (0 != m_pSurface->format->colorkey);
		}


		//
		void Image::convertPixelFormat()
		{
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
			Uint32 rmask = 0xff000000;
			Uint32 gmask = 0x00ff0000;
			Uint32 bmask = 0x0000ff00;
			Uint32 amask = 0x000000ff;
#else
			Uint32 rmask = 0x000000ff;
			Uint32 gmask = 0x0000ff00;
			Uint32 bmask = 0x00ff0000;
			Uint32 amask = 0xff000000;
#endif	
			//SDL_SetAlpha(m_pSurface, 0, SDL_ALPHA_TRANSPARENT);
			SDL_Surface* pAlphaSurface = SDL_CreateRGBSurface(SDL_SWSURFACE, getWidth(), getHeight(),
				32, rmask, gmask, bmask, amask);
			SDL_BlitSurface(m_pSurface, NULL, pAlphaSurface, NULL);
			SDL_FreeSurface(m_pSurface);
			m_pSurface = pAlphaSurface;
		}

	}
}