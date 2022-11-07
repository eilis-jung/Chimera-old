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

#ifndef __RENDERING_IMAGE_H__
#define __RENDERING_IMAGE_H__
#pragma once

/************************************************************************/
/* Rendering                                                           */
/************************************************************************/
#include "ChimeraCore.h"
/** SDL */
#include "SDL/SDL_image.h"
/** Resources */
#include "Resources/Resource.h"

using namespace std;

namespace Chimera {

	using namespace Core;

	namespace Resources {

		class Image : public Resource {

		public:
			Image(const string & strFilePath);
			~Image() { SDL_FreeSurface(m_pSurface); m_pSurface = NULL; }

			bool isTransparent() const;
			//	PROMISE: return true if the alpha mask or the colorkey is not null.
			void convertPixelFormat();
			//	PROMISE: convert the pixel format to RGBA for OpenGL.

			const GLint getWidth() const { testInvariant(); return m_pSurface->w; }
			//	REQUIRE: the image must be loaded.
			const GLint getHeight() const { testInvariant(); return m_pSurface->h; }
			//	REQUIRE: the image must be loaded.
			const void* getPixels() const { testInvariant(); return m_pSurface->pixels; }
			//	REQUIRE: the image must be loaded.

			inline void testInvariant() const;

		private:
			Image(const Image& rhs);
			//  Don't copy ! Use shared_ptr<Image> and copy shared pointers instead.
			Image& operator=(const Image& rhs);
			//  Idem.

			SDL_Surface *m_pSurface;
		};


		//
		void Image::testInvariant() const
		{
			assert(NULL != m_pSurface);
		}

	}


}
#endif // __IMAGE_HPP
