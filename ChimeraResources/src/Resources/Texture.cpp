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


#include "Resources/Texture.h"
#include "Resources/Image.h"
#include "Resources/ResourceManager.h"

namespace Chimera {
	namespace Resources {


		const GLuint Texture::INVALID_ID = (GLuint) ~1;

		/************************************************************************/
		/* Ctors                                                                */
		/************************************************************************/
		Texture::Texture(const string& strFilePath) : m_textureID(INVALID_ID) {
			m_pImage = ResourceManager::getInstance()->loadImage(strFilePath);
			m_isTransparent = m_pImage->isTransparent();
		}

		Texture::Texture(const textureParams_t &textureParams) : m_textureID(INVALID_ID), m_params(textureParams)  {
				glGenTextures(1, &m_textureID);
				glBindTexture(textureParams.textureType, m_textureID);

				glTexParameteri(textureParams.textureType, GL_TEXTURE_MAG_FILTER, textureParams.filterMode);
				glTexParameteri(textureParams.textureType, GL_TEXTURE_MIN_FILTER, textureParams.filterMode);

				glTexParameteri(textureParams.textureType, GL_TEXTURE_WRAP_S, textureParams.clampMode);
				glTexParameteri(textureParams.textureType, GL_TEXTURE_WRAP_T, textureParams.clampMode);

				glTexImage2D(textureParams.textureType, 0, textureParams.internalFormat, 
								textureParams.width, textureParams.height, 0, textureParams.textureFormat, GL_FLOAT, 0);
		}


		/************************************************************************/
		/* Operators                                                            */
		/************************************************************************/
		//
		const bool Texture::operator==(const Texture& rhs) const
		{
			return (m_pImage.get() == rhs.m_pImage.get()) && (getID() == rhs.getID());
		}


		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void Texture::registerGL() {
			testInvariant();

			glGenTextures (1, &m_textureID);
			glBindTexture (GL_TEXTURE_2D, m_textureID);

			glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			m_pImage->convertPixelFormat();
			const int w = m_pImage->getWidth();
			const int h = m_pImage->getHeight();
			//	if w and h are powers of 2, build mipmaps
			if (((w & (w-1)) == 0) && ((h & (h-1)) == 0))  {
				gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, w, h, GL_RGBA, GL_UNSIGNED_BYTE, m_pImage->getPixels());
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_NEAREST);
			}
			else {
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_pImage->getPixels());
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			}

			
			//	we don't need the image anymore
			m_pImage.reset();

			testInvariant();
		}





	}
}
