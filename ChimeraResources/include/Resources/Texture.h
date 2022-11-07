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


#ifndef __RENDERING_TEXTURE_H_
#define __RENDERING_TEXTURE_H_
#pragma once

#include "ChimeraCore.h"
#include "Resources/Image.h"

using namespace std;

namespace Chimera {

	using namespace Core;

	namespace Resources {
		
		class Texture : public Resource {
		public:

			/************************************************************************/
			/* Internal structures                                                  */
			/************************************************************************/
			typedef struct textureParams_t {
				GLenum textureType;
				GLint internalFormat;
				GLint filterMode;
				GLenum textureFormat;
				GLenum  clampMode;
				int width;
				int height;

				textureParams_t() {
					textureType = GL_TEXTURE_2D;
					internalFormat = GL_RGBA16F_ARB;
					filterMode = GL_LINEAR;
					textureFormat = GL_RGBA;
					clampMode = GL_CLAMP_TO_EDGE;
					width = height = 0;
				}
			} textureParams_t;

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			/** Default texture creation: a string file is passed and loaded with SDL library */
			Texture(const string& strFilePath);

			/** Customized texture creation: create an empty texture with configurable parameters. */
			Texture(const textureParams_t &textureParams);
			

			/************************************************************************/
			/* Custom operators                                                     */
			/************************************************************************/
			const bool operator==(const Texture& rhs) const;

			/************************************************************************/
			/* Access functions                                                    */
			/************************************************************************/
			const GLuint	getID() const { return m_textureID; }
			const bool		isTransparent() const { return m_isTransparent; }
			const bool		isRegistred() const { return NULL == m_pImage.get(); }
			const textureParams_t & getTextureParams() const { return m_params; }

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			void	registerGL();

			static const GLuint	INVALID_ID;

		private:
			
			/************************************************************************/
			/* Private Functionalities                                              */
			/************************************************************************/
			inline void	testInvariant() const;

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			GLuint	m_textureID;
			bool	m_isTransparent;
			shared_ptr<Image>	m_pImage;
			textureParams_t m_params;
		};


		//
		void Texture::testInvariant() const
		{
			assert(NULL !=  m_pImage.get() || INVALID_ID != m_textureID);
		}

#endif //_TEXTURE_H_
	}
}
