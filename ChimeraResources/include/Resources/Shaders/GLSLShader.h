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


#ifndef __RENDERING_GLSLSHADER_H_
#define __RENDERING_GLSLSHADER_H_

#pragma once


#include "ChimeraCore.h"
#include "Resources/Resource.h"

namespace Chimera {

	using namespace std;

	namespace Resources {


		class GLSLShader : public Resource {

		public:

			/************************************************************************/
			/* Ctors                                                                */
			/************************************************************************/
			/** Default shaders */
			GLSLShader(const string &vertexShader, const string &pixelShader);

			/** Transform feedback shaders*/
			GLSLShader(GLuint shaderType, const string &fileString, bool transformFeeback = true);
			GLSLShader(GLuint shaderType, const string &fileString, int nVaryings, GLchar const * Strings[], GLenum attribsTypes);

			


			/************************************************************************/
			/* Shader Functions                                                     */
			/************************************************************************/
			void applyShader() const;
			void removeShader() const ;

			virtual void updateParameters();


			/************************************************************************/
			/* Access functions														*/
			/************************************************************************/

			inline GLuint getProgramID() const {
				return m_programID;
			}

			inline GLuint getShaderType() const {
				return m_shaderType;
			}

			inline GLuint getVertexShaderID() const {
				return m_vertexShaderID;
			}

			inline GLuint getPixelShaderID() const {
				return m_pixelShaderID;
			}
		protected:

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			std::string loadShaderFile(std::string const & Filename) const;
			bool checkShader(GLuint ShaderName, const char* Source);
			GLuint createShader(GLenum Type, std::string const & Source);
			bool checkProgram(GLuint programID) const;


			virtual void initParameters();

			/************************************************************************/
			/* Class members	                                                    */
			/************************************************************************/
			GLuint m_shaderType;
			GLuint m_programID;
			
			GLuint m_vertexShaderID;
			GLuint m_pixelShaderID;

			bool m_isValid;

		};
	}
}

#endif

