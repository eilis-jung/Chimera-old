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

#include "Resources/Shaders/GLSLShader.h"

namespace Chimera {

	namespace Resources {

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		GLSLShader::GLSLShader(const string &vertexShader, const string &pixelShader) {
			m_isValid = false;
			m_vertexShaderID = createShader(GL_VERTEX_SHADER, vertexShader);
			m_pixelShaderID = createShader(GL_FRAGMENT_SHADER, pixelShader);
			m_programID = glCreateProgram();
			
			glAttachShader(m_programID, m_vertexShaderID);
			glAttachShader(m_programID, m_pixelShaderID);
			glDeleteShader(m_vertexShaderID);
			glDeleteShader(m_pixelShaderID);

			glLinkProgram(m_programID);
			m_isValid = checkProgram(m_programID);
		}
		GLSLShader::GLSLShader(GLuint shaderType, const string &filename, bool transformFeedback) {
			m_isValid = false;
			m_vertexShaderID = createShader(shaderType, filename);
			m_programID = glCreateProgram();
			
			glAttachShader(m_programID, m_vertexShaderID);
			glDeleteShader(m_vertexShaderID);

			if(transformFeedback) {
				GLchar const * Strings[] = {"vOut"}; 
				glTransformFeedbackVaryings(m_programID, 1, Strings, GL_SEPARATE_ATTRIBS);
			}

			glLinkProgram(m_programID);
			m_isValid = checkProgram(m_programID);
		}

		GLSLShader::GLSLShader(GLuint shaderType, const string &filename, int nVaryings, GLchar const * Strings[], GLenum attribsTypes) {
			m_isValid = false;
			m_vertexShaderID = createShader(shaderType, filename);
			m_programID = glCreateProgram();

			glAttachShader(m_programID, m_vertexShaderID);
			glDeleteShader(m_vertexShaderID);

			glTransformFeedbackVaryings(m_programID, nVaryings, Strings, attribsTypes);
			glLinkProgram(m_programID);
			m_isValid = checkProgram(m_programID);
		}

		/************************************************************************/
		/* Shader Functions                                                     */
		/************************************************************************/
		void GLSLShader::applyShader() const {
			glUseProgram(m_programID);
		}

		void GLSLShader::removeShader() const {
			glUseProgram(0);
		}

		void GLSLShader::updateParameters() {

		}

		void GLSLShader::initParameters() {

		}

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		std::string GLSLShader::loadShaderFile(std::string const & Filename) const {
			std::ifstream stream(Filename.c_str(), std::ios::in);

			if(!stream.is_open())
				return "";

			std::string Line = "";
			std::string Text = "";

			while(getline(stream, Line))
				Text += "\n" + Line;

			stream.close();

			return Text;
		}

		bool GLSLShader::checkShader(GLuint ShaderName, const char* Source) {
			if(!ShaderName)
				return false;

			GLint Result = GL_FALSE;
			glGetShaderiv(ShaderName, GL_COMPILE_STATUS, &Result);

			Logger::getInstance()->log("Compiling shader...", Log_LowPriority);
			Logger::getInstance()->log(Source, Log_LowPriority);

			int InfoLogLength;
			glGetShaderiv(ShaderName, GL_INFO_LOG_LENGTH, &InfoLogLength);
			if (InfoLogLength != 0) {
				std::vector<char> Buffer(InfoLogLength);
				glGetShaderInfoLog(ShaderName, InfoLogLength, NULL, &Buffer[0]);

				if (Result != GL_TRUE) {
					Logger::getInstance()->log(&Buffer[0], Log_HighPriority);
				}
				else {
					Logger::getInstance()->log(&Buffer[0], Log_LowPriority);
				}
			}
			
			return Result == GL_TRUE;
		}

		GLuint GLSLShader::createShader(GLenum Type, string const & Source) {
			bool Validated = true;
			GLuint Name = 0;

			if(!Source.empty()) {
				std::string SourceContent = loadShaderFile(Source);
				char const * SourcePointer = SourceContent.c_str();
				Name = glCreateShader(Type);
				glShaderSource(Name, 1, &SourcePointer, NULL);
				glCompileShader(Name);
				Validated = checkShader(Name, SourcePointer);
			}

			return Name;
		}
		bool GLSLShader::checkProgram(GLuint ProgramName) const {
			if(!ProgramName)
				return false;

			GLint Result = GL_FALSE;
			glGetProgramiv(ProgramName, GL_LINK_STATUS, &Result);

			fprintf(stdout, "Linking program\n");
			int InfoLogLength;
			glGetProgramiv(ProgramName, GL_INFO_LOG_LENGTH, &InfoLogLength);
			std::vector<char> Buffer(std::max(InfoLogLength, int(1)));
			glGetProgramInfoLog(ProgramName, InfoLogLength, NULL, &Buffer[0]);
			fprintf(stdout, "%s\n", &Buffer[0]);

			return Result == GL_TRUE;
		}
	}
}