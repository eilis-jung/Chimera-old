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

#ifndef __CHIMERA_RESOURCE_MANAGER
#define __CHIMERA_RESOURCE_MANAGER
#pragma once

#include "ChimeraCore.h"
#include "Resources/Resource.h"

/** Materials, textures */
#include "Resources/Image.h"
#include "Resources/Texture.h"
#include "Resources/Shaders/GLSLShader.h"

/** Docs */
#include "Resources/XMLDoc.h"

using namespace std;

namespace Chimera {
	
	using namespace Core;

	namespace Resources {

		class ResourceManager : public Singleton<ResourceManager> {

		protected:

			//Accessing the objects through strings
			map<string, weak_ptr<Resource> > m_objStringMap;
			string m_branchPath;

		public:

			/************************************************************************/
			/* ctors and dtors					                                    */
			/************************************************************************/
			//Default constructor
			explicit ResourceManager();

			//Copy constructor
			explicit ResourceManager(const ResourceManager &rhs);

			//Assignment constructor
			ResourceManager& operator=(const ResourceManager &rhs);

			//Destructor
			~ResourceManager();

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			inline weak_ptr<Resource> getObject(const string &objName) {
				map<string, weak_ptr<Resource> >::const_iterator iter = m_objStringMap.find(objName);
				if(iter == m_objStringMap.end()) 
					return weak_ptr<Resource>();
				return(iter->second);
			}


			inline void createObject(const string &objName, weak_ptr<Resource> pResource) {
				m_objStringMap[objName] = pResource;
			}

			/************************************************************************/
			/* Functionalities                                                      */	
			/************************************************************************/
			/* Material */
			shared_ptr<Image>			loadImage(const string &filePath);
			shared_ptr<Texture>			loadTexture(const string &filePath);
			shared_ptr<GLSLShader>		loadGLSLShader(const string &vertexShader, const string &pixelShader);
			shared_ptr<GLSLShader>		loadGLSLShader(GLuint shaderType, const string &filePath, bool transformFeedback = true);
			shared_ptr<GLSLShader>		loadGLSLShader(GLuint shaderType, const string &fileString, int nVaryings, 
														GLchar const * Strings[], GLenum attribsTypes);

			/**XML doc: throws CFileNotFoundException.*/
			shared_ptr<XMLDoc> loadXMLDocument(const string &filePath);
		};

	}
}




#endif