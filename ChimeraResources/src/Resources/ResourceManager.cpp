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

/************************************************************************/
/* Resources                                                            */
/************************************************************************/
#include "Resources/ResourceManager.h"


namespace Chimera {

	namespace Resources {

		/************************************************************************/
		/* ctors and dtors                                                      */
		/************************************************************************/

		ResourceManager::ResourceManager() {
			m_branchPath = "";
		}

		ResourceManager::ResourceManager(const ResourceManager &rhs) {

		}
		
		ResourceManager & ResourceManager::operator= (const ResourceManager &rhs) {
			return(*this);
		}
		ResourceManager::~ResourceManager() {

		}

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		shared_ptr<Image> ResourceManager::loadImage(const string &filePath) {
			/** Using relative file path */
			shared_ptr<Resource> pResource = getObject(filePath).lock();
			shared_ptr<Image> pImg;
			if(pResource == NULL) {
				try {
					pImg = shared_ptr<Image>(new Image(filePath));
				} catch(exception e) {
					Logger::get() << "#### Caught exception: " << e.what() << endl;
				}
				createObject(filePath, pImg);
			} else {
				pImg = shared_ptr<Image>(dynamic_pointer_cast<Image>(pResource));
			}

			return(pImg);
		}

		shared_ptr<Texture> ResourceManager::loadTexture(const string &filePath) {
			shared_ptr<Resource> pResource = getObject(filePath).lock();
			shared_ptr<Texture> pTexture;
			if(pResource == NULL) {
				Logger::get() << "Loading texture: " << filePath << endl;
				pTexture = shared_ptr<Texture>(new Texture(filePath));
				createObject(filePath, pTexture);
			} else {
				Logger::get() << "Retrieving texture: " << filePath << endl;
				pTexture = shared_ptr<Texture>(dynamic_pointer_cast<Texture>(pResource));
			}

			return(pTexture);
		}


		shared_ptr<GLSLShader> ResourceManager::loadGLSLShader(const string &vertexShader, const string &pixelShader) {
			shared_ptr<Resource> pResource = getObject(vertexShader + pixelShader).lock();
			shared_ptr<GLSLShader> pShader;
			if(pResource == NULL) {
				pShader = shared_ptr<GLSLShader>(new GLSLShader(vertexShader, pixelShader));
				createObject(vertexShader + pixelShader, pShader);
			} else {
				pShader = shared_ptr<GLSLShader>(dynamic_pointer_cast<GLSLShader>(pResource));
			}

			return(pShader);
		}

		shared_ptr<GLSLShader> ResourceManager::loadGLSLShader(GLuint shaderType, const string &filePath, bool transformFeedback) {
			shared_ptr<Resource> pResource = getObject(filePath).lock();
			shared_ptr<GLSLShader> pShader;
			if(pResource == NULL) {
				pShader = shared_ptr<GLSLShader>(new GLSLShader(shaderType, filePath, transformFeedback));
				createObject(filePath, pShader);
			} else {
				pShader = shared_ptr<GLSLShader>(dynamic_pointer_cast<GLSLShader>(pResource));
			}

			return(pShader);
		}

		shared_ptr<GLSLShader> ResourceManager::loadGLSLShader(GLuint shaderType, const string &filePath, int nVaryings, 
																GLchar const * Strings[], GLenum attribsTypes) {
			shared_ptr<Resource> pResource = getObject(filePath).lock();
			shared_ptr<GLSLShader> pShader;
			if(pResource == NULL) {
				pShader = shared_ptr<GLSLShader>(new GLSLShader(shaderType, filePath, nVaryings, Strings, attribsTypes));
				createObject(filePath, pShader);
			} else {
				pShader = shared_ptr<GLSLShader>(dynamic_pointer_cast<GLSLShader>(pResource));
			}

			return(pShader);
		}


		shared_ptr<XMLDoc> ResourceManager::loadXMLDocument(const string &filePath)
		{
			shared_ptr<Resource> pResource = getInstance()->getObject(filePath).lock();
			shared_ptr<XMLDoc> pXML;
			if(pResource == NULL) {
				pXML = shared_ptr<XMLDoc>(new XMLDoc(filePath));
				
				if(!pXML->loadFile())
					throw exception(filePath.c_str());

				getInstance()->createObject(filePath, pXML);
			} else {
				pXML = shared_ptr<XMLDoc>(dynamic_pointer_cast<XMLDoc>(pResource));
			}
			return(pXML);
		}
	}



}