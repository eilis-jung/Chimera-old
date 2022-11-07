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

#ifndef __RENDERING_MATERIAL_HPP__
#define __RENDERING_MATERIAL_HPP__
#pragma once

/************************************************************************/
/* Rendering                                                           */
/************************************************************************/
#include "ChimeraCore.h"
#include "Resources/Resource.h"
#include "Resources/Texture.h"

using namespace std;

namespace Chimera {

	using namespace Core;
	
	namespace Rendering {

		typedef enum shaderType_t {
			TexturedPhong,
			DiffusePhong,
			NoShading
		} shaderType_t;

		class Material : public Resource {

		public:

			GLfloat	Ns;	//	shininess
			GLfloat	Ka[3];	//	ambient color
			GLfloat	Kd[3];	//	diffuse color
			GLfloat	Ks[3];	//	specular color


			int	illum;	//	not supported yet, defines the type of illumination
			float d;	//	not supported yet, defines the opacity of the material

			string	map_Kd;
			shaderType_t m_shader;
			Material() { 
				reset(); 
			}

			void	reset() {
				Ka[0] = Ka[1] = Ka[2] = 0.05f;
				Kd[0] = Kd[1] = Kd[2] = 1;
				Ks[0] = Ks[1] = Ks[2] = .8;
				m_shader = NoShading;
				Ns = 32;
				illum = 2;	//	fully 
				d = 1;	//	opaque
				map_Kd = "";
			}

			void initResources();

			void applyMaterial();
			void removeMaterial();

		private:

			/** Resources */
			shared_ptr<CgFXShader> m_pShader;
			shared_ptr<Texture> m_pTexture;

		};

		typedef map<string, Material> MaterialMap_t;

	}

}
#endif // __MATERIAL_HPP__
