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


#ifndef __CHIMERA_RENDERING_LINE_MESH_RENDERER_H_
#define __CHIMERA_RENDERING_LINE_MESH_RENDERER_H_
#pragma once

#include "Visualization/MeshRenderer.h"

namespace Chimera {
	namespace Rendering {

		/** Line Renderer 2-D */
		template<class VectorType>
		class MeshRendererT<VectorType, Edge, true> : public MeshRendererBase<MeshRendererT<VectorType, Edge, true>, VectorType, Edge, true> {

		public:
			MeshRendererT(const vector<Mesh<VectorType, Edge> *> &g_meshes, const Vector3 &cameraPosition) : MeshRendererBase(g_meshes, cameraPosition) {
				m_drawMeshVertices = false;
				m_drawMeshNormals = false;
				m_drawLineMeshes = false;
			}
			virtual void drawElements(uint selectedPolygonMesh, Color color = Color::BLACK) override;

			virtual void drawMeshVertices(uint selectedPolygonMesh, Color color = Color::BLACK) override;

		protected:
			bool m_drawLineMeshes;
		};

		/** Line Renderer 3-D */
		template<class VectorType>
		class MeshRendererT<VectorType, Edge, false> : public MeshRendererBase<MeshRendererT<VectorType, Edge, false>, VectorType, Edge, false> {

		public:
			MeshRendererT(const vector<Mesh<VectorType, Edge> *> &g_meshes, const Vector3 &cameraPosition) : MeshRendererBase(g_meshes, cameraPosition) {
				m_drawMeshVertices = false;
				m_drawMeshNormals = false;
				m_drawLineMeshes = true;
			}
			virtual void drawElements(uint selectedPolygonMesh, Color color = Color::BLACK) override;

			virtual void drawMeshVertices(uint selectedPolygonMesh, Color color = Color::BLACK) override;

		protected:
			bool m_drawLineMeshes;
		};
	}
}

#endif

		
