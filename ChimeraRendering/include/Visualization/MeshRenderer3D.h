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


#ifndef _CHIMERA_POLYGON_MESH_RENDERER_3D_
#define _CHIMERA_POLYGON_MESH_RENDERER_3D_
#pragma once

#include "ChimeraWindows.h"
#include "ChimeraRenderingCore.h"
#include "RenderingUtils.h"

#include "Mesh/Mesh.h"

namespace Chimera {

	using namespace Meshes;

	namespace Rendering {

		template <class VectorType, template <class> class ElementType>
		class MeshRenderer3D {

		public:
			#pragma region Constructors
			MeshRenderer3D(const vector<Mesh<VectorType, ElementType> *> &g_meshes, const Vector3 &cameraPosition);
			#pragma endregion

			#pragma region InitializationFunctions
			void initializeWindowControls(BaseWindow *pBaseWindow);
			#pragma endregion

			#pragma region AccessFunctions
			int getSelectedMesh() const {
				return m_selectedIndex;
			}
			void setSelectedMesh(int selectedCell) {
				m_selectedIndex = selectedCell;
			}
			void setDrawing(bool drawing) {
				m_draw = drawing;
			}
			bool isDrawing() const {
				return m_draw;
			}

			void setDrawingVertices(bool drawing) {
				m_drawVertices = drawing;
			}
			#pragma endregion

			#pragma region Functionalities
			void draw();

			void update();
			#pragma endregion

		private:
			#pragma region ClassMembers
			vector<Mesh<VectorType, ElementType> *> m_meshes;

			/** Rendering attributes */
			vector<GLuint *> m_pVerticesVBOs;
			vector<GLuint *> m_pNormalsVBOs;
			vector<GLuint *> m_pIndicesVBOs;
			vector<int> m_numElementsToDraw;

			GLint m_lightPosLoc;
			GLint m_lightPosLocWire;

			/** Shaders */
			shared_ptr<GLSLShader> m_pWireframeShader;
			shared_ptr<GLSLShader> m_pPhongShader;
			shared_ptr<GLSLShader> m_pPhongWireframeShader;

			/** Camera Position Reference */
			const Vector3 &m_cameraPosition;

			/** Selected mesh for rendering */
			int m_selectedIndex;

			bool m_draw;

			bool m_drawVertices;
			#pragma endregion

			#pragma region PrivateFunctionalities
			void initializeVBOs();
			void initializeVBO(unsigned int meshIndex);
			void initializeShaders();
			void drawMesh(int selectedPolygonMesh);
			void drawMeshVertices(int selectedPolygonMesh);
			
			/** Draw polygons: old OpenGL push functions */
			void drawPolygons(int selectedPolygonMesh);
			/** Draw triangles: if the mesh is a triangle mesh, draws using VBOs */
			void drawTrianglesVBOs(int selectedPolygonMesh);

			void drawMeshNormals(int selectedPolygonMesh);
			void drawMeshFaceNormals(int selectedPolygonMesh);
			#pragma endregion
		};



	}
}
#endif