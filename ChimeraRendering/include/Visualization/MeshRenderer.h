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


#ifndef _CHIMERA_POLYGON_MESH_RENDERER_
#define _CHIMERA_POLYGON_MESH_RENDERER_
#pragma once

#include "ChimeraResources.h"
#include "ChimeraRenderingCore.h"
#include "RenderingUtils.h"
#include "ChimeraMesh.h"
#include "ChimeraCutCells.h"

namespace Chimera {

	using namespace Meshes;
	using namespace Resources;
	using namespace CutCells;

	namespace Rendering {

		/** Using Curiously Recurring Template Pattern https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
		for sharing explicit template specializations */
		template <class ChildT, class VectorType, template <class> class ElementType, bool isVector2>
		class MeshRendererBase {

		public:
			#pragma region Constructors
			MeshRendererBase(const vector<Mesh<VectorType, ElementType> *> &g_meshes, const Vector3 &cameraPosition);
			#pragma endregion

			#pragma region AccessFunctions
			int getSelectedMesh() const {
				return m_selectedIndex;
			}
			void setSelectedMesh(int selectedCell) {
				m_selectedIndex = selectedCell;
			}

			bool & isDrawing() {
				return m_draw;
			}

			bool & isDrawingVertices() {
				return m_drawMeshVertices;
			}

			bool & isDrawingNormals() {
				return m_drawMeshNormals;
			}

			void setDrawing(bool drawing) {
				m_draw = drawing;
			}
			
			void setDrawingVertices(bool drawVertices) {
				m_drawMeshVertices = drawVertices;
			}

			void setDrawingNormals(bool drawNormals) {
				m_drawMeshNormals = drawNormals;
			}

			void setVertexColor(const Color &color) {
				m_vertexColor = color;
			}

			void setElementColor(const Color &color) {
				m_elementColor = color;
			}

			#pragma endregion

			#pragma region Functionalities
			virtual void draw();
			#pragma endregion

		protected:
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

			/** Options for drawing */
			bool m_draw;
			bool m_drawMeshVertices;
			bool m_drawMeshNormals;
			Scalar m_vectorLengthSize;

			Color m_vertexColor;
			Color m_elementColor;
			#pragma endregion

			#pragma region PrivateFunctionalities
			void initializeVBOs();
			void initializeVertexVBO(uint meshIndex);
			void initiliazeNormalVBO(uint meshIndex);
			void initializeVBO(uint meshIndex);
			void initializeShaders();
			void drawMesh(int selectedPolygonMesh);
			
			/** Draw elements and vertices: each derived class must implement this*/
			virtual void drawElements(uint selectedPolygonMesh, Color color = Color::BLACK) = 0;
			virtual void drawMeshVertices(uint selectedPolygonMesh, Color color = Color::BLACK) = 0;

			/** Draw triangles: if the mesh is a triangle mesh, draws using VBOs */
			void drawTrianglesVBOs(int selectedPolygonMesh);

			void drawMeshNormals(uint selectedPolygonMesh);
			#pragma endregion
		};


		template<class VectorType, template <class> class ElementType, bool isVector2>
		class MeshRendererT : public MeshRendererBase<MeshRendererT<VectorType, ElementType, isVector2>, VectorType, ElementType, isVector2> {
			
		};


		template <class VectorType>
		using LineMeshRenderer = MeshRendererT<VectorType, Edge, isVector2<VectorType>::value>;
		
		
	}


}
#endif