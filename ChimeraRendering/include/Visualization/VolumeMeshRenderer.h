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


#ifndef __CHIMERA_RENDERING_VOLUME_MESH_RENDERER_H_
#define __CHIMERA_RENDERING_VOLUME_MESH_RENDERER_H_
#pragma once

#include "Visualization/MeshRenderer.h"
#include "Visualization/LineMeshRenderer.h"
#include "Visualization/PolygonMeshRenderer.h"


namespace Chimera {
	namespace Rendering {

		/** Volume mesh renderer: only makes sense in 3-D */
		template<class VectorType>
		class MeshRendererT<VectorType, Volume, false> : public MeshRendererBase<MeshRendererT<VectorType, Volume, false>, VectorType, Volume, false> {

		public:
			MeshRendererT(const vector<Mesh<VectorType, Volume> *> &g_meshes, const Vector3 &cameraPosition) : MeshRendererBase(g_meshes, cameraPosition) {
				m_pCutVoxels = dynamic_cast<CutVoxels3D<VectorType>*>(g_meshes[0]);
				m_selectedCutVoxel = 0;
				m_drawCutVoxels = true;
				m_drawSelectedCutVoxel = true;
				m_drawMeshVertices = false;
				m_drawMeshNormals = false;
				m_drawCutVoxelNormals = false;
				m_drawCutFacesCentroids = false;
				m_drawNeighbors = false;
				m_drawVertexHalfFaces = false;

				m_selectedVertex = 0;
				m_cutVoxelsColor = Color(56, 155, 255);
				m_nodeNeighborsColor = Color(255, 30, 109);
				initializeLineRenderers();
				initializeCutCellsRenderers();
			}
			
			#pragma region AccessFunctions
			void setSelectedCutVoxel(int cutVoxelIndex) {
				m_selectedCutVoxel = cutVoxelIndex;
			}
			int & getSelectedCutVoxel() {
				return m_selectedCutVoxel;
			}

			int & getSelectedNode() {
				return m_selectedVertex;
			}

			/** Drawing access functions */
			bool & isDrawingSelectedCutVoxel() {
				return m_drawSelectedCutVoxel;
			}

			bool & isDrawingCutVoxels() {
				return m_drawCutVoxels;
			}

			bool & isDrawingCutVoxelNormals() {
				return m_drawCutVoxelNormals;
			}

			bool & isDrawingCutFacesCentroids() {
				return m_drawCutFacesCentroids;
			}

			bool & isDrawingNeighbors() {
				return m_drawNeighbors;
			}

			bool & isDrawingMixedNodeNeighbors() {
				return m_drawVertexHalfFaces;
			}

			void setDrawingCutVoxels(bool drawCutVoxels) {
				m_drawCutVoxels = drawCutVoxels;
			}

			CutVoxels3D<VectorType> * getCutVoxels() {
				return m_pCutVoxels;
			}

			const vector<LineMeshRenderer<VectorType> *> & getLineMeshRenderers() {
				return m_lineRenderers;
			}

			#pragma endregion
			
			#pragma region DrawingFunctions
			virtual void drawElements(uint selectedPolygonMesh, Color color = Color::BLACK) override;

			virtual void drawMeshVertices(uint selectedPolygonMesh, Color color = Color::BLACK) override;

			void drawCutVoxelNormals(Color color = Color::BLACK);
			void drawCutFacesCentroid(Color color = Color::BLACK);
			#pragma endregion
			
		protected:
			
			#pragma region ClassMembers
			/** Drawing options */
			int m_selectedCutVoxel;
			bool m_drawSelectedCutVoxel;
			bool m_drawNeighbors;
			bool m_drawCutVoxels;
			
			bool m_drawCutVoxelNormals;
			bool m_drawCutFacesCentroids;
			int m_selectedCutCell;

			int m_selectedVertex;
			bool m_drawVertexHalfFaces;

			Color m_cutVoxelsColor;
			Color m_nodeNeighborsColor;

			/** To add: 
				- cut-cells planes to render (XY, XZ, YZ, all)
				- cut-lines to render (xAligned, yAligned, all) 
			*/

			/** Cut-Voxels */
			CutVoxels3D<VectorType> *m_pCutVoxels;

			/** Internal renderers */
			vector<LineMeshRenderer<VectorType> *> m_lineRenderers;
			vector<PolygonMeshRenderer<VectorType> *> m_cutCellsRenderers;
			#pragma endregion

			
			#pragma region InitializationFunctions
			void initializeLineRenderers();
			void initializeCutCellsRenderers();
			#pragma endregion

			#pragma region PrivateDrawingFunctions
			virtual void drawCutVoxel(uint cutVoxelIndex);
			virtual void drawCutFace(HalfFace<VectorType> *pHalfFace);
			#pragma endregion
		};

		template <class VectorType>
		using VolumeMeshRenderer = MeshRendererT<VectorType, Volume, isVector2<VectorType>::value>;
	}
}
#endif
