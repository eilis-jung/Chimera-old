////  Copyright (c) 2013, Vinicius Costa Azevedo
////	All rights reserved.
////
////	Redistribution and use in source and binary forms, with or without
////	modification, are permitted provided that the following conditions are met: 
////
////1. Redistributions of source code must retain the above copyright notice, this
////	list of conditions and the following disclaimer. 
////	2. Redistributions in binary form must reproduce the above copyright notice,
////	this list of conditions and the following disclaimer in the documentation
////	and/or other materials provided with the distribution. 
////
////	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
////	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
////	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
////	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
////	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
////	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
////LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
////	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
////	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
////	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////
////	The views and conclusions contained in the software and documentation are those
////	of the authors and should not be interpreted as representing official policies, 
////	either expressed or implied, of the FreeBSD Project.
//
//
//#ifndef _CHIMERA_CUT_VOXEL_RENDERER_3D_
//#define _CHIMERA_CUT_VOXEL_RENDERER_3D_
//#pragma once
//
///************************************************************************/
///* Data                                                                 */
///************************************************************************/
//#include "Grids/CutCells3D.h"
//
///************************************************************************/
///* Rendering                                                            */
///************************************************************************/
//#include "ChimeraRenderingCore.h"
//#include "Windows/GridVisualizationWindow.h"
//
//namespace Chimera {
//	using namespace Data;
//
//	namespace Rendering {
//
//		class CutVoxelRenderer3D {
//
//		public:
//			#pragma region Constructors
//			CutVoxelRenderer3D(CutCells3D *pCutCells, LinearInterpolant3D<Vector3> *pLinearInterpolant, const Scalar *pVelocityScale, const string &gridName);
//			#pragma endregion
//
//			#pragma region InitializationFunctions
//			void initializeWindowControls(GridVisualizationWindow<Vector3> *pGridVisualizationWindow);
//			#pragma endregion
//
//			#pragma region AccessFunctions
//			int getSelectedVoxel() const {
//				return m_selectedCell;
//			}
//			void setSelectedVoxel(int selectedCell) {
//				m_selectedCell = selectedCell;
//			}
//
//			dimensions_t getSelectedCellDimensions() const {
//				return m_pCutCells->getCutVoxel(m_selectedCell).regularGridIndex;
//			}
//			#pragma endregion
//
//			#pragma region Functionalities
//			void draw();
//			void update();
//			#pragma endregion
//
//		private:
//			#pragma region InternalStructures
//			typedef struct velocityNode_t {
//				CutVoxel specialCell;
//				vector<Vector3> velocityPoints;
//				vector<Vector3> velocities;
//			} velocityNode_t;
//			#pragma endregion
//
//			#pragma region ClassMembers
//			CutCells3D *m_pCutCells;
//			GridData3D *m_pGridData3D;
//			LinearInterpolant3D<Vector3> *m_pLinearInterpolant;
//			nodeVelocityField3D_t *m_pNodeVelocityField;
//			const Scalar *m_pVelocityScale;
//			string m_gridName;
//			
//			/** Drawing vars */
//			bool m_drawSelectedVoxel;
//			int m_selectedCell;
//			int m_numSelectedVoxels;
//			bool m_drawFaceNormals;
//			bool m_drawOnTheFaceLines;
//			
//			/** Velocities */
//			//Interior velocities
//			bool m_drawInteriorVelocities;
//			int m_dimVelocityInterpolationSubdivision;
//			vector<velocityNode_t> m_velocitiesNodes;
//			bool m_drawFaceVelocities;
//			bool m_drawNodeVelocities;
//			bool m_drawMixedNodeVelocities;
//			bool m_drawGeometryVelocities;
//			bool m_drawAllVelocities;
//			
//			/** Mixed nodes*/
//			int m_selectedMixedNode;
//			bool m_drawMixedNodeFaces;
//
//			//Grid visualization window
//			GridVisualizationWindow<Vector3> *m_pGridVisualizationWindow;
//			#pragma endregion
//
//			#pragma region UpdatingFunctions
//			void updateVelocityPoints();
//			void updateVelocities();
//			#pragma endregion
//
//			#pragma region DrawingFunctions
//			
//			/** General */
//			void drawFace(CutFace<Vector3D> *pFace);
//			void drawCell(int ithCell);			
//			void drawSelectedCells();
//			void drawFaceNormals();
//			void drawOnTheFaceLines();
//			
//			/** Velocities */
//			void drawFaceVelocities();
//			void drawNodeVelocities();
//			void drawMixedNodeVelocities();
//			void drawGeometryVelocities();
//			void drawAllVelocities();
//			void drawInteriorVelocities();
//
//			/** Mixed node*/
//			void drawSelectedMixedNodeFaces(int selectedMixedNode);
//			#pragma endregion
//		};
//
//
//
//	}
//}
//#endif