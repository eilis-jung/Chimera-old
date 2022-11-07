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
//#ifndef __RENDERING_MESH_PATCH_MAP_SPLITTER_H_
//#define __RENDERING_MESH_PATCH_MAP_SPLITTER_H_
//
//#include "ChimeraData.h"
//#include "ChimeraCGALWrapper.h"
//
///************************************************************************/
///* Rendering                                                            */
///************************************************************************/
//#include "ChimeraRenderingCore.h"
//#include "CGAL/PolygonSurface.h"
//
//namespace Chimera {
//	namespace Rendering {
//
//
//
//		class MeshPatchMapSplitter {
//
//			//Current edge map stores pairs of faces adjacent to an edge 
//			map<unsigned long int, pair<unsigned int, unsigned int>> m_edgeMap;
//
//			vector<bool> m_visitedFaces;
//
//			MeshPatchMap *m_pMeshPatchMap;
//
//			//This hash doesn't depend on the orientation of the edge. Thus its not a halfedge hash
//			unsigned long int edgeHash(unsigned int i, unsigned int j, unsigned int numVertices) const {
//				if (i < j)
//					return numVertices*i + j;
//				return numVertices*j + i;
//			}
//
//			void breadthFirstSearch(unsigned int currFaceID, MeshPatchMap *pNewPatchMap);
//
//			int getFirstNonVisitedFace() {
//				for (int i = 0; i < m_visitedFaces.size(); i++) {
//					if (!m_visitedFaces[i])
//						return i;
//				}
//				return -1;
//			}
//
//		public:
//			/************************************************************************/
//			/* ctors                                                                */
//			/************************************************************************/
//			MeshPatchMapSplitter(MeshPatchMap *pMeshPatchMap);
//
//			/************************************************************************/
//			/* Functionalities                                                      */
//			/************************************************************************/
//			vector<MeshPatchMap *> split();
//		};
//#pragma endregion
//	}
//}
//#endif