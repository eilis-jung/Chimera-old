//#include "CGAL/MeshPatchMapSplitter.h" 
//
//
//namespace Chimera {
//	namespace Rendering {
//		MeshPatchMapSplitter::MeshPatchMapSplitter(MeshPatchMap *pMeshPatchMap) {
//			m_pMeshPatchMap = pMeshPatchMap;
//			const vector<simpleFace_t> & simpleFaces = pMeshPatchMap->pPolySurface->getFaces();
//			for (int i = 0; i < pMeshPatchMap->faces.size(); i++) {
//				m_visitedFaces.push_back(false);
//				int currFaceIndex = pMeshPatchMap->faces[i];
//				for (int j = 0; j < simpleFaces[currFaceIndex].edges.size(); j++) {
//					unsigned long int currEdgeHash = edgeHash(simpleFaces[currFaceIndex].edges[j].first, simpleFaces[currFaceIndex].edges[j].second, pMeshPatchMap->pPolySurface->getVertices().size());
//					if (m_edgeMap.find(currEdgeHash) != m_edgeMap.end()) {
//						m_edgeMap[currEdgeHash].second = i;
//					}
//					else {
//						m_edgeMap[currEdgeHash].first = i;
//					}
//				}
//			}
//		}
//
//		void MeshPatchMapSplitter::breadthFirstSearch(unsigned int faceID, MeshPatchMap *pNewPatchMap) {
//			if (m_visitedFaces[faceID])
//				return;
//
//			m_visitedFaces[faceID] = true;
//
//			int currFaceIndex = m_pMeshPatchMap->faces[faceID];
//			pNewPatchMap->faces.push_back(currFaceIndex);
//			const simpleFace_t &currFace = m_pMeshPatchMap->pPolySurface->getFaces()[currFaceIndex];
//
//			if (currFace.borderFace) {
//				pNewPatchMap->danglingPatch = true;
//			}
//			for (int j = 0; j < currFace.edges.size(); j++) {
//				unsigned long int currEdgeHash = edgeHash(currFace.edges[j].first, currFace.edges[j].second, m_pMeshPatchMap->pPolySurface->getVertices().size());
//				pair<unsigned int, unsigned int> currEdgeFaces = m_edgeMap[currEdgeHash];
//				int nextFaceID = currEdgeFaces.first == faceID ? currEdgeFaces.second : currEdgeFaces.first;
//				breadthFirstSearch(nextFaceID, pNewPatchMap);
//			}
//		}
//
//		vector<MeshPatchMap *> MeshPatchMapSplitter::split() {
//			vector<MeshPatchMap *> meshPatchMaps;
//			int firstIndex = getFirstNonVisitedFace();
//			while (firstIndex != -1) {
//				MeshPatchMap *pNewPatchMap = new MeshPatchMap();
//				pNewPatchMap->pPolySurface = m_pMeshPatchMap->pPolySurface;
//				breadthFirstSearch(firstIndex, pNewPatchMap);
//				meshPatchMaps.push_back(pNewPatchMap);
//				firstIndex = getFirstNonVisitedFace();
//			}
//			return meshPatchMaps;
//			
//		}
//	}
//}