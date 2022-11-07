#include "Mesh/PolygonalMesh.h"

namespace Chimera {

	namespace Meshes {

		#pragma region Constructors
		template <class VectorType>
		PolygonalMesh<VectorType>::PolygonalMesh(const VectorType &position, const string &objFilename, dimensions_t gridDimensions = dimensions_t(0, 0, 0), Scalar gridDx = 0.0f, bool perturbPoints = true) :
			m_regularGridPatches(gridDimensions) {
			m_centroid = position;
			m_pCgalPolyhedron = new CGALWrapper::CgalPolyhedron();
			CGALWrapper::IO::importOBJ(position, objFilename, m_pCgalPolyhedron, gridDx);
			if (gridDx) {
				m_pMeshSlicer = new CGALWrapper::MeshSlicer<VectorType>(m_pCgalPolyhedron, gridDimensions, gridDx);
				m_pMeshSlicer->sliceMesh();
				CGALWrapper::triangulatePolyhedron(m_pCgalPolyhedron);
			}
			CGALWrapper::Conversion::polyhedron3ToFacesAndVertices(m_pCgalPolyhedron, m_vertices, m_elements);
			initializeGhostVertices();
			if (gridDx) {
				classifyVertices(gridDx);
				m_pMeshSlicer->initializeLineMeshes(m_vertices);
			}

			createHalfFaces();
			computeVerticesNormals();
			
			if (gridDx) {
				initializeRegularGridPatches(gridDx);
			}
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template <class VectorType>
		void PolygonalMesh<VectorType>::classifyVertices(Scalar gridDx) {
			for (int i = 0; i < m_vertices.size(); i++) {
				bool vx = (m_vertices[i]->getPosition().x / gridDx) - floor(m_vertices[i]->getPosition().x / gridDx) == 0;
				bool vy = (m_vertices[i]->getPosition().y / gridDx) - floor(m_vertices[i]->getPosition().y / gridDx) == 0;
				bool vz = (m_vertices[i]->getPosition().z / gridDx) - floor(m_vertices[i]->getPosition().z / gridDx) == 0;
				if (vx && vy && vz) {
					m_vertices[i]->setVertexType(geometryVertex);
				}
				else if ((vx && vy) || (vx && vz) || (vy && vz)) {
					m_vertices[i]->setVertexType(edgeVertex);
				}
				else if (vx || vy || vz) {
					m_vertices[i]->setVertexType(faceVertex);
				}
			}

			for (auto iter = m_ghostVerticesMap.begin(); iter != m_ghostVerticesMap.end(); iter++) {
				bool vx = (iter->second->getPosition().x / gridDx) - floor(iter->second->getPosition().x / gridDx) == 0;
				bool vy = (iter->second->getPosition().y / gridDx) - floor(iter->second->getPosition().y / gridDx) == 0;
				bool vz = (iter->second->getPosition().z / gridDx) - floor(iter->second->getPosition().z / gridDx) == 0;
				if (vx && vy && vz) {
					iter->second->setVertexType(geometryVertex);
				}
				else if ((vx && vy) || (vx && vz) || (vy && vz)) {
					iter->second->setVertexType(edgeVertex);
				}
				else if (vx || vy || vz) {
					iter->second->setVertexType(faceVertex);
				}
			}
		}

		template <class VectorType>
		void PolygonalMesh<VectorType>::computeVerticesNormals() {
			for (int i = 0; i < m_vertices.size(); i++) {
				VectorType accumulatedNormal;
				for (int j = 0; j < m_vertices[i]->getConnectedFaces().size(); j++) {
					accumulatedNormal += m_vertices[i]->getConnectedFaces()[j]->getHalfFaces().back()->getNormal();
				}
				accumulatedNormal /= m_vertices[i]->getConnectedFaces().size();
				accumulatedNormal.normalize();
				m_vertices[i]->setNormal(accumulatedNormal);
			}

			for (auto iter = m_ghostVerticesMap.begin(); iter != m_ghostVerticesMap.end(); iter++) {
				VectorType accumulatedNormal;
				for (int j = 0; j < iter->second->getConnectedFaces().size(); j++) {
					accumulatedNormal += iter->second->getConnectedFaces()[j]->getHalfFaces().front()->getNormal();
				}
				accumulatedNormal /= iter->second->getConnectedFaces().size();
				accumulatedNormal.normalize();
				iter->second->setNormal(accumulatedNormal);
			}
		}

		template<class VectorType>
		void PolygonalMesh<VectorType>::initializeRegularGridPatches(Scalar dx) {
			for (uint j = 0; j < m_elements.size(); j++) {
				Face<VectorType> *pCurrFace = m_elements[j];
				VectorType gridSpacePosition = pCurrFace->getCentroid() / dx;
				m_regularGridPatches(floor(gridSpacePosition.x), floor(gridSpacePosition.y), floor(gridSpacePosition.z)).push_back(j);
			}
		}

		template<class VectorType>
		void PolygonalMesh<VectorType>::initializeGhostVertices() {
			for(uint i = 0; i < m_vertices.size(); i++) {
				m_ghostVerticesMap[m_vertices[i]->getID()] = new Vertex<VectorType>(*m_vertices[i]);
			}
		}

		template<class VectorType>
		void PolygonalMesh<VectorType>::createHalfFaces() {
			for (int i = 0; i < m_elements.size(); i++) {
				vector<HalfEdge<VectorType> *> halfEdges1;
				vector<HalfEdge<VectorType> *> halfEdges2;
				halfEdges2.resize(m_elements[i]->getEdges().size());

				auto halfFace = m_elements[i]->getHalfFaces().front();
				auto halfEdges = halfFace->getHalfEdges();
				for (int j = 0; j < halfEdges.size(); j++) {
					if (halfEdges[j]->getEdge()->getHalfEdges().first->getID() == halfEdges[j]->getID()) {
						halfEdges2[halfEdges.size() - 1 - j] = halfEdges[j]->getEdge()->getHalfEdges().second;
					}
					else {
						halfEdges2[halfEdges.size() - 1 - j] = halfEdges[j]->getEdge()->getHalfEdges().first;
					}
				}
				/*for (int j = 0; j < m_elements[i]->getEdges().size(); j++) {
					Edge<VectorType> * pEdge = m_elements[i]->getEdges()[j];
					halfEdges1.push_back(pEdge->getHalfEdges().first);
					halfEdges2[m_elements[i]->getEdges().size() - 1 - j] = pEdge->getHalfEdges().second;
				}*/
				//m_elements[i]->addHalfFace(new HalfFace<VectorType>(halfEdges1, m_elements[i], geometryHalfFace));

				//Substituing halfedges 2 by ghost vertices
				/*for (int j = 0; j < halfEdges2.size(); j++) {
					Vertex<VectorType> *pNewVertex = nullptr;
					auto iter = m_ghostVerticesMap.find(halfEdges2[j]->getVertices().first->getID());
					if (iter == m_ghostVerticesMap.end())
						throw(exception("HalfFace reversed copy error: no ghost vertex found on map"));

					pNewVertex = iter->second;
					if (pNewVertex->hasUpdated())
						continue;

					auto connectedEdges = pNewVertex->getConnectedEdges();
					for (int k = 0; k < connectedEdges.size(); k++) {
						if (connectedEdges[k]->getHalfEdges().first->getVertices().first->getID() == pNewVertex->getID()) {
							connectedEdges[k]->getHalfEdges().first->setFirstVertex(pNewVertex);
						}
						else if (connectedEdges[k]->getHalfEdges().second->getVertices().first->getID() == pNewVertex->getID()) {
							connectedEdges[k]->getHalfEdges().second->setFirstVertex(pNewVertex);
						}
					}

					pNewVertex->setUpdated(true);*/

					/*halfEdges2[j]->setFirstVertex(pNewVertex);

					int prevJ = roundClamp<int>(j - 1, 0, halfEdges2.size());

					halfEdges2[prevJ]->setSecondVertex(pNewVertex);*/
				//}
				//Reversed half-copye ;0
				m_elements[i]->addHalfFace(new HalfFace<VectorType>(halfEdges2, m_elements[i], geometryHalfFace));

				/** Only the first half-face is connected to the edge. In this way the splitting algorithm just
				sees "one-side" of the mesh (half-face) and can resolve geometry-to-gridfaces transition */
				for (int j = 0; j < m_elements[i]->getEdges().size(); j++) {
					m_elements[i]->getEdges()[j]->addConnectedHalfFace(m_elements[i]->getHalfFaces().front());
				}
				
				m_elements[i]->setCentroid(m_elements[i]->getHalfFaces().front()->getCentroid());
			}
			
			//Reset vertices update
			for (int i = 0; i < m_vertices.size(); i++) {
				m_vertices[i]->setUpdated(false);
			}
			for (auto iter = m_ghostVerticesMap.begin(); iter != m_ghostVerticesMap.end(); iter++) {
				iter->second->setUpdated(false);
			}


			for (int i = 0; i < m_vertices.size(); i++) {
				auto faces = m_vertices[i]->getConnectedFaces();
				for (int j = 0; j < faces.size(); j++) {
					m_vertices[i]->addConnectedHalfFace(faces[j]->getHalfFaces().back());
				}
			}

			for (auto iter = m_ghostVerticesMap.begin(); iter != m_ghostVerticesMap.end(); iter++) {
				auto faces = iter->second->getConnectedFaces();
				for (int j = 0; j < faces.size(); j++) {
					iter->second->addConnectedHalfFace(faces[j]->getHalfFaces().front());
				}
			}

			////Add half-faces adjacencies here
			//for (int i = 0; i < m_elements.size(); i++) {
			//	auto halfEdges1 = m_elements[i]->getHalfFaces().back()->getHalfEdges();
			//	for (int j = 0; j < halfEdges1.size(); j++) {
			//		halfEdges1[j]->getVertices().first->addConnectedHalfFace(m_elements[i]->getHalfFaces().back());

			//		int prevJ = roundClamp<int>(j - 1, 0, halfEdges1.size());
			//		halfEdges1[prevJ]->getVertices().second->addConnectedHalfFace(m_elements[i]->getHalfFaces().back());
			//	}
			//	
			//	auto halfEdges2 = m_elements[i]->getHalfFaces().front()->getHalfEdges();
			//	for (int j = 0; j < halfEdges2.size(); j++) {
			//		halfEdges2[j]->getVertices().first->addConnectedHalfFace(m_elements[i]->getHalfFaces().front());

			//		int prevJ = roundClamp<int>(j - 1, 0, halfEdges2.size());
			//		halfEdges2[prevJ]->getVertices().second->addConnectedHalfFace(m_elements[i]->getHalfFaces().front());
			//	}
			//}

		}
		#pragma endregion

		template class PolygonalMesh<Vector3>;
		template class PolygonalMesh<Vector3D>;
	}
}