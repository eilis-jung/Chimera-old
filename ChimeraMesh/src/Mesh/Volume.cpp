#include "Mesh/Volume.h"

namespace Chimera {
	namespace Meshes {


		/************************************************************************/
		/* Half-volumes implementation                                          */
		/************************************************************************/

		#pragma region PrivateFunctionalities
		template<class VectorType>
		void HalfVolume<VectorType>::initializeVerticesMap() {
			for (int i = 0; i < m_halfFaces.size(); i++) {
				for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
					uint vertexID = m_halfFaces[i]->getHalfEdges()[j]->getVertices().first->getID();
					if (m_verticesMap.find(vertexID) == m_verticesMap.end()) {
						m_verticesMap[vertexID] = m_halfFaces[i]->getHalfEdges()[j]->getVertices().first;
					}
				}
				if (m_halfFaces[i]->getLocation() != geometryHalfFace) {
					for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
						if(m_halfFaces[i]->getHalfEdges()[j]->getVertices().first->getVertexType() == edgeVertex) {
							uint vertexID = m_halfFaces[i]->getHalfEdges()[j]->getVertices().first->getID();
							if (m_onEdgeVertices.find(vertexID) == m_onEdgeVertices.end()) {
								m_onEdgeVertices[vertexID] = m_halfFaces[i]->getHalfEdges()[j]->getVertices().first;
							}
						}
					}
				}
			}
		}

		template<class VectorType>
		void HalfVolume<VectorType>::classifyHalfFaces() {
			const dimensions_t & cellLocation = m_pParentVolume->getGridCellLocation();
			Scalar dx = m_pParentVolume->getGridSpacing();
			for (int i = 0; i < m_halfFaces.size(); i++) {
				if (m_halfFaces[i]->getLocation() == backHalfFace) {
					if (floor(m_halfFaces[i]->getCentroid().z/ dx) != cellLocation.z) {
						m_halfFaces[i]->setLocation(frontHalfFace);
					}
				} else if (m_halfFaces[i]->getLocation() == leftHalfFace) {
					if (floor(m_halfFaces[i]->getCentroid().x / dx) != cellLocation.x) {
						m_halfFaces[i]->setLocation(rightHalfFace);
					}
				} else if (m_halfFaces[i]->getLocation() == bottomHalfFace) {
					if (floor(m_halfFaces[i]->getCentroid().y / dx) != cellLocation.y) {
						m_halfFaces[i]->setLocation(topHalfFace);
					}
				}
			}
		}
		#pragma endregion
		#pragma region Functionalities
		template<class VectorType>
		bool HalfVolume<VectorType>::isInside(const VectorType &position, VectorType direction) const {
			bool insidePolygon = false;
			for (int i = 0; i < m_halfFaces.size(); i++) {
				if (m_halfFaces[i]->getLocation() == geometryHalfFace) {
					if (m_halfFaces[i]->rayIntersect(position, direction)) {
						insidePolygon = !insidePolygon;
					}
				} else {
					if (direction == VectorType(0, 0, 1) && m_halfFaces[i]->getLocation() == frontHalfFace) { //XY Plane
						VectorType projectedPoint(position.x, position.y, 0);
						if (m_halfFaces[i]->isInside(projectedPoint)) {
							insidePolygon = !insidePolygon;
						}
					}
					else if (direction == VectorType(0, 1, 0) && m_halfFaces[i]->getLocation() == topHalfFace) { //XZ Plane
						VectorType projectedPoint(position.x, 0, position.z);
						if (m_halfFaces[i]->isInside(projectedPoint)) {
							insidePolygon = !insidePolygon;
						}
					} else if (direction == VectorType(1, 0, 0) && m_halfFaces[i]->getLocation() == leftHalfFace) { //YZ Plane
						VectorType projectedPoint(0, position.y, position.z);
						if (m_halfFaces[i]->isInside(projectedPoint)) {
							insidePolygon = !insidePolygon;
						}
					}
				}
			}
			return insidePolygon;
		}

		template<class VectorType>
		bool HalfVolume<VectorType>::crossedThroughGeometry(const VectorType &v1, const VectorType &v2, VectorType &crossingPoint) {
			for (int i = 0; i < m_halfFaces.size(); i++) {
				if (m_halfFaces[i]->getLocation() == geometryHalfFace) {
					VectorType triPoints[3];
					triPoints[0] = m_halfFaces[i]->getHalfEdges()[0]->getVertices().first->getPosition();
					triPoints[1] = m_halfFaces[i]->getHalfEdges()[1]->getVertices().first->getPosition();
					triPoints[2] = m_halfFaces[i]->getHalfEdges()[2]->getVertices().first->getPosition();
					if (segmentTriangleIntersect(v1, v2, triPoints)) {
						return true;
					}
				}
			}
			return false;
		}
		#pragma endregion
		
		/************************************************************************/
		/* Volume functions implementation                                      */
		/************************************************************************/

		#pragma region Functionalities
		template<class VectorType>
		const vector<HalfVolume<VectorType>*> & Volume<VectorType>::split() {
			m_halfVolumes.clear();

			vector<HalfFace<VectorType> *> halfFaces;
			Face<VectorType> *pFace = nullptr;

			for (int i = 0; i < m_faces.size(); i++) {
				m_faces[i]->setVisited(false);
			}
			
			while ((pFace = hasUnvisitedFaces()) != nullptr) {
				halfFaces.clear();

				for (int i = 0; i < m_faces.size(); i++) {
					if (m_faces[i]->getLocation() == geometricFace)
						m_faces[i]->setVisited(false);
				}

				/** After closing a loop with breadthFirstSearch, halfEdgesVector will have a closed half-face */
				breadthFirstSearch(getHalfFace(pFace, m_gridCellLocation), halfFaces, nullptr);

				m_halfVolumes.push_back(new HalfVolume<VectorType>(halfFaces, this));

				/** Adding connectivity information for half-volumes */
				for (int i = 0; i < m_halfVolumes.back()->getHalfFaces().size(); i++) {
					Face<VectorType> * pCurrFace = m_halfVolumes.back()->getHalfFaces()[i]->getFace();
					pCurrFace->addConnectedHalfVolume(m_halfVolumes.back());
				}
			}

			//Leave all edges Unvisited, will be used for ghost vertices initialization
			for (int i = 0; i < m_faces.size(); i++) {
				m_faces[i]->setVisited(false);
			}

			return m_halfVolumes;
		}

		template<class VectorType>
		HalfVolume<VectorType> * Volume<VectorType>::getHalfVolume(const VectorType &position) {
			for (int i = 0; i < m_halfVolumes.size(); i++) {
				if (m_halfVolumes[i]->isInside(position)) {
					return m_halfVolumes[i];
				}
			}
			//Didn't found cells using the standard orientations, try other ones
			Logger::getInstance()->get() << "Half-volume not found with standard direction, try another ones" << endl;
			for (int i = 0; i < m_halfVolumes.size(); i++) {
				if (m_halfVolumes[i]->isInside(position, VectorType(0, 1, 0))) {
					return m_halfVolumes[i];
				}
			}

			Logger::getInstance()->get() << "Half-volume not found with standard direction, try last one" << endl;
			for (int i = 0; i < m_halfVolumes.size(); i++) {
				if (m_halfVolumes[i]->isInside(position, VectorType(1, 0, 0))) {
					return m_halfVolumes[i];
				}
			}

			return m_halfVolumes[0];
			//Didn't find any, return null

			return nullptr;
		}
		#pragma endregion

		#pragma region PrivateFunctionalities

		template<class VectorType>
		HalfFace<VectorType>* Volume<VectorType>::getHalfFace(Face<VectorType>* pFace, dimensions_t cellLocation) {
			
			//Any point will work for checking where the face is
			const VectorType facePoint = pFace->getEdges().front()->getCentroid();

			switch (pFace->getLocation()) {
			case geometricFace:
				return nullptr;

			case XZFace: //Bottom Face
				if (floor(facePoint.y / m_gridDx) == cellLocation.y) {
					return pFace->getHalfFaces().back();
				}
				break;
			case YZFace: //Left Face
				if (floor(facePoint.x / m_gridDx) == cellLocation.x) {
					return pFace->getHalfFaces().back();
				}
				break;
			case XYFace: //Back Face
				if (floor(facePoint.z / m_gridDx) == cellLocation.z) {
					return pFace->getHalfFaces().back();
				}
				break;
			}
			return pFace->getHalfFaces().front();

		}

		template<class VectorType>
		void Volume<VectorType>::breadthFirstSearch(HalfFace<VectorType> * pHalfFace, vector<HalfFace<VectorType>*>& halfFaces, HalfEdge<VectorType> *pPrevHalfEdge) {
			if (pHalfFace->getFace()->isVisited())
				return;

			//Tag this face as visited
			pHalfFace->getFace()->setVisited(true);			
			halfFaces.push_back(pHalfFace);

			for (int i = 0; i < pHalfFace->getHalfEdges().size(); i++) {
				HalfEdge<VectorType> *pCurrHalfEdge = pHalfFace->getHalfEdges()[i];
				Edge<VectorType> *pCurrEdge = pCurrHalfEdge->getEdge();
				//Choosing next faces
				const vector<Face<VectorType> *> &nextFaces = m_edgeToFaceMap.getFaces(pCurrEdge);

				if (nextFaces.size() > 3) { //Case that is currently not supported, abort
					throw(exception("Face split: high frequency feature found on top of mesh."));
				}
				else if (nextFaces.size() == 3) { //Going from gridEdges to geometryEdges or vice-versa
					if (pHalfFace->getLocation() == geometryHalfFace) {
						bool followThrough = false;
						for (int j = 0; j < 3; j++) {
							if (nextFaces[j]->getLocation() != geometricFace) {
								HalfFace<VectorType> *pNextHalfFace = getHalfFace(nextFaces[j], m_gridCellLocation);
								if (!pNextHalfFace->hasHalfedge(pCurrHalfEdge)) {
									breadthFirstSearch(pNextHalfFace, halfFaces, pCurrHalfEdge);
									followThrough = true;
									break;
								}
							}
						}
						if (!followThrough) {
							throw(exception("Error on edge path choosing algorithm: unexpected geometry to grid edge case."));
						}
					} else {
						if (nextFaces[0]->getLocation() == geometricFace) {
							HalfFace<VectorType> *pNextHalfFace = nextFaces[0]->getHalfFaces().front();
							if (!pNextHalfFace->hasHalfedge(pCurrHalfEdge)) {
								breadthFirstSearch(pNextHalfFace, halfFaces, pCurrHalfEdge);
							}
							else {
								breadthFirstSearch(nextFaces[0]->getHalfFaces().back(), halfFaces, pCurrHalfEdge);
							}
							
						}
						else if (nextFaces[1]->getLocation() == geometricFace) {
							HalfFace<VectorType> *pNextHalfFace = nextFaces[1]->getHalfFaces().front();
							if (!pNextHalfFace->hasHalfedge(pCurrHalfEdge)) {
								breadthFirstSearch(pNextHalfFace, halfFaces, pCurrHalfEdge);
							}
							else {
								breadthFirstSearch(nextFaces[1]->getHalfFaces().back(), halfFaces, pCurrHalfEdge);
							}
						}
						else if (nextFaces[2]->getLocation() == geometricFace) {
							HalfFace<VectorType> *pNextHalfFace = nextFaces[2]->getHalfFaces().front();
							if (!pNextHalfFace->hasHalfedge(pCurrHalfEdge)) {
								breadthFirstSearch(pNextHalfFace, halfFaces, pCurrHalfEdge);
							}
							else {
								breadthFirstSearch(nextFaces[2]->getHalfFaces().back(), halfFaces, pCurrHalfEdge);
							}
						}
						else {
							throw(exception("Error on vertex path choosing algorithm: not connected to geometric face."));
						}
					}
				} 
				else if (nextFaces.size() == 2) {
					if (nextFaces[0]->getID() == pHalfFace->getFace()->getID()) {
						if (nextFaces[1]->getHalfFaces().back()->hasHalfedge(pCurrHalfEdge)) {
							breadthFirstSearch(nextFaces[1]->getHalfFaces().front(), halfFaces, pCurrHalfEdge);
						} else {
							breadthFirstSearch(nextFaces[1]->getHalfFaces().back(), halfFaces, pCurrHalfEdge);
						}
					}
					else {
						if (nextFaces[0]->getHalfFaces().back()->hasHalfedge(pCurrHalfEdge)) {
							breadthFirstSearch(nextFaces[0]->getHalfFaces().front(), halfFaces, pCurrHalfEdge);
						}
						else {
							breadthFirstSearch(nextFaces[0]->getHalfFaces().back(), halfFaces, pCurrHalfEdge);
						}
					}
				}
				else { //This is a open ended point, go back through same geometric edges

				}

			}
		}
		#pragma endregion

		template class Volume<Vector3D>;
		template class Volume<Vector3>;
		template class HalfVolume<Vector3D>;
		template class HalfVolume<Vector3>;

	}
}