#include "Mesh/Face.h"

namespace Chimera {

	namespace Meshes {

		template <class VectorType>
		bool HalfFace<VectorType>::crossedThroughGeometry(const VectorType &v1, const VectorType &v2, VectorType &crossingPoint) const {
			bool crossedThroughGeometry = false;
			for (int i = 0; i < m_halfEdges.size(); i++) {
				if (m_halfEdges[i]->getLocation() == geometryHalfEdge) {
					if (DoLinesIntersect(v1, v1, m_halfEdges[i]->getVertices().first->getPosition(), m_halfEdges[i]->getVertices().second->getPosition())) {
						crossedThroughGeometry = true;
						break;
					}
				}
			}
			return crossedThroughGeometry;
		}

		template <class VectorType>
		Scalar HalfFace<VectorType>::getDistanceToBoundary(const VectorType &position) const {
			Scalar minDistance = FLT_MAX;
			for (int i = 0; i < m_halfEdges.size(); i++) {
				if (m_halfEdges[i]->getLocation() == geometryHalfEdge) {
					int nextI = roundClamp(i + 1, 0, (int)m_halfEdges.size());
					Scalar currDistance = distanceToLineSegment(position, m_halfEdges[i]->getVertices().first->getPosition(), 
																m_halfEdges[i]->getVertices().second->getPosition());
					if (currDistance < minDistance) {
						minDistance = currDistance;
					}
				}
			}
			return minDistance;
		}

		template<class VectorType>
		bool HalfFace<VectorType>::isInside(const VectorType & position) const {
			/** Checking if points are livin on the edge */
			for (unsigned i = 0; i < m_halfEdges.size(); i++) {
				int nextI = roundClamp<int>(i + 1, 0, m_halfEdges.size());
				if (isOnEdge(m_halfEdges[i]->getVertices().first->getPosition(), m_halfEdges[nextI]->getVertices().first->getPosition(), position, 1e-7))
					return true;
			}

			//http://alienryderflex.com/polygon/ polygon function
			int i = 0, j = static_cast<int>(m_halfEdges.size() - 1);
			bool insidePolygon = false;

			if (m_faceLocation == backHalfFace || m_faceLocation == frontHalfFace) { //XY plane
				for (unsigned i = 0; i < m_halfEdges.size(); i++) {
					const VectorType &pointPositionI = m_halfEdges[i]->getVertices().first->getPosition();
					const VectorType &pointPositionJ = m_halfEdges[j]->getVertices().first->getPosition();
					if ((pointPositionI.y < position.y && pointPositionJ.y >= position.y)
						|| (pointPositionJ.y < position.y && pointPositionI.y >= position.y)) {
						if (pointPositionI.x + (position.y - pointPositionI.y) / (pointPositionJ.y - pointPositionI.y)*(pointPositionJ.x - pointPositionI.x) < position.x) {
							insidePolygon = !insidePolygon;
						}
					}
					j = i;
				}
			} else if (m_faceLocation == leftHalfFace || m_faceLocation == rightHalfFace) { //YZ plane 
				for (unsigned i = 0; i < m_halfEdges.size(); i++) {
					const VectorType &pointPositionI = m_halfEdges[i]->getVertices().first->getPosition();
					const VectorType &pointPositionJ = m_halfEdges[j]->getVertices().first->getPosition();
					if ((pointPositionI.y < position.y && pointPositionJ.y >= position.y)
						|| (pointPositionJ.y < position.y && pointPositionI.y >= position.y)) {
						if (pointPositionI[2] + (position.y - pointPositionI.y) / (pointPositionJ.y - pointPositionI.y)*(pointPositionJ[2] - pointPositionI[2]) < position[2]) {
							insidePolygon = !insidePolygon;
						}
					}
					j = i;
				}
			} else if (m_faceLocation == topHalfFace || m_faceLocation == bottomHalfFace) { //XZ plane 
				for (unsigned i = 0; i < m_halfEdges.size(); i++) {
					const VectorType &pointPositionI = m_halfEdges[i]->getVertices().first->getPosition();
					const VectorType &pointPositionJ = m_halfEdges[j]->getVertices().first->getPosition();
					if ((pointPositionI[2] < position[2] && pointPositionJ[2] >= position[2])
						|| (pointPositionJ[2] < position[2] && pointPositionI[2] >= position[2])) {
						if (pointPositionI.x + (position[2] - pointPositionI[2]) / (pointPositionJ[2] - pointPositionI[2])*(pointPositionJ.x - pointPositionI.x) < position.x) {
							insidePolygon = !insidePolygon;
						}
					}
					j = i;
				}
			} else { //Geometric half-face
				return false; //In 3-d we wont deal with this 
			}

			return insidePolygon;
		}


		template<>
		bool HalfFace<Vector2>::rayIntersect(const Vector2 &point, const Vector2 &rayDirection) {
			return false;
		}

		template<>
		bool HalfFace<Vector2D>::rayIntersect(const Vector2D &point, const Vector2D &rayDirection) {
			return false;
		}

		template<>
		bool HalfFace<Vector3>::rayIntersect(const Vector3 &point, const Vector3 &rayDirection) {
			if (m_faceLocation != geometryHalfFace) {
				return false;
			}

			//Barycentric weights 
			DoubleScalar b[2];

			// Edge vectors
			const Vector3& e_1 = m_halfEdges[1]->getVertices().first->getPosition() - m_halfEdges[0]->getVertices().first->getPosition();
			const Vector3& e_2 = m_halfEdges[2]->getVertices().first->getPosition() - m_halfEdges[0]->getVertices().first->getPosition();

			Vector3 q = rayDirection.cross(e_2);
			DoubleScalar a = e_1.dot(q);

			if ((abs(a) <= 1e-10))
				return false;

			DoubleScalar invA = 1 / a;
			const Vector3& s = (point - m_halfEdges[0]->getVertices().first->getPosition());
			b[0] = s.dot(q)*invA;

			if ((b[0] < 0.0) || (b[0] > 1.0))
				return false;

			const Vector3& r = s.cross(e_1);
			b[1] = r.dot(rayDirection)*invA;
			if ((b[1] < 0.0) || (b[0] + b[1] > 1.0))
				return false;

			DoubleScalar t = invA*e_2.dot(r);
			if (t >= 0)
				return true;

			return false;
		}

		template<>
		bool HalfFace<Vector3D>::rayIntersect(const Vector3D &point, const Vector3D &rayDirection) {
			if (m_faceLocation != geometryHalfFace) {
				return false;
			}

			//Barycentric weights 
			DoubleScalar b[2];

			// Edge vectors
			const Vector3D& e_1 = m_halfEdges[1]->getVertices().first->getPosition() - m_halfEdges[0]->getVertices().first->getPosition();
			const Vector3D& e_2 = m_halfEdges[2]->getVertices().first->getPosition() - m_halfEdges[0]->getVertices().first->getPosition();

			Vector3D q = rayDirection.cross(e_2);
			DoubleScalar a = e_1.dot(q);

			if ((abs(a) <= 1e-10))
				return false;

			DoubleScalar invA = 1 / a;
			const Vector3D& s = (point - m_halfEdges[0]->getVertices().first->getPosition());
			b[0] = s.dot(q)*invA;

			if ((b[0] < 0.0) || (b[0] > 1.0))
				return false;

			const Vector3D& r = s.cross(e_1);
			b[1] = r.dot(rayDirection)*invA;
			if ((b[1] < 0.0) || (b[0] + b[1] > 1.0))
				return false;

			DoubleScalar t = invA*e_2.dot(r);
			if (t >= 0)
				return true;

			return false;
		}

		template<class VectorType>
		bool HalfFace<VectorType>::hasHalfedge(HalfEdge<VectorType> *pHalfEdge) {
			for (int i = 0; i < m_halfEdges.size(); i++) {
				if (pHalfEdge->getID() == m_halfEdges[i]->getID()) {
					return true;
				}
			}
			return false;
		}

		template<class VectorType>
		bool HalfFace<VectorType>::hasReverseHalfedge(HalfEdge<VectorType> *pHalfEdge) {
			for (int i = 0; i < m_halfEdges.size(); i++) {
				if (pHalfEdge->getVertices().first->getID() == m_halfEdges[i]->getVertices().second->getID() && 
					pHalfEdge->getVertices().second->getID() == m_halfEdges[i]->getVertices().first->getID()) {
					return true;
				}
			}
			return false;
		}

		template<class VectorType>
		bool HalfFace<VectorType>::hasEdge(Edge<VectorType> *pEdge) {
			auto connectedHalfFaces = pEdge->getConnectedHalfFaces();
			for (int i = 0; i < connectedHalfFaces.size(); i++) {
				if (connectedHalfFaces[i]->getID() == getID())
					return true;
			}
			return false;
		}

		template<class VectorType>
		HalfFace<VectorType> * HalfFace<VectorType>::reversedCopy() {
			vector<HalfEdge<VectorType> *> inverseHalfEdges;
			for(int i = 0; i < m_halfEdges.size(); i++) {
				uint rI = m_halfEdges.size() - (i + 1);

				if (*m_halfEdges[rI]->getEdge()->getHalfEdges().first == *m_halfEdges[rI]) {
					inverseHalfEdges.push_back(m_halfEdges[rI]->getEdge()->getHalfEdges().second);
				}
				else if (*m_halfEdges[rI]->getEdge()->getHalfEdges().second == *m_halfEdges[rI]) {
					inverseHalfEdges.push_back(m_halfEdges[rI]->getEdge()->getHalfEdges().first);
				}
				else {
					throw(exception("HalfFace reversed copy error: halfedges inside edge do not match"));
				}
			}

			return new HalfFace<VectorType>(inverseHalfEdges, m_pFace, m_faceLocation);
		}

		template<class VectorType>
		HalfFace<VectorType> * HalfFace<VectorType>::reversedCopy(const map<uint, Vertex<VectorType> *> &ghostVerticesMap) {
			vector<HalfEdge<VectorType> *> inverseHalfEdges;
			for (int i = 0; i < m_halfEdges.size(); i++) {
				uint rI = m_halfEdges.size() - (i + 1);

				if (*m_halfEdges[rI]->getEdge()->getHalfEdges().first == *m_halfEdges[rI]) {
					inverseHalfEdges.push_back(m_halfEdges[rI]->getEdge()->getHalfEdges().second);
				}
				else if (*m_halfEdges[rI]->getEdge()->getHalfEdges().second == *m_halfEdges[rI]) {
					inverseHalfEdges.push_back(m_halfEdges[rI]->getEdge()->getHalfEdges().first);
				}
				else {
					throw(exception("HalfFace reversed copy error: halfedges inside edge do not match"));
				}
			}

			
			for (int i = 0; i < inverseHalfEdges.size(); i++) {
				
				Vertex<VectorType> *pNewVertex = nullptr;
				auto iter = ghostVerticesMap.find(inverseHalfEdges[i]->getVertices().first->getID());
				if(iter == ghostVerticesMap.end())
					throw(exception("HalfFace reversed copy error: no ghost vertex found on map"));
				pNewVertex = iter->second;
				
				inverseHalfEdges[i]->setFirstVertex(pNewVertex);

				int prevI = roundClamp<int>(i - 1, 0, inverseHalfEdges.size());
				//inverseHalfEdges[prevI]->getVertices().second = pNewVertex;
				inverseHalfEdges[prevI]->setSecondVertex(pNewVertex);
			}

			return new HalfFace<VectorType>(inverseHalfEdges, m_pFace, m_faceLocation);
		}

		template<class VectorType>
		const vector<HalfFace<VectorType>*> & Face<VectorType>::split() {
			m_halfFaces.clear();

			vector<HalfEdge<VectorType> *> halfEdges;
			Edge<VectorType> *pEdge = nullptr;
			for (int i = 0; i < m_edges.size(); i++) {
				m_edges[i]->setVisited(false);
			}
			while ((pEdge = hasUnvisitedEdges()) != nullptr) {
				halfEdges.clear();

				for (int i = 0; i < m_edges.size(); i++) {
					if(m_edges[i]->getType() == geometricEdge)
						m_edges[i]->setVisited(false);
				}
				
				/** After closing a loop with breadthFirstSearch, halfEdgesVector will have a closed half-face */
				breadthFirstSearch(pEdge, halfEdges);

				if (halfEdges.size() <= 2)
					throw(exception("Face split: Insufficient number of half-edges after breadthFirstSearch"));

				//Half-face temporary position: just to get centroid computed correctly. They should be fixed separately
				//for each half-volume inserted after

				halfFaceLocation_t halfFaceLocation;
				if (m_faceLocation == XYFace)
					halfFaceLocation = backHalfFace;
				else if (m_faceLocation == XZFace)
					halfFaceLocation = bottomHalfFace;
				else if (m_faceLocation == YZFace)
					halfFaceLocation = leftHalfFace;

				m_halfFaces.push_back(new HalfFace<VectorType>(halfEdges, this, halfFaceLocation));
				for (int i = 0; i < m_halfFaces.back()->getHalfEdges().size(); i++) {
					Edge<VectorType> * pCurrEdge = m_halfFaces.back()->getHalfEdges()[i]->getEdge();
					pCurrEdge->addConnectedHalfFace(m_halfFaces.back());
				}
			}

			//Leave all edges Unvisited, will be used for ghost vertices initialization
			for (int i = 0; i < m_edges.size(); i++) {
				m_edges[i]->setVisited(false);
			}

			return m_halfFaces;
		}

		#pragma region Functionalities
		template <class VectorType>
		HalfFace<VectorType> * Face<VectorType>::getHalfFace(const VectorType &position) {
			for (uint i = 0; i < m_halfFaces.size(); i++) {
				if (m_halfFaces[i]->isInside(position)) {
					return m_halfFaces[i];
				}
			}
			return nullptr;
		}

		template <>
		DoubleScalar Face<Vector3>::calculateArea() {
			DoubleScalar areaSum = 0.0f;
			auto halfFace = m_halfFaces.front();
			auto halfEdges = halfFace->getHalfEdges();
			for (int i = 0; i < halfEdges.size(); i++) {
				areaSum += halfFace->getNormal().dot(halfEdges[i]->getVertices().first->getPosition().cross(halfEdges[i]->getVertices().second->getPosition()));
			}
			return abs(areaSum*0.5);
		}

		template <>
		DoubleScalar Face<Vector3D>::calculateArea() {
			DoubleScalar areaSum = 0.0f;
			auto halfFace = m_halfFaces.front();
			auto halfEdges = halfFace->getHalfEdges();
			for (int i = 0; i < halfEdges.size(); i++) {
				areaSum += halfFace->getNormal().dot(halfEdges[i]->getVertices().first->getPosition().cross(halfEdges[i]->getVertices().second->getPosition()));
			}
			return abs(areaSum*0.5);
		}

		template <class VectorType>
		vector<Face<VectorType> *> Face<VectorType>::convertToFaces3D() {
			vector<Face<VectorType> *> faces;
			for (int i = 0; i < m_halfFaces.size(); i++) {
				vector<Edge<VectorType> *> hfedges;
				for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
					hfedges.push_back(m_halfFaces[i]->getHalfEdges()[j]->getEdge());
				}
				Face<VectorType> *pNewFace = new Face<VectorType>(hfedges, m_gridCellLocation, m_gridDx, m_faceLocation);
				faces.push_back(pNewFace);
				//Fixing half-faces pointer to parent face
				m_halfFaces[i]->setFace(pNewFace);
				faces.back()->addHalfFace(m_halfFaces[i]);
				faces.back()->addHalfFace(m_halfFaces[i]->reversedCopy());
			}
			return faces;
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template<class VectorType>
		void Face<VectorType>::breadthFirstSearch(Edge<VectorType> *pEdge, vector<HalfEdge<VectorType> *> &halfEdges) {
			if (pEdge->isVisited())
				return;

			halfEdgeLocation_t currLocation = Edge<VectorType>::classifyEdge(pEdge, m_gridCellLocation, m_gridDx, m_faceLocation);
			//Tag this edge as visited
			pEdge->setVisited(true);
			if (currLocation == geometryHalfEdge) {
				if (halfEdges.back()->getVertices().second->getID() == pEdge->getHalfEdges().first->getVertices().first->getID()) {
					halfEdges.push_back(pEdge->getHalfEdges().first);
				}
				else if (halfEdges.back()->getVertices().second->getID() == pEdge->getHalfEdges().second->getVertices().first->getID()) {
					halfEdges.push_back(pEdge->getHalfEdges().second);
				}
				else {
					throw(exception("Error on vertex path choosing algorithm: wrong connection information on geometry edges."));
				}
				//Update halfedge location
				halfEdges.back()->setLocation(currLocation);
			}
			else if (currLocation == bottomHalfEdge || currLocation == rightHalfEdge) {
				halfEdges.push_back(pEdge->getHalfEdges().first);
				//Update halfedge location
				halfEdges.back()->setLocation(currLocation);
			}
			else if (currLocation == topHalfEdge || currLocation == leftHalfEdge) {
				halfEdges.push_back(pEdge->getHalfEdges().second);
				//Update halfedge location
				halfEdges.back()->setLocation(currLocation);
			}

			//Choosing next Edge
			const vector<Edge<VectorType> *> &nextEdges = m_vertexToEdgeMap.getEdges(halfEdges.back()->getVertices().second);
			if (nextEdges.size() > 3) { //Case that is currently not supported, abort
				throw(exception("Face split: high frequency feature found on top of mesh."));
			}
			else if (nextEdges.size() == 3) { //Going from gridEdges to geometryEdges or vice-versa
				if (currLocation == geometryHalfEdge) {
					bool followThroughOneEdge = false;
					if (nextEdges[0]->getType() != geometricEdge) {
						halfEdgeLocation_t nextLocation = Edge<VectorType>::classifyEdge(nextEdges[0], m_gridCellLocation, m_gridDx, m_faceLocation);
						HalfEdge<VectorType> *pHalfEdge = getOrientedHalfEdge(nextEdges[0], nextLocation);
						if (pHalfEdge->getVertices().first->getID() == halfEdges.back()->getVertices().second->getID()) { //This is the right halfedge
							followThroughOneEdge = true;
							breadthFirstSearch(nextEdges[0], halfEdges);
						}
					}
					if (!followThroughOneEdge && nextEdges[1]->getType() != geometricEdge) {
						halfEdgeLocation_t nextLocation = Edge<VectorType>::classifyEdge(nextEdges[1], m_gridCellLocation, m_gridDx, m_faceLocation);
						HalfEdge<VectorType> *pHalfEdge = getOrientedHalfEdge(nextEdges[1], nextLocation);
						if (pHalfEdge->getVertices().first->getID() == halfEdges.back()->getVertices().second->getID()) { //This is the right halfedge
							followThroughOneEdge = true;
							breadthFirstSearch(nextEdges[1], halfEdges);
						}
					}
					if (!followThroughOneEdge && nextEdges[2]->getType() != geometricEdge) {
						halfEdgeLocation_t nextLocation = Edge<VectorType>::classifyEdge(nextEdges[2], m_gridCellLocation, m_gridDx, m_faceLocation);
						HalfEdge<VectorType> *pHalfEdge = getOrientedHalfEdge(nextEdges[2], nextLocation);
						if (pHalfEdge->getVertices().first->getID() == halfEdges.back()->getVertices().second->getID()) { //This is the right halfedge
							followThroughOneEdge = true;
							breadthFirstSearch(nextEdges[2], halfEdges);
						}
					}
					if (!followThroughOneEdge) {
						throw(exception("Error on vertex path choosing algorithm: unexpected geometry to grid edge case."));
					}
				}
				else {
					if (nextEdges[0]->getType() == geometricEdge) {
						breadthFirstSearch(nextEdges[0], halfEdges);
					}
					else if (nextEdges[1]->getType() == geometricEdge) {
						breadthFirstSearch(nextEdges[1], halfEdges);
					}
					else if (nextEdges[2]->getType() == geometricEdge) {
						breadthFirstSearch(nextEdges[2], halfEdges);
					}
					else {
						throw(exception("Error on vertex path choosing algorithm: not connected to geometric edge."));
					}
				}
			}
			else if (nextEdges.size() == 2) {
				nextEdges[0] == pEdge ? breadthFirstSearch(nextEdges[1], halfEdges) : breadthFirstSearch(nextEdges[0], halfEdges);
			}
			else { //This is a open ended point, go back through same geometric edges

			}
		}

		template<class VectorType>
		HalfEdge<VectorType>* Face<VectorType>::getOrientedHalfEdge(Edge<VectorType>* pEdge, halfEdgeLocation_t halfEdgeLocation) {
			if (halfEdgeLocation == topHalfEdge || halfEdgeLocation == leftHalfEdge)
				return pEdge->getHalfEdges().second;
			else if (halfEdgeLocation == bottomHalfEdge || halfEdgeLocation == rightHalfEdge)
				return pEdge->getHalfEdges().first;
			else {
				throw exception("Invalid half-edge location for getOrientedHalfEdge function. ");
			}

			return nullptr;
		}


		template<class VectorType> 
		VectorType HalfFace<VectorType>::computeCentroid() {
			VectorType centroid;
			if (m_faceLocation == geometryHalfFace) {
				for (int i = 0; i < m_halfEdges.size(); i++) {
					centroid += m_halfEdges[i]->getVertices().first->getPosition();
				}
				return centroid / m_halfEdges.size();
			}

			//Centroid x and y
			double cx = 0, cy = 0;
			
			double signedArea = 0.0;
			double x0 = 0.0; // Current vertex X
			double y0 = 0.0; // Current vertex Y
			double x1 = 0.0; // Next vertex X
			double y1 = 0.0; // Next vertex Y
			double a = 0.0;  // Partial signed area
			
			// For all vertices
			int i = 0;
			for (i = 0; i < m_halfEdges.size(); ++i) {
				switch (m_faceLocation) {
					case backHalfFace:
					case frontHalfFace:
						x0 = m_halfEdges[i]->getVertices().first->getPosition()[0];
						y0 = m_halfEdges[i]->getVertices().first->getPosition()[1];
						x1 = m_halfEdges[i]->getVertices().second->getPosition()[0];
						y1 = m_halfEdges[i]->getVertices().second->getPosition()[1];
					break;

					case leftHalfFace:
					case rightHalfFace:
						x0 = m_halfEdges[i]->getVertices().first->getPosition()[2];
						y0 = m_halfEdges[i]->getVertices().first->getPosition()[1];
						x1 = m_halfEdges[i]->getVertices().second->getPosition()[2];
						y1 = m_halfEdges[i]->getVertices().second->getPosition()[1];
					break;

					case bottomHalfFace:
					case topHalfFace:
						x0 = m_halfEdges[i]->getVertices().first->getPosition()[0];
						y0 = m_halfEdges[i]->getVertices().first->getPosition()[2];
						x1 = m_halfEdges[i]->getVertices().second->getPosition()[0];
						y1 = m_halfEdges[i]->getVertices().second->getPosition()[2];
					break;
				}
				
				a = x0*y1 - x1*y0;
				signedArea += a;
				cx += (x0 + x1)*a;
				cy += (y0 + y1)*a;
			}

			signedArea *= 0.5;
			cx /= (6.0*signedArea);
			cy /= (6.0*signedArea);

			if (isVector2<VectorType>::value) {
				centroid.x = cx;
				centroid.y = cy;
			}
			else {
				switch (m_faceLocation) {
				case backHalfFace:
				case frontHalfFace:
					centroid.x = cx;
					centroid.y = cy;
					centroid[2] = m_halfEdges.front()->getVertices().first->getPosition()[2];
					break;

				case leftHalfFace:
				case rightHalfFace:
					centroid[2] = cx;
					centroid.y = cy;
					centroid.x = m_halfEdges.front()->getVertices().first->getPosition()[0];
					break;

				case bottomHalfFace:
				case topHalfFace:
					centroid.x = cx;
					centroid[2] = cy;
					centroid.y = m_halfEdges.front()->getVertices().first->getPosition()[1];
					break;
				}
			}
			return centroid;
		}

		template<>
		Vector2 HalfFace<Vector2>::computeNormal() {
			return Vector2(0, 0);
		}

		template<>
		Vector2D HalfFace<Vector2D>::computeNormal() {
			return Vector2D(0, 0);
		}

		template<>
		Vector3 HalfFace<Vector3>::computeNormal() {
			Vector3 v1, v2;
			v1 = m_halfEdges[0]->getVertices().second->getPosition() - m_halfEdges[0]->getVertices().first->getPosition();
			v2 = m_halfEdges[1]->getVertices().second->getPosition() - m_halfEdges[1]->getVertices().first->getPosition();
			Vector3 normal = v1.cross(v2);
			normal.normalize();
			return normal;
		}

		template<>
		Vector3D HalfFace<Vector3D>::computeNormal() {
			Vector3D v1, v2;
			v1 = m_halfEdges[0]->getVertices().second->getPosition() - m_halfEdges[0]->getVertices().first->getPosition();
			v2 = m_halfEdges[1]->getVertices().second->getPosition() - m_halfEdges[1]->getVertices().first->getPosition();
			Vector3D normal = v1.cross(v2);
			normal.normalize();
			return normal;
		}
		#pragma endregion

		#pragma region HelperStructuress
		template <class VectorType>
		void EdgeToFaceMap<VectorType>::initializeMap(const vector<Face<VectorType> *> &faces) {
			m_edgeToFaceMap.clear();
			for (int i = 0; i < faces.size(); i++) {
				for (int j = 0; j < faces[i]->getEdges().size(); j++) {
					m_edgeToFaceMap[faces[i]->getEdges()[j]->getID()].push_back(faces[i]);
				}
			}
		}
		#pragma endregion
		template class HalfFace<Vector2>;
		template class HalfFace<Vector2D>;
		template class HalfFace<Vector3>;
		template class HalfFace<Vector3D>;

		template class Face<Vector2>;
		template class Face<Vector2D>;
		template class Face<Vector3>;
		template class Face<Vector3D>;

		template class EdgeToFaceMap<Vector2>;
		template class EdgeToFaceMap<Vector2D>;
		template class EdgeToFaceMap<Vector3>;
		template class EdgeToFaceMap<Vector3D>;
	}
}