#include "Mesh/NonManifoldMesh2D.h"
#include "Grids/GridUtils.h"
//Partial dependency
#include "CutCells/CutFace.h"
#include "Mesh/MeshUtils.h"

namespace Chimera {
	namespace Meshes {


		#pragma region InternalStructures
		NonManifoldMesh2D::DisconnectedRegion::DisconnectedRegion(const lineSegment_t<Vector3D> &lineSegment, faceLocation_t faceLocation) {
			for (int j = 0; j < lineSegment.pEdges->size(); j++)  {
				m_points.push_back(lineSegment.pLine->getPoints()[lineSegment.pEdges->at(j).first]);
				m_centroid += m_points.back();
			}
			m_centroid /= m_points.size();
			m_closedRegion = lineSegment.pLine->isClosedMesh();
		}

		Vector3D NonManifoldMesh2D::DisconnectedRegion::getClosestPoint(const Vector3D &point) const {
			DoubleScalar minDistance = FLT_MAX;
			Vector3D closestPoint;
			for (int i = 0; i < m_points.size(); i++) {
				DoubleScalar currDistance = (m_points[i] - point).length();
				if (currDistance < minDistance) {
					minDistance = currDistance;
					closestPoint = m_points[i];
				}
			}
			return closestPoint;
		}

		pair<Vector3D, Vector3D> NonManifoldMesh2D::DisconnectedRegion::getClosestPoint(const DisconnectedRegion &disconnectedRegion) const {
			pair<Vector3D, Vector3D> closestPoints;
			DoubleScalar minDistance = FLT_MAX;
			for (int i = 0; i < m_points.size(); i++) {
				for (int j = 0; j < disconnectedRegion.getNumberOfPoints(); j++) {
					DoubleScalar currDistance = (m_points[i] - disconnectedRegion.getPoint(j)).length();
					if (currDistance < minDistance) {
						minDistance = currDistance;
						closestPoints.first = m_points[i];
						closestPoints.second = disconnectedRegion.getPoint(j);
					}
				}
 			} 
			return closestPoints;
		}	
		#pragma endregion
		#pragma region Constructors
		NonManifoldMesh2D::NonManifoldMesh2D(dimensions_t cellDim, faceLocation_t faceSliceLocation, const vector<lineSegment_t<Vector3D>> &lineSegments, Scalar dx) : 
			m_lineSegments(lineSegments), m_lineEndsOnCell(lineSegments.size(), false) {
			m_dx = dx;
			m_faceDim = cellDim;
			m_distanceProximity = doublePrecisionThreshold;
			
			m_faceLocation = faceSliceLocation;

			initializeGridVertices();
			initializeEdgeVertices();

			for (int k = 0; k < m_lineSegments.size(); k++) {
				vector<pair<unsigned int, unsigned int>> *pEdges = m_lineSegments[k].pEdges;
				LineMesh<Vector3D> *pLine = m_lineSegments[k].pLine;
				for (int i = 0; i < pEdges->size(); i++) {
					int vertexID1 = addVertex(pLine->getPoints()[pEdges->at(i).first], m_edgeVerticesOffset);
					int vertexID2 = addVertex(pLine->getPoints()[pEdges->at(i).second], m_edgeVerticesOffset);
				}
			}

			//Adding regular cell edges
			addBorderEdge(0, 1, bottomEdge);
			addBorderEdge(1, 2, rightEdge);
			addBorderEdge(2, 3, topEdge);
			addBorderEdge(3, 0, leftEdge);

			

			initializeInsideVerticesAndEdges();

			connectDisconnectedRegions();

			//Verify edges sizes for projected points inconsistencies
			for (int i = 0; i < m_edges.size();) {
				DoubleScalar edgeDistance = (m_vertices[m_edges[i].indices[1]] - m_vertices[m_edges[i].indices[0]]).length();
				if (edgeDistance < doublePrecisionThreshold) { //Beyond numerical error for float numbers
					Logger::getInstance()->get() << " Removing invalid edge " << endl;
					std::cout.precision(15);
					Logger::getInstance()->get() << edgeDistance << endl;
					Logger::getInstance()->get() << " Invalid edge indices " << m_edges[i].indices[0] << ", " << m_edges[i].indices[1] << endl;
					Logger::getInstance()->get() << " P1" << vector3ToStr(m_vertices[m_edges[i].indices[0]]) << "; " << vector3ToStr(m_vertices[m_edges[i].indices[1]]) << endl;

					m_edges.erase(m_edges.begin() + i);
					m_visitedEdges.pop_back();
					m_visitedEdgesCount.pop_back();
					//string exceptionError = "NonManifoldMesh2D: Invalid edge size at " + intToStr(cellDim.x) + " " + intToStr(cellDim.y) + " " + intToStr(cellDim.z);
					//throw exception(exceptionError.c_str());
				}
				else {
					i++;
				}
			}

			

			//Adding all edges references to the vertex map
			for (int i = 0; i < m_edges.size(); i++) {
				m_vertexMap[m_edges[i].indices[0]].push_back(&m_edges[i]);
				m_vertexMap[m_edges[i].indices[1]].push_back(&m_edges[i]);
			}

			
		}
		#pragma endregion

		#pragma region InitializationFunctions
		void NonManifoldMesh2D::initializeGridVertices() {
			Vector3D gridPoint;
			switch (m_faceLocation) {
			case leftFace:
				m_vertices.push_back(Vector3D(m_faceDim.x*m_dx, m_faceDim.y*m_dx, m_faceDim.z*m_dx));
				m_vertices.push_back(Vector3D(m_faceDim.x*m_dx, m_faceDim.y*m_dx, (m_faceDim.z + 1)*m_dx));
				m_vertices.push_back(Vector3D(m_faceDim.x*m_dx, (m_faceDim.y + 1)*m_dx, (m_faceDim.z + 1)*m_dx));
				m_vertices.push_back(Vector3D(m_faceDim.x*m_dx, (m_faceDim.y + 1)*m_dx, m_faceDim.z*m_dx));
				break;
			case bottomFace:
				m_vertices.push_back(Vector3D(m_faceDim.x*m_dx, m_faceDim.y*m_dx, m_faceDim.z*m_dx));
				m_vertices.push_back(Vector3D((m_faceDim.x + 1)*m_dx, m_faceDim.y*m_dx, m_faceDim.z*m_dx));
				m_vertices.push_back(Vector3D((m_faceDim.x + 1)*m_dx, m_faceDim.y*m_dx, (m_faceDim.z + 1)*m_dx));
				m_vertices.push_back(Vector3D(m_faceDim.x*m_dx, m_faceDim.y*m_dx, (m_faceDim.z + 1)*m_dx));
				break;
			case backFace:
				m_vertices.push_back(Vector3D(m_faceDim.x*m_dx, m_faceDim.y*m_dx, m_faceDim.z*m_dx));
				m_vertices.push_back(Vector3D((m_faceDim.x + 1)*m_dx, m_faceDim.y*m_dx, m_faceDim.z*m_dx));
				m_vertices.push_back(Vector3D((m_faceDim.x + 1)*m_dx, (m_faceDim.y + 1)*m_dx, m_faceDim.z*m_dx));
				m_vertices.push_back(Vector3D(m_faceDim.x*m_dx, (m_faceDim.y + 1)*m_dx, m_faceDim.z*m_dx));
				break;

				default:
					throw exception("NonManifoldMesh2D: Invalid face location");
				break;
			}
			m_edgeVerticesOffset = m_vertices.size();
		}

		void NonManifoldMesh2D::initializeEdgeVertices() {
			for (int k = 0; k < m_lineSegments.size(); k++) {
				vector<pair<unsigned int, unsigned int>> *pEdges = m_lineSegments[k].pEdges;
				LineMesh<Vector3D> *pLine = m_lineSegments[k].pLine;
				for (int i = 0; i < pEdges->size(); i++) {
					Vector3D vertex = pLine->getPoints()[pEdges->at(i).first];
					//Refactor the way this function works
					if (isOnGridEdge(vertex, m_dx)) {
						addVertex(vertex, m_edgeVerticesOffset);
					}
					vertex = pLine->getPoints()[pEdges->at(i).second];
					if (isOnGridEdge(vertex, m_dx)) {
						addVertex(vertex, m_edgeVerticesOffset);
					}
				}
			}
			m_insideVerticesOffset = m_vertices.size();
		}

		void NonManifoldMesh2D::initializeInsideVerticesAndEdges() {
			m_geometryEdgesOffset = m_edges.size();
			for (int k = 0; k < m_lineSegments.size(); k++) {
				vector<pair<unsigned int, unsigned int>> *pEdges = m_lineSegments[k].pEdges;
				LineMesh<Vector3D> *pLine = m_lineSegments[k].pLine;
				for (int i = 0; i < pEdges->size(); i++) {
					int vertexID1 = addVertex(pLine->getPoints()[pEdges->at(i).first], m_edgeVerticesOffset);
					int vertexID2 = addVertex(pLine->getPoints()[pEdges->at(i).second], m_edgeVerticesOffset);
					addEdge(vertexID1, vertexID2, geometryEdge);
					m_edges.back().crossesGrid = m_lineSegments[k].crossesGrid;
				}
				//if (!pLine->isClosedMesh()) {
				//	if (pEdges->front().first == 0 || pEdges->front().first == pLine->getPoints().size() - 1 ||
				//		pEdges->back().second == 0 || pEdges->back().second == pLine->getPoints().size() - 1) {
				//		//Adding back all the edges in the inverse order (open-ended cell)
				//		for (int i = pEdges->size() - 1; i >= 0; i--) {
				//			int vertexID1 = addVertex(pLine->getPoints()[pEdges->at(i).second], m_edgeVerticesOffset);
				//			int vertexID2 = addVertex(pLine->getPoints()[pEdges->at(i).first], m_edgeVerticesOffset);
				//			addEdge(vertexID1, vertexID2, geometryEdge);
				//		}
				//		m_lineEndsOnCell[k] = true;
				//	}
				//}
			}
		}
		#pragma endregion


		#pragma region Functionalities
		vector<CutFace<Vector3D> *> NonManifoldMesh2D::split() {
			int ithNonVisitedFace;

			vector<CutFace<Vector3D> *> cutFaces;
			while ((ithNonVisitedFace = findNonVisitedEdge()) != -1) {
				//The face ID is calculated locally. Be careful on global applications.
				CutFace<Vector3D> *pFace = new CutFace<Vector3D>(cutFaces.size());
				pFace->m_regularGridIndex = m_faceDim;

				//Tag all geometry edges and connection edges as not visited
				for (int i = m_geometryEdgesOffset; i < m_edges.size(); i++) {
					if (m_edges[i].edgeLocation == geometryEdge || m_edges[i].edgeLocation == connectionEdge) {
						m_visitedEdges[i] = false;
					}
				}

				int regionTag = signedDistanceTag(m_edges[ithNonVisitedFace].centroid);

				breadthFirstSearch(m_edges[ithNonVisitedFace], pFace, regionTag, -1);

				pFace->m_interiorPoint = findValidInteriorPoint(pFace);

				if (pFace->m_cutEdges.size() > 0) { //Discard invalid 0 edges faces
					pFace->updateCentroid();
					cutFaces.push_back(pFace);
				}
				else {
					Logger::getInstance()->get() << " Breaking from invalid face configuration" << endl;
					break;
				}

				
			}

			//If, after the all breadthFirstSearch, there's any unvisited geometry edges, they are from
			//a disconnected part inside the cell, so they must be added to the edges structure
			
			vector<CutFace<Vector3D> *> disconnectedRegionFaces;
			int prevEdgeIndex = -1;
			for (int i = m_geometryEdgesOffset; i < m_edges.size(); i++) {
				if (!m_edges[i].crossesGrid) {
					CutFace<Vector3D> *pNewInteriorFace = new CutFace<Vector3D>(cutFaces.size());
					pNewInteriorFace->m_regularGridIndex = m_faceDim;
					int prevEdgeIndex = -1;
					int j = i;
					while (j < m_edges.size()) {
						CutEdge<Vector3D> *tempOppCutEdge = m_edges[j].pCutEdge;
						if (prevEdgeIndex != -1 && m_edges[j].indices[0] != prevEdgeIndex) { //Adding the inverted point
							tempOppCutEdge = new CutEdge<Vector3D>(m_edges[j].pCutEdge->getID(), m_edges[j].pCutEdge->m_finalPoint, m_edges[j].pCutEdge->m_initialPoint, m_dx, geometryEdge);
							prevEdgeIndex = m_edges[j].indices[0];
						} else {
							prevEdgeIndex = m_edges[j].indices[1];
						}
						if (!m_edges[j].crossesGrid && (pNewInteriorFace->m_cutEdges.size() == 0 || tempOppCutEdge->m_initialPoint == pNewInteriorFace->m_cutEdges.back()->m_finalPoint)) {
							pNewInteriorFace->m_cutEdgesLocations.push_back(geometryEdge);
							pNewInteriorFace->m_cutEdges.push_back(tempOppCutEdge);
						} else {
							disconnectedRegionFaces.push_back(pNewInteriorFace);
							i = j - 1; //The previous one, since on the end of this loop, i++
							break;
						}
						j++;
					}
				}
			}

			for (int i = 0; i < disconnectedRegionFaces.size(); i++) {
				disconnectedRegionFaces[i]->m_interiorPoint = findValidInteriorPoint(disconnectedRegionFaces[i]);
				Vector3D currCentroid;
				for (int j = 0; j < disconnectedRegionFaces[i]->m_cutEdges.size(); j++) {
					currCentroid += disconnectedRegionFaces[i]->m_cutEdges[j]->m_initialPoint;
				}
				currCentroid /= disconnectedRegionFaces[i]->m_cutEdges.size();
				disconnectedRegionFaces[i]->m_centroid = currCentroid;
				cutFaces.push_back(disconnectedRegionFaces[i]);
			}

			return cutFaces;
		}

		Vector3D NonManifoldMesh2D::findValidInteriorPoint(CutFace<Vector3D> *pCutFace) {
			DoubleScalar smallerEdgeRatio = FLT_MAX;
			Vector3D bestInteriorPoint;

			if (pCutFace->m_cutEdges.size() < 3)
				return Vector3D(0, 0, 0);

			for (int j = 0; j < pCutFace->m_cutEdges.size(); j++) {
				int nextJ = roundClamp<int>(j + 1, 0, pCutFace->m_cutEdges.size());
				int nextNextJ = roundClamp<int>(j + 2, 0, pCutFace->m_cutEdges.size());

				Vector3D p1 = pCutFace->m_cutEdges[j]->m_initialPoint;
				Vector3D p2 = pCutFace->m_cutEdges[nextJ]->m_initialPoint;
				Vector3D p3 = pCutFace->m_cutEdges[nextNextJ]->m_initialPoint;

				DoubleScalar minEdgeSize, maxEdgeSize;
				Vector3D e1 = p2 - p1;
				Vector3D e2 = p3 - p2;
				Vector3D e3 = p1 - p3;
				DoubleScalar e1Size = e1.length();
				DoubleScalar e2Size = e2.length();
				DoubleScalar e3Size = e3.length();
				minEdgeSize = min(min(e1Size, e2Size), e3Size);
				maxEdgeSize = max(max(e1Size, e2Size), e3Size);
				DoubleScalar edgeRatio = maxEdgeSize / minEdgeSize;

				DoubleScalar areaFraction = calculateTriangleArea(p1, p2, p3);

				edgeRatio /= areaFraction;

				Vector3D centroid = (p1 + p2 + p3) / 3;
				//isInsideFace(centroid, *pCutFace, m_faceLocation) &&
				if (isInsideFace(centroid, *pCutFace, m_faceLocation) && edgeRatio < smallerEdgeRatio) {
					smallerEdgeRatio = edgeRatio;
					bestInteriorPoint = centroid;
					pCutFace->m_normal = (e1).cross(e2);
					pCutFace->m_normal.normalize();
				}
			}
			if (smallerEdgeRatio == FLT_MAX) {
				throw exception("NonManifoldMesh: valid point inside cell face not found");
			}
			return bestInteriorPoint;
		}

		bool NonManifoldMesh2D::isInsideFace(const Vector3D & point, const CutFace<Vector3D> &face, faceLocation_t faceLocation) {
			vector<Vector2D> polygonTempPoints;
			Vector2D projectedPoint;
			switch (faceLocation) {
			case leftFace:
			case rightFace:
				for (int i = 0; i < face.m_cutEdges.size(); i++) {
					Vector3D originalPoint = face.m_cutEdges[i]->m_initialPoint;
					//Projecting onto the YZ plane
					Vector2D convertedPoint(originalPoint.y, originalPoint.z);
					polygonTempPoints.push_back(convertedPoint);
				}
				projectedPoint.x = point.y;
				projectedPoint.y = point.z;
				break;

			case bottomFace:
			case topFace:
				for (int i = 0; i < face.m_cutEdges.size(); i++) {
					Vector3D originalPoint = face.m_cutEdges[i]->m_initialPoint;
					//Projecting onto the XZ plane
					Vector2D convertedPoint(originalPoint.x, originalPoint.z);
					polygonTempPoints.push_back(convertedPoint);
				}
				projectedPoint.x = point.x;
				projectedPoint.y = point.z;
				break;

			case backFace:
			case frontFace:
				for (int i = 0; i < face.m_cutEdges.size(); i++) {
					Vector3D originalPoint = face.m_cutEdges[i]->m_initialPoint;
					//Projecting onto the XY plane
					Vector2D convertedPoint(originalPoint.x, originalPoint.y);
					polygonTempPoints.push_back(convertedPoint);
				}
				projectedPoint.x = point.x;
				projectedPoint.y = point.y;
				break;

			default:
				throw exception("NonManifoldMesh: invalid call to isInsideFace");
				break;
			}
			return isInsidePolygon(projectedPoint, polygonTempPoints);
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		int NonManifoldMesh2D::crossesThroughGeometryFaces(const Vector3D &p1, const Vector3D &p2) {
			Scalar dx = m_dx;
			int crossedIndex = 0;
			DoubleScalar crossPointDistance = FLT_MAX;

			for (int i = m_geometryEdgesOffset; i < m_edges.size(); i++) {
				Vector3D v1 = m_vertices[m_edges[i].indices[0]];
				Vector3D v2 = m_vertices[m_edges[i].indices[1]];

				if (p1 == v1 || p1 == v2 || p2 == v1 || p2 == v2 )
					continue;

				Vector3D crossedPoint;
				if (m_edges[i].edgeLocation == geometryEdge && DoLinesIntersect(p1, p2, m_vertices[m_edges[i].indices[0]], m_vertices[m_edges[i].indices[1]], crossedPoint)) {
					DoubleScalar currDistance = (p2 - crossedPoint).length();
					if (currDistance < crossPointDistance) {
						crossPointDistance = currDistance;
						if (isOnGridEdge(m_vertices[m_edges[i].indices[0]], dx)) {
							crossedIndex = m_edges[i].indices[1];
						}
						else {
							crossedIndex = m_edges[i].indices[0];
						}
					}
					
				}
			}

			return crossedIndex;
		}
		int NonManifoldMesh2D::splitEdge(const Vector3D &splitPoint) {
			for (int i = 0; i < m_edges.size(); i++) {
				if (distanceToLineSegment(splitPoint, m_vertices[m_edges[i].indices[0]], m_vertices[m_edges[i].indices[1]]) < doublePrecisionThreshold) {
					edge_t oldEdge = m_edges[i];
					removeEdge(i);
					int vertexID = addVertex(splitPoint, 0);
					insertEdge(oldEdge.indices[0], vertexID, i, oldEdge.edgeLocation);
					insertEdge(vertexID, oldEdge.indices[1], i + 1, oldEdge.edgeLocation);
					fixEdgesIDs();
					return vertexID;
				}
			}
			return -1;
		}


		int NonManifoldMesh2D::getClosestOuterPointID(const Vector3D &point) const{
			DoubleScalar smallerDistance = FLT_MAX;
			int currIndexID = -1;
			for (int i = 0; i < m_edgeVerticesOffset; i++) {
				if (isOnGridPoint(m_vertices[i], m_dx)) {
					DoubleScalar currLength = (m_vertices[i] - point).length();
					if (currLength < smallerDistance) {
						smallerDistance = currLength;
						currIndexID = i;
					}
				}
			}
			return currIndexID;
		}

		void NonManifoldMesh2D::connectDisconnectedRegions() {
			m_disconnectedRegions.clear();

			int outerSmallerIndex, outerGreaterIndex;
			Vector3D smallerPoint(FLT_MAX, FLT_MAX, FLT_MAX), greaterPoint(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			for (int i = 0; i < m_vertices.size(); i++) {
				if (m_vertices[i] < smallerPoint) {
					smallerPoint = m_vertices[i];
					outerSmallerIndex = i;
				}
				if (m_vertices[i] > greaterPoint) {
					greaterPoint = m_vertices[i];
					outerGreaterIndex = i;
				}
			}

			for (int i = 0; i < m_lineSegments.size(); i++) {
				if (!m_lineSegments[i].crossesGrid) {
					m_disconnectedRegions.push_back(DisconnectedRegion(m_lineSegments[i], m_faceLocation));
				}
			}

			if (m_faceLocation == backFace || m_faceLocation == leftFace) { //YZ or XY, choose Y 
				sort(m_disconnectedRegions.begin(), m_disconnectedRegions.end(), compareDisconnectedRegionsA);
			} else {
				sort(m_disconnectedRegions.begin(), m_disconnectedRegions.end(), compareDisconnectedRegionsB);
			}

			for (int i = 0; i < m_disconnectedRegions.size(); i++) {
				if (i == 0) {
					//outerSmallerIndex = getClosestOuterPointID(m_disconnectedRegions[i].getCentroid());
					int closestPointIndex = findVertexID(m_disconnectedRegions[i].getClosestPoint(m_vertices[outerSmallerIndex]));
					int crossesThrough = crossesThroughGeometryFaces(m_vertices[outerSmallerIndex], m_vertices[closestPointIndex]);
					if (crossesThrough == 0) {
						addEdge(outerSmallerIndex, closestPointIndex, connectionEdge);
					}
					else {
						addEdge(crossesThrough, closestPointIndex, connectionEdge);
					}
				}
				if (i < m_disconnectedRegions.size() - 1) {
					pair<Vector3D, Vector3D> closestPointPairs = m_disconnectedRegions[i].getClosestPoint(m_disconnectedRegions[i + 1]);
					
					int closestPointIndexCurrent = findVertexID(closestPointPairs.first);
					int closestPointIndexNext = findVertexID(closestPointPairs.second);

					int crossesThrough = crossesThroughGeometryFaces(m_vertices[closestPointIndexNext], m_vertices[closestPointIndexCurrent]);
					if (crossesThrough == 0) {
						addEdge(closestPointIndexCurrent, closestPointIndexNext, connectionEdge);
					}
					else {
						addEdge(closestPointIndexCurrent, crossesThrough, connectionEdge);
						
						//We have to add the next edge that didn't make through the geometry
						int crossesThrough2 = crossesThroughGeometryFaces(m_vertices[closestPointIndexCurrent], m_vertices[closestPointIndexNext]);
						if (crossesThrough2 == 0) {
							throw exception("NonManifoldMesh2D: invalid configuration on connectDisconnectRegions");
						}

						if (crossesThrough2 == crossesThrough) {
							Logger::getInstance()->get() << "Warning: Fixing connectDisconnectRegions crosses through function" << endl;
							crossesThrough2++;
						}
						addEdge(closestPointIndexNext, crossesThrough2, connectionEdge);
					}
				}
				else if (i == m_disconnectedRegions.size() - 1) {
					//outerGreaterIndex = getClosestOuterPointID(m_disconnectedRegions[i].getCentroid());
					int closestPointIndex = findVertexID(m_disconnectedRegions[i].getClosestPoint(m_vertices[outerGreaterIndex]));

					int crossesThrough = crossesThroughGeometryFaces(m_vertices[outerGreaterIndex], m_vertices[closestPointIndex]);
					if (crossesThrough == 0) {
						addEdge(outerGreaterIndex, closestPointIndex, connectionEdge);
					}
					else {
						addEdge(crossesThrough, closestPointIndex, connectionEdge);
					}
				}
			}
		}

		void NonManifoldMesh2D::connectDisconnectedRegions2() {
			
			m_disconnectedRegions.clear();

			vector<Vector3D> minDCCRegionsPoints;
			vector<Vector3D> maxDCCRegionsPoints;
			vector<int> minDCCRegionIndices;
			vector<int> maxDCCRegionIndices;
			vector<Vector3D> dccRegionsMinProjections;
			vector<Vector3D> dccRegionsMaxProjections;

			int outerSmallerIndex, outerGreaterIndex;
			Vector3D smallerPoint(FLT_MAX, FLT_MAX, FLT_MAX), greaterPoint(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			for (int i = 0; i < m_vertices.size(); i++) {
				if (m_vertices[i] < smallerPoint) {
					smallerPoint = m_vertices[i];
					outerSmallerIndex = i;
				}
				if (m_vertices[i] > greaterPoint) {
					greaterPoint = m_vertices[i];
					outerGreaterIndex = i;
				}
			}

			for (int i = 0; i < m_lineSegments.size(); i++) {
				if (!m_lineSegments[i].crossesGrid) {
					Vector3D smallestCoordVec(FLT_MAX, FLT_MAX, FLT_MAX), greatestCoordVec(-FLT_MAX, -FLT_MAX, -FLT_MAX);

					for (int j = 0; j < m_lineSegments[i].pEdges->size(); j++)  {
						Vector3D linePoint = m_lineSegments[i].pLine->getPoints()[m_lineSegments[i].pEdges->at(j).first];
						if (m_faceLocation == backFace || m_faceLocation == leftFace) { //YZ or XY, choose Y 
							if (linePoint.y < smallestCoordVec.y) {
								smallestCoordVec = linePoint;
							}
							if (linePoint.y > greatestCoordVec.y) {
								greatestCoordVec = linePoint;
							}
						}
						else if (m_faceLocation == bottomFace) { //XZ, choose between X and Z
							if (linePoint.z < smallestCoordVec.z) {
								smallestCoordVec = linePoint;
							}
							if (linePoint.z > greatestCoordVec.z) {
								greatestCoordVec = linePoint;
							}
						}
					}
					minDCCRegionsPoints.push_back(smallestCoordVec);
					maxDCCRegionsPoints.push_back(greatestCoordVec);

					int smallestCoordIndex = -1;
					int greatestCoordIndex = -1;
					for (int j = m_edgeVerticesOffset; j < m_vertices.size(); j++) {
						if (m_vertices[j] == smallestCoordVec) {
							smallestCoordIndex = j;
						}
						else if (m_vertices[j] == greatestCoordVec) {
							greatestCoordIndex = j;
						}
					}
					minDCCRegionIndices.push_back(smallestCoordIndex);
					maxDCCRegionIndices.push_back(greatestCoordIndex);
				}
			}

			for (int i = 0; i < minDCCRegionIndices.size(); i++) {
				//Finding the projections of the ray
				if (m_faceLocation == backFace || m_faceLocation == leftFace) { //YZ or XY, choose Y 
					Vector3D projectedPointMin = minDCCRegionsPoints[i];
					projectedPointMin.y = floor(minDCCRegionsPoints[i].y);
					Vector3D projectedPointMax = projectedPointMin;
					projectedPointMax.y += 1;
					dccRegionsMinProjections.push_back(projectedPointMin);
					dccRegionsMaxProjections.push_back(projectedPointMax);
				}
				else if (m_faceLocation == bottomFace) {
					Vector3D projectedPointMin = minDCCRegionsPoints[i];
					projectedPointMin.z = floor(minDCCRegionsPoints[i].z);
					Vector3D projectedPointMax = projectedPointMin;
					projectedPointMax.z += 1;
					dccRegionsMinProjections.push_back(projectedPointMin);
					dccRegionsMaxProjections.push_back(projectedPointMax);
				}

				int crossesThrough = crossesThroughGeometryFaces(dccRegionsMinProjections[i], minDCCRegionsPoints[i]);
				//int crossesThrough = crossesThroughGeometryFaces(m_vertices[0], minDCCRegionsPoints[i]);
				if (crossesThrough == 0) {
					int vertexID = splitEdge(dccRegionsMinProjections[i]);
					//addEdge(outerSmallerIndex, minDCCRegionIndices[i], connectionEdge);
					addEdge(vertexID, minDCCRegionIndices[i], connectionEdge);
				}
				else {
					addEdge(crossesThrough, minDCCRegionIndices[i], connectionEdge);
				}
				crossesThrough = crossesThroughGeometryFaces(dccRegionsMaxProjections[i], maxDCCRegionsPoints[i]);
				//crossesThrough = crossesThroughGeometryFaces(m_vertices[2], maxDCCRegionsPoints[i]);
				if (crossesThrough == 0) {
					int vertexID = splitEdge(dccRegionsMaxProjections[i]);
					addEdge(vertexID, maxDCCRegionIndices[i], connectionEdge);
					//addEdge(outerGreaterIndex, maxDCCRegionIndices[i], connectionEdge);
				}
				else {
					addEdge(crossesThrough, maxDCCRegionIndices[i], connectionEdge);
				}
			}
		}

		NonManifoldMesh2D::edge_t * NonManifoldMesh2D::chooseEdge(const Vector2D &direction, int currEdgeID, int currVertexID, vector<edge_t *> vertexEdges) {
			edge_t *pE1 = vertexEdges[0]->ID != currEdgeID ? vertexEdges[0] : vertexEdges[1];
			edge_t *pE2 = vertexEdges[2]->ID != currEdgeID ? vertexEdges[2] : vertexEdges[1];


			Vector2D pE1Vec, pE2Vec;
			if (pE1->indices[0] == currVertexID) {
				pE1Vec = convertToVec2(m_vertices[pE1->indices[1]] - m_vertices[pE1->indices[0]]);
			}
			else {
				pE1Vec = convertToVec2(m_vertices[pE1->indices[0]] - m_vertices[pE1->indices[1]]);
			}

			if (pE2->indices[0] == currVertexID) {
				pE2Vec = convertToVec2(m_vertices[pE2->indices[1]] - m_vertices[pE2->indices[0]]);
			}
			else {
				pE2Vec = convertToVec2(m_vertices[pE2->indices[0]] - m_vertices[pE2->indices[1]]);
			}

			Scalar angle1 = angle2D(direction, pE1Vec);
			Scalar angle2 = angle2D(direction, pE2Vec);

		/*	if (m_faceLocation == leftFace || m_faceLocation == bottomFace) {
				if (angle1 < angle2) {
					return pE1;
				}
				return pE2;
			} else*/ if (angle1 > angle2) {
				return pE1;
			}

			return pE2;
		}
		void NonManifoldMesh2D::breadthFirstSearch(edge_t &edge, CutFace<Vector3D> *pFace, int regionTag, int prevEdgeIndex) {
			if (m_visitedEdges[edge.ID])
				return;

			m_visitedEdges[edge.ID] = true;
			m_visitedEdgesCount[edge.ID]++;
			
			//Check prevEdgeIndex to verify if the current half-edge has to be flipped
			if (edge.edgeLocation == connectionEdge && m_visitedEdgesCount[edge.ID] == 2) {
				if (edge.addedInverted) {
					pFace->m_cutEdgesLocations.push_back(edge.edgeLocation);
					pFace->m_cutEdges.push_back(edge.pCutEdge);
					prevEdgeIndex = edge.indices[0];
				}
				else {
					CutEdge<Vector3D> *tempCutEdge = new CutEdge<Vector3D>(edge.pCutEdge->getID(), edge.pCutEdge->m_finalPoint, edge.pCutEdge->m_initialPoint, m_dx, geometryEdge);
					pFace->m_cutEdgesLocations.push_back(edge.edgeLocation);
					pFace->m_cutEdges.push_back(tempCutEdge);
					prevEdgeIndex = edge.indices[1];
				}
			} else if ((edge.edgeLocation == geometryEdge || edge.edgeLocation == connectionEdge) && prevEdgeIndex != -1 && edge.indices[0] != prevEdgeIndex) {
				//Create a new pEdgePointer with the inverse of the previous edge, but use the same ID
				CutEdge<Vector3D> *tempCutEdge = new CutEdge<Vector3D>(edge.pCutEdge->getID(), edge.pCutEdge->m_finalPoint, edge.pCutEdge->m_initialPoint, m_dx, geometryEdge);
				pFace->m_cutEdgesLocations.push_back(edge.edgeLocation);
				pFace->m_cutEdges.push_back(tempCutEdge);
				edge.addedInverted = true;
			}
			else {
				pFace->m_cutEdgesLocations.push_back(edge.edgeLocation);
				pFace->m_cutEdges.push_back(edge.pCutEdge);
				edge.addedInverted = false;
			}
			
			

			//If the edge is not part of the geometry, following the index 1 will guarantee that the cell will be 
			//traversed on a CCW manner
			if (edge.edgeLocation != geometryEdge && edge.edgeLocation != connectionEdge) {
				vector<edge_t *> currVertexEdges = m_vertexMap[edge.indices[1]];
				if (currVertexEdges.size() > 2) {
					edge_t *pE1 = currVertexEdges[0]->ID != edge.ID ? currVertexEdges[0] : currVertexEdges[1];
					edge_t *pE2 = currVertexEdges[2]->ID != edge.ID ? currVertexEdges[2] : currVertexEdges[1];
					if (pE1->edgeLocation == geometryEdge || pE1->edgeLocation == connectionEdge) {
						breadthFirstSearch(*pE1, pFace, regionTag, edge.indices[1]);
					}
					else if (pE2->edgeLocation == geometryEdge || pE2->edgeLocation == connectionEdge) {
						breadthFirstSearch(*pE2, pFace, regionTag, edge.indices[1]);
					} else {
						throw exception("NonManifoldMesh: invalid edge neighbor configuration");
					}
				}
				else if (currVertexEdges.size() > 1) {
					breadthFirstSearch(currVertexEdges[0]->ID != edge.ID ? *currVertexEdges[0] : *currVertexEdges[1], pFace, regionTag, edge.indices[1]);
				} else {
					throw exception("NonManifoldMesh: invalid number of edge neighbors");
				}
			} else {
				vector<edge_t *> currVertexEdges;
				//If our initial edge vertex, which is the same as the previous edge final point,
				//is the same as the vertex that we are searching for on the edge.indices[0],
				//we dont need to check that direction, since we already went through that.
				int nextVertexIndex = 0;
				if (prevEdgeIndex == edge.indices[0]) {
					currVertexEdges = m_vertexMap[edge.indices[1]];
					nextVertexIndex = edge.indices[1];
				}
				else {
					currVertexEdges = m_vertexMap[edge.indices[0]];
					nextVertexIndex = edge.indices[0];
				}
				if (currVertexEdges.size() > 3) {
					throw exception("NonManifoldMesh2D: High frequency feature found on top of a vertex");
				}
				else if (currVertexEdges.size() > 2) {
					Vector2D iniPoint = convertToVec2(pFace->m_cutEdges.back()->m_initialPoint);
					Vector2D finalPoint = convertToVec2(pFace->m_cutEdges.back()->m_finalPoint);
					Vector2D direction = finalPoint - iniPoint;
					edge_t *pTempEdge = chooseEdge(direction, edge.ID, nextVertexIndex, currVertexEdges);
					breadthFirstSearch(*pTempEdge, pFace, regionTag, nextVertexIndex);
				}
				else if (currVertexEdges.size() > 1) {
					breadthFirstSearch(currVertexEdges[0]->ID != edge.ID ? *currVertexEdges[0] : *currVertexEdges[1], pFace, regionTag, nextVertexIndex);
				} else {
					//We got to roll back all the geometry edges added, adding them on the inverse order
					//First we store all geometry edges in the stack until we find a non-geometry edge
					vector<CutEdge<Vector3D> *> geometryCutEdges;
					int lastNonGeometryEdge = -1;
					for (int i = pFace->m_cutEdges.size() - 1; i >= 0; i--) {
						if (pFace->m_cutEdgesLocations[i] != geometryEdge) {
							lastNonGeometryEdge = i;
							break;
						}
						geometryCutEdges.push_back(pFace->m_cutEdges[i]);
					}
					
					//Then for all those edges, we revert and add them back to the faces, in an inverse order
					for (int i = 0; i < geometryCutEdges.size(); i++) {
						CutEdge<Vector3D> *pInvertedCutEdge = new CutEdge<Vector3D>(m_edges.size() + i, geometryCutEdges[i]->m_finalPoint, geometryCutEdges[i]->m_initialPoint, m_dx, geometryEdge);
						pFace->m_cutEdgesLocations.push_back(geometryEdge);
						pFace->m_cutEdges.push_back(pInvertedCutEdge);
					}

					//Then, for the last added, we choose based on the geometry which side we should go
					int vertexID = findVertex(pFace->m_cutEdges.back()->m_finalPoint);
					vector<edge_t *> tempCurrVertexEdges = m_vertexMap[vertexID];
					for (int i = 0; i < tempCurrVertexEdges.size(); i++) {
						if (tempCurrVertexEdges[i]->edgeLocation != geometryEdge) {
							if (tempCurrVertexEdges[i]->ID != pFace->m_cutEdges[lastNonGeometryEdge]->getID()) {
								breadthFirstSearch(*tempCurrVertexEdges[i], pFace, regionTag, nextVertexIndex);
							}
						}
					}

					Vector2D iniPoint = convertToVec2(pFace->m_cutEdges.back()->m_initialPoint);
					Vector2D finalPoint = convertToVec2(pFace->m_cutEdges.back()->m_finalPoint);
					Vector2D direction = finalPoint - iniPoint;


					

					edge_t *pTempEdge = chooseEdge(direction, edge.ID, vertexID, tempCurrVertexEdges);
					breadthFirstSearch(*pTempEdge, pFace, regionTag, nextVertexIndex);
				}

			}
		}

		int NonManifoldMesh2D::findNonVisitedEdge() {
			for (int i = 0; i < m_visitedEdges.size(); i++) {
				if (m_edges[i].edgeLocation == connectionEdge) {
					if (m_visitedEdgesCount[i] < 2) {
						return i;
					}
				}
				else if (!m_visitedEdges[i] && m_edges[i].edgeLocation != geometryEdge && m_edges[i].edgeLocation != connectionEdge)
					return i;
			}
			//All faces are visited
			return -1;
		}
		
		int NonManifoldMesh2D::signedDistanceTag(const Vector3D & point) {
			int regionTag = 0;
			for (int k = 0; k < m_lineSegments.size(); k++) {
				Scalar minDistance = FLT_MAX;
				int ithEdge = -1;
				
				if (!m_lineEndsOnCell[k] && m_lineSegments[k].crossesGrid) {
					vector<pair<unsigned int, unsigned int>> *pEdges = m_lineSegments[k].pEdges;
					LineMesh<Vector3D> *pLine = m_lineSegments[k].pLine;
					for (int i = 0; i < pEdges->size(); i++) {
						Vector3D e1 = pLine->getPoints()[pEdges->at(i).first];
						Vector3D e2 = pLine->getPoints()[pEdges->at(i).second];
						Scalar tempDistance = distanceToLineSegment(point, e1, e2);
						if (tempDistance < minDistance) {
							minDistance = tempDistance;
							ithEdge = i;
						}
					}
					Vector3D edgeNormal = -MeshUtils::getEdgeNormal(pLine->getPoints()[pEdges->at(ithEdge).first],
														pLine->getPoints()[pEdges->at(ithEdge).second], m_faceLocation);
					
					Vector3D edgeCentroid = (pLine->getPoints()[pEdges->at(ithEdge).first] + pLine->getPoints()[pEdges->at(ithEdge).second])*0.5;
					if (edgeNormal.dot(point - edgeCentroid) < 0) {
						minDistance = -minDistance;
					}
					if (minDistance > 0) {
						regionTag++;
					}
				}
			}

			return regionTag;
		}

		Vector2D NonManifoldMesh2D::convertToVec2(const Vector3D &vec) {
			Vector2D vec2;
			switch (m_faceLocation) {
				case rightFace:
				case leftFace:
					vec2.x = vec.z;
					vec2.y = vec.y;
				break;
				case topFace:
				case bottomFace:
					vec2.x = vec.x;
					vec2.y = vec.z;
				break;

				case frontFace:
				case backFace:
					vec2.x = vec.x;
					vec2.y = vec.y;
				break;
			}
			return vec2;
		}
		#pragma endregion
		#pragma region AddingFunctions
		void NonManifoldMesh2D::addBorderEdge(unsigned int vertexID1, unsigned int vertexID2, edgeLocation_t edgeLocation) {
			vector<pair<Scalar, unsigned int>> onTheLineVertices;
			for (unsigned int i = m_edgeVerticesOffset; i < m_insideVerticesOffset; i++) {
				if (distanceToLine(m_vertices[i], m_vertices[vertexID1], m_vertices[vertexID2]) < m_distanceProximity) {
					Scalar lineAlpha = linearInterpolationWeight(m_vertices[i], m_vertices[vertexID1], m_vertices[vertexID2]);
					onTheLineVertices.push_back(pair<Scalar, unsigned int>(lineAlpha, i));
				}
			}
			if (onTheLineVertices.size() > 0) {
				if (onTheLineVertices.size() > 1) {
					sort(onTheLineVertices.begin(), onTheLineVertices.end(), compareEdges);
				}
				addEdge(vertexID1, onTheLineVertices.front().second, edgeLocation);
				for (unsigned int i = 1; i < onTheLineVertices.size(); i++) {
					addEdge(onTheLineVertices[i - 1].second, onTheLineVertices[i].second, edgeLocation);
				}
				addEdge(onTheLineVertices.back().second, vertexID2, edgeLocation);
			}
			else {
				addEdge(vertexID1, vertexID2, edgeLocation);
			}
			
		}
		void NonManifoldMesh2D::addEdge(unsigned int vertexID1, unsigned int vertexID2, edgeLocation_t edgeLocation) {
			edge_t edge(m_edges.size());

			edge.indices[0] = vertexID1;
			edge.indices[1] = vertexID2;
			edge.edgeLocation = edgeLocation;
			edge.centroid = (m_vertices[vertexID1] + m_vertices[vertexID2]) * 0.5f;

			edge.pCutEdge = new CutEdge<Vector3D>(m_edges.size(), m_vertices[vertexID1], m_vertices[vertexID2], m_dx, edgeLocation);

			m_edges.push_back(edge);
			m_visitedEdges.push_back(false);
			m_visitedEdgesCount.push_back(0);
		}
		
		void NonManifoldMesh2D::insertEdge(unsigned int vertexID1, unsigned int vertexID2, unsigned int edgePosition, edgeLocation_t edgeLocation) {
			edge_t edge(edgePosition);

			edge.indices[0] = vertexID1;
			edge.indices[1] = vertexID2;
			edge.edgeLocation = edgeLocation;
			edge.centroid = (m_vertices[vertexID1] + m_vertices[vertexID2]) * 0.5f;

			edge.pCutEdge = new CutEdge<Vector3D>(m_edges.size(), m_vertices[vertexID1], m_vertices[vertexID2], m_dx, edgeLocation);

			m_edges.insert(m_edges.begin() + edgePosition, edge);
			m_visitedEdges.push_back(false);
			m_visitedEdgesCount.push_back(0);
		}

		void NonManifoldMesh2D::removeEdge(unsigned int edgeID) {
			m_edges.erase(m_edges.begin() + edgeID);
			m_visitedEdges.pop_back();
			m_visitedEdgesCount.pop_back();
		}

		void NonManifoldMesh2D::fixEdgesIDs() {
			for (int i = 0; i < m_edges.size(); i++) {
				m_edges[i].ID = i;
			}
		}
		unsigned int NonManifoldMesh2D::addVertex(const Vector3D &newVertex, unsigned int edgesOffset) {
			for (int i = edgesOffset; i < m_vertices.size(); i++) {
				if ((m_vertices[i] - newVertex).length() < m_distanceProximity) {
					return i;
				}
			}
			m_vertices.push_back(newVertex);
			return m_vertices.size() - 1;
		}

		int NonManifoldMesh2D::findVertex(const Vector3D &vertex) {
			for (int i = 0; i < m_vertices.size(); i++) {
				if ((m_vertices[i] - vertex).length() < m_distanceProximity) {
					return i;
				}
			}
			return -1;
		}
		#pragma endregion

		#pragma region comparisonFunctions
		bool NonManifoldMesh2D::compareEdges(pair<Scalar, unsigned int> a, pair<Scalar, unsigned int>b) {
			return a.first < b.first;
		}
		bool NonManifoldMesh2D::compareDisconnectedRegionsA(DisconnectedRegion a, DisconnectedRegion b) {
			return a.getCentroid().y < b.getCentroid().y;
		}
		bool NonManifoldMesh2D::compareDisconnectedRegionsB(DisconnectedRegion a, DisconnectedRegion b) {
			return a.getCentroid().x < b.getCentroid().x;
		}
		#pragma endregion
	}
}
