#include "Mesh/TriangleMesh.h"
#include "Grids/CutCells3D.h"
#include "CGAL/PolygonSurface.h"

namespace Chimera {

	namespace Data {

		bool TriangleMesh::comparePairs(pair<Vector3D, int *> a, pair<Vector3D, int *> b) {
			return a.first < b.first;
		}

		bool TriangleMesh::comparePairs_b(pair<Vector3D, int *> a, pair<Vector3D, int *> b) {
			return *a.second < *b.second;
		}

		bool TriangleMesh::uniqueVectors(Vector3D a, Vector3D b) {
			return a == b;
		}

		TriangleMesh::TriangleMesh(const vector<Vector3D> &vertices, Scalar dx) : m_points(vertices) {
			//Calculating centroid and initialized nodes types
			m_centroid = Vector3D(0, 0, 0);

			for (int i = 0; i < m_points.size(); i++) {
				m_centroid += m_points[i];
				if (isOnGridPoint(m_points[i], dx)) {
					m_nodeTypes.push_back(gridNode);
				}
				else if (isOnGridEdge(m_points[i], dx) > 0) {
					m_nodeTypes.push_back(mixedNode);
				}
				else {
					m_nodeTypes.push_back(geometryNode);
				}
			}

			m_centroid /= m_points.size();


			m_gridSpacing = dx;
		}

		TriangleMesh::TriangleMesh(const CutVoxel &cutVoxel, CutCells3D *pCutCells) {
			vector<pair<Vector3D, int *>> allPairs;
			vector<trianglePointers_t> tempTriangles;
			vector<Vector3D> tempPoints;
			vector<nodeType_t> tempNodeTypes;
			Scalar dx = pCutCells->getGridSpacing();
			int initialIndex = 0;
			for(int i = 0; i < cutVoxel.cutFaces.size(); i++) {
				CutFace<Vector3D> *pFace = cutVoxel.cutFaces[i];
				Vector3D centroid = pFace->m_centroid;
				int cellPointsSize = pFace->m_cutEdges.size();
				bool rectangularFace = isRectangularFace(pFace);
				//rectangularFace = true;
				if (rectangularFace) {
					trianglePointers_t t1, t2;
						
					int * newIndex = new int;
					*newIndex = initialIndex + 0;
					t1.pointsIndexes[0] = newIndex;
					Vector3D currPoint = pFace->getEdgeInitialPoint(0);
					allPairs.push_back(pair<Vector3D, int *>(currPoint, newIndex));
					tempPoints.push_back(currPoint);
					if (isOnGridPoint(currPoint, dx)) {
						tempNodeTypes.push_back(gridNode);
					}
					else {
						tempNodeTypes.push_back(geometryNode);
					}

					newIndex = new int;
					*newIndex = initialIndex + 1;
					t1.pointsIndexes[1] = newIndex;
					currPoint = pFace->getEdgeFinalPoint(0);
					allPairs.push_back(pair<Vector3D, int *>(currPoint, newIndex));
					tempPoints.push_back(currPoint);
					if (isOnGridPoint(currPoint, dx)) {
						tempNodeTypes.push_back(gridNode);
					}
					else {
						tempNodeTypes.push_back(geometryNode);
					}
					
					newIndex = new int;
					*newIndex = initialIndex + 2;
					t1.pointsIndexes[2] = newIndex;
					currPoint = pFace->getEdgeFinalPoint(1);
					allPairs.push_back(pair<Vector3D, int *>(currPoint, newIndex));
					tempPoints.push_back(currPoint);
					if (isOnGridPoint(currPoint, dx)) {
						tempNodeTypes.push_back(gridNode);
					}
					else {
						tempNodeTypes.push_back(geometryNode);
					}
					
					tempTriangles.push_back(t1);

					//T2
					newIndex = new int;
					*newIndex = initialIndex + 2;
					t2.pointsIndexes[0] = newIndex;
					currPoint = pFace->getEdgeInitialPoint(2);
					allPairs.push_back(pair<Vector3D, int *>(currPoint, newIndex));
				
					newIndex = new int;
					*newIndex = initialIndex + 3;
					t2.pointsIndexes[1] = newIndex;
					currPoint = pFace->getEdgeFinalPoint(2);
					allPairs.push_back(pair<Vector3D, int *>(currPoint, newIndex));
					tempPoints.push_back(currPoint);
					if (isOnGridPoint(currPoint, dx)) {
						tempNodeTypes.push_back(gridNode);
					}
					else {
						tempNodeTypes.push_back(geometryNode);
					}
				
					newIndex = new int;
					*newIndex = initialIndex + 0;
					t2.pointsIndexes[2] = newIndex;
					currPoint = pFace->getEdgeFinalPoint(3);
					allPairs.push_back(pair<Vector3D, int *>(currPoint, newIndex));
					
					tempTriangles.push_back(t2);

					initialIndex += 4;
				}
				else {
					for (int j = 0; j < pFace->m_cutEdges.size(); j++) {
						Vector3D currPoint, nextPoint;
						int nextIndex = roundClamp<int>(j + 1, 0, cellPointsSize);

						currPoint = pFace->getEdgeInitialPoint(j);
						nextPoint = pFace->getEdgeFinalPoint(j);

						tempPoints.push_back(currPoint);
						if (isOnGridPoint(currPoint, dx)) {
							tempNodeTypes.push_back(gridNode);
						}
						else {
							tempNodeTypes.push_back(geometryNode);
						}

						trianglePointers_t currTriangle;

						int * newIndex = new int;
						*newIndex = initialIndex + j;
						allPairs.push_back(pair<Vector3D, int *>(currPoint, newIndex));
						currTriangle.pointsIndexes[0] = newIndex;

						newIndex = new int;
						*newIndex = initialIndex + nextIndex;
						allPairs.push_back(pair<Vector3D, int *>(nextPoint, newIndex));
						currTriangle.pointsIndexes[1] = newIndex;

						newIndex = new int;
						*newIndex = initialIndex + cellPointsSize;
						allPairs.push_back(pair<Vector3D, int *>(centroid, newIndex));
						currTriangle.pointsIndexes[2] = newIndex;

						tempTriangles.push_back(currTriangle);
					}
					tempPoints.push_back(centroid);
					tempNodeTypes.push_back(centroidNode);
					initialIndex += cellPointsSize + 1;
				}
			}

			sort(allPairs.begin(), allPairs.end(), TriangleMesh::comparePairs);
			sort(allPairs.begin(), allPairs.end(), TriangleMesh::comparePairs_b);

			map<int, bool> eraseList;
			for(int i = 0; i < allPairs.size(); i++) {
				int nextIndex = roundClamp<int>(i + 1, 0, allPairs.size()); 
				while((allPairs[i].first - allPairs[nextIndex].first).length() < 1e-5) {
					if(*allPairs[nextIndex].second > *allPairs[i].second) {
						eraseList[*allPairs[nextIndex].second] = true;
						*allPairs[nextIndex].second = *allPairs[i].second;
					} else if(*allPairs[nextIndex].second < *allPairs[i].second){
						eraseList[*allPairs[i].second] = true;
						*allPairs[i].second = *allPairs[nextIndex].second;
					}
					i++;
					if(i >= allPairs.size())
						break;
					nextIndex = roundClamp<int>(i + 1, 0, allPairs.size()); 
				}
			}

 			for(int i = 0; i < allPairs.size();) {
				int nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
				m_points.push_back(tempPoints[*allPairs[i].second]);
				m_nodeTypes.push_back(tempNodeTypes[*allPairs[i].second]);
				if (*allPairs[i].second == *allPairs[nextIndex].second) {
					while (*allPairs[i].second == *allPairs[nextIndex].second) {
						*allPairs[i].second = m_points.size() - 1;
						i++;
						nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
					}
					*allPairs[i].second = m_points.size() - 1;
					i++;
				} else {
					i++;
				}
			}

			for(int i = 0; i < tempTriangles.size(); i++) {
				triangle_t currTriangle;
				currTriangle.pointsIndexes[0] = *tempTriangles[i].pointsIndexes[0];
				currTriangle.pointsIndexes[1] = *tempTriangles[i].pointsIndexes[1];
				currTriangle.pointsIndexes[2] = *tempTriangles[i].pointsIndexes[2];
				Vector3D p1 = m_points[currTriangle.pointsIndexes[0]];
				Vector3D p2 = m_points[currTriangle.pointsIndexes[1]];
				Vector3D p3 = m_points[currTriangle.pointsIndexes[2]];
				Vector3D trianguleNormal = (p2 - p1).cross(p3 - p1);
				currTriangle.normal = trianguleNormal.normalized();
				currTriangle.centroid = (p1 + p2 + p3) / 3;
				m_triangles.push_back(currTriangle);
			}

			//Calculating centroid
			m_centroid = Vector3D(0, 0, 0);
			for(int i = 0; i < m_points.size(); i++) {
				m_centroid += m_points[i];
			}
			m_centroid /= m_points.size();

			m_gridSpacing = pCutCells->getGridSpacing();
			alignTriangleNormals();
		}

		TriangleMesh::TriangleMesh(const vector<Vector3D> &points, const vector<triangle_t> & triangles, Scalar dx, const CutVoxel &cutVoxel) {
			m_points = points;
			m_triangles = triangles;

			for (int i = 0; i < m_triangles.size(); i++) {
				if (isOnGridFace(m_triangles[i].centroid, dx)) {
					m_triangles[i].triangleType = gridFaceTriangle;
				}
				else {
					m_triangles[i].triangleType = geometryTriangle;
				}
			}

			//Calculating centroid and initialized nodes types
			m_centroid = Vector3D(0, 0, 0);
			
			//Removing unused points before classifying node types
			removeUnusedPoints();
			for(int i = 0; i < m_points.size(); i++) {
				m_centroid += m_points[i];
				if (m_nodeTypes.size() < m_points.size()) {
					if (isOnGridPoint(m_points[i], dx)) {
						m_nodeTypes.push_back(gridNode);
					}
					else if (isOnGridEdge(m_points[i], dx) > 0) {
						m_nodeTypes.push_back(mixedNode);
					}
					else {
						m_nodeTypes.push_back(geometryNode);
					}
				}
			}

			m_centroid /= m_points.size();


			m_gridSpacing = dx;
			initializeVertexNormals(cutVoxel);
			//alignTriangleNormals();
		}

		TriangleMesh::TriangleMesh(const vector<Vector3D> &points, const vector<triangle_t> & triangles, Scalar dx) {
			m_points = points;
			m_triangles = triangles;

			for (int i = 0; i < m_triangles.size(); i++) {
				if (isOnGridFace(m_triangles[i].centroid, dx)) {
					m_triangles[i].triangleType = gridFaceTriangle;
				}
				else {
					m_triangles[i].triangleType = geometryTriangle;
				}
			}

			//Calculating centroid and initialized nodes types
			m_centroid = Vector3D(0, 0, 0);

			//Removing unused points before classifying node types
			removeUnusedPoints();
			for (int i = 0; i < m_points.size(); i++) {
				m_centroid += m_points[i];
				if (m_nodeTypes.size() < m_points.size()) {
					if (isOnGridPoint(m_points[i], dx)) {
						m_nodeTypes.push_back(gridNode);
					}
					else if (isOnGridEdge(m_points[i], dx) > 0) {
						m_nodeTypes.push_back(mixedNode);
					}
					else {
						m_nodeTypes.push_back(geometryNode);
					}
				}
			}

			m_centroid /= m_points.size();


			m_gridSpacing = dx;
			//alignTriangleNormals();
		}


		void TriangleMesh::initializeVertexNormals(const CutVoxel &cutVoxel) {
			m_normals.resize(m_points.size());
			for (int i = 0; i < m_triangles.size(); i++) {
				if (m_triangles[i].cutFaceID != -1) { //This is a geometry triangle and has a cut-face ID attached to it
					CutFace<Vector3D> *pFace = cutVoxel.cutFaces[m_triangles[i].cutFaceID];
					int polygonFaceIndices[3];
					Vector3D v1, v2, v3;
					Vector3D a1, a2, a3;
					v1 = pFace->m_pPolygonSurface->getVertices()[pFace->m_originalPolyIndices[0]];
					v2 = pFace->m_pPolygonSurface->getVertices()[pFace->m_originalPolyIndices[1]];
					v3 = pFace->m_pPolygonSurface->getVertices()[pFace->m_originalPolyIndices[2]];

					a1 = m_points[m_triangles[i].pointsIndexes[0]];
					a2 = m_points[m_triangles[i].pointsIndexes[1]];
					a3 = m_points[m_triangles[i].pointsIndexes[2]];
					if ((m_points[m_triangles[i].pointsIndexes[0]] - pFace->m_pPolygonSurface->getVertices()[pFace->m_originalPolyIndices[0]]).length() < doublePrecisionThreshold) {
						polygonFaceIndices[0] = pFace->m_originalPolyIndices[0];
						polygonFaceIndices[2] = pFace->m_originalPolyIndices[2];
					}
					else { //swapped
						polygonFaceIndices[0] = pFace->m_originalPolyIndices[2];
						polygonFaceIndices[2] = pFace->m_originalPolyIndices[0];
					}
					polygonFaceIndices[1] = pFace->m_originalPolyIndices[1];
					m_normals[m_triangles[i].pointsIndexes[0]] = pFace->m_pPolygonSurface->getVerticesNormals()[polygonFaceIndices[0]];
					m_normals[m_triangles[i].pointsIndexes[1]] = pFace->m_pPolygonSurface->getVerticesNormals()[polygonFaceIndices[1]];
					m_normals[m_triangles[i].pointsIndexes[2]] = pFace->m_pPolygonSurface->getVerticesNormals()[polygonFaceIndices[2]];
				}
			}

		}

		void TriangleMesh::alignTriangleNormals() {
			//All normals face the outside of the Triangle Mesh, so the centroid must be on "bottom" of the plane
			for(int i = 0; i < m_triangles.size(); i++) {
				Vector3D p1 = m_points[m_triangles[i].pointsIndexes[0]];
				Vector3D p2 = m_points[m_triangles[i].pointsIndexes[1]];
				Vector3D p3 = m_points[m_triangles[i].pointsIndexes[2]];
				Vector3D planeOrigin = p1;
				Vector3D planeNormal = (p2 - p1).cross(p3 - p1);
				planeNormal.normalize();

				if(isTop(planeOrigin, planeNormal, m_centroid)) {
					swap(m_triangles[i].pointsIndexes[0], m_triangles[i].pointsIndexes[1]);
					m_triangles[i].normal = -m_triangles[i].normal;
				}
				p1 = m_points[m_triangles[i].pointsIndexes[0]];
				p2 = m_points[m_triangles[i].pointsIndexes[1]];
				p3 = m_points[m_triangles[i].pointsIndexes[2]];
				planeOrigin = p1;
				planeNormal = (p2 - p1).cross(p3 - p1);
				planeNormal.normalize();
				if(isTop(planeOrigin, planeNormal, m_centroid)) {
					cout << "Oops" << endl;
				}
			}

			/*for(int i = 0; i < m_triangles.size(); i++) {
				Vector3 p1 = m_points[m_triangles[i].pointsIndexes[0]];
				Vector3 p2 = m_points[m_triangles[i].pointsIndexes[1]];
				Vector3 p3 = m_points[m_triangles[i].pointsIndexes[2]];
				Vector3 planeOrigin = p1;
				Vector3 planeNormal = (p2 - p1).cross(p3 - p1);

				if(isTop(planeOrigin, planeNormal, m_centroid)) {
					swap(m_triangles[i].pointsIndexes[0], m_triangles[i].pointsIndexes[1]);
				}
			}*/
		}

		void TriangleMesh::initializeFaceLines(const CutVoxel &cutVoxel) {
			vector<int> currentLine;
			int numMixedNodes = 0;
			m_faceLinesIndices.clear();
			for (int i = 0; i < m_nodeTypes.size(); i++) {
				if (m_nodeTypes[i] == mixedNode) {
					numMixedNodes++;
					//m_tempVisitedNodes.clear();
					m_faceLinesIndices.push_back(vector<int>());
					//We still dont know for which face we are going to travel
					followNextPointOnFace(cutVoxel.getPointsToMeshMap(), i, -2);
					if (m_faceLinesIndices.back().size() == 1) {
						m_faceLinesIndices.pop_back();
					}
				}
			}
			if (m_faceLinesIndices.size() != numMixedNodes) {
				//Rerun the algorithm until we find the necessary number of face line indices
				for (int i = 0; i < m_nodeTypes.size(); i++) {
					if (m_nodeTypes[i] == mixedNode) {
						//m_tempVisitedNodes.clear();
						m_faceLinesIndices.push_back(vector<int>());
						//We still dont know for which face we are going to travel
						followNextPointOnFace(cutVoxel.getPointsToMeshMap(), i, -2);
						if (m_faceLinesIndices.back().size() == 1) {
							m_faceLinesIndices.pop_back();
						}
						if (numMixedNodes == m_faceLinesIndices.size()) {
							break;
						}
					}
				}
			}

			for (int i = 0; i < m_faceLinesIndices.size(); i++) {
				Scalar totalLineSize = 0;
				for (int j = 1; j < m_faceLinesIndices[i].size(); j++) {
					totalLineSize += (m_points[m_faceLinesIndices[i][j]] - m_points[m_faceLinesIndices[i][j - 1]]).length();
				}
				m_faceLinesTotalSizes.push_back(totalLineSize);
			}
		}

		void TriangleMesh::clearFaceLines() {
			m_faceLinesIndices.clear();
			m_faceLinesTotalSizes.clear();
		}
		void TriangleMesh::followNextPointOnFace(const map<int, vector<int>> & map, int currID, int currFaceLocationID) {
			if (m_tempVisitedNodes.find(currID) != m_tempVisitedNodes.end()) {
				//Node already visited
				return;
			}
			else {
				if (m_nodeTypes[currID] != mixedNode) {
					m_tempVisitedNodes[currID] = true;
				}
			}
			m_faceLinesIndices.back().push_back(currID);
			const vector<int> &geometryIndices = map.at(currID);
			for (int j = 0; j < geometryIndices.size(); j++) {
				int triangleID = geometryIndices[j];
				for (int k = 0; k < 3; k++) {
					int tempLocation = isOnGridFace(m_points[m_triangles[triangleID].pointsIndexes[k]], m_gridSpacing);
					if (currID != m_triangles[triangleID].pointsIndexes[k] && tempLocation > 0) {
						if (m_nodeTypes[m_triangles[triangleID].pointsIndexes[k]] == mixedNode) {
							if (m_faceLinesIndices.back().front() != m_triangles[triangleID].pointsIndexes[k]) {
								m_faceLinesIndices.back().push_back(m_triangles[triangleID].pointsIndexes[k]);
								return;
							}
							else {
								continue;
							}
						}

						if (currFaceLocationID > 0 && currFaceLocationID != tempLocation) {
							continue;
						}
						
						if (m_tempVisitedNodes.find(m_triangles[triangleID].pointsIndexes[k]) == m_tempVisitedNodes.end()) {
							//Node not visited
							followNextPointOnFace(map, m_triangles[triangleID].pointsIndexes[k], tempLocation);
							return;
						}
					}
				}

			}
		}

		void TriangleMesh::removeUnusedPoints() {
			map<int, bool> pointsMap;
			vector<int> unusedIndices;
			//First-pass: add all points on the pointsMap - those points not found
			//on the pointsMap are the ones that should be removed
			for (int i = 0; i < m_triangles.size(); i++) {
				for (int j = 0; j < 3; j++)
					pointsMap[m_triangles[i].pointsIndexes[j]] = true;
			}

			//Second-pass: add all indices that should be removed to a vector
			for (int i = 0; i < m_points.size(); i++) {
				if (pointsMap.find(i) == pointsMap.end()) {
					unusedIndices.push_back(i);
				}
			}

			//Third-pass: iterate through all indices and count the number of unused indices
			//below it. Subtract that number from the current index.
			for (int i = 0; i < m_triangles.size(); i++) {
				for (int j = 0; j < 3; j++) {
					int k = 0;
					int currIndex = m_triangles[i].pointsIndexes[j];
					for (; k < unusedIndices.size(); k++) {
						if (unusedIndices[k] > currIndex) {
							break;
						}
					}
					m_triangles[i].pointsIndexes[j] -= k;
				}
			}

			//Last-pass: now that all indices are correct, remove the points from the points vector
			//Here we use the fact that the unusedIndices is a ordered list of indices that will
			//be used to remove the unused points from the m_points vector
			for (int i = unusedIndices.size() - 1; i >= 0; i--) {
				m_points.erase(m_points.begin() + unusedIndices[i]);
			}
		}

		bool TriangleMesh::isInsideMesh(const Vector3D & position) {
			for(int i = 0; i < m_triangles.size(); i++) {
				Vector3D p1 = m_points[m_triangles[i].pointsIndexes[0]];
				Vector3D p2 = m_points[m_triangles[i].pointsIndexes[1]];
				Vector3D p3 = m_points[m_triangles[i].pointsIndexes[2]];

				if (distanceToTriangle(position, p1, p2, p3) < 1e-4)
					return true;

				Vector3D planeOrigin = p1;
				Vector3D planeNormal = (p2 - p1).cross(p3 - p1);
				planeNormal.normalize();

				Vector3D v1 = position - planeOrigin;
				//Testing which side of the plane the point is on
				Scalar dprod = planeNormal.dot(v1);
				
				if (dprod > 0)
					return false;
			}
			return true;
		}
		
		bool TriangleMesh::isInsideMesh2(const Vector3D & position) {
			return signedDistanceFunction(position) <= 0 + 1e-4;
		}

		bool TriangleMesh::isInsideMesh3(const Vector3D & position) {
			int numCrossingsX = 0, numCrossingsY = 0, numCrossingsZ = 0, numCrossingsR = 0;
			//int numCrossingsX1 = 0, numCrossingsX2 = 0;

			//Treat special cases first

			Vector3D rayRandom(rand() / ((double)RAND_MAX), rand() / ((double)RAND_MAX), rand() / ((double)RAND_MAX));
			rayRandom.normalize();
			
			DoubleScalar minDistanceToTriangle = FLT_MAX;
			for (int i = 0; i < m_triangles.size(); i++) {
				Vector3D trianglePoints[3];
				for (int j = 0; j < 3; j++) {
					trianglePoints[j] = m_points[m_triangles[i].pointsIndexes[j]];
				}

				DoubleScalar disToTriangle = distanceToTriangle(position, trianglePoints[0], trianglePoints[1], trianglePoints[2]);
				if (disToTriangle < minDistanceToTriangle)
					minDistanceToTriangle = disToTriangle;

				if (disToTriangle < 1e-4) {
					return true;
				}
				
				/*if (rayTriangleIntersect(position, Vector3D(1, 0, 0), trianglePoints, m_triangles[i].normal)) {
					numCrossingsX++;
				}*/

				if (rayTriangleIntersect(position, rayRandom, trianglePoints, m_triangles[i].normal)) {
					numCrossingsR++;
				}
				
				/*if (rayTriangleIntersect(position, Vector3D(0, 1, 0), trianglePoints, m_triangles[i].normal)) {
					numCrossingsY++;
				}
				if (rayTriangleIntersect(position, Vector3D(0, 0, 1), trianglePoints, m_triangles[i].normal)) {
					numCrossingsZ++;
				}
*/
				
			}
			if (numCrossingsX % 2 == 0 && numCrossingsY % 2 == 0 && numCrossingsZ % 2 == 0 && numCrossingsR % 2 == 0) {
				return false;
			}
			return true;
		}

		Scalar TriangleMesh::signedDistanceFunction(const Vector3D & point) {
			DoubleScalar minDistance = FLT_MAX;
			int ithFace = -1;
			for (int i = 0; i < m_triangles.size(); i++) {
				const triangle_t &currTri = m_triangles[i];
				Vector3D v1, v2, v3;
				v1 = m_points[currTri.pointsIndexes[0]];
				v2 = m_points[currTri.pointsIndexes[1]];
				v3 = m_points[currTri.pointsIndexes[2]];
				DoubleScalar tempDistance = distanceToTriangle(point, v1, v2, v3);
				if (tempDistance < minDistance) {
					minDistance = tempDistance;
					ithFace = i;
				}
			}
			Vector3D faceNormal = m_triangles[ithFace].normal;
			Vector3D faceCentroid = m_triangles[ithFace].centroid;
			if (faceNormal.dot(point - faceCentroid) < 0) {
				return -minDistance;
			}
			return minDistance;
		}

		bool TriangleMesh::isRectangularFace(const CutFace<Vector3D> *pCutFace) {
			if (pCutFace->m_cutEdges.size() != 4)
				return false;

			for (int i = 0; i < pCutFace->m_cutEdges.size(); i++) {
				Vector3D e1 = (pCutFace->getEdgeFinalPoint(i) - pCutFace->getEdgeInitialPoint(i)).normalized();
				int nextEdge = roundClamp<int>(i + 1, 0, pCutFace->m_cutEdges.size());
				Vector3D e2 = (pCutFace->getEdgeFinalPoint(nextEdge) - pCutFace->getEdgeInitialPoint(nextEdge)).normalized();
				if (e1.dot(e2) > 1e-5) {
					return false;
				}
			}
			return true;
		}
	}
}