#include "Mesh/MeshUtils.h"
#include "ChimeraCGALWrapper.h"

namespace Chimera {
	namespace Data{

		namespace MeshUtils {

			#pragma region Deprecated Functions
			//TriangleAdjacencyGraph::TriangleAdjacencyGraph(const vector<TriangleMesh3D::triangle_t> &triangles, int numVertices) : m_visitedTriangles(triangles.size(), false),
			//	m_flippedTriangles(triangles.size(), false) {
			//	m_numVertices = numVertices;

			//	for (int i = 0; i < triangles.size(); i++) {
			//		triangleAdj_t triangle;
			//		triangle.ID = i;
			//		triangle.indices[0] = triangles[i].pointsIndexes[0];
			//		triangle.indices[1] = triangles[i].pointsIndexes[1];
			//		triangle.indices[2] = triangles[i].pointsIndexes[2];
			//		m_triangleAdjacencyVector.push_back(triangle);
			//	}

			//	//Initialize adjacency triangles
			//	findAdjacentTriangles(0);
			//}

			//void TriangleAdjacencyGraph::orientNormals(map<int, bool> &initialEdgeMap) {
			//	//Start in a node that has at least one sborder with the initial edge map
			//	int currNode = getBorderNode();
			//	int numVisitedNodes = 0;

			//	m_visitedTriangles.assign(m_visitedTriangles.size(), false);
			//	if (currNode != -1) {
			//		checkTriangleNormal(currNode, initialEdgeMap);
			//	}
			//	else {
			//		Logger::get() << "No border node found on TriangleAdjacencyGraph." << endl;
			//	}

			//	//Check if there's any unvisited node from a disconnected triangle mesh
			//	for (int i = 0; i < m_visitedTriangles.size(); i++) {
			//		if (!m_visitedTriangles[i]) {
			//			//Get border node starting for the first unvisited node
			//			currNode = getBorderNode(i);
			//			//Visit unvisited triangle
			//			checkTriangleNormal(currNode, initialEdgeMap);
			//		}
			//	}
			//}

			//void TriangleAdjacencyGraph::checkTriangleNormal(int currNode, map<int, bool> &initialEdgeMap) {
			//	triangleAdj_t &triangle = m_triangleAdjacencyVector[currNode];
			//	m_visitedTriangles[currNode] = true;

			//	//First pass is to verify if the triangle edges need to be flipped
			//	for (int i = 0; i < 3; i++) {
			//		int nextI = roundClamp<int>(i + 1, 0, 3);
			//		int currHash = hash(triangle.indices[i], triangle.indices[nextI], m_numVertices);
			//		int inverseHash = hash(triangle.indices[nextI], triangle.indices[i], m_numVertices);
			//		if (initialEdgeMap.find(currHash) != initialEdgeMap.end()) {
			//			if (!m_flippedTriangles[currNode]) {
			//				flipTriangleEdges(currNode);
			//				m_flippedTriangles[currNode] = true;
			//			}

			//			////Store the flipped edge on the initialEdgeMap
			//			//currHash = hash(triangle.indices[i], triangle.indices[nextI], m_numVertices);
			//			//if (initialEdgeMap.find(currHash) != initialEdgeMap.end()) {
			//			//	Logger::get() << "Error on TriangleAdjacencyGraph orientNormals." << endl;
			//			//}
			//			//else {
			//			//	initialEdgeMap[currHash] = true;
			//			//}
			//		}
			//		/*else {
			//			initialEdgeMap[currHash] = true;
			//			}*/
			//	}

			//	//Second pass is to store the triangles half-edges on the edge map
			//	for (int i = 0; i < 3; i++) {
			//		int nextI = roundClamp<int>(i + 1, 0, 3);
			//		int currHash = hash(triangle.indices[i], triangle.indices[nextI], m_numVertices);
			//		initialEdgeMap[currHash] = true;
			//	}

			//	if (triangle.neighbors[0] != NULL && !m_visitedTriangles[triangle.neighbors[0]->ID]) {
			//		checkTriangleNormal(triangle.neighbors[0]->ID, initialEdgeMap);
			//	}
			//	if (triangle.neighbors[1] != NULL && !m_visitedTriangles[triangle.neighbors[1]->ID]) {
			//		checkTriangleNormal(triangle.neighbors[1]->ID, initialEdgeMap);
			//	}
			//	if (triangle.neighbors[2] != NULL && !m_visitedTriangles[triangle.neighbors[2]->ID]) {
			//		checkTriangleNormal(triangle.neighbors[2]->ID, initialEdgeMap);
			//	}

			//}

			//void TriangleAdjacencyGraph::findAdjacentTriangles(int ithTriangle) {
			//	triangleAdj_t &triangleAdj = m_triangleAdjacencyVector[ithTriangle];
			//	m_visitedTriangles[triangleAdj.ID] = true;

			//	for (int i = 0; i < m_triangleAdjacencyVector.size(); i++) {
			//		if (i != triangleAdj.ID && !alreadyConnected(triangleAdj.ID, i)) {
			//			int numCommonVertices = 0;
			//			//Find the number of common vertices
			//			for (int j = 0; j < 3; j++) {
			//				if (m_triangleAdjacencyVector[i].indices[j] == triangleAdj.indices[0] ||
			//					m_triangleAdjacencyVector[i].indices[j] == triangleAdj.indices[1] ||
			//					m_triangleAdjacencyVector[i].indices[j] == triangleAdj.indices[2]) {
			//					numCommonVertices++;
			//				}

			//			}
			//			if (numCommonVertices > 1) {
			//				//Connect the two triangles and increase the node degree of each one
			//				m_triangleAdjacencyVector[i].neighbors[m_triangleAdjacencyVector[i].degree++] = &triangleAdj;
			//				triangleAdj.neighbors[triangleAdj.degree++] = &m_triangleAdjacencyVector[i];
			//				
			//				if (!m_visitedTriangles[i]) {
			//					findAdjacentTriangles(i);
			//				}
			//			}
			//		}
			//	}
			//}

			//TriangleMesh3D convertCutCell(CutVoxel &cutVoxel, CutCells3D *pCutCells) {
			//	if (cutVoxel.danglingVoxel) {
			//		//Returns empty triangle mesh
			//		return TriangleMesh3D(cutVoxel.m_vertices, pCutCells->getGridSpacing());
			//	}
			//	//return TriangleMesh(cutVoxel.m_vertices, pCutCells->getGridSpacing());
			//	/** Convert to an unified triangle mesh */
			//	vector<vector<pair<unsigned int, unsigned int>>> nonGeometryFaces;
			//	vector<int> originalIndexes;
			//	for (int i = 0; i < cutVoxel.cutFacesLocations.size(); i++) {
			//		if (cutVoxel.cutFacesLocations[i] != geometryFace) {
			//			nonGeometryFaces.push_back(cutVoxel.m_edgeIndices[i]);
			//			originalIndexes.push_back(i);
			//		}
			//	}

			//	/** Align nonGeometryFaces edges with mesh normals */
			//	//for (int i = 0; i < nonGeometryFaces.size(); i++) {
			//	//	//Find a valid normal inside the candidate vertices
			//	//	Vector3D planeNormal = cutVoxel.cutFaces[originalIndexes[i]]->m_normal;
			//	//	DoubleScalar planeDot = planeNormal.dot(pCutCells->getFaceNormal(cutVoxel.cutFacesLocations[originalIndexes[i]]));
			//	//	if (planeDot < 0) {
			//	//		if (cutVoxel.cutFaces[originalIndexes[i]]->m_discontinuousEdges) {
			//	//			//J finds the first discontinuity
			//	//			int lastDiscontinuousEdge = 0;
			//	//			//J finds the first discontinuity
			//	//			int nextDiscontinuousEdge = findDiscontinuity(nonGeometryFaces[i]);
			//	//			//We can have multiple discontinuous regions inside the same cell
			//	//			while (nextDiscontinuousEdge != -1) {
			//	//				reverse(nonGeometryFaces[i].begin() + lastDiscontinuousEdge, nonGeometryFaces[i].begin() + nextDiscontinuousEdge - 1);
			//	//				for (int j = lastDiscontinuousEdge; j < nextDiscontinuousEdge; j++) {
			//	//					swap(nonGeometryFaces[i][j].first, nonGeometryFaces[i][j].second);
			//	//				}
			//	//				lastDiscontinuousEdge = nextDiscontinuousEdge;
			//	//				nextDiscontinuousEdge = findDiscontinuity(nonGeometryFaces[i], nextDiscontinuousEdge);
			//	//			}
			//	//			reverse(nonGeometryFaces[i].begin() + lastDiscontinuousEdge, nonGeometryFaces[i].end());
			//	//			for (int j = lastDiscontinuousEdge; j < nonGeometryFaces[i].size(); j++) {
			//	//				swap(nonGeometryFaces[i][j].first, nonGeometryFaces[i][j].second);
			//	//			}
			//	//		}
			//	//		else {
			//	//			reverse(nonGeometryFaces[i].begin(), nonGeometryFaces[i].end());
			//	//			for (int j = 0; j < nonGeometryFaces[i].size(); j++) {
			//	//				swap(nonGeometryFaces[i][j].first, nonGeometryFaces[i][j].second);
			//	//			}
			//	//		}
			//	//	}	

			//	//}
			//	
			//	/** Convert CGAL polyhedron back to TriangleMesh class */
			//	CGALWrapper::CgalPolyhedron cgalPoly;
			//	CGALWrapper::convertToPoly(cutVoxel.m_vertices, nonGeometryFaces, &cgalPoly);
			//	cgalPoly.normalize_border();
			//	CGALWrapper::triangulatePolyhedron(&cgalPoly);

			//	vector<Vector3D> tempVertices;				
			//	vector<TriangleMesh3D::triangle_t> triangles;
			//	CGALWrapper::Conversion::polyToTriangleMesh(&cgalPoly, tempVertices, triangles);
			//	
			//	/** Add the remaining geometry faces to the triangles structure. 
			//		1 - Create an edge adjacency map with correct edges orientations */
			//	map<int, bool> edgeMap;
			//	vector<TriangleMesh3D::triangle_t> geometryTriangles;
			//	int numVertices = cutVoxel.m_vertices.size();
			//	for (int i = 0; i < triangles.size(); i++) {
			//		for (int j = 0; j < 3; j++) {
			//			int nextJ = roundClamp<int>(j + 1, 0, 3);
			//			int hash = TriangleAdjacencyGraph::hash(triangles[i].pointsIndexes[j], triangles[i].pointsIndexes[nextJ], numVertices);
			//			edgeMap[hash] = false;
			//		}
			//	}
			//	/** 2 - Starting geometry triangles data structure. Traversing edgeMap structure and marking as true edges that are shared
			//	*   between geometry and grid faces. */
			//	for (int i = 0; i < cutVoxel.cutFacesLocations.size(); i++) {
			//		if (cutVoxel.cutFacesLocations[i] == geometryFace) {
			//			TriangleMesh3D::triangle_t triangle;
			//			int numPointsOnPolyPointMap = 0;
			//			for (int j = 0; j < 3; j++) {
			//				triangle.pointsIndexes[j] = cutVoxel.m_edgeIndices[i][j].first;

			//				int nextJ = roundClamp<int>(j + 1, 0, 3);
			//				int hash = TriangleAdjacencyGraph::hash(cutVoxel.m_edgeIndices[i][j].first, cutVoxel.m_edgeIndices[i][j].second, numVertices);
			//				if (edgeMap.find(hash) != edgeMap.end()) {
			//					edgeMap[hash] = true;
			//					continue;
			//				}
			//				int inverseHash = TriangleAdjacencyGraph::hash(cutVoxel.m_edgeIndices[i][j].second, cutVoxel.m_edgeIndices[i][j].first, numVertices);
			//				if (edgeMap.find(inverseHash) != edgeMap.end()) {
			//					edgeMap[inverseHash] = true;
			//				}
			//			}

			//			Vector3D p1 = cutVoxel.m_vertices[triangle.pointsIndexes[0]];
			//			Vector3D p2 = cutVoxel.m_vertices[triangle.pointsIndexes[1]];
			//			Vector3D p3 = cutVoxel.m_vertices[triangle.pointsIndexes[2]];
			//			Vector3D planeOrigin = p1;
			//			triangle.normal = (p2 - p1).cross(p3 - p1);
			//			triangle.normal.normalize();
			//			triangle.centroid = (p1 + p2 + p3) / 3;
			//			triangle.pPolygonSurface = cutVoxel.cutFaces[i]->m_pPolygonSurface;
			//			triangle.cutFaceID = i;
			//			geometryTriangles.push_back(triangle);
			//		}
			//	}

			//	/**3- Deleting all unused edges from the edge map */
			//	for (map<int, bool>::iterator it = edgeMap.begin(); it != edgeMap.end();) {
			//		if (it->second == false) {
			//			it = edgeMap.erase(it);
			//		}
			//		else {
			//			it++;
			//		}
			//	}

			//	if (geometryTriangles.size() > 0) {
			//		/** 4 - Orient normals using TriangleAdjacencyGraph */
			//		TriangleAdjacencyGraph triangleAdjacencyGraph(geometryTriangles, tempVertices.size());
			//		triangleAdjacencyGraph.orientNormals(edgeMap);

			//		for (int i = 0; i < triangleAdjacencyGraph.getTriangles().size(); i++) {
			//			bool flipNormal = geometryTriangles[i].pointsIndexes[0] != triangleAdjacencyGraph.getTriangles()[i].indices[0];
			//			for (int j = 0; j < 3; j++) {
			//				geometryTriangles[i].pointsIndexes[j] = triangleAdjacencyGraph.getTriangles()[i].indices[j];
			//			}
			//			if (flipNormal) {
			//				geometryTriangles[i].normal = -geometryTriangles[i].normal;
			//			}
			//			CutFace<Vector3D> *pFace = cutVoxel.cutFaces[geometryTriangles[i].cutFaceID];
			//			pFace->m_normal = geometryTriangles[i].normal;
			//			cutVoxel.geometryFacesToMesh.push_back(triangles.size());
			//			triangles.push_back(geometryTriangles[i]);
			//		}
			//	}
			//	

			//	TriangleMesh3D triangleMesh(tempVertices, triangles, pCutCells->getGridSpacing(), cutVoxel);
			//	return triangleMesh;
			//}

			#pragma endregion
			

			bool comparePairs(pair<Vector3, unsigned int *> a, pair<Vector3, unsigned int *> b) {
				if (a.first != b.first) {
					return a.first < b.first;
				} else {
					return *a.second < *b.second;
				}
			}

			pair < vector<Vector3>, vector<vector<unsigned int>> > simplifyMesh(const vector<Vector3> &vertices, const vector<vector<unsigned int>> & faces) {
				pair < vector<Vector3>, vector<vector<unsigned int>> > simplifiedMesh;
				vector<vector<unsigned int>> localFaces(faces);
				vector<pair<Vector3, unsigned int *>> allPairs;
				for (int i = 0; i < faces.size(); i++) {
					for (int j = 0; j < faces[i].size(); j++) {
						unsigned int *pIndex = &localFaces[i][j];
						allPairs.push_back(pair<Vector3, unsigned int *>(vertices[*pIndex], pIndex));
					}
				}
				sort(allPairs.begin(), allPairs.end(), MeshUtils::comparePairs);

				for (int i = 0; i < allPairs.size(); i++) {
					int nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
					while (allPairs[i].first == allPairs[nextIndex].first) {
						if (*allPairs[nextIndex].second > *allPairs[i].second) {
							*allPairs[nextIndex].second = *allPairs[i].second;
						}
						else if (*allPairs[nextIndex].second < *allPairs[i].second){
							*allPairs[i].second = *allPairs[nextIndex].second;
						}
						i++;
						if (i >= allPairs.size())
							break;
						nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
					}
				}

				for (int i = 0; i < allPairs.size();) {
					int nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
					if (simplifiedMesh.first.size() > 0 && simplifiedMesh.first.back() != vertices[*allPairs[i].second]) {
						simplifiedMesh.first.push_back(vertices[*allPairs[i].second]);
					}
					else {
						simplifiedMesh.first.push_back(vertices[*allPairs[i].second]);
					}

					if (*allPairs[i].second == *allPairs[nextIndex].second) {
						while (allPairs[i].first == allPairs[nextIndex].first) {
							*allPairs[i].second = simplifiedMesh.first.size() - 1;
							i++;
							nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
						}
						*allPairs[i].second = simplifiedMesh.first.size() - 1;
						i++;
					}
					else {
						i++;
					}
				}

				for (int i = 0; i < localFaces.size(); i++) {
					simplifiedMesh.second.push_back(localFaces[i]);
				}

				//Removing points duplicates
				for (int i = 0; i < simplifiedMesh.second.size(); i++) {
					for (int j = 1; j < simplifiedMesh.second[i].size();) {
						if (simplifiedMesh.second[i][j] == simplifiedMesh.second[i][j - 1]) {
							simplifiedMesh.second[i].erase(simplifiedMesh.second[i].begin() + j);
						}
						else {
							j++;
						}
					}
				}

				Vector3 centroid;
				for (int i = 0; i < simplifiedMesh.first.size(); i++) {
					centroid += simplifiedMesh.first[i];
				}
				centroid /= simplifiedMesh.first.size();

				// Align face normals
				//map<int, bool> edgeMap;

				//for (int i = 0; i < simplifiedMesh.second.size(); i++) {
				//	for (int j = 0; j < simplifiedMesh.second[i].size(); j++) {
				//		int nextJ = roundClamp<int>(j + 1, 0, simplifiedMesh.second[i].size());
				//		int currPoint = simplifiedMesh.second[i][j], nextPoint = simplifiedMesh.second[i][nextJ];
				//		int hashKey = nextPoint*simplifiedMesh.first.size() + currPoint;
				//		if (edgeMap.find(hashKey) != edgeMap.end()) {

				//			//Removing all old entries of this face on the edgeMap
				//			for (int k = 0; k < j; k++) {
				//				int nextK = roundClamp<int>(k + 1, 0, simplifiedMesh.second[i].size());
				//				currPoint = simplifiedMesh.second[i][k], nextPoint = simplifiedMesh.second[i][nextK];
				//				hashKey = nextPoint*simplifiedMesh.first.size() + currPoint;
				//				auto edgeMapIt = edgeMap.find(hashKey);
				//				if (edgeMapIt != edgeMap.end()) {
				//					edgeMap.erase(edgeMapIt);
				//				}
				//			}
				//			//Flip all edges of current face current vertex
				//			reverse(simplifiedMesh.second[i].begin(), simplifiedMesh.second[i].end());

				//			//Adding back all the entries on the edgeMap
				//			for (int k = 0; k < simplifiedMesh.second[i].size(); k++) {
				//				int nextK = roundClamp<int>(k + 1, 0, simplifiedMesh.second[i].size());
				//				currPoint = simplifiedMesh.second[i][k], nextPoint = simplifiedMesh.second[i][nextK];
				//				hashKey = nextPoint*simplifiedMesh.first.size() + currPoint;
				//				if (edgeMap.find(hashKey) != edgeMap.end()) {
				//					cout << "chola " << endl;
				//				}
				//				edgeMap[hashKey] = true;
				//			}

				//			//Break from outer edge map
				//			break;
				//		}
				//		else {
				//			edgeMap[hashKey] = true;
				//		}
				//	} 
				//}

				return simplifiedMesh;
			}

			int findDiscontinuity(const vector<pair<unsigned int, unsigned int>> &edges, int initialIndex) {
				 for (int i = initialIndex; i < edges.size() - 1; i++) {
					 if (edges[i].second != edges[i + 1].first) {
						 return i + 1; //Returns the first edge after discontinuity 
					 }
				 }
				 return -1; //No discontinuity
			 }

			
			/*Vector2 calculateCentroid(const vector<Vector2> &points) {
				Scalar signedArea = 0;
				Vector2 centroid;
				for (int i = 0; i < points.size(); i++) {
					int nextI = roundClamp<int>(i + 1, 0, points.size());
					Scalar currA = points[i].cross(points[nextI]);
					signedArea += currA;
					centroid += ((points[i] + points[nextI])*0.5)*currA;
				}
				signedArea *= 0.5f;
				centroid /= (6.0f*signedArea);
				return centroid;
			}*/

		


		}
	}
}