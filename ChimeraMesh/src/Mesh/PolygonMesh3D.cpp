#include "Mesh/PolygonMesh3D.h"
#include "Mesh/TriangleMesh3D.h"
#include "Grids/CutCells3D.h"
#include "Mesh/MeshUtils.h"


namespace Chimera {

	namespace Meshes {

		PolygonMesh3D::PolygonMesh3D(const vector<Vector3D > &points, const vector<Vector3D> &normals, const vector<faceLocation_t> &faceLocations, vector<vector<pair<unsigned int, unsigned int>>> &polygons, Scalar dx) : m_points(points) {
			

			for (int i = 0; i < polygons.size(); i++) {
				m_polygons.push_back(polygon_t());
				m_polygons.back().edges = polygons[i];
				if (faceLocations[i] == geometryFace) {
					m_polygons.back().polygonType = geometryPolygon;
				}
				else {
					m_polygons.back().polygonType = gridFacePolygon;
				}
			}

			removedUnusedVertices();
			Vector3D tempCentroid;

			for (int i = 0; i < m_points.size(); i++) {
				tempCentroid += m_points[i];
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

			tempCentroid /= m_points.size();
			m_regularGridDimensions.x = floor(tempCentroid.x / dx);
			m_regularGridDimensions.y = floor(tempCentroid.y / dx);
			m_regularGridDimensions.z = floor(tempCentroid.z / dx);

			map<int, bool> edgeMap;
			for (int i = 0; i < polygons.size(); i++) {
				if (m_polygons[i].polygonType == gridFacePolygon) {
					Vector3D currNormal = getFaceNormal(i);
					if (currNormal.dot(normals[i]) > 0) {
						reverse(m_polygons[i].edges.begin(), m_polygons[i].edges.end());
						for (int j = 0; j < m_polygons[i].edges.size(); j++) {
							swap(m_polygons[i].edges[j].first, m_polygons[i].edges[j].second);
						}
						m_polygons[i].normal = -currNormal;
					}
					else {
						m_polygons[i].normal = currNormal;
					}

					m_polygons[i].normal.normalize();

					for (int j = 0; j < m_polygons[i].edges.size(); j++) {
						int currEdgeHash = edgeHash(m_polygons[i].edges[j].first, m_polygons[i].edges[j].second);
						if (m_edgesPolygonMap.find(currEdgeHash) != m_edgesPolygonMap.end()) {
							m_edgesPolygonMap[currEdgeHash].second = i;
						}
						else {
							m_edgesPolygonMap[currEdgeHash].first = i;
						}

						int hash = MeshUtils::TriangleAdjacencyGraph::hash(m_polygons[i].edges[j].first, m_polygons[i].edges[j].second, m_points.size());
						edgeMap[hash] = false;
					}
					
				}	

				for (int j = 0; j < m_polygons[i].edges.size(); j++) {
					m_polygons[i].centroid += m_points[m_polygons[i].edges[j].first];
				}
				m_polygons[i].centroid /= m_polygons[i].edges.size();
				if (m_polygons[i].polygonType == geometryPolygon) {
					m_polygons[i].normal = normals[i];
					m_polygons[i].normal.normalize();
				}
			}

			

			if (revertGeometryNormals()) {
				for (int i = 0; i < polygons.size(); i++) {
					if (m_polygons[i].polygonType == geometryPolygon) {
						reverse(m_polygons[i].edges.begin(), m_polygons[i].edges.end());
						for (int j = 0; j < m_polygons[i].edges.size(); j++) {
							swap(m_polygons[i].edges[j].first, m_polygons[i].edges[j].second);
						}
						m_polygons[i].normal = -m_polygons[i].normal;
					}					
				}
			}



			vector<vector<pair<unsigned int, unsigned int>>> allPolygonEdges;
			for (int i = 0; i < m_polygons.size(); i++) {
				allPolygonEdges.push_back(m_polygons[i].edges);
			}
			
			CGALWrapper::convertToPoly(m_points, allPolygonEdges, &m_cgalPoly);
			m_cgalPoly.normalize_border();
			CGALWrapper::triangulatePolyhedron(&m_cgalPoly);
			m_pAABBTree = new CGALWrapper::AABBTree(faces(m_cgalPoly).first, faces(m_cgalPoly).second, m_cgalPoly);
			//m_pAABBTree->accelerate_distance_queries();
			

			vector<Vector3D> tempVertices;
			vector<TriangleMesh3D::triangle_t> triangles;
			CGALWrapper::Conversion::polyToTriangleMesh(&m_cgalPoly, tempVertices, triangles);

			m_pTriangleMesh = new TriangleMesh3D(tempVertices, triangles, dx);

			initializeVerticesNormals();
			initializeEdgesNormals();
		}

		PolygonMesh3D::PolygonMesh3D(const vector<Vector3D > &points, const vector<polygon_t> &polygons, Scalar dx) : m_points(points), m_polygons(polygons) {
			Vector3D tempCentroid;
			for (int i = 0; i < m_points.size(); i++) {
				tempCentroid += m_points[i];
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
			tempCentroid /= m_points.size();

			m_regularGridDimensions.x = floor(tempCentroid.x / dx);
			m_regularGridDimensions.y = floor(tempCentroid.y / dx);
			m_regularGridDimensions.z = floor(tempCentroid.z / dx);

			vector<vector<pair<unsigned int, unsigned int>>> allPolygonEdges;
			for (int i = 0; i < m_polygons.size(); i++) {
				allPolygonEdges.push_back(m_polygons[i].edges);
				for (int j = 0; j < m_polygons[i].edges.size(); j++) {
					int currEdgeHash = edgeHash2(m_polygons[i].edges[j].first, m_polygons[i].edges[j].second);
					if (m_edgesPolygonCount[currEdgeHash] >= 1) {
						cout << "OH WOW" << endl;
 					}
					m_edgesPolygonCount[currEdgeHash] += 1;
				}
				
			}

			CGALWrapper::convertToPoly(m_points, allPolygonEdges, &m_cgalPoly);
			m_cgalPoly.normalize_border();
			CGALWrapper::triangulatePolyhedron(&m_cgalPoly);
			m_pAABBTree = new CGALWrapper::AABBTree(faces(m_cgalPoly).first, faces(m_cgalPoly).second, m_cgalPoly);
			m_pAABBTree->accelerate_distance_queries();

			vector<Vector3D> tempVertices;
			vector<TriangleMesh3D::triangle_t> triangles;
			CGALWrapper::Conversion::polyToTriangleMesh(&m_cgalPoly, tempVertices, triangles);

			m_pTriangleMesh = new TriangleMesh3D(tempVertices, triangles, dx);

			initializeVerticesNormals();
			initializeEdgesNormals();
		}

		bool PolygonMesh3D::revertGeometryNormals() {
			for (int k = 0; k < m_polygons.size(); k++) {
				if (m_polygons[k].polygonType != geometryPolygon) {
					continue;
				}
				for (int j = 0; j < m_polygons[k].edges.size(); j++) {
					unsigned int otherPolyIndex;
					int currEdgeHash = edgeHash(m_polygons[k].edges[j].first, m_polygons[k].edges[j].second);
					if (m_edgesPolygonMap[currEdgeHash].first == k) {
						otherPolyIndex = m_edgesPolygonMap[currEdgeHash].second;
					}
					else {
						otherPolyIndex = m_edgesPolygonMap[currEdgeHash].first;
					}
					if (m_polygons[otherPolyIndex].polygonType == gridFacePolygon) {
						for (int i = 0; i < m_polygons[otherPolyIndex].edges.size(); i++) {
							if (m_polygons[otherPolyIndex].edges[i].second == m_polygons[k].edges[j].first
								&& m_polygons[otherPolyIndex].edges[i].first == m_polygons[k].edges[j].second) {
								//Edges that are shared by a face have the reversed location
								//its the expected behavior
								return false;
							}
							else if (m_polygons[otherPolyIndex].edges[i].first == m_polygons[k].edges[j].first
								&& m_polygons[otherPolyIndex].edges[i].second == m_polygons[k].edges[j].second) {
								//Edges shared by face have the same orientation, reverse
								//geometry faces normals
								return true;
							}
						}
					}
				}
			}
			
			return true;
		}

		bool PolygonMesh3D::isInsideMesh(const Vector3D & position) {
			Vector3D direction(rand() / ((double)RAND_MAX), rand() / ((double)RAND_MAX), rand() / ((double)RAND_MAX));
			direction.normalize();
			CGALWrapper::Ray ray(CGALWrapper::Conversion::vecToPoint3(position), CGALWrapper::Conversion::vecToVec3(direction));

			int numIntersectedPrimitives = m_pAABBTree->number_of_intersected_primitives(ray);
			
			if (numIntersectedPrimitives % 2 == 0)
				return false;
			return true;
		}

		bool PolygonMesh3D::removedUnusedVertices() {
			map<unsigned int, bool> pointsMap;
			vector<unsigned int> unusedIndices;
			//First-pass: add all points on the pointsMap - those points not found
			//on the pointsMap are the ones that should be removed
			for (int i = 0; i < m_polygons.size(); i++) {
				for (int j = 0; j < m_polygons[i].edges.size(); j++)
					pointsMap[m_polygons[i].edges[j].first] = true;
			}

			//Second-pass: add all indices that should be removed to a vector
			for (int i = 0; i < m_points.size(); i++) {
				if (pointsMap.find(i) == pointsMap.end()) {
					unusedIndices.push_back(i);
				}
			}

			//Third-pass: iterate through all indices and count the number of unused indices
			//below it. Subtract that number from the current index.
			for (int i = 0; i < m_polygons.size(); i++) {
				for (int j = 0; j < m_polygons[i].edges.size(); j++) {
					int k = 0;
					int currIndex = m_polygons[i].edges[j].first;
					for (; k < unusedIndices.size(); k++) {
						if (unusedIndices[k] > currIndex) {
							break;
						}
					}
					m_polygons[i].edges[j].first -= k;

					k = 0;
					currIndex = m_polygons[i].edges[j].second;
					for (; k < unusedIndices.size(); k++) {
						if (unusedIndices[k] > currIndex) {
							break;
						}
					}
					m_polygons[i].edges[j].second -= k;

				}
			}
			
			//Last-pass: now that all indices are correct, remove the points from the points vector
			//Here we use the fact that the unusedIndices is a ordered list of indices that will
			//be used to remove the unused points from the m_points vector
			for (int i = unusedIndices.size() - 1; i >= 0; i--) {
				m_points.erase(m_points.begin() + unusedIndices[i]);
			}
			return true;
		}

		Vector3D PolygonMesh3D::getFaceNormal(unsigned int faceIndex) {
			DoubleScalar smallerDotProduct = FLT_MAX;
			int smallerIndex;
			for (int i = 0; i < m_polygons[faceIndex].edges.size(); i++) {
				int nextI = roundClamp<int>(i + 1, 0, m_polygons[faceIndex].edges.size());
				Vector3D v1 = m_points[m_polygons[faceIndex].edges[i].second] - m_points[m_polygons[faceIndex].edges[i].first];
				Vector3D v2 = m_points[m_polygons[faceIndex].edges[nextI].second] - m_points[m_polygons[faceIndex].edges[nextI].first];
				v1.normalize();
				v2.normalize();

				DoubleScalar currDotProduct = abs(v1.dot(v2));
				if (currDotProduct < smallerDotProduct) {
					smallerIndex = i;
				};
			}

			int nextI = roundClamp<int>(smallerIndex + 1, 0, m_polygons[faceIndex].edges.size());
			Vector3D v1 = m_points[m_polygons[faceIndex].edges[smallerIndex].second] - m_points[m_polygons[faceIndex].edges[smallerIndex].first];
			Vector3D v2 = m_points[m_polygons[faceIndex].edges[nextI].second] - m_points[m_polygons[faceIndex].edges[nextI].first];
			v1.normalize();
			v2.normalize();
			return v1.cross(v2);
		}

		void PolygonMesh3D::initializeVerticesNormals() {
			vector<DoubleScalar> verticesWeights(m_points.size(), 0);
			m_pointsNormals.clear();
			m_pointsNormals.resize(m_points.size());

			const vector<TriangleMesh3D::triangle_t> &triangles = m_pTriangleMesh->getTriangles();
			for (int i = 0; i < triangles.size(); i++) {
				for (int j = 0; j < 3; j++) { //Assuming that each face is a triangular face
					int nextJ = roundClamp<int>(j + 1, 0, 3);
					int nextNextJ = roundClamp<int>(j + 2, 0, 3);

					Vector3D e1 = m_points[triangles[i].pointsIndexes[nextJ]] - m_points[triangles[i].pointsIndexes[j]];
					Vector3D e2 = m_points[triangles[i].pointsIndexes[nextNextJ]]  - m_points[triangles[i].pointsIndexes[j]];

					DoubleScalar currAngle = angle3D(e1, e2);
					verticesWeights[triangles[i].pointsIndexes[j]] += currAngle;
					m_pointsNormals[triangles[i].pointsIndexes[j]] += triangles[i].normal * currAngle;
				}
			}
			for (int i = 0; i < verticesWeights.size(); i++) {
				m_pointsNormals[i] /= verticesWeights[i];
				m_pointsNormals[i].normalize();
			}
		}

		void PolygonMesh3D::initializeEdgesNormals() {
			m_edgeNormals.clear();
			const vector<TriangleMesh3D::triangle_t> &triangles = m_pTriangleMesh->getTriangles();
			for (int i = 0; i < triangles.size(); i++) {
				for (int j = 0; j < 3; j++) { //Assuming that each face is a triangular face
					int nextJ = roundClamp<int>(j + 1, 0, 3);
					int edgeHashVar = edgeHash(triangles[i].pointsIndexes[j], triangles[i].pointsIndexes[nextJ]);
					m_edgeNormals[edgeHashVar] += triangles[i].normal * 0.5;
				}
			}
			for (map<unsigned int, Vector3D>::iterator it = m_edgeNormals.begin(); it != m_edgeNormals.end(); it++) {
				//it->second /= m_edgeWeights[it->first];
				it->second.normalize();
			}

		}

	}
}