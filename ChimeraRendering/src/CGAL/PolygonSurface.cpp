//#include "CGAL/PolygonSurface.h"
//#include "CGAL/Utils.h"
//#include "Resources/ResourceManager.h"
//#include "RenderingUtils.h"
//#include "Physics/PhysicsCore.h"
//
//
//
//namespace Chimera {
//	CGALWrapper::CgalPolyhedron* CGALWrapper::Border_is_constrained_edge_map::sm_ptr = NULL;
//	namespace Rendering {
//		PolygonSurface::PolygonSurface(int polygonID, const Vector3 &position, const Vector3 &scale, const string &surfaceFilename, Scalar dx) : Object3D(position, Vector3(0, 0, 0), Vector3(1, 1, 1)) {
//			m_ID = polygonID;
//			m_pIndicesVBO = m_pVertexVBO = NULL;
//			m_triangleMesh = false;
//			m_lightPosLoc = -1;
//			m_dx = dx;
//			CGALWrapper::IO::importOBJ(position, surfaceFilename, &m_initialCgalPoly, dx);
//			m_pInitialCgalPolyTree = new CGALWrapper::AABBTree(faces(m_initialCgalPoly).first, faces(m_initialCgalPoly).second, m_initialCgalPoly);
//			m_cgalPoly = m_initialCgalPoly;
//			m_pCgalPolyTree = NULL;
//			m_rotationAngle = 0;
//			//Updating initial rotation
//			if (abs(m_rotationFunction.initialRotation) > 0) {
//				translatePolygon(&m_cgalPoly, convertToVector3D(-m_position));
//				rotatePolygonZ(&m_cgalPoly, m_rotationFunction.initialRotation);
//				translatePolygon(&m_cgalPoly, convertToVector3D(m_position));
//			}
//			
//
//			m_color[0] = 0.7922f; m_color[1] = 0.05f; m_color[2] = 0.084f;
//
//			updateLocalDataStructures();
//
//
//			m_initialCentroid.x = m_initialCentroid.y = m_initialCentroid.z = 0;
//			for (int i = 0; i < m_vertices.size(); i++) {
//				m_initialCentroid += m_vertices[i];
//			}
//
//			m_initialCentroid /= m_vertices.size();
//
//			m_pWireframeShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/Wireframe.frag");
//			m_pPhongShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/PhongShading.frag");
//			m_pPhongWireframeShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/PhongShadingWireframe.frag");
//			m_lightPosLoc = glGetUniformLocation(m_pPhongShader->getProgramID(), "lightPos");
//			m_lightPosLocWire = glGetUniformLocation(m_pPhongWireframeShader->getProgramID(), "lightPos");
//
//			initializeFaceCentroids();
//			initializeFaceNormals();
//			initializeVerticesNormals();
//			initializeEdgesNormals();
//			initializeVBOs();
//			initializeVAOs();
//		}
//
//		PolygonSurface::PolygonSurface(int polygonID, const pair<vector<CGALWrapper::CgalPolyhedron::Point_3>, vector<vector<polyEdge>>> &polygonInfo) : Object3D(Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(1, 1, 1)){
//			m_ID = polygonID;
//			m_pIndicesVBO = m_pVertexVBO = NULL;
//			m_triangleMesh = false;
//
//			for (int i = 0; i < polygonInfo.first.size(); i++) {
//				m_vertices.push_back(CGALWrapper::Conversion::pointToVec<Vector3D>(polygonInfo.first[i]));
//				simpleFace_t simpleFace;
//				simpleFace.edges = polygonInfo.second[i];
//				m_faces.push_back(simpleFace);
//			}
//			m_totalNumberOfVertices = 0;
//			for (int i = 0; i < m_faces.size(); i++) {
//				for (int j = 0; j < m_faces[i].edges.size(); j++) {
//					m_totalNumberOfVertices++;
//				}
//			}
//
//			m_pWireframeShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/Wireframe.frag");
//			m_pPhongShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/PhongShading.frag");
//			m_pPhongWireframeShader = ResourceManager::getInstance()->loadGLSLShader("Shaders/3D/PhongShading.glsl", "Shaders/3D/PhongShadingWireframe.frag");
//			m_lightPosLoc = glGetUniformLocation(m_pPhongShader->getProgramID(), "lightPos");
//			m_lightPosLocWire = glGetUniformLocation(m_pPhongWireframeShader->getProgramID(), "lightPos");
//
//			initializeFaceCentroids();
//			initializeFaceNormals();
//			initializeVerticesNormals();
//			initializeEdgesNormals();
//			initializeVBOs();
//			initializeVAOs();
//		}
//
//		/************************************************************************/
//		/* Functionalities                                                      */
//		/************************************************************************/
//
//		void PolygonSurface::draw() {
//			glColor4f(m_color[0], m_color[1], m_color[2], m_color[3]);
//			glLineWidth(1.0f);
//			
//			//Drawing type
//			if (m_drawShaded && m_drawWireframe) {
//				glDisable(GL_BLEND);
//				glEnable(GL_DEPTH_TEST);
//				glDepthMask(GL_TRUE);
//				m_pPhongWireframeShader->applyShader();
//				glUniform3f(m_lightPosLocWire, m_cameraPosition.x, m_cameraPosition.y, m_cameraPosition.z);
//			}
//			else if (m_drawShaded) {
//				glDisable(GL_BLEND);
//				glEnable(GL_DEPTH_TEST);
//				glDepthMask(GL_TRUE);
//				m_pPhongShader->applyShader();
//				glUniform3f(m_lightPosLoc, m_cameraPosition.x, m_cameraPosition.y, m_cameraPosition.z);
//			}
//			else if (m_drawWireframe) {
//				glEnable(GL_BLEND);
//				glDisable (GL_DEPTH_TEST);
//				//glDepthMask(GL_FALSE);
//				m_pWireframeShader->applyShader();
//			}
//
//			if (m_drawShaded || m_drawWireframe) {
//				glEnable(GL_LIGHTING);
//				glEnable(GL_LIGHT0);
//
//				for (int i = 0; i < m_faces.size(); i++) {
//					glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//					glBegin(GL_POLYGON);
//					glColor4f(m_color[0], m_color[1], m_color[2], m_color[3]);
//					for (int j = 0; j < m_faces[i].edges.size(); j++) {
//						int vertexID = m_faces[i].edges[j].first;
//						glNormal3f(m_verticesNormals[vertexID].x, m_verticesNormals[vertexID].y, m_verticesNormals[vertexID].z);
//						glVertex3f(m_vertices[vertexID].x, m_vertices[vertexID].y, m_vertices[vertexID].z);
//					}
//					glEnd();
//				}
//
//				glDisable(GL_LIGHTING);
//				glDisable(GL_LIGHT0);
//			}
//			
//			if (m_drawShaded && m_drawWireframe) {
//				m_pPhongWireframeShader->removeShader();
//			}
//			else if (m_drawShaded) {
//				m_pPhongShader->removeShader();
//			}
//			else if (m_drawWireframe) {
//				m_pWireframeShader->removeShader();
//			}
//
//
//			//Drawing points
//			if (m_drawPoints) {
//				glPointSize(5.0f);
//				glColor3f(0.05, 0.5, 0.71);
//				for (int i = 0; i < m_faces.size(); i++) {
//
//					for (int j = 0; j < m_faces[i].edges.size(); j++) {
//						int vertexID = m_faces[i].edges[j].first;
//						glBegin(GL_POINTS);
//						glVertex3f(m_vertices[vertexID].x, m_vertices[vertexID].y, m_vertices[vertexID].z);
//						glEnd();
//					}
//
//				}
//			}
//
//
//			if (m_drawNormals) {
//				for (int i = 0; i < m_facesNormals.size(); i++) {
//					Vector3 iniPoint = convertToVector3F(m_facesCentroids[i]);
//					Vector3 finalPoint = convertToVector3F(m_facesCentroids[i] + m_facesNormals[i] * 0.05);
//					RenderingUtils::getInstance()->drawVector(iniPoint, finalPoint);
//				}
//			}
//
//			if (m_drawVertexNormals) {
//				for (int i = 0; i < m_verticesNormals.size(); i++) {
//					Vector3 iniPoint = convertToVector3F(m_vertices[i]);
//					Vector3 finalPoint = convertToVector3F(m_vertices[i] + m_verticesNormals[i] * 0.05);
//					RenderingUtils::getInstance()->drawVector(iniPoint, finalPoint);
//				}
//			}
//
//			if (m_drawEdgeNormals) {
//				for (int i = 0; i < m_faces.size(); i++) {
//					for (int j = 0; j < 3; j++) { //Assuming that each face is a triangular face
//						Vector3 iniPoint = convertToVector3F(m_vertices[m_faces[i].edges[j].first] + m_vertices[m_faces[i].edges[j].second]);
//						iniPoint *= 0.5;
//						int edgeHashVar = edgeHash(m_faces[i].edges[j].first, m_faces[i].edges[j].second);
//						Vector3 finalPoint = iniPoint + convertToVector3F(m_edgeNormals[edgeHashVar] * 0.05);
//						RenderingUtils::getInstance()->drawVector(iniPoint, finalPoint);
//					}
//				}
//			}
//
//
//			//Clean up
//			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//			glBindVertexArray(0);
//			glBindBuffer(GL_ARRAY_BUFFER, 0);
//			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
//			glDisableClientState(GL_INDEX_ARRAY);			
//		}
//
//		void PolygonSurface::treatVerticesOnGridPoints() {
//			bool hasVertexOnGridPoint = false;
//			DoubleScalar perturbationSize = 1e-4;
//			Vector3D perturbationVec = Vector3D(1, 0, 0)*perturbationSize;
//			/*do {
//				for (auto it = m_cgalPoly.vertices_begin(); it != m_cgalPoly.vertices_end(); it++) {
//					CGALWrapper::CgalPolyhedron::Vertex_handle vh(it);
//					Vector3D currVertex = CGALWrapper::Conversion::pointToVec<Vector3D>(it->point());
//					if (isOnGridPoint(currVertex, m_dx, 1e-4)) {
//						hasVertexOnGridPoint = true;
//					}
//				}
//				if (hasVertexOnGridPoint) {
//					Logger::getInstance()->get() << "PolygonSurface: Perturbing mesh on top of grid points" << endl;
//					translatePolygon(&m_cgalPoly, perturbationVec);
//				}
//			} while (hasVertexOnGridPoint);*/
//		}
//
//		void PolygonSurface::treatVerticesOnGridEdges() {
//			bool hasVertexOnGridEdge = false;
//			DoubleScalar perturbationSize = 1e-4;
//			Vector3D perturbationVec = Vector3D(1, 0, 0)*perturbationSize;
//			do {
//				for (auto it = m_cgalPoly.vertices_begin(); it != m_cgalPoly.vertices_end(); it++) {
//					CGALWrapper::CgalPolyhedron::Vertex_handle vh(it);
//					Vector3D currVertex = CGALWrapper::Conversion::pointToVec<Vector3D>(it->point());
//					if (isOnGridEdge(currVertex, m_dx)) {
//						hasVertexOnGridEdge = true;
//					}
//				}
//				if (hasVertexOnGridEdge) {
//					Logger::getInstance()->get() << "PolygonSurface: Perturbing mesh on top of grid points" << endl;
//					translatePolygon(&m_cgalPoly, perturbationVec);
//				}
//			} while (hasVertexOnGridEdge);
//		}
//
//		void PolygonSurface::simplifyMesh() {
//			
//			Logger::getInstance()->get() << "Starting mesh simplification" << endl;
//			// Contract the surface mesh as much as possible
//			//CGAL::Surface_mesh_simplification::Count_stop_predicate<CGALWrapper::CgalPolyhedron> stop(2500);
//			CGAL::Surface_mesh_simplification::Count_ratio_stop_predicate<CGALWrapper::CgalPolyhedron> stop(0.95);
//			CGALWrapper::Border_is_constrained_edge_map bem;
//			bem.sm_ptr = &m_cgalPoly;
//			// This the actual call to the simplification algorithm.
//			// The surface mesh and stop conditions are mandatory arguments.
//			// The index maps are needed because the vertices and edges
//			// of this surface mesh lack an "id()" field.
//			/*int r = CGAL::Surface_mesh_simplification::edge_collapse
//				(m_cgalPoly
//				, stop
//				, CGAL::parameters::vertex_index_map(get(CGAL::vertex_external_index, m_cgalPoly))
//				.edge_index_map(get(CGAL::halfedge_external_index, m_cgalPoly))
//				.edge_is_constrained_map(bem)
//				.get_placement(CGALWrapper::Placement(bem))
//				);*/
//			int r = 0;
//			/*int r = CGAL::Surface_mesh_simplification::edge_collapse
//			(surface_mesh
//				, stop
//				, CGAL::parameters::vertex_index_map(get(CGAL::vertex_external_index, m_cgalPoly))
//				.halfedge_index_map(get(CGAL::halfedge_external_index, m_cgalPoly))
//				.get_cost(edge_is_constrained_map(bem))
//				.get_placement(CGAL::Surface_mesh_simplification::Midpoint_placement<Surface_mesh>())
//			);*/
//			/*int r = CGAL::Surface_mesh_simplification::edge_collapse
//				(m_cgalPoly
//				, stop
//				, CGAL::vertex_index_map(get(CGAL::vertex_external_index, m_cgalPoly))
//				.halfedge_index_map(get(CGAL::halfedge_external_index, m_cgalPoly))
//				.get_placement(CGAL::Surface_mesh_simplification::Midpoint_placement<CGALWrapper::CgalPolyhedron>())
//				);*/
//			Logger::getInstance()->get() << "Mesh simplification done" << endl;
//			Logger::getInstance()->get() << "Number of removed edges: " << r << endl  
//										 << "Mesh remaining edges: " << (m_cgalPoly.size_of_halfedges() / 2)  << endl;
//		}
//
//		void PolygonSurface::updateInitialRotation() {
//			Scalar rotationAngle = m_rotationFunction.initialRotation;
//			translatePolygon(&m_initialCgalPoly, convertToVector3D(-m_position));
//			rotatePolygonZ(&m_initialCgalPoly, rotationAngle);
//			translatePolygon(&m_initialCgalPoly, convertToVector3D(m_position));
//			m_cgalPoly = m_initialCgalPoly;
//			treatVerticesOnGridEdges();
//			m_totalNumberOfVertices = CGALWrapper::Conversion::polyhedron3ToFaceAndVertex(&m_cgalPoly, m_vertices, m_faces);
//			initializeFaceCentroids();
//			initializeVerticesNormals();
//			initializeEdgesNormals();
//		}
//
//		void PolygonSurface::update(Scalar dt) {
//			Vector3 positionOffset;
//			DoubleScalar elapsedTime = PhysicsCore::getInstance()->getElapsedTime() + dt;
//			if (elapsedTime > m_velocityFunction.startingTime && elapsedTime < m_velocityFunction.endingTime) {
//				positionOffset = updateVelocityFunction(elapsedTime);
//			} 
//			
//			DoubleScalar rotationAngle = 0;
//			if (elapsedTime > m_rotationFunction.startingTime && elapsedTime < m_rotationFunction.endingTime) {
//				rotationAngle = updateRotationAngle(elapsedTime);
//			}
//			std::cout.precision(20);
//			//cout << positionOffset.x << " " << positionOffset.y << " " << positionOffset.z << endl;
//			//cout << rotationAngle << endl;
//			if (abs(rotationAngle) > 0 || positionOffset.length() > 0) {
//				m_cgalPoly = m_initialCgalPoly;
//				translatePolygon(&m_cgalPoly, convertToVector3D(-m_position));
//				rotatePolygonZ(&m_cgalPoly, rotationAngle);
//				translatePolygon(&m_cgalPoly, convertToVector3D(m_position));
//				translatePolygon(&m_cgalPoly, convertToVector3D(positionOffset));
//				treatVerticesOnGridEdges();
//				m_totalNumberOfVertices = CGALWrapper::Conversion::polyhedron3ToFaceAndVertex(&m_cgalPoly, m_vertices, m_faces);
//				initializeFaceCentroids();
//				initializeVerticesNormals();
//				initializeEdgesNormals();
//			}	
//		}
//		void PolygonSurface::updateLocalDataStructures() {
//			m_totalNumberOfVertices = CGALWrapper::Conversion::polyhedron3ToFaceAndVertex(&m_cgalPoly, m_vertices, m_faces);
//			initializeFaceCentroids();
//			initializeFaceNormals();
//			initializeVerticesNormals();
//			initializeEdgesNormals();
//			updateEdgesCount();
//
//			m_centroid.x = m_centroid.y = m_centroid.z = 0;
//			for (int i = 0; i < m_vertices.size(); i++) {
//				m_centroid += m_vertices[i];
//			}
//			m_centroid /= m_vertices.size();
//
//			//Rebuilding CGALs tree:
//			if (m_pCgalPolyTree != NULL)
//				delete m_pCgalPolyTree;
//
//			m_pCgalPolyTree = new CGALWrapper::AABBTree(faces(m_cgalPoly).first, faces(m_cgalPoly).second, m_cgalPoly);
//			
//			m_pCgalPolyTree->accelerate_distance_queries();
//		}
//		void PolygonSurface::reinitializeVBOBuffers() {
//			glDeleteBuffers(1, m_pVertexVBO);
//			glDeleteBuffers(1, m_pIndicesVBO);
//			initializeVBOs();
//		}
//
//		bool sortByPosition(pair<int, Vector3D> a, pair<int, Vector3D> b) {
//			return a.second < b.second;
//		}
//
//		bool sortByID(pair<int, Vector3D> a, pair<int, Vector3D> b) {
//			return a.first < b.first;
//		}
//
//		void PolygonSurface::fixDuplicatedVertices() {
//			int tempIndex = 0;
//			vector<Vector3D> tempVertices = m_vertices;
//			
//			/**Initializing vector of pairs */
//			vector<pair<int, Vector3D>> tempPairs;
//			for (int i = 0; i < m_vertices.size(); i++) {
//				tempPairs.push_back(pair<int, Vector3D>(i, m_vertices[i]));
//			}
//
//			/**Sorting vector to remove duplicates */
//			sort(tempPairs.begin(), tempPairs.end(), sortByPosition);
//			
//			/** Creating vertex map and removing duplicates */
//			map<int, int> vertexIDMap;
//			vertexIDMap[tempPairs.front().first] = tempPairs.front().first;
//			for (int i = 1; i < tempPairs.size();) {
//				bool erased = false;
//				Scalar tempLength = (tempPairs[i].second - tempPairs[i - 1].second).length();
//				while (tempLength < singlePrecisionThreshold) {
//					vertexIDMap[tempPairs[i].first] = tempPairs[i - 1].first;
//					tempPairs.erase(tempPairs.begin() + i);
//					tempLength = (tempPairs[i].second - tempPairs[i - 1].second).length();
//					erased = true;
//				} 
//				if(!erased) {
//					vertexIDMap[tempPairs[i].first] = tempPairs[i].first;
//					i++;
//				}
//			}
//
//			/** Sorting the vector back to its original positions */
//			sort(tempPairs.begin(), tempPairs.end(), sortByID);
//
//			/** Update faces edges first to get the remove indices right */
//			for (int i = 0; i < m_faces.size(); i++) {
//				for (int j = 0; j < m_faces[i].edges.size(); j++) {
//					m_faces[i].edges[j].first = vertexIDMap[m_faces[i].edges[j].first];
//					m_faces[i].edges[j].second = vertexIDMap[m_faces[i].edges[j].second];
//				}
//			}
//			m_oldVertexMap = vertexIDMap;
//
//			//Clear cos everythin is wrong, we gonna have to rebuild that shit
//			vertexIDMap.clear();
//
//			/** Packing tempPairs and updating vertexID map with the packed vertices */
//			int numRemovedVerticesDetected = 0;
//			for (int i = 0; i < tempPairs.size(); i++) {
//				if (tempPairs[i].first != i) { //we have to update the map
//					numRemovedVerticesDetected = tempPairs[i].first - i;
//					int newID = tempPairs[i].first - numRemovedVerticesDetected;
//					vertexIDMap[tempPairs[i].first] = newID;
//				}
//				else {
//					vertexIDMap[tempPairs[i].first] = i;
//				}
//			}
//
//			/** Update faces edges first to get the remove indices right */
//			for (int i = 0; i < m_faces.size(); i++) {
//				for (int j = 0; j < m_faces[i].edges.size(); j++) {
//					m_faces[i].edges[j].first = vertexIDMap[m_faces[i].edges[j].first];
//					m_faces[i].edges[j].second = vertexIDMap[m_faces[i].edges[j].second];
//				}
//			}
//
//			m_vertices.clear();
//			for (int i = 0; i < tempPairs.size(); i++) {
//				m_vertices.push_back(tempPairs[i].second);
//			}
//			for (auto it = m_oldVertexMap.begin(); it != m_oldVertexMap.end(); it++) {
//				it->second = vertexIDMap[it->second];
//			}
//
//			//m_oldVertexMap = vertexIDMap;
//			initializeFaceCentroids();
//			initializeFaceNormals();
//			updateEdgesCount();
//			initializeVerticesNormals();
//			initializeEdgesNormals();
//		}
//
//		bool PolygonSurface::doesEdgeIntersect(const Vector3 &e1, const Vector3 &e2) {
//			if (m_pCgalPolyTree != NULL) {
//				CGALWrapper::Kernel::Segment_3 cgalSegment(CGALWrapper::Conversion::vecToPoint3(e1), CGALWrapper::Conversion::vecToPoint3(e2));
//				auto anyIntersection = m_pCgalPolyTree->any_intersection(cgalSegment);
//				if (anyIntersection) {
//					return true;
//				}
//			}
//			return false;
//		}
//
//		bool PolygonSurface::doesEdgeIntersect(const Vector3 &e1, const Vector3 &e2, Vector3 &intersectionPoint) {
//			if (m_pCgalPolyTree != NULL) {
//				CGALWrapper::Kernel::Segment_3 cgalSegment(CGALWrapper::Conversion::vecToPoint3(e1), CGALWrapper::Conversion::vecToPoint3(e2));
//				auto anyIntersection = m_pCgalPolyTree->any_intersection(cgalSegment);
//				
//				if (anyIntersection) {
//					CGALWrapper::Kernel::Point_3 *point = get<CGALWrapper::Kernel::Point_3>(&(anyIntersection->first));
//					if (point) {
//						intersectionPoint = CGALWrapper::Conversion::pointToVec<Vector3>(*point);
//					}
//					else {
//						intersectionPoint = e1;
//					}
//
//					return true;
//				}
//			}
//			return false;
//		}
//
//		vector<unsigned int> PolygonSurface::getListOfPossibleTrianglesCollision(const Vector3 &initialPoint, const Vector3 &finalPoint, Scalar distanceThreshold) {
//			vector<unsigned int> possibleCollisions;
//			
//			for (int i = 0; i < m_faces.size(); i++) {
//				if ((initialPoint - convertToVector3F(m_faces[i].centroid)).length() < distanceThreshold) {
//					possibleCollisions.push_back(i);
//				} else if ((finalPoint - convertToVector3F(m_faces[i].centroid)).length() < distanceThreshold) {
//					possibleCollisions.push_back(i);
//				}
//			}
//
//			return possibleCollisions;
//		}
//
//		Vector3D PolygonSurface::getClosestPoint(const Vector3D &point) {
//			Vector3D closestPoint = CGALWrapper::Conversion::pointToVec<Vector3D>(m_pCgalPolyTree->closest_point(CGALWrapper::Conversion::vecToPoint3(point))); 
//			return closestPoint;
//		}
//
//		void PolygonSurface::getClosestPoint(const Vector3D &point, Vector3D &resultingPoint, Vector3D &resultingNormal) {
//			CGALWrapper::AABBTree::Point_and_primitive_id pointAndPrimitiveID = m_pCgalPolyTree->closest_point_and_primitive(CGALWrapper::Conversion::vecToPoint3(point));
//			CGALWrapper::AABBTree::Primitive_id primitiveId = pointAndPrimitiveID.second;
//			auto hfc = primitiveId->facet_begin();
//			
//			simpleFace_t currFace;
//			Vector3D trianglePoints[3];
//			
//			int pointsIDs[3];
//			for (int j = 0; j < 3; ++j, ++hfc) {
//				CGALWrapper::CgalPolyhedron::Vertex_handle vh(hfc->vertex());
//				pointsIDs[j] = m_oldVertexMap[vh->id];
//				trianglePoints[j] = m_vertices[pointsIDs[j]];
//			}
//
// 			resultingPoint = closesPointOnTriangle(point, trianglePoints[0], trianglePoints[1], trianglePoints[2]);
//
//   			Vector3D barycentricCoordinates = barycentricWeights(resultingPoint, trianglePoints[0], trianglePoints[1], trianglePoints[2]);
//			resultingNormal = primitiveId->normal;
//			if (abs(barycentricCoordinates.x) >= 1 - singlePrecisionThreshold) { //Its on top of first vertex
//				resultingNormal = m_verticesNormals[pointsIDs[0]];
//			}
//			else if (abs(barycentricCoordinates.y) >= 1 - singlePrecisionThreshold) { //Its on top of second vertex
//				resultingNormal = m_verticesNormals[pointsIDs[1]];
//			}
//			else if (abs(barycentricCoordinates.z) >= 1 - singlePrecisionThreshold) { //Its on top of third vertex
//				resultingNormal = m_verticesNormals[pointsIDs[2]];
//			}
//			else if (abs(barycentricCoordinates.x) < singlePrecisionThreshold) { //Its on top of first edge
//				resultingNormal = m_verticesNormals[pointsIDs[1]] * barycentricCoordinates.y + m_verticesNormals[pointsIDs[2]] * barycentricCoordinates.z;
//				//resultingNormal.normalize();
//				resultingNormal = getEdgeNormal(pointsIDs[1], pointsIDs[2]);
//			}
//			else if (abs(barycentricCoordinates.y) < singlePrecisionThreshold) { //On top of the second edge
//				resultingNormal = m_verticesNormals[pointsIDs[0]] * barycentricCoordinates.x + m_verticesNormals[pointsIDs[2]] * barycentricCoordinates.z;
//				//resultingNormal.normalize();
//				resultingNormal = getEdgeNormal(pointsIDs[2], pointsIDs[0]);
//			}
//			else if (abs(barycentricCoordinates.z) < singlePrecisionThreshold) { //On top of the third edge
//				resultingNormal = m_verticesNormals[pointsIDs[0]] * barycentricCoordinates.x + m_verticesNormals[pointsIDs[1]] * barycentricCoordinates.y;
//				//resultingNormal.normalize();
//				resultingNormal = getEdgeNormal(pointsIDs[0], pointsIDs[1]);
//			}
//		}
//
//		#pragma region PrivateFunctionalities
//		void PolygonSurface::updateEdgesCount() {
//			m_edgesRefCount.clear();
//			for (int i = 0; i < m_faces.size(); i++) {
//				for (int j = 0; j < m_faces[i].edges.size(); j++) {
//					int currEdgeHash = edgeHash(m_faces[i].edges[j].first, m_faces[i].edges[j].second);
//					if (m_edgesRefCount.find(currEdgeHash) != m_edgesRefCount.end()) {
//						m_edgesRefCount[currEdgeHash] += 1;
//					} else {
//						m_edgesRefCount[currEdgeHash] = 1;
//					}
//				}
//			}
//
//			int numEqualVertices = 0;
//			for (int i = 0; i < m_vertices.size(); i++) {
//				for (int j = 0; j < m_vertices.size(); j++) {
//					if (i != j) {
//						if ((m_vertices[i] - m_vertices[j]).length() < singlePrecisionThreshold) {
//							++numEqualVertices;
//						}
//					}
//				}
//			}
//			//Updating border polygons
//			for (int i = 0; i < m_faces.size(); i++) {
//				m_faces[i].borderFace = false;
//				for (int j = 0; j < m_faces[i].edges.size(); j++) {
//					int currEdgeHash = edgeHash(m_faces[i].edges[j].first, m_faces[i].edges[j].second);
//					if (m_edgesRefCount[currEdgeHash] == 1) {
//						m_faces[i].borderFace = true;
//						break;
//					}
//				}
//			}
//		}
//		#pragma endregion 
//
//		#pragma region InitializationFunctions
//		void PolygonSurface::initializeFaceCentroids() {
//			m_facesCentroids.clear();
//			for (int i = 0; i < m_faces.size(); i++) {
//				Vector3D faceCentroid;
//				for (int j = 0; j < m_faces[i].edges.size(); j++) {
//					faceCentroid += m_vertices[m_faces[i].edges[j].first];
//				}
//				faceCentroid /= m_faces[i].edges.size();
//				m_facesCentroids.push_back(faceCentroid);
//				m_faces[i].centroid = faceCentroid;
//			}
//		}
//		void PolygonSurface::initializeFaceNormals() {
//			m_facesNormals.clear();
//			for (CGALWrapper::CgalPolyhedron::Facet_iterator it = m_cgalPoly.facets_begin(); it != m_cgalPoly.facets_end(); ++it) {
//				CGALWrapper::CgalPolyhedron::Face_handle fit = it;
//				m_facesNormals.push_back(fit->normal);
//			}
//			for (int i = 0; i < m_faces.size(); i++) {
//				Vector3D e1 = m_vertices[m_faces[i].edges[1].first] - m_vertices[m_faces[i].edges[0].first];
//				Vector3D e2 = m_vertices[m_faces[i].edges[2].first] - m_vertices[m_faces[i].edges[1].first];
//				m_facesNormals[i] = e1.cross(e2).normalized();
//				m_faces[i].normal = m_facesNormals[i];
//			}
//			
//		}
//		void PolygonSurface::initializeVBOs() {
//			m_pVertexVBO = new GLuint();
//			glGenBuffers(1, m_pVertexVBO);
//			glBindBuffer(GL_ARRAY_BUFFER, *m_pVertexVBO);
//
//			void *pVertices = reinterpret_cast<void *>(&m_vertices[0]);
//			unsigned int sizeVertices = m_vertices.size()*sizeof(Vector3);
//
//			glBufferData(GL_ARRAY_BUFFER, sizeVertices, pVertices, GL_DYNAMIC_DRAW);
//			glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//			m_pIndicesVBO = new GLuint();
//			glGenBuffers(1, m_pIndicesVBO);
//			glBindBuffer(GL_ARRAY_BUFFER, *m_pIndicesVBO);
//
//			int *pIndices = new int[m_totalNumberOfVertices];
//			int indicesIndex = 0;
//			for (int i = 0; i < m_faces.size(); i++) {
//				for (int j = 0; j < m_faces[i].edges.size(); j++) {
//					pIndices[indicesIndex++] = m_faces[i].edges[j].first;
//				}
//			}
//			unsigned int sizeIndices = indicesIndex * sizeof(int);
//
//			glBufferData(GL_ARRAY_BUFFER, sizeIndices, pIndices, GL_DYNAMIC_DRAW);
//			glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//			delete[] pIndices;
//		}
//
//		void PolygonSurface::initializeVAOs() {
//			m_pPolygonVAO =  new GLuint();
//			glGenVertexArrays(1, m_pPolygonVAO);
//
//			glBindVertexArray(*m_pPolygonVAO);
//			glBindBuffer(GL_ARRAY_BUFFER, *m_pVertexVBO);
//			glVertexPointer(3, GL_FLOAT, 0, 0);
//			glEnableClientState(GL_VERTEX_ARRAY);
//			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pIndicesVBO);
//			glBindVertexArray(0);
//		}
//
//		void PolygonSurface::initializeVerticesNormals() {
//			m_verticesWeights.clear();
//			m_verticesNormals.clear();
//			m_verticesWeights.resize(m_vertices.size());
//			m_verticesNormals.resize(m_vertices.size());
//			for (int i = 0; i < m_faces.size(); i++) {
//				for (int j = 0; j < 3; j++) { //Assuming that each face is a triangular face
//					int nextJ = roundClamp<int>(j + 1, 0, 3);
//					int nextNextJ = roundClamp<int>(j + 2, 0, 3);
//
//					Vector3D e1 = m_vertices[m_faces[i].edges[nextJ].first] - m_vertices[m_faces[i].edges[j].first];
//					Vector3D e2 = m_vertices[m_faces[i].edges[nextNextJ].first] - m_vertices[m_faces[i].edges[j].first];
//					
//					DoubleScalar currAngle = angle3D(e1, e2);
//					m_verticesWeights[m_faces[i].edges[j].first] += currAngle;
//					m_verticesNormals[m_faces[i].edges[j].first] += m_facesNormals[i]*currAngle;
//				}
//			}
//			for (int i = 0; i < m_verticesWeights.size(); i++) {
//				m_verticesNormals[i] /= m_verticesWeights[i];
//				m_verticesNormals[i].normalize();
//			}
//		}
//
//		void PolygonSurface::initializeEdgesNormals() {
//			m_edgeNormals.clear();
//			m_edgeWeights.clear();
//			for (int i = 0; i < m_faces.size(); i++) {
//				for (int j = 0; j < 3; j++) { //Assuming that each face is a triangular face
//					int nextJ = roundClamp<int>(j + 1, 0, 3);
//					int nextNextJ = roundClamp<int>(j + 2, 0, 3);
//
//					Vector3D e1 = m_vertices[m_faces[i].edges[nextJ].first] - m_vertices[m_faces[i].edges[j].first];
//					Vector3D e2 = m_vertices[m_faces[i].edges[nextNextJ].first] - m_vertices[m_faces[i].edges[j].first];
//
//					int edgeHashVar = edgeHash(m_faces[i].edges[j].first, m_faces[i].edges[j].second);
//					DoubleScalar currAngle = angle3D(e1, e2);
//					m_edgeWeights[edgeHashVar] += currAngle;
//					m_edgeNormals[edgeHashVar] += m_facesNormals[i] * 0.5;
//				}
//			}
//			for (map<unsigned int, Vector3D>::iterator it = m_edgeNormals.begin(); it != m_edgeNormals.end(); it++) {
//				//it->second /= m_edgeWeights[it->first];
//				it->second.normalize();
//			}
//			
//		}
//
//		bool PolygonSurface::isInside(const Vector3D &point) {
//			for (int i = 0; i < m_faces.size(); i++) {
//				Vector3D p1 = m_vertices[m_faces[i].edges[0].first];
//				Vector3D p2 = m_vertices[m_faces[i].edges[1].first];
//				Vector3D p3 = m_vertices[m_faces[i].edges[2].first];
//
//				if (distanceToTriangle(point, p1, p2, p3) < 1e-5)
//					return true;
//
//				Vector3D planeOrigin = p1;
//				Vector3D planeNormal = (p2 - p1).cross(p3 - p1);
//				planeNormal.normalize();
//
//				Vector3D v1 = point - planeOrigin;
//				//Testing which side of the plane the point is on
//				Scalar dprod = planeNormal.dot(v1);
//
//				if (dprod > 0)
//					return false;
//			}
//			return true;
//		}
//		#pragma endregion
//	}
//}