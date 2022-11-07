#include "Utils/MeshSlicer.h"
#include "Utils/ConversionManager.h"

namespace Chimera {

	namespace CGALWrapper {
		#pragma region Functionalities

		template<class VectorType>
		void MeshSlicer<VectorType>::sliceMesh() {
			for (CgalPolyhedron::Vertex_iterator iter = m_pCgalPoly->vertices_begin(); iter != m_pCgalPoly->vertices_end(); iter++) {
				for (CgalPolyhedron::Vertex_iterator iter2 = m_pCgalPoly->vertices_begin(); iter2 != m_pCgalPoly->vertices_end(); iter2++) {
					if (iter->id != iter2->id && iter->point() == iter2->point()) {
						cout << "Found duplicated point " << iter->point() - Kernel::Point_3(3, 3, 3) << endl;
					}
				}
			}

			sliceMesh(VectorType(0, 1, 0), VectorType(0, m_gridDx, 0), m_gridDx, m_gridDimensions.y - 1);
			
			for (CgalPolyhedron::Vertex_iterator iter = m_pCgalPoly->vertices_begin(); iter != m_pCgalPoly->vertices_end(); iter++) {
				for (CgalPolyhedron::Vertex_iterator iter2 = m_pCgalPoly->vertices_begin(); iter2 != m_pCgalPoly->vertices_end(); iter2++) {
					if (iter->id != iter2->id && iter->point() == iter2->point()) {
						cout << "Found duplicated point " << iter->point() - Kernel::Point_3(3, 3, 3) << endl;
					}
				}
			}

			sliceMesh(VectorType(1, 0, 0), VectorType(m_gridDx, 0, 0), m_gridDx, m_gridDimensions.x - 1);

			for (CgalPolyhedron::Vertex_iterator iter = m_pCgalPoly->vertices_begin(); iter != m_pCgalPoly->vertices_end(); iter++) {
				for (CgalPolyhedron::Vertex_iterator iter2 = m_pCgalPoly->vertices_begin(); iter2 != m_pCgalPoly->vertices_end(); iter2++) {
					if (iter->id != iter2->id && iter->point() == iter2->point()) {
						cout << "Found duplicated point " << iter->point() - Kernel::Point_3(3, 3, 3) << endl;
					}
				}
			}

			sliceMesh(VectorType(0, 0, 1), VectorType(0, 0, m_gridDx), m_gridDx, m_gridDimensions.z - 1);
			

			for (CgalPolyhedron::Vertex_iterator iter = m_pCgalPoly->vertices_begin(); iter != m_pCgalPoly->vertices_end(); iter++) {
				for (CgalPolyhedron::Vertex_iterator iter2 = m_pCgalPoly->vertices_begin(); iter2 != m_pCgalPoly->vertices_end(); iter2++) {
					if (iter->id != iter2->id && iter->point() == iter2->point()) {
						cout << "Found duplicated point " << iter->point() - Kernel::Point_3(3, 3, 3) << endl;
					}
				}
			}
		}

		template<class VectorType>
		void MeshSlicer<VectorType>::initializeLineMeshes(vector<Vertex<VectorType> *> vertices) {
			m_vertices = vertices;
			for (int i = 0; i < m_gridDimensions.z; i++) {
				initializePlaneLineMeshes(i, XYFace);
			}
			for (int i = 0; i < m_gridDimensions.y; i++) {
				initializePlaneLineMeshes(i, XZFace);
			}
			for (int i = 0; i < m_gridDimensions.x; i++) {
				initializePlaneLineMeshes(i, YZFace);
			}
		}
		#pragma endregion
		
		#pragma region PrivateFunctionalities
		template<class VectorType>
		void MeshSlicer<VectorType>::sliceMesh(const VectorType & normal, const VectorType & origin, Scalar increment, uint numSlices) {
			typedef std::vector<Kernel::Point_3> Polyline;
			typedef std::list< Polyline > Polylines;

			Polylines polylines;
			Kernel::Vector_3 planeNormal = Conversion::vecToVec3(normal);
			for (int i = 0; i < numSlices; i++) {
				Kernel::Point_3 planeOrigin = Conversion::vecToPoint3(origin + normal*increment*i);
				Kernel::Plane_3 intersectionPlane(planeOrigin, planeNormal);
				CGAL::Polygon_mesh_slicer_3<CgalPolyhedron, Kernel> slicer(*m_pCgalPoly);
				slicer(intersectionPlane, std::back_inserter(polylines));
			}	
		}

		template<class VectorType>
		void MeshSlicer<VectorType>::followThroughVerticesOnPlane(CgalPolyhedron::Vertex *pCurrVertex, CgalPolyhedron::Vertex *pInitialVertex, uint ithPlane, faceLocation_t planeType, vector<VectorType>& linePoints) {
			if (pCurrVertex != nullptr && pCurrVertex->id == pInitialVertex->id)
				return;
			
			if (pCurrVertex == nullptr)
				pCurrVertex = pInitialVertex;

			//Visit vertices only once
			m_visitedVertices[pCurrVertex->id] = true;
			linePoints.push_back(Conversion::pointToVec<VectorType>(pCurrVertex->point()));

			CgalPolyhedron::Halfedge_around_vertex_circulator vertexHalfEdges = pCurrVertex->vertex_begin();
			do 
			{
				auto currVertex = vertexHalfEdges->opposite()->vertex();
				switch (planeType) {
				case XYFace:
					if (currVertex->point().z() / m_gridDx - ithPlane == 0) {
						if (m_visitedVertices.find(currVertex->id) == m_visitedVertices.end()) {
							followThroughVerticesOnPlane(&(*currVertex), pInitialVertex, ithPlane, planeType, linePoints);
						}
					}
					break;
				case XZFace:
					if (currVertex->point().y() / m_gridDx - ithPlane == 0) {
						if (m_visitedVertices.find(currVertex->id) == m_visitedVertices.end()) {
							followThroughVerticesOnPlane(&(*currVertex), pInitialVertex, ithPlane, planeType, linePoints);
						}
					}
					break;
				case YZFace:
					if (currVertex->point().x() / m_gridDx - ithPlane == 0) {
						if (m_visitedVertices.find(currVertex->id) == m_visitedVertices.end()) {
							followThroughVerticesOnPlane(&(*currVertex), pInitialVertex, ithPlane, planeType, linePoints);
						}
					}
					break;
				}
				vertexHalfEdges++;
			} while (vertexHalfEdges != pCurrVertex->vertex_begin());
		}

		template<class VectorType>
		void MeshSlicer<VectorType>::followThroughVerticesOnPlane(Vertex<VectorType> *pCurrVertex, Vertex<VectorType> *pInitialVertex, uint ithPlane, faceLocation_t planeType, 
																	vector<Vertex<VectorType> *> &verticesVec, vector<Edge<VectorType> *> &edgesVec) {
			if (pCurrVertex != nullptr && pCurrVertex->getID() == pInitialVertex->getID()) {
				return;
			}

			if (pCurrVertex == nullptr)
				pCurrVertex = pInitialVertex;
			//else //Visit vertices only once, dont mark the first vertex, this is a trick to add the last edge
				
			m_visitedVertices[pCurrVertex->getID()] = true;

			
			verticesVec.push_back(pCurrVertex);
			
			const vector<Edge<VectorType> *> &edges = pCurrVertex->getConnectedEdges();
			for (int i = 0; i < edges.size(); i++) {
				Vertex<VectorType> *pNextVertex = pCurrVertex->getID() == edges[i]->getVertex1()->getID() ? 
													edges[i]->getVertex2() : edges[i]->getVertex1();

				switch (planeType) {
					case XYFace:
						if (pNextVertex->getPosition().z / m_gridDx - ithPlane == 0) {
							if (m_visitedVertices.find(pNextVertex->getID()) == m_visitedVertices.end()) {
								edgesVec.push_back(edges[i]);
								followThroughVerticesOnPlane(pNextVertex, pInitialVertex, ithPlane, planeType, verticesVec, edgesVec);
								return;
							}
							else if (pNextVertex->getID() == pInitialVertex->getID() && edgesVec.size() > 1) {
								edgesVec.push_back(edges[i]);
								return;
							}
						}			
					break;
					case XZFace:
						if (pNextVertex->getPosition().y / m_gridDx - ithPlane == 0) {
							if (m_visitedVertices.find(pNextVertex->getID()) == m_visitedVertices.end()) {
								edgesVec.push_back(edges[i]);
								followThroughVerticesOnPlane(pNextVertex, pInitialVertex, ithPlane, planeType, verticesVec, edgesVec);
								return;
							}
							else if (pNextVertex->getID() == pInitialVertex->getID() && edgesVec.size() > 1) {
								edgesVec.push_back(edges[i]);
								return;
							}
						}		
					break;
					case YZFace:
						if (pNextVertex->getPosition().x / m_gridDx - ithPlane == 0) {
							if (m_visitedVertices.find(pNextVertex->getID()) == m_visitedVertices.end()) {
								edgesVec.push_back(edges[i]);
								followThroughVerticesOnPlane(pNextVertex, pInitialVertex, ithPlane, planeType, verticesVec, edgesVec);
								return;
							}
							else if (pNextVertex->getID() == pInitialVertex->getID() && edgesVec.size() > 1) {
								edgesVec.push_back(edges[i]);
								return;
							}
						}
					break;
				}
			}
		}


		template <class VectorType>
		void MeshSlicer<VectorType>::initializePlaneLineMeshes(uint ithPlane, Meshes::faceLocation_t planeType) {
			vector<vector<Vertex<VectorType> *>> planeVertices;
			vector<vector<Edge<VectorType> *>> planeEdges;

			m_visitedVertices.clear();

			//Finding all lines of a plane and adding it to planeLines

			for (int i = 0; i < m_vertices.size(); i++) {
				if (planeType == XYFace && m_vertices[i]->getPosition().z / m_gridDx - ithPlane == 0 && m_visitedVertices.find(m_vertices[i]->getID()) == m_visitedVertices.end()) {
					planeVertices.push_back(vector<Vertex<VectorType> *>());
					planeEdges.push_back(vector<Edge<VectorType> *>());
					followThroughVerticesOnPlane(nullptr, m_vertices[i], ithPlane, planeType, planeVertices.back(), planeEdges.back());
				}
				else if (planeType == XZFace && m_vertices[i]->getPosition().y / m_gridDx - ithPlane == 0 && m_visitedVertices.find(m_vertices[i]->getID()) == m_visitedVertices.end()) {
					planeVertices.push_back(vector<Vertex<VectorType> *>());
					planeEdges.push_back(vector<Edge<VectorType> *>());
					followThroughVerticesOnPlane(nullptr, m_vertices[i], ithPlane, planeType, planeVertices.back(), planeEdges.back());
				}
				else if (planeType == YZFace && m_vertices[i]->getPosition().x / m_gridDx - ithPlane == 0 && m_visitedVertices.find(m_vertices[i]->getID()) == m_visitedVertices.end()) {
					planeVertices.push_back(vector<Vertex<VectorType> *>());
					planeEdges.push_back(vector<Edge<VectorType> *>());
					followThroughVerticesOnPlane(nullptr, m_vertices[i], ithPlane, planeType, planeVertices.back(), planeEdges.back());
				}
			}

			/*for (CgalPolyhedron::Vertex_iterator iter = m_pCgalPoly->vertices_begin(); iter != m_pCgalPoly->vertices_end(); iter++) {
				if (planeType == XYFace && iter->point().z() / m_gridDx - ithPlane == 0 && m_visitedVertices.find(iter->id) == m_visitedVertices.end()) {
					planeLines.push_back(vector<VectorType>());
					followThroughVerticesOnPlane(nullptr, &(*iter), ithPlane, planeType, planeLines.back());
				}
				else if (planeType == XZFace && iter->point().y() / m_gridDx - ithPlane == 0 && m_visitedVertices.find(iter->id) == m_visitedVertices.end()) {
					planeLines.push_back(vector<VectorType>());
					followThroughVerticesOnPlane(nullptr, &(*iter), ithPlane, planeType, planeLines.back());
				}
				else if (planeType == YZFace && iter->point().x() / m_gridDx - ithPlane == 0 && m_visitedVertices.find(iter->id) == m_visitedVertices.end()) {
					planeLines.push_back(vector<VectorType>());
					followThroughVerticesOnPlane(nullptr, &(*iter), ithPlane, planeType, planeLines.back());
				}
			}*/

			for (int i = 0; i < planeVertices.size(); i++) {
				if (planeType == XYFace) {
					m_XYLineMeshes[ithPlane].push_back(new LineMesh<VectorType>(planeVertices[i], planeEdges[i], planeType, m_gridDimensions, m_gridDx));
				} else if (planeType == XZFace) {
					m_XZLineMeshes[ithPlane].push_back(new LineMesh<VectorType>(planeVertices[i], planeEdges[i], planeType, m_gridDimensions, m_gridDx));
				} else if (planeType == YZFace) {
					m_YZLineMeshes[ithPlane].push_back(new LineMesh<VectorType>(planeVertices[i], planeEdges[i], planeType, m_gridDimensions, m_gridDx));
				}
			}
		}

		#pragma endregion

		template class MeshSlicer<Vector3>;
		template class MeshSlicer<Vector3D>;
	}
}
