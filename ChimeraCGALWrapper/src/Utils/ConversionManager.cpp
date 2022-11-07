#include "Utils/ConversionManager.h"
#include "Utils/Utils.h"

namespace Chimera {

	bool simpleFace_t::isConnectedTo(const simpleFace_t &other) {
		for (int i = 0; i < edges.size(); i++) {
			for (int j = 0; j < other.edges.size(); j++) {
				if (edges[i].second == other.edges[j].first && edges[i].first == other.edges[j].second) {
					return true;
				}
			}
		}
		return false;
	}

	namespace CGALWrapper {
		namespace Conversion {
			#pragma region GeneralConversions
			template <>
			int polyhedron3ToHalfFacesAndVertices(CgalPolyhedron *pPoly, vector<Vertex<Vector3> *> &vertices, vector<HalfFace<Vector3> *> &faces) {
				int totalNumberOfVertices = 0;
				/** Clean-up structures */
				vertices.clear();
				faces.clear();
				
				int tempIndex = 0;
				for (auto it = pPoly->vertices_begin(); it != pPoly->vertices_end(); it++) {
					CgalPolyhedron::Vertex_handle vh(it);
					vh->id = tempIndex++;
					vertices.push_back(new Vertex<Vector3>(pointToVec<Vector3>(it->point()), geometryVertex));
				}

				for (CgalPolyhedron::Facet_iterator it = pPoly->facets_begin(); it != pPoly->facets_end(); ++it) {
					auto hfc = it->facet_begin();
					vector<unsigned int> faceIndices;
					Vector3D faceNormal = it->normal;
					//Empty face container
					Face<Vector3> *pNewFace = new Face<Vector3>();
					for (int j = 0; j < it->size(); ++j, ++hfc) {
						CgalPolyhedron::Vertex_handle vh(hfc->vertex());
						faceIndices.push_back(vh->id);

						if (hfc->vertex()->halfedge()->is_border_edge()) {
							//currFace.borderFace = true;
						}

						//Adding face connection to the vertex 
						vertices[vh->id]->addConnectedFace(pNewFace);
						totalNumberOfVertices++;
					}
					faces.push_back(new HalfFace<Vector3>(vertices, faceIndices, pNewFace));
					pNewFace->addHalfFace(faces.back());
					//faces.back()->setNormal(convertToVector3F(faceNormal));
				}
				return totalNumberOfVertices;
			}

			template <>
			int polyhedron3ToHalfFacesAndVertices(CgalPolyhedron *pPoly, vector<Vertex<Vector3D> *> &vertices, vector<HalfFace<Vector3D> *> &faces) {
				int totalNumberOfVertices = 0;

				/** Clean-up structures */
				vertices.clear();
				faces.clear();

				int tempIndex = 0;
				for (auto it = pPoly->vertices_begin(); it != pPoly->vertices_end(); it++) {
					CgalPolyhedron::Vertex_handle vh(it);
					vh->id = tempIndex++;
					vertices.push_back(new Vertex<Vector3D>(pointToVec<Vector3D>(it->point()), geometryVertex));
				}

				for (CgalPolyhedron::Facet_iterator it = pPoly->facets_begin(); it != pPoly->facets_end(); ++it) {
					auto hfc = it->facet_begin();
					vector<unsigned int> faceIndices;
					Vector3D faceNormal = it->normal;
					Face<Vector3D> *pNewFace = new Face<Vector3D>();
					for (int j = 0; j < it->size(); ++j, ++hfc) {
						CgalPolyhedron::Vertex_handle vh(hfc->vertex());
						faceIndices.push_back(vh->id);
						
						if (hfc->vertex()->halfedge()->is_border_edge()) {
							//currFace.borderFace = true;
						}

						//Adding face connection to the vertex 
						vertices[vh->id]->addConnectedFace(pNewFace);
						totalNumberOfVertices++;
					}
					
					faces.push_back(new HalfFace<Vector3D>(vertices, faceIndices, pNewFace));
					pNewFace->addHalfFace(faces.back());
				}
				return totalNumberOfVertices;
			}


			template <>
			int polyhedron3ToFacesAndVertices(CgalPolyhedron *pPoly, vector<Vertex<Vector3> *> &vertices, vector<Face<Vector3> *> &faces) {
				int totalNumberOfVertices = 0;
				/** Clean-up structures */
				vertices.clear();
				faces.clear();

				map<uint, Edge<Vector3> *> edgeMap;

				int tempIndex = 0;
				/** Create all vertices first */
				for (auto it = pPoly->vertices_begin(); it != pPoly->vertices_end(); it++) {
					CgalPolyhedron::Vertex_handle vh(it);
					vh->id = tempIndex++;
					vertices.push_back(new Vertex<Vector3>(pointToVec<Vector3>(it->point()), geometryVertex, &(*it)));
				}

				//Filler dimensions, geometric faces need no grid dimensions
				dimensions_t zeroDimensions;

				/** Create all edges, halfEdges and faces at the same time*/
				for (CgalPolyhedron::Facet_iterator it = pPoly->facets_begin(); it != pPoly->facets_end(); ++it) {
					auto hfc = it->facet_begin();
					vector<Edge<Vector3> *> edges;
					vector<HalfEdge<Vector3>*> halfedges;
					Vector3D faceNormal = it->normal;

					//Creating edges and half-edges for the current face 
					for (int j = 0; j < it->size(); ++j, ++hfc) {
						/** Setting up vertices temporaries */
						CgalPolyhedron::Vertex_handle vh(hfc->vertex());
						auto nextHfc = hfc;
						nextHfc++;
						CgalPolyhedron::Vertex_handle nextVh(nextHfc->vertex());
						Vertex<Vector3> *pVertex = vertices[vh->id], *pNextVertex = vertices[nextVh->id];

						/** Setting up edges temporaries*/
						uint mapHash = edgeHash(vh->id, nextVh->id, vertices.size());
						Edge<Vector3> *pEdge = nullptr;
						
						if (edgeMap.find(mapHash) == edgeMap.end()) {
							edgeMap[mapHash] = new Edge<Vector3>(pVertex, pNextVertex, geometricEdge);
						}
						else {
							Scalar faith = 0;
							faith += 2;
						}
						edges.push_back(edgeMap[mapHash]);
						const pair<HalfEdge<Vector3> *, HalfEdge<Vector3> *> &halfEdgesPair = edgeMap[mapHash]->getHalfEdges();
						if (halfEdgesPair.first->getVertices().first->getID() == pVertex->getID()) { //First halfEdge is aligned with curr halfedge
							halfedges.push_back(halfEdgesPair.first);
						} else {
							halfedges.push_back(halfEdgesPair.second);
						}
					}

					Face<Vector3> *pFace = new Face<Vector3>(edges, zeroDimensions, 0, geometricFace);
					pFace->addHalfFace(new HalfFace<Vector3>(halfedges, pFace, geometryHalfFace));
					//pFace->addHalfFace(pFace->getHalfFaces().front()->reversedCopy(false));
					//pFace->setCentroid(pFace->getHalfFaces().front()->getCentroid());
					faces.push_back(pFace);
					
					/** Add face connections to vertices*/
					for (int i = 0; i < edges.size(); i++) {
						edges[i]->getVertex1()->addConnectedFace(pFace);
						edges[i]->getVertex2()->addConnectedFace(pFace);
					}

					
				}
				
				return totalNumberOfVertices;
			}


			template <>
			int polyhedron3ToFacesAndVertices(CgalPolyhedron *pPoly, vector<Vertex<Vector3D> *> &vertices, vector<Face<Vector3D> *> &faces) {
				int totalNumberOfVertices = 0;
				/** Clean-up structures */
				vertices.clear();
				faces.clear();

				map<uint, Edge<Vector3D> *> edgeMap;

				int tempIndex = 0;
				/** Create all vertices first */
				for (auto it = pPoly->vertices_begin(); it != pPoly->vertices_end(); it++) {
					CgalPolyhedron::Vertex_handle vh(it);
					vh->id = tempIndex++;
					vertices.push_back(new Vertex<Vector3D>(pointToVec<Vector3D>(it->point()), geometryVertex, &(*it)));
				}

				//Filler dimensions, geometric faces need no grid dimensions
				dimensions_t zeroDimensions;

				/** Create all edges, halfEdges and faces at the same time*/
				for (CgalPolyhedron::Facet_iterator it = pPoly->facets_begin(); it != pPoly->facets_end(); ++it) {
					auto hfc = it->facet_begin();
					vector<Edge<Vector3D> *> edges;
					vector<HalfEdge<Vector3D>*> halfedges;
					Vector3D faceNormal = it->normal;

					for (int j = 0; j < it->size(); ++j, ++hfc) {
						/** Setting up vertices temporaries */
						CgalPolyhedron::Vertex_handle vh(hfc->vertex());
						auto nextHfc = hfc;
						nextHfc++;
						CgalPolyhedron::Vertex_handle nextVh(hfc->vertex());
						Vertex<Vector3D> *pVertex = vertices[vh->id], *pNextVertex = vertices[nextVh->id];

						/** Setting up edges*/
						uint mapHash = edgeHash(vh->id, nextVh->id, vertices.size());
						Edge<Vector3D> *pEdge = nullptr;

						if (edgeMap.find(mapHash) == edgeMap.end()) {
							edgeMap[mapHash] = new Edge<Vector3D>(pVertex, pNextVertex, geometricEdge);
						}
						edges.push_back(edgeMap[mapHash]);
						const pair<HalfEdge<Vector3D> *, HalfEdge<Vector3D> *> &halfEdgesPair = edgeMap[mapHash]->getHalfEdges();
						if (halfEdgesPair.first->getVertices().first->getID() == pVertex->getID()) { //First halfEdge is aligned with curr halfedge
							halfedges.push_back(halfEdgesPair.first);
						}
						else {
							halfedges.push_back(halfEdgesPair.second);
						}
						Face<Vector3D> *pFace = new Face<Vector3D>(edges, zeroDimensions, 0, geometricFace);
						pFace->addHalfFace(new HalfFace<Vector3D>(halfedges, pFace, geometryHalfFace));
						pFace->addHalfFace(pFace->getHalfFaces().front()->reversedCopy());
						faces.push_back(pFace);
					}
				}

				return totalNumberOfVertices;
			}

			

			template <>
			int polyhedron3ToHalfFaces(CgalPolyhedron *pPoly, const map<uint, Vertex<Vector3> *> &verticesMap, vector<HalfFace<Vector3> *> &halffaces) {
				int totalNumberOfVertices = 0;

				/** Clean-up structures */
				halffaces.clear();

				map<uint, Edge<Vector3> *> edgeMap;

				int tempIndex = 0;
				/** Check if all vertices are around first */
				for (auto it = pPoly->vertices_begin(); it != pPoly->vertices_end(); it++) {
					CgalPolyhedron::Vertex_handle vh(it);
					if (verticesMap.find(vh->id) == verticesMap.end()) {
						throw(exception("polyhedron3ToHalfFaces: vertex not found on verticesMap"));
					}
				}

				//Filler dimensions, geometric faces need no grid dimensions
				dimensions_t zeroDimensions;

				/** Create all edges, halfEdges and faces at the same time*/
				for (CgalPolyhedron::Facet_iterator it = pPoly->facets_begin(); it != pPoly->facets_end(); ++it) {
					auto hfc = it->facet_begin();
					vector<Edge<Vector3> *> edges;
					vector<HalfEdge<Vector3>*> halfedges;
					Vector3D faceNormal = it->normal;

					//Creating edges and half-edges for the current face 
					for (int j = 0; j < it->size(); ++j, ++hfc) {
						/** Setting up vertices temporaries */
						CgalPolyhedron::Vertex_handle vh(hfc->vertex());
						auto nextHfc = hfc;
						nextHfc++;
						CgalPolyhedron::Vertex_handle nextVh(nextHfc->vertex());
						auto iter = verticesMap.find(vh->id);
						auto nextIter = verticesMap.find(nextVh->id);
						if (iter == verticesMap.end() || nextIter == verticesMap.end()) 
							throw(exception("polyhedron3ToHalfFaces: vertex not found on verticesMap"));

						Vertex<Vector3> *pVertex = iter->second, *pNextVertex = nextIter->second;

						/** Setting up edges temporaries*/
						uint mapHash = edgeHash(vh->id, nextVh->id, verticesMap.size());
						Edge<Vector3> *pEdge = nullptr;

						if (edgeMap.find(mapHash) == edgeMap.end()) {
							edgeMap[mapHash] = new Edge<Vector3>(pVertex, pNextVertex, geometricEdge);
						}
						edges.push_back(edgeMap[mapHash]);
						const pair<HalfEdge<Vector3> *, HalfEdge<Vector3> *> &halfEdgesPair = edgeMap[mapHash]->getHalfEdges();
						if (halfEdgesPair.first->getVertices().first->getID() == pVertex->getID()) { //First halfEdge is aligned with curr halfedge
							halfedges.push_back(halfEdgesPair.first);
						}
						else {
							halfedges.push_back(halfEdgesPair.second);
						}
					}

					HalfFace<Vector3> *pHalfFace = new HalfFace<Vector3>(halfedges, nullptr, geometryHalfFace);
					//pFace->addHalfFace(new HalfFace<Vector3>(halfedges, pFace, geometryHalfFace));
					//pFace->addHalfFace(pFace->getHalfFaces().front()->reversedCopy(false));
					//pFace->setCentroid(pFace->getHalfFaces().front()->getCentroid());
					halffaces.push_back(pHalfFace);

					/** Add face connections to vertices*/
					for (int i = 0; i < edges.size(); i++) {
						edges[i]->getVertex1()->addConnectedHalfFace(pHalfFace);
						edges[i]->getVertex2()->addConnectedHalfFace(pHalfFace);
					}


				}

				return totalNumberOfVertices;
			}
			

			template <>
			int polyhedron3ToHalfFaces(CgalPolyhedron *pPoly, const map<uint, Vertex<Vector3D> *> &verticesMap, vector<HalfFace<Vector3D> *> &halffaces) {
				int totalNumberOfVertices = 0;

				/** Clean-up structures */
				halffaces.clear();

				map<uint, Edge<Vector3D> *> edgeMap;

				int tempIndex = 0;
				/** Check if all vertices are around first */
				for (auto it = pPoly->vertices_begin(); it != pPoly->vertices_end(); it++) {
					CgalPolyhedron::Vertex_handle vh(it);
					if (verticesMap.find(vh->id) == verticesMap.end()) {
						throw(exception("polyhedron3ToHalfFaces: vertex not found on verticesMap"));
					}
				}

				//Filler dimensions, geometric faces need no grid dimensions
				dimensions_t zeroDimensions;

				/** Create all edges, halfEdges and faces at the same time*/
				for (CgalPolyhedron::Facet_iterator it = pPoly->facets_begin(); it != pPoly->facets_end(); ++it) {
					auto hfc = it->facet_begin();
					vector<Edge<Vector3D> *> edges;
					vector<HalfEdge<Vector3D>*> halfedges;
					Vector3D faceNormal = it->normal;

					//Creating edges and half-edges for the current face 
					for (int j = 0; j < it->size(); ++j, ++hfc) {
						/** Setting up vertices temporaries */
						CgalPolyhedron::Vertex_handle vh(hfc->vertex());
						auto nextHfc = hfc;
						nextHfc++;
						CgalPolyhedron::Vertex_handle nextVh(nextHfc->vertex());
						auto iter = verticesMap.find(vh->id);
						auto nextIter = verticesMap.find(nextVh->id);
						if (iter == verticesMap.end() || nextIter == verticesMap.end())
							throw(exception("polyhedron3ToHalfFaces: vertex not found on verticesMap"));

						Vertex<Vector3D> *pVertex = iter->second, *pNextVertex = nextIter->second;

						/** Setting up edges temporaries*/
						uint mapHash = edgeHash(vh->id, nextVh->id, verticesMap.size());
						Edge<Vector3D> *pEdge = nullptr;

						if (edgeMap.find(mapHash) == edgeMap.end()) {
							edgeMap[mapHash] = new Edge<Vector3D>(pVertex, pNextVertex, geometricEdge);
						}
						edges.push_back(edgeMap[mapHash]);
						const pair<HalfEdge<Vector3D> *, HalfEdge<Vector3D> *> &halfEdgesPair = edgeMap[mapHash]->getHalfEdges();
						if (halfEdgesPair.first->getVertices().first->getID() == pVertex->getID()) { //First halfEdge is aligned with curr halfedge
							halfedges.push_back(halfEdgesPair.first);
						}
						else {
							halfedges.push_back(halfEdgesPair.second);
						}
					}

					HalfFace<Vector3D> *pHalfFace = new HalfFace<Vector3D>(halfedges, nullptr, geometryHalfFace);
					//pFace->addHalfFace(new HalfFace<Vector3D>(halfedges, pFace, geometryHalfFace));
					//pFace->addHalfFace(pFace->getHalfFaces().front()->reversedCopy(false));
					//pFace->setCentroid(pFace->getHalfFaces().front()->getCentroid());
					halffaces.push_back(pHalfFace);

					/** Add face connections to vertices*/
					for (int i = 0; i < edges.size(); i++) {
						edges[i]->getVertex1()->addConnectedHalfFace(pHalfFace);
						edges[i]->getVertex2()->addConnectedHalfFace(pHalfFace);
					}


				}

				return totalNumberOfVertices;
			}
			#pragma endregion

			/*template int polyhedron3ToHalfFacesAndVertices<Vector3>(CgalPolyhedron *pPoly, vector<Vertex<Vector3> *> &vertices, vector<HalfFace<Vector3> *> &faces);

			template int polyhedron3ToHalfFacesAndVertices<Vector3D>(CgalPolyhedron *pPoly, vector<Vertex<Vector3D> *> &vertices, vector<HalfFace<Vector3D> *> &faces);*/
		}
	}
}
