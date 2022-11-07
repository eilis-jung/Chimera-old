#include "CutCells/CutCellsBase.h"


/** Cut-Cells base functions implementation. */
namespace Chimera {


	namespace CutCells {
	
		#pragma region Constructors
		template <class VectorType>
		CutCellsBase<VectorType>::CutCellsBase(const vector<LineMesh<VectorType> *> &lineMeshes, Scalar gridSpacing, const dimensions_t &gridDimensions, faceLocation_t faceLocation)  
			: m_lineMeshes(lineMeshes), m_gridSpacing(gridSpacing), m_gridDimensions(gridDimensions), m_faceLocation(faceLocation), 
			m_pFacesArray(new Array2D<vector<Face<VectorType> *>>(gridDimensions)),
			m_pVerticalEdges(new Array2D<vector<Edge<VectorType> *> *>(gridDimensions, true)),
			m_pHorizontalEdges(new Array2D<vector<Edge<VectorType> *> *>(gridDimensions, true)),
			m_pNodeVertices(new Array2D<Vertex<VectorType> *>(gridDimensions)),
			m_facesArray(*m_pFacesArray), m_verticalEdges(*m_pVerticalEdges),
			m_horizontalEdges(*m_pHorizontalEdges), m_nodeVertices(*m_pNodeVertices) {
			m_sliceIndex = 0;

			//Array2<Vector2 *> tubu(gridDimensions);
			
			/**2-D initialization of own structures*/
			/*m_pVerticalEdges = new Array2D<vector<Edge<VectorType> *> *>(gridDimensions);
			m_pHorizontalEdges = new Array2D<vector<Edge<VectorType> *> *>(gridDimensions);*/
			/*for (int i = 0; i < gridDimensions.x; i++) {
				for (int j = 0; j < gridDimensions.y; j++) {
					(*m_pVerticalEdges)(i, j) = new vector <Edge<VectorType> *>();
					(*m_pHorizontalEdges)(i, j) = new vector <Edge<VectorType> *>();
				}
			}*/

			//m_pNodeVertices->assign(nullptr);

			m_pLinePatches = createLinePatches(m_faceLocation);
		}


		template <class VectorType>
		CutCellsBase<VectorType>::CutCellsBase(const extStructures_t & extStructures, const vector<LineMesh<VectorType> *> &lineMeshes, Scalar gridSpacing, const dimensions_t &gridDimensions, faceLocation_t faceLocation)
			: m_lineMeshes(lineMeshes), m_gridSpacing(gridSpacing), m_gridDimensions(gridDimensions), m_faceLocation(faceLocation),
			m_pFacesArray(new Array2D<vector<Face<VectorType> *>>(gridDimensions)),
			m_pVerticalEdges(extStructures.pVerticalEdges),
			m_pHorizontalEdges(extStructures.pHorizontalEdges),
			m_pNodeVertices(extStructures.pNodeVertices),
			m_facesArray(*m_pFacesArray), m_verticalEdges(*m_pVerticalEdges),
			m_horizontalEdges(*m_pHorizontalEdges), m_nodeVertices(*m_pNodeVertices) {
			
			m_pLinePatches = createLinePatches(m_faceLocation);
		}
		#pragma endregion


		#pragma region Functionalities
		template<class VectorType>
		void CutCellsBase<VectorType>::reinitialize(const vector<LineMesh<VectorType>*> &lineMeshes) {
			//Resets IDs for faces and edges
			Face<VectorType>::resetIDs();
			HalfFace<VectorType>::resetIDs();
			Edge<VectorType>::resetIDs();
			HalfEdge<VectorType>::resetIDs();

			for (int i = 0; i < m_elements.size(); i++) {
				delete m_elements[i];
			}
			m_elements.clear();
			//Do not delete elements because we are deallocating vertices later
			m_vertices.clear();

			for (int i = 0; i < m_pFacesArray->getDimensions().x; i++) {
				for (int j = 0; j < m_pFacesArray->getDimensions().y; j++) {
					(*m_pFacesArray)(i, j).clear();
				}
			}

			//Deletes only the vertices that were exclusivly created by this class
			for (int i = 0; i < m_pNodeVertices->getDimensions().x; i++) {
				for (int j = 0; j < m_pNodeVertices->getDimensions().y; j++) {
					delete (*m_pNodeVertices)(i, j);
					(*m_pNodeVertices)(i, j) = nullptr;
				}
			}

			for (int i = 0; i < m_pVerticalEdges->getDimensions().x; i++) {
				for (int j = 0; j < m_pVerticalEdges->getDimensions().y; j++) {
					for (int k = 0; k < (*m_pVerticalEdges)(i, j)->size(); k++) {
						delete (*m_pVerticalEdges)(i, j)->at(k);
					}
					(*m_pVerticalEdges)(i, j)->clear();
					for (int k = 0; k < (*m_pHorizontalEdges)(i, j)->size(); k++) {
						delete (*m_pHorizontalEdges)(i, j)->at(k);
					}
					(*m_pHorizontalEdges)(i, j)->clear();
				}
			}

			//Deletes only the vertices that were exclusivly created by this class
			for (int i = 0; i < (*m_pLinePatches).getDimensions().x; i++) {
				for (int j = 0; j < (*m_pLinePatches).getDimensions().y; j++) {
					(*m_pLinePatches)(i, j).clear();
				}
			}

			for (int i = 0; i < m_halfFaces.size(); i++) {
				delete m_halfFaces[i];
			}
			m_halfFaces.clear();

			m_lineMeshes = lineMeshes;
			initialize();
		}

		#pragma endregion
		#pragma region InitializationFunctions
		template <class VectorType>
		Array2D<vector<pair<uint, vector<uint>>>> * CutCellsBase<VectorType>::createLinePatches(faceLocation_t faceLocation) {
			return new Array2D<vector<pair<uint, vector<uint>>>>(dimensions_t(m_gridDimensions.x, m_gridDimensions.y));
			/*if (faceLocation == XYFace) {
				return new Array2D<vector<pair<uint, vector<uint>>>>(dimensions_t(m_gridDimensions.x, m_gridDimensions.y));
			}
			else if (faceLocation == YZFace) {
				return new Array2D<vector<pair<uint, vector<uint>>>>(dimensions_t(m_gridDimensions.z, m_gridDimensions.y));
			}
			else if (faceLocation == XZFace) {
				return  new Array2D<vector<pair<uint, vector<uint>>>>(dimensions_t(m_gridDimensions.x, m_gridDimensions.z));
			}*/
		}
		
		template <class VectorType>
		void CutCellsBase<VectorType>::buildLinePatches() {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					for (int k = 0; k < m_lineMeshes.size(); k++) {
						if (m_lineMeshes[k]->getPatchesIndices(i, j).size() > 0) {
							(*m_pLinePatches)(i, j).push_back(pair<uint, vector<uint>>(k, m_lineMeshes[k]->getPatchesIndices(i, j)));
						}
					}
				}
			}
		}

		template <class VectorType>
		void CutCellsBase<VectorType>::buildNodeVertices() {
			for (int i = 0; i < m_gridDimensions.x - 1; i++) {
				for (int j = 0; j < m_gridDimensions.y - 1; j++) {
					if ((*m_pLinePatches)(i, j).size() > 0) {

						//Check if any of the vertices of the line mesh are on top of a grid nodes
						for (int k = 0; k < (*m_pLinePatches)(i, j).size(); k++) {
							uint lineMeshIndex = (*m_pLinePatches)(i, j)[k].first;
							auto lineMeshPatches = (*m_pLinePatches)(i, j)[k].second;

							for (int l = 0; l < lineMeshPatches.size(); l++) {
								Edge<VectorType> *pEdge = m_lineMeshes[lineMeshIndex]->getElements()[lineMeshPatches[l]];
								if (pEdge->getVertex1()->isOnGridNode()) { //If it is, initialize the node vertex to be this edge vertex
									const VectorType &position = pEdge->getVertex1()->getPosition();
									dimensions_t nodeVertexDim(position.x / m_gridSpacing, position.y / m_gridSpacing);
									m_nodeVertices(nodeVertexDim) = pEdge->getVertex1();
								}
								if (pEdge->getVertex2()->isOnGridNode()) {
									const VectorType &position = pEdge->getVertex2()->getPosition();
									dimensions_t nodeVertexDim(position.x / m_gridSpacing, position.y / m_gridSpacing);
									m_nodeVertices(nodeVertexDim) = pEdge->getVertex2();
								}
							}
						}

					
						if (!validVertex(i, j)) {
							m_vertices.push_back(createVertex(i, j));
							m_nodeVertices(i, j) = m_vertices.back();
						}

						if ((*m_pLinePatches)(i + 1, j).size() == 0 && !validVertex(i + 1, j)) {
							m_vertices.push_back(createVertex(i + 1, j));
							m_nodeVertices(i + 1, j) = m_vertices.back();
						}

						if ((*m_pLinePatches)(i, j + 1).size() == 0 && !validVertex(i, j + 1)) {
							m_vertices.push_back(createVertex(i, j + 1));
							m_nodeVertices(i, j + 1) = m_vertices.back();
						}

						if ((*m_pLinePatches)(i + 1, j + 1).size() == 0 && !validVertex(i + 1, j + 1)) {
							m_vertices.push_back(createVertex(i + 1, j + 1));
							m_nodeVertices(i + 1, j + 1) = m_vertices.back();
						}
					}
				}
			}
		}

		template <class VectorType>
		void CutCellsBase<VectorType>::buildGridEdges() {
			//Adding all edge vertices to be accessible from the m_vertices vector
			for (int i = 0; i < m_lineMeshes.size(); i++) {
				for (int j = 0; j < m_lineMeshes[i]->getVertices().size(); j++) {
					m_vertices.push_back(m_lineMeshes[i]->getVertices()[j]);
				}
			}

			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					if ((*m_pLinePatches)(i, j).size() > 0) {
						vector <Vertex<VectorType> *> bottomVertices, leftVertices, topVertices, rightVertices;
						//Add all vertices from all line patches
						for (int k = 0; k < (*m_pLinePatches)(i, j).size(); k++) {
							uint lineMeshIndex = (*m_pLinePatches)(i, j)[k].first;
							auto lineMeshPatches = (*m_pLinePatches)(i, j)[k].second;

							for (int l = 0; l < lineMeshPatches.size(); l++) {
								Edge<VectorType> *pEdge = m_lineMeshes[lineMeshIndex]->getElements()[lineMeshPatches[l]];
								pEdge->setRelativeFraction(pEdge->getLength() / m_gridSpacing);
								if (pEdge->getVertex1()->getVertexType() == edgeVertex && !pEdge->getVertex1()->isOnGridNode())  {
									classifyVertex(dimensions_t(i, j), pEdge->getVertex1(), bottomVertices, leftVertices, topVertices, rightVertices);
								}
								if (pEdge->getVertex2()->getVertexType() == edgeVertex && !pEdge->getVertex2()->isOnGridNode()) {
									classifyVertex(dimensions_t(i, j), pEdge->getVertex2(), bottomVertices, leftVertices, topVertices, rightVertices);
								}
							}
						}
						//Sort crossings
						sortVertices(bottomVertices, leftVertices, topVertices, rightVertices);

						/** Bottom-Top edges initialization */
						Vertex<VectorType> *pIniVertex = m_nodeVertices(i, j);
						Vertex<VectorType> *pFinalVertex = m_nodeVertices(i + 1, j);
						dimensions_t currDimension(i, j);
						if ((*m_pHorizontalEdges)(i, j)->size() == 0) { //Only initialize if another cut-slice didn't initialized this edge vector (valid for 3-D only)
							if (bottomVertices.size() == 0) { //Create full grid edge
								if (!hasAlignedEdges(pIniVertex->getID(), pFinalVertex->getID(), currDimension, bottomHalfEdge)) { //Only add if this edge is not aligned with geometry edges
									(*m_pHorizontalEdges)(i, j)->push_back(createGridEdge(pIniVertex, pFinalVertex, bottomHalfEdge));
									(*m_pHorizontalEdges)(i, j)->back()->setRelativeFraction((*m_pHorizontalEdges)(i, j)->back()->getLength() / m_gridSpacing);
								}
							}
							else {
								Vertex<VectorType> *pCurrVertex = pIniVertex;
								for (int k = 0; k < bottomVertices.size(); k++) {
									Vertex<VectorType> *pEdgeVertex = bottomVertices[k];
									if (!hasAlignedEdges(pCurrVertex->getID(), pEdgeVertex->getID(), currDimension, bottomHalfEdge)) { //Only add if this edge is not aligned with geometry edges
										(*m_pHorizontalEdges)(i, j)->push_back(createGridEdge(pCurrVertex, pEdgeVertex, bottomHalfEdge));
										(*m_pHorizontalEdges)(i, j)->back()->setRelativeFraction((*m_pHorizontalEdges)(i, j)->back()->getLength() / m_gridSpacing);
									}
									pCurrVertex = pEdgeVertex;
								}
								if (!hasAlignedEdges(pCurrVertex->getID(), pFinalVertex->getID(), currDimension, bottomHalfEdge)) { //Only add if this edge is not aligned with geometry edges
									(*m_pHorizontalEdges)(i, j)->push_back(createGridEdge(pCurrVertex, pFinalVertex, bottomHalfEdge));
									(*m_pHorizontalEdges)(i, j)->back()->setRelativeFraction((*m_pHorizontalEdges)(i, j)->back()->getLength() / m_gridSpacing);
								}
							}
						}
						else {
							auto debugVec = (*m_pHorizontalEdges)(i, j);
							for (int i = 0; i < debugVec->size(); i++) {
								debugVec[i];
							}
						}

						/** Left edges initialization */
						pFinalVertex = m_nodeVertices(i, j + 1);
						if ((*m_pVerticalEdges)(i, j)->size() == 0) {//Only initialize if another cut-slice didn't initialized this edge vector (valid for 3-D only)
							if (leftVertices.size() == 0) { //Create full grid edge
								if (!hasAlignedEdges(pIniVertex->getID(), pFinalVertex->getID(), currDimension, leftHalfEdge)) { //Only add if this edge is not aligned with geometry edges
									(*m_pVerticalEdges)(i, j)->push_back(createGridEdge(pIniVertex, pFinalVertex, leftHalfEdge));
									(*m_pVerticalEdges)(i, j)->back()->setRelativeFraction(1.0);
								}
							}
							else {
								Vertex<VectorType> *pCurrVertex = pIniVertex;
								for (int k = 0; k < leftVertices.size(); k++) {
									Vertex<VectorType> *pEdgeVertex = leftVertices[k];
									if (!hasAlignedEdges(pCurrVertex->getID(), pEdgeVertex->getID(), currDimension, leftHalfEdge)) { //Only add if this edge is not aligned with geometry edges
										(*m_pVerticalEdges)(i, j)->push_back(createGridEdge(pCurrVertex, pEdgeVertex, leftHalfEdge));
										(*m_pVerticalEdges)(i, j)->back()->setRelativeFraction((*m_pVerticalEdges)(i, j)->back()->getLength() / m_gridSpacing);
									}
									pCurrVertex = pEdgeVertex;
								}
								if (!hasAlignedEdges(pCurrVertex->getID(), pFinalVertex->getID(), currDimension, leftHalfEdge)) { //Only add if this edge is not aligned with geometry edges
									(*m_pVerticalEdges)(i, j)->push_back(createGridEdge(pCurrVertex, pFinalVertex, leftHalfEdge));
									(*m_pVerticalEdges)(i, j)->back()->setRelativeFraction((*m_pVerticalEdges)(i, j)->back()->getLength() / m_gridSpacing);
								}
							}
						}
						

						pIniVertex = m_nodeVertices(i, j + 1);
						pFinalVertex = m_nodeVertices(i + 1, j + 1);
						if ((*m_pHorizontalEdges)(i, j + 1)->size() == 0 && topVertices.size() == 0 && (*m_pLinePatches)(i, j + 1).size() == 0) {
							if (!hasAlignedEdges(pIniVertex->getID(), pFinalVertex->getID(), currDimension, topHalfEdge)) { //Only add if this edge is not aligned with geometry edges
								(*m_pHorizontalEdges)(i, j + 1)->push_back(createGridEdge(pIniVertex, pFinalVertex, topHalfEdge));
								(*m_pHorizontalEdges)(i, j + 1)->back()->setRelativeFraction(1.0);
							}
						}

						pIniVertex = m_nodeVertices(i + 1, j);
						pFinalVertex = m_nodeVertices(i + 1, j + 1);
						if ((*m_pVerticalEdges)(i + 1, j)->size() == 0 && rightVertices.size() == 0 && (*m_pLinePatches)(i + 1, j).size() == 0) {
							if (!hasAlignedEdges(pIniVertex->getID(), pFinalVertex->getID(), currDimension, rightHalfEdge)) { //Only add if this edge is not aligned with geometry edges
								(*m_pVerticalEdges)(i + 1, j)->push_back(createGridEdge(pIniVertex, pFinalVertex, rightHalfEdge));
								(*m_pVerticalEdges)(i + 1, j)->back()->setRelativeFraction(1.0);
							}
						} 
					}
				}
			}	
		}

		template <class VectorType>
		void CutCellsBase<VectorType>::buildFaces() {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					if ((*m_pLinePatches)(i, j).size() > 0) {
						vector<Edge<VectorType> *> faceEdges;
						//Bottom, right, top, left edges
						faceEdges.insert(faceEdges.end(), (*m_pHorizontalEdges)(i, j)->begin(), (*m_pHorizontalEdges)(i, j)->end());
						int tempSize = (*m_pHorizontalEdges)(i, j)->size();
						faceEdges.insert(faceEdges.end(), (*m_pVerticalEdges)(i + 1, j)->begin(), (*m_pVerticalEdges)(i + 1, j)->end());
						tempSize = (*m_pVerticalEdges)(i + 1, j)->size();
						faceEdges.insert(faceEdges.end(), (*m_pHorizontalEdges)(i, j + 1)->begin(), (*m_pHorizontalEdges)(i, j + 1)->end());
						tempSize = (*m_pHorizontalEdges)(i, j + 1)->size();
						faceEdges.insert(faceEdges.end(), (*m_pVerticalEdges)(i, j)->begin(), (*m_pVerticalEdges)(i, j)->end());
						tempSize = (*m_pVerticalEdges)(i, j)->size();

						for (int k = 0; k < (*m_pLinePatches)(i, j).size(); k++) {
							uint lineMeshIndex = (*m_pLinePatches)(i, j)[k].first;
							auto lineMeshPatches = (*m_pLinePatches)(i, j)[k].second;

							//LineMesh vertices
							for (int l = 0; l < lineMeshPatches.size(); l++) {
								faceEdges.push_back(m_lineMeshes[lineMeshIndex]->getElements()[lineMeshPatches[l]]);
							}
						}
						m_elements.push_back(new Face<VectorType>(faceEdges, dimensions_t(i, j), m_gridSpacing, m_faceLocation));
						m_facesArray(i, j).push_back(m_elements.back());
					}
				}
			}
		}
	
		template <class VectorType>
		void CutCellsBase<VectorType>::buildHalfFaces() {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					if ((*m_pLinePatches)(i, j).size() > 0) {
						auto currHalfFaces = m_facesArray(i, j).back()->split();
						m_halfFaces.insert(m_halfFaces.end(), currHalfFaces.begin(), currHalfFaces.end());
					}
				}
			}
			
			if (m_sliceIndex == 0) {
				buildGhostVertices();
				//Creates adjcency map after ghost vertices: in this way, ghost vertices will only refer to edges that are
				//on the same side of the cut-cell geometry edge
				buildVertexEdgesAdjacencyMaps();
			}
			
		}

		template <class VectorType>
		void CutCellsBase<VectorType>::buildGhostVertices() {
			//Initially set all vertices to not visited
			for (int i = 0; i < m_halfFaces.size(); i++) {
				for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
					m_halfFaces[i]->getHalfEdges()[j]->getVertices().first->setUpdated(false);
				}
			} 
			/* First split all half edges */
			for (int i = 0; i < m_lineMeshes.size(); i++) {

				for (int j = 0; j < m_lineMeshes[i]->getElements().size(); j++) {
					Edge<VectorType> *pEdge = m_lineMeshes[i]->getElements()[j];
					HalfEdge<VectorType> *pHalfEdge = pEdge->getHalfEdges().first;
					if (!pHalfEdge->getVertices().first->hasUpdated()) { //Each vertex has to be split only once
						Vertex<VectorType> *pNewVertex = new Vertex<VectorType>(*pHalfEdge->getVertices().first);
						pNewVertex->setUpdated(true);
						m_vertices.push_back(pNewVertex);
						pHalfEdge->getVertices().first = pNewVertex;

						if (j == 0 && !m_lineMeshes[i]->isClosedMesh())
							continue;

						int prevJ = roundClamp<int>(j - 1, 0, m_lineMeshes[i]->getElements().size());
						pEdge = m_lineMeshes[i]->getElements()[prevJ];
						pHalfEdge = pEdge->getHalfEdges().first;
						pHalfEdge->getVertices().second = pNewVertex;
					}
				}
			}

			//Then visit all cut-cells and ensure that half-edges that were split have correct vertices pointers 
			//The idea is to fix vertices of grid edges which have vertices on top of the geometry. These are replaced
			//by the newly split ghost vertices
			for (int i = 0; i < m_halfFaces.size(); i++) {
				for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
					vector<HalfEdge<VectorType> *> & halfEdges = m_halfFaces[i]->getHalfEdges();
					if (halfEdges[j]->getLocation() != geometryHalfEdge) {
						if (halfEdges[j]->getVertices().first->isOnGeometryVertex()) {
							int prevJ = roundClamp<int>(j - 1, 0, m_halfFaces[i]->getHalfEdges().size());
							halfEdges[j]->getVertices().first = halfEdges[prevJ]->getVertices().second;
						}
						if (halfEdges[j]->getVertices().second->isOnGeometryVertex()) {
							int nextJ = roundClamp<int>(j + 1, 0, m_halfFaces[i]->getHalfEdges().size());
							halfEdges[j]->getVertices().second = halfEdges[nextJ]->getVertices().first;
						}
					}
				}
			}

			//Check if vertices split worked
			for (int i = 0; i < m_halfFaces.size(); i++) {
				for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {

					if (m_halfFaces[i]->getHalfEdges()[j]->getLocation() == geometryHalfEdge) {
						
						HalfEdge<VectorType> *pCurrEdge = m_halfFaces[i]->getHalfEdges()[j];
						HalfEdge<VectorType> *pOtherEdge = pCurrEdge->getEdge()->getHalfEdges().first == pCurrEdge ? pCurrEdge->getEdge()->getHalfEdges().second :
																										 pCurrEdge->getEdge()->getHalfEdges().first;
						if (pOtherEdge->getVertices().second == pCurrEdge->getVertices().first || pOtherEdge->getVertices().first == pCurrEdge->getVertices().second) {
							throw("Ghost vertices subdivision failed");
						}
													
					}
				}
			}
		}

		template <class VectorType>
		void CutCellsBase<VectorType>::buildVertexEdgesAdjacencyMaps() {
			//Clear connected edges first
			for (int i = 0; i < m_halfFaces.size(); i++) {
				for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
					m_halfFaces[i]->getHalfEdges()[j]->getVertices().first->clearConnectedEdges();
					m_halfFaces[i]->getHalfEdges()[j]->getVertices().second->clearConnectedEdges();
				}
			}

			for (int i = 0; i < m_lineMeshes.size(); i++) {
				for (int j = 0; j < m_lineMeshes[i]->getElements().size(); j++) {
					HalfEdge<VectorType> *pHalfEdge = m_lineMeshes[i]->getElements()[j]->getHalfEdges().first;
					pHalfEdge->getVertices().first->addConnectedEdge(pHalfEdge->getEdge());
					pHalfEdge->getVertices().second->addConnectedEdge(pHalfEdge->getEdge());

					pHalfEdge = m_lineMeshes[i]->getElements()[j]->getHalfEdges().second;
					pHalfEdge->getVertices().first->addConnectedEdge(pHalfEdge->getEdge());
					pHalfEdge->getVertices().second->addConnectedEdge(pHalfEdge->getEdge());
				}
			}

			for (int i = 0; i < m_pVerticalEdges->getDimensions().x; i++) {
				for (int j = 0; j < m_pVerticalEdges->getDimensions().y; j++) {
					for (int k = 0; k < (*m_pVerticalEdges)(i, j)->size(); k++) {
						(*m_pVerticalEdges)(i, j)->at(k)->getVertex1()->addConnectedEdge((*m_pVerticalEdges)(i, j)->at(k));
						(*m_pVerticalEdges)(i, j)->at(k)->getVertex2()->addConnectedEdge((*m_pVerticalEdges)(i, j)->at(k));
					}
					for (int k = 0; k < (*m_pHorizontalEdges)(i, j)->size(); k++) {
						(*m_pHorizontalEdges)(i, j)->at(k)->getVertex1()->addConnectedEdge((*m_pHorizontalEdges)(i, j)->at(k));
						(*m_pHorizontalEdges)(i, j)->at(k)->getVertex2()->addConnectedEdge((*m_pHorizontalEdges)(i, j)->at(k));
					}
				}
			}

			/*for (int i = 0; i < m_halfFaces.size(); i++) {
				for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
					m_halfFaces[i]->getHalfEdges()[j]->getVertices().first->addConnectedEdge(m_halfFaces[i]->getHalfEdges()[j]->getEdge());
					m_halfFaces[i]->getHalfEdges()[j]->getVertices().second->addConnectedEdge(m_halfFaces[i]->getHalfEdges()[j]->getEdge());
				}
			}*/
		}
		#pragma endregion


		#pragma region AuxiliaryHelperFunctions 
		template<class VectorType>
		bool CutCellsBase<VectorType>::hasAlignedEdges(uint vertex1, uint vertex2, dimensions_t linePatchesDim) {
			for (int k = 0; k < (*m_pLinePatches)(linePatchesDim).size(); k++) {
				uint lineMeshIndex = (*m_pLinePatches)(linePatchesDim)[k].first;
				auto lineMeshPatches = (*m_pLinePatches)(linePatchesDim)[k].second;

				for (int l = 0; l < lineMeshPatches.size(); l++) {
					Edge<VectorType> *pEdge = m_lineMeshes[lineMeshIndex]->getElements()[lineMeshPatches[l]];
					if ((pEdge->getVertex1()->getID() == vertex1 && pEdge->getVertex2()->getID() == vertex2) ||
						(pEdge->getVertex2()->getID() == vertex1 && pEdge->getVertex1()->getID() == vertex2)) {
						return true;
					}
				}
			}
			return false;
		}
		#pragma endregion

		template class CutCellsBase<Vector2>;
		template class CutCellsBase<Vector2D>;
		template class CutCellsBase<Vector3>;
		template class CutCellsBase<Vector3D>;
	}
}