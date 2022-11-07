#include "CutCells/CutCells2D.h"


/** Cut-Cells 2D-2D implementation. */
namespace Chimera {


	namespace CutCells {
	
		#pragma region Constructors
		template<class ChildT, class VectorType, bool isVector2>
		CutCells2DBase<ChildT, VectorType, isVector2>::CutCells2DBase(const vector<LineMesh<VectorType> *> &lineMeshes, Scalar gridSpacing, const dimensions_t &gridDimensions, faceLocation_t cutCellsPlane)
			: m_facesArray(gridDimensions), m_verticalEdges(gridDimensions), m_horizontalEdges(gridDimensions), m_nodeVertices(gridDimensions), m_linePatches(gridDimensions) {
			m_nodeVertices.assign(nullptr);
			m_facesArray.assign(nullptr);
			m_gridSpacing = gridSpacing;
			m_gridDimensions = gridDimensions;
			m_lineMeshes = lineMeshes;
			m_cutCellsPlane = cutCellsPlane;

		}
		#pragma endregion

		#pragma region Functionalities
		template<class ChildT, class VectorType, bool isVector2>
		void CutCells2DBase<ChildT, VectorType, isVector2>::reinitialize(const vector<LineMesh<VectorType>*> &lineMeshes) {
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

			for (int i = 0; i < m_facesArray.getDimensions().x; i++) {
				for (int j = 0; j < m_facesArray.getDimensions().y; j++) {
					m_facesArray(i, j) = nullptr;
				}
			}

			//Deletes only the vertices that were exclusivly created by this class
			for (int i = 0; i < m_nodeVertices.getDimensions().x; i++) {
				for (int j = 0; j < m_nodeVertices.getDimensions().y; j++) {
					delete m_nodeVertices(i, j);
					m_nodeVertices(i, j) = nullptr;
				}
			}

			for (int i = 0; i < m_verticalEdges.getDimensions().x; i++) {
				for (int j = 0; j < m_verticalEdges.getDimensions().y; j++) {
					for (int k = 0; k < m_verticalEdges(i, j).size(); k++) {
						delete m_verticalEdges(i, j)[k];
					}
					m_verticalEdges(i, j).clear();
					for (int k = 0; k < m_horizontalEdges(i, j).size(); k++) {
						delete m_horizontalEdges(i, j)[k];
					}
					m_horizontalEdges(i, j).clear();
				}
			}

			//Deletes only the vertices that were exclusivly created by this class
			for (int i = 0; i < m_linePatches.getDimensions().x; i++) {
				for (int j = 0; j < m_linePatches.getDimensions().y; j++) {
					m_linePatches(i, j).clear();
				}
			}

			for (int i = 0; i < m_halfFaces.size(); i++) {
				delete m_halfFaces[i];
			}
			m_halfFaces.clear();

			m_lineMeshes = lineMeshes;
			buildLinePatches();
			buildNodeVertices();
			buildGridEdges();
			buildFaces();
			buildHalfFaces();
			//Mesh points for rendering
			initializePoints();
		}
		#pragma endregion
		
		#pragma region AccessFunctions
		template<class ChildT, class VectorType, bool isVector2>
		uint CutCells2DBase<ChildT, VectorType, isVector2>::getCutCellIndex(const VectorType & position) {
			dimensions_t currCellIndex(floor(position.x), floor(position.y));
			if (isCutCell(currCellIndex)) {
				return m_facesArray(currCellIndex)->getHalfFace(position*m_gridSpacing)->getID();
			}
			throw("Error: invalid cut-cell look-up on getCutCellIndex.");
			return uint(UINT_MAX);
		}
		#pragma endregion

		#pragma region InitializationFunctions
		template<class ChildT, class VectorType, bool isVector2>
		void CutCells2DBase<ChildT, VectorType, isVector2>::buildLinePatches() {
			for (int i = 0; i < m_gridDimensions.x - 1; i++) {
				for (int j = 0; j < m_gridDimensions.y - 1; j++) {
					for (int k = 0; k < m_lineMeshes.size(); k++) {
						if (m_lineMeshes[k]->getPatchesIndices(i, j).size() > 0) {
							m_linePatches(i, j).push_back(pair<uint, vector<uint>>(k, m_lineMeshes[k]->getPatchesIndices(i, j)));
						}
					}
				}
			}
		}

		template<class ChildT, class VectorType, bool isVector2>
		void CutCells2DBase<ChildT, VectorType, isVector2>::buildGridEdges() {
			//Adding all edge vertices to be accessible from the m_vertices vector
			for (int i = 0; i < m_lineMeshes.size(); i++) {
				for (int j = 0; j < m_lineMeshes[i]->getVertices().size(); j++) {
					m_vertices.push_back(m_lineMeshes[i]->getVertices()[j]);
				}
			}

			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					if (m_linePatches(i, j).size() > 0) {
						vector <Vertex<VectorType> *> bottomVertices, leftVertices, topVertices, rightVertices;
						//Add all vertices from all line patches
						for (int k = 0; k < m_linePatches(i, j).size(); k++) {
							uint lineMeshIndex = m_linePatches(i, j)[k].first;
							auto lineMeshPatches = m_linePatches(i, j)[k].second;

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
						if (bottomVertices.size() == 0) {
							if (!hasAlignedEdges(pIniVertex->getID(), pFinalVertex->getID(), currDimension, bottomHalfEdge)) { //Only add if this edge is not aligned with geometry edges
								m_horizontalEdges(i, j).push_back(new Edge<VectorType>(pIniVertex, pFinalVertex, bottomTopEdge));
								m_horizontalEdges(i, j).back()->setRelativeFraction(1.0);
							}
						}
						else {
							Vertex<VectorType> *pCurrVertex = pIniVertex;
							for (int k = 0; k < bottomVertices.size(); k++) {
								Vertex<VectorType> *pEdgeVertex = bottomVertices[k];
								if (!hasAlignedEdges(pCurrVertex->getID(), pEdgeVertex->getID(), currDimension, bottomHalfEdge)) { //Only add if this edge is not aligned with geometry edges
									m_horizontalEdges(i, j).push_back(new Edge<VectorType>(pCurrVertex, pEdgeVertex, bottomTopEdge));
									m_horizontalEdges(i, j).back()->setRelativeFraction(m_horizontalEdges(i, j).back()->getLength() / m_gridSpacing);
								}
								pCurrVertex = pEdgeVertex;
							}
							if (!hasAlignedEdges(pCurrVertex->getID(), pFinalVertex->getID(), currDimension, bottomHalfEdge)) { //Only add if this edge is not aligned with geometry edges
								m_horizontalEdges(i, j).push_back(new Edge<VectorType>(pCurrVertex, pFinalVertex, bottomTopEdge));
								m_horizontalEdges(i, j).back()->setRelativeFraction(m_horizontalEdges(i, j).back()->getLength() / m_gridSpacing);
							}
						}

						/** Left edges initialization */
						pFinalVertex = m_nodeVertices(i, j + 1);
						if (leftVertices.size() == 0) {
							if (!hasAlignedEdges(pIniVertex->getID(), pFinalVertex->getID(), currDimension, leftHalfEdge)) { //Only add if this edge is not aligned with geometry edges
								m_verticalEdges(i, j).push_back(new Edge<VectorType>(pIniVertex, pFinalVertex, leftRightEdge));
								m_verticalEdges(i, j).back()->setRelativeFraction(1.0);
							}	
						}
						else {
							Vertex<VectorType> *pCurrVertex = pIniVertex;
							for (int k = 0; k < leftVertices.size(); k++) {
								Vertex<VectorType> *pEdgeVertex = leftVertices[k];
								if (!hasAlignedEdges(pCurrVertex->getID(), pEdgeVertex->getID(), currDimension, leftHalfEdge)) { //Only add if this edge is not aligned with geometry edges
									m_verticalEdges(i, j).push_back(new Edge<VectorType>(pCurrVertex, pEdgeVertex, leftRightEdge));
									m_verticalEdges(i, j).back()->setRelativeFraction(m_verticalEdges(i, j).back()->getLength() / m_gridSpacing);
								}
								pCurrVertex = pEdgeVertex;
							}
							if (!hasAlignedEdges(pCurrVertex->getID(), pFinalVertex->getID(), currDimension, leftHalfEdge)) { //Only add if this edge is not aligned with geometry edges
								m_verticalEdges(i, j).push_back(new Edge<VectorType>(pCurrVertex, pFinalVertex, leftRightEdge));
								m_verticalEdges(i, j).back()->setRelativeFraction(m_verticalEdges(i, j).back()->getLength() / m_gridSpacing);
							}
						}

						pIniVertex = m_nodeVertices(i, j + 1);
						pFinalVertex = m_nodeVertices(i + 1, j + 1);
						if (topVertices.size() == 0 && m_linePatches(i, j + 1).size() == 0) {
							if (!hasAlignedEdges(pIniVertex->getID(), pFinalVertex->getID(), currDimension, topHalfEdge)) { //Only add if this edge is not aligned with geometry edges
								m_horizontalEdges(i, j + 1).push_back(new Edge<VectorType>(pIniVertex, pFinalVertex, bottomTopEdge));
								m_horizontalEdges(i, j + 1).back()->setRelativeFraction(1.0);
							}
						}

						pIniVertex = m_nodeVertices(i + 1, j);
						pFinalVertex = m_nodeVertices(i + 1, j + 1);
						if (rightVertices.size() == 0 && m_linePatches(i + 1, j).size() == 0) {
							if (!hasAlignedEdges(pIniVertex->getID(), pFinalVertex->getID(), currDimension, rightHalfEdge)) { //Only add if this edge is not aligned with geometry edges
								m_verticalEdges(i + 1, j).push_back(new Edge<VectorType>(pIniVertex, pFinalVertex, leftRightEdge));
								m_verticalEdges(i + 1, j).back()->setRelativeFraction(1.0);
							}
						}
					}
				}
			}	
		}

		template<class ChildT, class VectorType, bool isVector2>
		void CutCells2DBase<ChildT, VectorType, isVector2>::buildFaces() {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					if (m_linePatches(i, j).size() > 0) {
						vector<Edge<VectorType> *> faceEdges;
						//Bottom, right, top, left edges
						faceEdges.insert(faceEdges.end(), m_horizontalEdges(i, j).begin(), m_horizontalEdges(i, j).end());
						int tempSize = m_horizontalEdges(i, j).size();
						faceEdges.insert(faceEdges.end(), m_verticalEdges(i + 1, j).begin(), m_verticalEdges(i + 1, j).end());
						tempSize = m_verticalEdges(i + 1, j).size();
						faceEdges.insert(faceEdges.end(), m_horizontalEdges(i, j + 1).begin(), m_horizontalEdges(i, j + 1).end());
						tempSize = m_horizontalEdges(i, j + 1).size();
						faceEdges.insert(faceEdges.end(), m_verticalEdges(i, j).begin(), m_verticalEdges(i, j).end());
						tempSize = m_verticalEdges(i, j).size();

						for (int k = 0; k < m_linePatches(i, j).size(); k++) {
							uint lineMeshIndex = m_linePatches(i, j)[k].first;
							auto lineMeshPatches = m_linePatches(i, j)[k].second;

							//LineMesh vertices
							for (int l = 0; l < lineMeshPatches.size(); l++) {
								faceEdges.push_back(m_lineMeshes[lineMeshIndex]->getElements()[lineMeshPatches[l]]);
							}
						}
						m_elements.push_back(new Face<VectorType>(faceEdges, dimensions_t(i, j), m_gridSpacing));
						m_facesArray(i, j) = m_elements.back();
					}
				}
			}
		}
	
		template<class ChildT, class VectorType, bool isVector2>
		void CutCells2DBase<ChildT, VectorType, isVector2>::buildHalfFaces() {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					if (m_linePatches(i, j).size() > 0) {
						auto currHalfFaces = m_facesArray(i, j)->split();
						m_halfFaces.insert(m_halfFaces.end(), currHalfFaces.begin(), currHalfFaces.end());
					}
				}
			}
			
			
			buildGhostVertices();
			//Creates adjcency map after ghost vertices: in this way, ghost vertices will only refer to edges that are
			//on the same side of the cut-cell geometry edge
			buildVertexEdgesAdjacencyMaps();
		}

		template<class ChildT, class VectorType, bool isVector2>
		void CutCells2DBase<ChildT, VectorType, isVector2>::buildGhostVertices() {
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
					Vertex<VectorType> *pNewVertex = new Vertex<VectorType>(*pHalfEdge->getVertices().first);
					m_vertices.push_back(pNewVertex);
					pHalfEdge->getVertices().first = pNewVertex;

					if(j == 0 && !m_lineMeshes[i]->isClosedMesh())
						continue;

					int prevJ = roundClamp<int>(j - 1, 0, m_lineMeshes[i]->getElements().size());
					pEdge = m_lineMeshes[i]->getElements()[prevJ];
					pHalfEdge = pEdge->getHalfEdges().first;
					pHalfEdge->getVertices().second = pNewVertex;
				}
			}

			//Then visit all cut-cells and ensure that half-edges that were split have correct vertices pointers 
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

			//for (int i = 0; i < m_halfFaces.size(); i++) {
			//	for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
			//		vector<HalfEdge<VectorType> *> & halfEdges = m_halfFaces[i]->getHalfEdges();
			//		if (halfEdges[j]->getVertices().first->isOnGeometryVertex() && !halfEdges[j]->getEdge()->isVisited()) {
			//			int k = j;
			//			
			//			//if (k == j) {
			//			//	//Creating a new ghost vertex: it starts with same attributes, but keeps separated copies
			//			//	Vertex<VectorType> *pNewVertex = new Vertex<VectorType>(*pHalfEdge->getVertices().first);
			//			//	pHalfEdge->getVertices().first = pNewVertex;
			//			//	halffaces[i]->getHalfEdges()[k - 1]->getVertices().second = pNewVertex;
			//			//}
			//			do 
			//			{
			//				HalfEdge<VectorType> *pHalfEdge = halfEdges[k];
			//				if (!pHalfEdge->getEdge()->isVisited() && !pHalfEdge->getVertices().first->hasUpdated()) {
			//					//Creating a new ghost vertex: it starts with same attributes, but keeps separated copies
			//					Vertex<VectorType> *pNewVertex = new Vertex<VectorType>(*pHalfEdge->getVertices().first);
			//					m_vertices.push_back(pNewVertex);
			//					pHalfEdge->getVertices().first->setUpdated(true);
			//					pHalfEdge->getVertices().first = pNewVertex;
			//					pNewVertex->setUpdated(true);
			//					pHalfEdge->getEdge()->setVisited(true);
			//					int prevK = roundClamp<int>(k - 1, 0, m_halfFaces[i]->getHalfEdges().size());
			//					halfEdges[prevK]->getVertices().second->setUpdated(true);
			//					halfEdges[prevK]->getVertices().second = pNewVertex;
			//					halfEdges[prevK]->getEdge()->setVisited(true);
			//				}							
			//			} while (	k < m_halfFaces[i]->getHalfEdges().size() - 1 &&
			//						halfEdges[++k]->getVertices().first->isOnGeometryVertex());
			//			

			//			
			//			//SSj = k;
			//		}
			//	}
			//}

			//Check if vertices split worked
			for (int i = 0; i < m_halfFaces.size(); i++) {
				for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
					
					/*m_halfFaces[i]->getHalfEdges()[j]->getEdge()->setVisited(false);
					m_halfFaces[i]->getHalfEdges()[j]->getVertices().first->setUpdated(false);*/

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

		template<class ChildT, class VectorType, bool isVector2>
		void CutCells2DBase<ChildT, VectorType, isVector2>::buildVertexEdgesAdjacencyMaps() {
			for (int i = 0; i < m_halfFaces.size(); i++) {
				for (int j = 0; j < m_halfFaces[i]->getHalfEdges().size(); j++) {
					m_halfFaces[i]->getHalfEdges()[j]->getVertices().first->addConnectedEdge(m_halfFaces[i]->getHalfEdges()[j]->getEdge());
					m_halfFaces[i]->getHalfEdges()[j]->getVertices().second->addConnectedEdge(m_halfFaces[i]->getHalfEdges()[j]->getEdge());
				}
			}
		}
		#pragma endregion

		#pragma region AuxiliaryHelperFunctions 
		template<class ChildT, class VectorType, bool isVector2>
		void CutCells2DBase<ChildT, VectorType, isVector2>::classifyVertex(const dimensions_t &gridDim, Vertex<VectorType> *pVertex, vector<Vertex<VectorType> *> &bottomVertices, vector<Vertex<VectorType> *> &leftVertices,
																				 vector<Vertex<VectorType> *> &topVertices, vector<Vertex<VectorType> *> &rightVertices) {
			if (pVertex->getVertexType() != edgeVertex)
				return;

			VectorType gridSpaceVertex = pVertex->getPosition() / m_gridSpacing;
			Scalar vx = gridSpaceVertex.x - gridDim.x;
			Scalar vy = gridSpaceVertex.y - gridDim.y;
			if (vx == 0) { //Left-edge
				leftVertices.push_back(pVertex);
			}
			else if (vy == 0) { //Bottom-edge
				bottomVertices.push_back(pVertex);
			}
			else if (vx == 1) {
				rightVertices.push_back(pVertex);
			}
			else if (vy == 1) {
				topVertices.push_back(pVertex);
			}
		}

		#pragma region ComparisonCallbacks
		template<class VectorType>
		bool compareVerticesHorizontal(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2) {
			return pV1->getPosition().x < pV2->getPosition().x;
		}

		template<class VectorType>
		bool compareVerticesVertical(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2) {
			return pV1->getPosition().y < pV2->getPosition().y;
		}
		#pragma endregion

		template<class ChildT, class VectorType, bool isVector2>
		void CutCells2DBase<ChildT, VectorType, isVector2>::sortVertices(	vector<Vertex<VectorType>*>& bottomVertices, vector<Vertex<VectorType>*>& leftVertices,
													vector<Vertex<VectorType> *> &topVertices, vector<Vertex<VectorType> *> &rightVertices) {
			sort(bottomVertices.begin(), bottomVertices.end(), compareVerticesHorizontal<VectorType>);
			sort(leftVertices.begin(), leftVertices.end(), compareVerticesVertical<VectorType>);
			sort(topVertices.begin(), topVertices.end(), compareVerticesHorizontal<VectorType>);
			sort(rightVertices.begin(), rightVertices.end(), compareVerticesVertical<VectorType>);
		}

		template<class ChildT, class VectorType, bool isVector2>
		bool CutCells2DBase<ChildT, VectorType, isVector2>::hasAlignedEdges(uint vertex1, uint vertex2, dimensions_t linePatchesDim) {
			for (int k = 0; k < m_linePatches(linePatchesDim).size(); k++) {
				uint lineMeshIndex = m_linePatches(linePatchesDim)[k].first;
				auto lineMeshPatches = m_linePatches(linePatchesDim)[k].second;

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

		#pragma region VectorAwareSpecializedFunctions
		/** Vector2 function specialization */
		template<class VectorType>
		void CutCells2DT<VectorType, true>::buildNodeVertices() {
			for (int i = 0; i < m_gridDimensions.x - 1; i++) {
				for (int j = 0; j < m_gridDimensions.y - 1; j++) {
					if (m_linePatches(i, j).size() > 0) {

						//Check if any of the vertices of the line mesh are on top of a grid nodes
						for (int k = 0; k < m_linePatches(i, j).size(); k++) {
							uint lineMeshIndex = m_linePatches(i, j)[k].first;
							auto lineMeshPatches = m_linePatches(i, j)[k].second;

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


						if (m_nodeVertices(i, j) == nullptr) {
							VectorType vertexPosition(i*m_gridSpacing, j*m_gridSpacing);
							m_vertices.push_back(new Vertex<VectorType>(vertexPosition, gridVertex));
							m_nodeVertices(i, j) = m_vertices.back();
						}

						if (m_linePatches(i + 1, j).size() == 0 && m_nodeVertices(i + 1, j) == nullptr) {
							VectorType vertexPosition = VectorType((i + 1)*m_gridSpacing, j*m_gridSpacing);
							m_vertices.push_back(new Vertex<VectorType>(vertexPosition, gridVertex));
							m_nodeVertices(i + 1, j) = m_vertices.back();
						}

						if (m_linePatches(i, j + 1).size() == 0 && m_nodeVertices(i, j + 1) == nullptr) {
							VectorType vertexPosition = VectorType(i*m_gridSpacing, (j + 1)*m_gridSpacing);
							m_vertices.push_back(new Vertex<VectorType>(vertexPosition, gridVertex));
							m_nodeVertices(i, j + 1) = m_vertices.back();
						}

						if (m_linePatches(i + 1, j + 1).size() == 0 && m_nodeVertices(i + 1, j + 1) == nullptr) {
							VectorType vertexPosition = VectorType((i + 1)*m_gridSpacing, (j + 1)*m_gridSpacing);
							m_vertices.push_back(new Vertex<VectorType>(vertexPosition, gridVertex));
							m_nodeVertices(i + 1, j + 1) = m_vertices.back();
						}
					}
				}
			}
		}

		/** Vector3 function specialization */
		template<class VectorType>
		void CutCells2DT<VectorType, false>::buildNodeVertices() {
			for (int i = 0; i < m_gridDimensions.x - 1; i++) {
				for (int j = 0; j < m_gridDimensions.y - 1; j++) {
					if (m_linePatches(i, j).size() > 0) {

						//Check if any of the vertices of the line mesh are on top of a grid nodes
						for (int k = 0; k < m_linePatches(i, j).size(); k++) {
							uint lineMeshIndex = m_linePatches(i, j)[k].first;
							auto lineMeshPatches = m_linePatches(i, j)[k].second;

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


						/*if (m_nodeVertices(i, j) == nullptr) {
							VectorType vertexPosition(i*m_gridSpacing, j*m_gridSpacing, k*m_gridSpacing);
							m_vertices.push_back(new Vertex<VectorType>(vertexPosition, gridVertex));
							m_nodeVertices(i, j) = m_vertices.back();
						}

						if (m_linePatches(i + 1, j).size() == 0 && m_nodeVertices(i + 1, j) == nullptr) {
							VectorType vertexPosition = VectorType((i + 1)*m_gridSpacing, j*m_gridSpacing, k*m_gridSpacing);
							m_vertices.push_back(new Vertex<VectorType>(vertexPosition, gridVertex));
							m_nodeVertices(i + 1, j) = m_vertices.back();
						}

						if (m_linePatches(i, j + 1).size() == 0 && m_nodeVertices(i, j + 1) == nullptr) {
							VectorType vertexPosition = VectorType(i*m_gridSpacing, (j + 1)*m_gridSpacing, k*m_gridSpacing);
							m_vertices.push_back(new Vertex<VectorType>(vertexPosition, gridVertex));
							m_nodeVertices(i, j + 1) = m_vertices.back();
						}

						if (m_linePatches(i + 1, j + 1).size() == 0 && m_nodeVertices(i + 1, j + 1) == nullptr) {
							VectorType vertexPosition = VectorType((i + 1)*m_gridSpacing, (j + 1)*m_gridSpacing, k*m_gridSpacing);
							m_vertices.push_back(new Vertex<VectorType>(vertexPosition, gridVertex));
							m_nodeVertices(i + 1, j + 1) = m_vertices.back();
						}*/
					}
				}
			}
		}
		#pragma endregion


		template class CutCells2DT<Vector2, isVector2<Vector2>::value>;
		template class CutCells2DT<Vector2D, isVector2<Vector2D>::value>;

		template class CutCells2DBase<CutCells2DT<Vector2, isVector2<Vector2>::value>, Vector2, isVector2<Vector2>::value>;
		template class CutCells2DBase<CutCells2DT<Vector2D, isVector2<Vector2D>::value>, Vector2D, isVector2<Vector2D>::value>;

	}
}