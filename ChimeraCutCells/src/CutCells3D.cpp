#include "CutCells/CutCells3D.h"

namespace Chimera {


	namespace CutCells {
		
		#pragma region Constructors
		template<class VectorType>
		CutCells3D<VectorType>::CutCells3D(extStructures_t extStructures, const vector<LineMesh<VectorType>*>& lineMeshes, Scalar gridSpacing, const dimensions_t & gridDimensions, faceLocation_t faceLocation) 
			:	CutCellsBase(extStructures, lineMeshes, gridSpacing, gridDimensions, faceLocation) {
			
			switch (faceLocation) {
				case XYFace:
					m_sliceIndex = floor(lineMeshes.front()->getParams()->initialPoints.front().z / m_gridSpacing);
				break;
				case XZFace:
					m_sliceIndex = floor(lineMeshes.front()->getParams()->initialPoints.front().y / m_gridSpacing);
				break;
				case YZFace:
					m_sliceIndex = floor(lineMeshes.front()->getParams()->initialPoints.front().x / m_gridSpacing);
				break;
			}
		}
		#pragma endregion

		#pragma region AccessFunctions
		template<class VectorType>
		uint CutCells3D<VectorType>::getCutCellIndex(const VectorType & position) {
			dimensions_t currCellIndex;
			switch (m_faceLocation) {
				case XYFace:
					currCellIndex.x = floor(position.x);
					currCellIndex.y = floor(position.y);
				break;
				case XZFace:
					currCellIndex.x = floor(position.x);
					currCellIndex.y = floor(position.z);
				break;
				case YZFace:
					currCellIndex.x = floor(position.z);
					currCellIndex.y = floor(position.y);
				break;
			}

			if (CutCellsBase<VectorType>::isCutCell(currCellIndex)) {
				return m_facesArray(currCellIndex).front()->getHalfFace(position*m_gridSpacing)->getID();
			}
			throw("Error: invalid cut-cell look-up on getCutCellIndex.");
			return uint(UINT_MAX);
		}

		template<class VectorType>
		void CutCells3D<VectorType>::initialize() {
			buildLinePatches();
			buildNodeVertices();
			buildGridEdges();
			buildFaces();
			buildHalfFaces();
			buildGhostVertices();
			mirrorFaces();
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template<class VectorType>
		void CutCells3D<VectorType>::mirrorFaces() {
			//Remove previous elements initialized 
			m_elements.clear();

			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					if (m_facesArray(i, j).size() > 0) {
						if (m_facesArray(i, j).size() > 1) { //Invalid number of faces
							throw(exception("CutCells3D mirrorFaces: invalid number of faces on facesArray"));
						} 
						Face<VectorType> *pPreviousFace = m_facesArray(i, j)[0];
						m_facesArray(i, j) = m_facesArray(i, j)[0]->convertToFaces3D();
						//Replaces pointer and delete pPreviousFace
						replaceElement(pPreviousFace, m_facesArray(i, j).front());

						/** Calculating relative fraction and pushing it to elements */
						for (int k = 0; k < m_facesArray(i, j).size(); k++) {
							m_facesArray(i, j)[k]->setRelativeFraction(m_facesArray(i, j)[k]->calculateArea() / (m_gridSpacing*m_gridSpacing));
							m_elements.push_back(m_facesArray(i, j)[k]);
						}
					}
				}
			}
		}
		#pragma endregion

		#pragma region PureVirtualFunctions
		template<class VectorType>
		bool compareVerticesHorizontal(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2) {
			return pV1->getPosition().x < pV2->getPosition().x;
		}

		template<class VectorType>
		bool compareVerticesVertical(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2) {
			return pV1->getPosition().y < pV2->getPosition().y;
		}
		
		template <class VectorType>
		bool compareVerticesTransversal(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2) {
			return pV1->getPosition().z < pV2->getPosition().z;
		}


		template<class VectorType>
		Edge<VectorType> * CutCells3D<VectorType>::createGridEdge(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2, halfEdgeLocation_t halfEdgeLocation) {
			switch (m_faceLocation) {
				case XYFace:
					if (halfEdgeLocation == bottomHalfEdge || halfEdgeLocation == topHalfEdge) {
						return new Edge<VectorType>(pV1, pV2, xAlignedEdge);
					}
					else if (halfEdgeLocation == leftHalfEdge || halfEdgeLocation == rightHalfEdge) {
						return new Edge<VectorType>(pV1, pV2, yAlignedEdge);
					}
					else {
						throw(exception("CutCells2D createGridEdge: invalid edge location"));
					}
					return nullptr;
				break;
				case XZFace:
					if (halfEdgeLocation == bottomHalfEdge || halfEdgeLocation == topHalfEdge) {
						return new Edge<VectorType>(pV1, pV2, xAlignedEdge);
					}
					else if (halfEdgeLocation == leftHalfEdge || halfEdgeLocation == rightHalfEdge) {
						return new Edge<VectorType>(pV1, pV2, zAlignedEdge);
					}
					else {
						throw(exception("CutCells2D createGridEdge: invalid edge location"));
					}
					return nullptr;
				break;
				case YZFace:
					if (halfEdgeLocation == bottomHalfEdge || halfEdgeLocation == topHalfEdge) {
						return new Edge<VectorType>(pV1, pV2, zAlignedEdge);
					}
					else if (halfEdgeLocation == leftHalfEdge || halfEdgeLocation == rightHalfEdge) {
						return new Edge<VectorType>(pV1, pV2, yAlignedEdge);
					}
					else {
						throw(exception("CutCells2D createGridEdge: invalid edge location"));
					}
					return nullptr;
				break;
			}

			return nullptr;
		}



		template<class VectorType>
		void CutCells3D<VectorType>::sortVertices(	vector<Vertex<VectorType>*>& bottomVertices, vector<Vertex<VectorType>*>& leftVertices,
													vector<Vertex<VectorType> *> &topVertices, vector<Vertex<VectorType> *> &rightVertices) {
			switch (m_faceLocation) {
				case XYFace:
					sort(bottomVertices.begin(), bottomVertices.end(), compareVerticesHorizontal<VectorType>);
					sort(leftVertices.begin(), leftVertices.end(), compareVerticesVertical<VectorType>);
					sort(topVertices.begin(), topVertices.end(), compareVerticesHorizontal<VectorType>);
					sort(rightVertices.begin(), rightVertices.end(), compareVerticesVertical<VectorType>);
				break;
				case XZFace:
					sort(bottomVertices.begin(), bottomVertices.end(), compareVerticesHorizontal<VectorType>);
					sort(leftVertices.begin(), leftVertices.end(), compareVerticesTransversal<VectorType>);
					sort(topVertices.begin(), topVertices.end(), compareVerticesHorizontal<VectorType>);
					sort(rightVertices.begin(), rightVertices.end(), compareVerticesTransversal<VectorType>);
				break;
				case YZFace:
					sort(bottomVertices.begin(), bottomVertices.end(), compareVerticesTransversal<VectorType>);
					sort(leftVertices.begin(), leftVertices.end(), compareVerticesVertical<VectorType>);
					sort(topVertices.begin(), topVertices.end(), compareVerticesTransversal<VectorType>);
					sort(rightVertices.begin(), rightVertices.end(), compareVerticesVertical<VectorType>);
				break;
			}
		}

		template <class VectorType>
		void CutCells3D<VectorType>::classifyVertex(const dimensions_t &gridDim, Vertex<VectorType> *pVertex, vector<Vertex<VectorType> *> &bottomVertices, vector<Vertex<VectorType> *> &leftVertices,
																											  vector<Vertex<VectorType> *> &topVertices, vector<Vertex<VectorType> *> &rightVertices) {
			if (pVertex->getVertexType() != edgeVertex)
				return;

			VectorType gridSpaceVertex = pVertex->getPosition() / m_gridSpacing;
			Scalar vx, vy;
			switch (m_faceLocation) {
			case XYFace:
				vx = gridSpaceVertex.x - gridDim.x;
				vy = gridSpaceVertex.y - gridDim.y;
				break;
			case XZFace:
				vx = gridSpaceVertex.x - gridDim.x;
				//grimDim.y acts like z here
				vy = gridSpaceVertex.z - gridDim.y;
				break;
			case YZFace:
				//grimDim.x acts like z here
				vx = gridSpaceVertex.z - gridDim.x;
				vy = gridSpaceVertex.y - gridDim.y;
				break;
			}

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
		#pragma endregion

		template class CutCells3D<Vector3>;
		template class CutCells3D<Vector3D>;
	}
}