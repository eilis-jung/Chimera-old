#include "CutCells/CutCells2D.h"

namespace Chimera {


	namespace CutCells {
	
		#pragma region Constructors
		template <class VectorType>
		CutCells2D<VectorType>::CutCells2D(const vector<LineMesh<VectorType> *> &lineMeshes, Scalar gridSpacing, const dimensions_t &gridDimensions)
			:  CutCellsBase(lineMeshes, gridSpacing, gridDimensions) {

			
		}
		#pragma endregion

		
		#pragma region AccessFunctions
		template<class VectorType>
		uint CutCells2D<VectorType>::getCutCellIndex(const VectorType & position) {
			dimensions_t currCellIndex(floor(position.x), floor(position.y));
			if (CutCellsBase<VectorType>::isCutCell(currCellIndex)) {
				return m_facesArray(currCellIndex).front()->getHalfFace(position*m_gridSpacing)->getID();
			}
			throw("Error: invalid cut-cell look-up on getCutCellIndex.");
			return uint(UINT_MAX);
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

		template<class VectorType>
		Edge<VectorType> * CutCells2D<VectorType>::createGridEdge(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2, halfEdgeLocation_t halfEdgeLocation) {
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
		}

		template<class VectorType>
		void CutCells2D<VectorType>::sortVertices(vector<Vertex<VectorType>*>& bottomVertices, vector<Vertex<VectorType>*>& leftVertices,
			vector<Vertex<VectorType> *> &topVertices, vector<Vertex<VectorType> *> &rightVertices) {
			sort(bottomVertices.begin(), bottomVertices.end(), compareVerticesHorizontal<VectorType>);
			sort(leftVertices.begin(), leftVertices.end(), compareVerticesVertical<VectorType>);
			sort(topVertices.begin(), topVertices.end(), compareVerticesHorizontal<VectorType>);
			sort(rightVertices.begin(), rightVertices.end(), compareVerticesVertical<VectorType>);
		}


		template <class VectorType>
		void CutCells2D<VectorType>::classifyVertex(const dimensions_t &gridDim, Vertex<VectorType> *pVertex, vector<Vertex<VectorType> *> &bottomVertices, vector<Vertex<VectorType> *> &leftVertices,
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
		#pragma endregion
		template class CutCells2D<Vector2>;
		template class CutCells2D<Vector2D>;
	}
}