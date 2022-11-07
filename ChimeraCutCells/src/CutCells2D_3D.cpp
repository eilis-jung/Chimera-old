#include "CutCells/CutCells2D.h"

namespace Chimera {


	namespace CutCells {
	
		#pragma region VectorAwareSpecializedFunctions
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

		template class CutCells2DT<Vector3, isVector2<Vector3>::value>;
		template class CutCells2DT<Vector3D, isVector2<Vector3D>::value>;

		template class CutCells2DBase<CutCells2DT<Vector3, isVector2<Vector3>::value>, Vector3, isVector2<Vector3>::value>;
		template class CutCells2DBase<CutCells2DT<Vector3D, isVector2<Vector3D>::value>, Vector3D, isVector2<Vector3D>::value>;

	}
}