#include "Mesh/Edge.h"

namespace Chimera {
	namespace Meshes {

		template <class VectorType>
		void VertexToEdgeMap<VectorType>::initializeMap(const vector<Edge<VectorType> *> &edges) {
			m_vertexToEdgeMap.clear();
			for (int i = 0; i < edges.size(); i++) {
				m_vertexToEdgeMap[edges[i]->getVertex1()->getID()].push_back(edges[i]);
				m_vertexToEdgeMap[edges[i]->getVertex2()->getID()].push_back(edges[i]);
			}
		}

		template <class VectorType>
		halfEdgeLocation_t Edge<VectorType>::classifyEdge(	Edge<VectorType> *pEdge,
																	const dimensions_t &gridDimensions, 
																	DoubleScalar gridDx, faceLocation_t faceLocation) {
			if (pEdge->getType() == geometricEdge) {
				return geometryHalfEdge;
			}

			switch (faceLocation) {
				case XYFace:
					return classifyEdgeXY(pEdge, gridDimensions, gridDx);
				break;
				case XZFace:
					return classifyEdgeXZ(pEdge, gridDimensions, gridDx);
				break;
				case YZFace:
					return classifyEdgeYZ(pEdge, gridDimensions, gridDx);
				break;
			}
		}

		template <class VectorType>
		halfEdgeLocation_t Edge<VectorType>::classifyEdgeXY(	Edge<VectorType> *pEdge,
																	const dimensions_t &gridDimensions,
																	DoubleScalar gridDx) {
			if (pEdge->getType() == xAlignedEdge) {
				if (floor(pEdge->getVertex1()->getPosition()[1] / gridDx) == gridDimensions.y) {
					return bottomHalfEdge;
				}
				else { 
					return topHalfEdge;
				}
			}
			else {
				if (floor(pEdge->getVertex1()->getPosition()[0] / gridDx) == gridDimensions.x) {
					return leftHalfEdge;
				}
				else {
					return rightHalfEdge;
				}
			}
		}

		template <class VectorType>
		halfEdgeLocation_t Edge<VectorType>::classifyEdgeXZ(	Edge<VectorType> *pEdge,
																	const dimensions_t &gridDimensions,
																	DoubleScalar gridDx) {
			if (pEdge->getType() == zAlignedEdge) {
				if (floor(pEdge->getVertex1()->getPosition()[0] / gridDx) == gridDimensions.x) {
					return leftHalfEdge;
				}
				else {
					return rightHalfEdge;
				}
			}
			else {
				if (floor(pEdge->getVertex1()->getPosition()[2] / gridDx) == gridDimensions.y) {
					return bottomHalfEdge;
				}
				else {
					return topHalfEdge;
				}
			}
		}

		template <class VectorType>
		halfEdgeLocation_t Edge<VectorType>::classifyEdgeYZ(	Edge<VectorType> *pEdge,
																	const dimensions_t &gridDimensions,
																	DoubleScalar gridDx) {
			if (pEdge->getType() == zAlignedEdge) {
				if (floor(pEdge->getVertex1()->getPosition()[1] / gridDx) == gridDimensions.y) {
					return bottomHalfEdge;
				}
				else {
					return topHalfEdge;
				}
			}
			else {
				if (floor(pEdge->getVertex1()->getPosition()[2] / gridDx) == gridDimensions.x) {
					return leftHalfEdge;
				}
				else {
					return rightHalfEdge;
				}
			}
		}


		template class Edge<Vector2>;
		template class Edge<Vector2D>;
		template class Edge<Vector3>;
		template class Edge<Vector3D>;

		template class VertexToEdgeMap<Vector2>;
		template class VertexToEdgeMap<Vector2D>;
		template class VertexToEdgeMap<Vector3>;
		template class VertexToEdgeMap<Vector3D>;
	}
}