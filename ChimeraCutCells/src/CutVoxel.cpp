#include "Grids/CutVoxel.h"
#include "Mesh/Mesh.h"
#include "Grids/GridUtils.h"

namespace Chimera {
	namespace CutCells {

		void CutVoxel::initializeMesh(Scalar dx, const vector<Vector3D> &normals, bool onEdgeMixedNode) {
			m_pMesh = new Mesh<Vector3D>(m_vertices, normals, *this, dx, onEdgeMixedNode);
			for (int i = 0; i < m_pMesh->getMeshPolygons().size(); i++) {
				if (m_pMesh->getMeshPolygons()[i].polygonType == Mesh<Vector3D>::geometryPolygon) {
					geometryFacesToMesh.push_back(i);
				}
			}
			
		}

		void CutVoxel::initializePointsToMeshMap(const Mesh<Vector3D> &mesh) {
			pointsToTriangleMeshMap.clear();
			const vector<Vector3D> & points = mesh.getPoints();
			const vector<typename Mesh<Vector3D>::nodeType_t> & nodeTypes = mesh.getNodeTypes();
			const vector<typename Mesh<Vector3D>::meshPolygon_t> & polygons = mesh.getMeshPolygons();
			for (int i = 0; i < polygons.size(); i++) {
				if (polygons[i].polygonType == Mesh<Vector3D>::geometryPolygon) {
					for (int j = 0; j < 3; j++) {
						pointsToTriangleMeshMap[polygons[i].edges[j].first].push_back(i);
					}
				}
			}
		}

		void CutVoxel::initializeFacesEdgeMap(const Mesh<Vector3D> &mesh) {
			facesEdgesToMeshMap.clear();
			for (int i = 0; i < cutFaces.size(); i++) {
				if (cutFacesLocations[i] != geometryFace) {
					vector<int> meshIndices;
					for (int j = 0; j < cutFaces[i]->m_cutEdges.size(); j++) {
						int foundIndex = findOnMesh(cutFaces[i]->m_cutEdges[j]->m_initialPoint, mesh);
						if (foundIndex != -1) {
							meshIndices.push_back(foundIndex);
						}
					}
					facesEdgesToMeshMap[i] = meshIndices;
				}	
			}
		}
		int CutVoxel::findOnMesh(const Vector3D &point, const Mesh<Vector3D> &mesh) {
			const vector<Vector3D> & points = mesh.getPoints();
			const vector<typename Mesh<Vector3D>::nodeType_t> & nodeTypes = mesh.getNodeTypes();
			for (int i = 0; i < points.size(); i++) {
				if (nodeTypes[i] == Mesh<Vector3D>::mixedNode || nodeTypes[i] == Mesh<Vector3D>::geometryNode) {
					DoubleScalar length = (points[i] - point).length();
					if (length < doublePrecisionThreshold) {
						return i;
					}
				}
			}
			return -1;
		}
	}
}