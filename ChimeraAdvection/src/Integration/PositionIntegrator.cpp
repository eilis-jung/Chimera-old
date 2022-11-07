#include "Integration/PositionIntegrator.h"


namespace Chimera {
	
	namespace Advection {
		template<>
		bool PositionIntegrator<Vector2, Array2D>::checkCollision(const Vector2 &p1, const Vector2 &p2) {
			if (m_pCutCell) {
				CutCells2D<Vector2> *pCutCells2D = dynamic_cast<CutCells2D<Vector2> *>(m_pCutCell);
				const vector<Meshes::LineMesh<Vector2> *> &lineMeshes = pCutCells2D->getLineMeshes();
				dimensions_t gridPosition(p1.x / pCutCells2D->getGridSpacing(), p1.y / pCutCells2D->getGridSpacing());
				if (!pCutCells2D->isCutCellAt(gridPosition.x, gridPosition.y)) {
					return false;
				}
				for (int i = 0; i < lineMeshes.size(); i++) {
					if (lineMeshes[i]->segmentIntersection(p1, p2)) {
						return true;
					}
				}
			}
			return false;
		}


		template<>
		bool PositionIntegrator<Vector3, Array3D>::checkCollision(const Vector3 &p1, const Vector3 &p2) {
			if (m_pCutVoxels) {
				Scalar dx = m_pCutVoxels->getGridSpacing();
				Vector3 crossingPoint;
				if (m_pCutVoxels) {
					dimensions_t gridPositionP1(p1.x / dx, p1.y / dx, p1.z / dx);
					if (m_pCutVoxels->isCutVoxel(gridPositionP1)) {
						auto cutVoxel = m_pCutVoxels->getCutVoxel(m_pCutVoxels->getCutVoxelIndex(p1 / dx));
						if (cutVoxel.crossedThroughGeometry(p1, p2, crossingPoint)) {
							return true;
						}
					}
					dimensions_t gridPositionP2(p2.x / dx, p2.y / dx, p2.z / dx);
					if (gridPositionP1 != gridPositionP2) {
						if (m_pCutVoxels->isCutVoxel(gridPositionP2)) {
							auto cutVoxel = m_pCutVoxels->getCutVoxel(m_pCutVoxels->getCutVoxelIndex(p2 / dx));
							if (cutVoxel.crossedThroughGeometry(p1, p2, crossingPoint)) {
								return true;
							}
						}
					}
				}
			}
			return false;
		}

		template<>
		void PositionIntegrator<Vector2, Array2D>::clampPosition(Vector2 &position) {
			Vector2 gridBoundaries(	(m_pInterpolant->getGridDimensions().x - (1e-5))*m_dx, 
									(m_pInterpolant->getGridDimensions().y - (1e-5))*m_dx);
			position.x = clamp(position.x, 0.0f, gridBoundaries.x);
			position.y = clamp(position.y, 0.0f, gridBoundaries.y);
		}

		template<>
		void PositionIntegrator<Vector3, Array3D>::clampPosition(Vector3 &position) {
			Vector3 gridBoundaries(	(m_pInterpolant->getGridDimensions().x - (1e-5))*m_dx,
									(m_pInterpolant->getGridDimensions().y - (1e-5))*m_dx,
									(m_pInterpolant->getGridDimensions().z - (1e-5))*m_dx);
			position.x = clamp(position.x, 0.0f, gridBoundaries.x);
			position.y = clamp(position.y, 0.0f, gridBoundaries.y);
			position.z = clamp(position.z, 0.0f, gridBoundaries.z);
		}
	}
}