#include "Mesh/PlaneMesh.h"

namespace Chimera {
	namespace Data {

		#pragma region Constructors
		PlaneMesh::PlaneMesh(const Vector3 & position, const Vector3 &upVec, const Vector2 &planeSize, Scalar tilingSpacing) {
			m_upVector = upVec;
			m_upVector.normalize();
			m_planeSize = planeSize;
			m_tilingAmount.x = static_cast<int>(m_planeSize.x/tilingSpacing);
			m_tilingAmount.y = static_cast<int>(m_planeSize.y/tilingSpacing);
			m_tilingAmount.z = 0;

			//Adjusting plane size
			m_planeSize.x = m_tilingAmount.x*tilingSpacing;
			m_planeSize.y = m_tilingAmount.y*tilingSpacing; 
			m_pPoints = new Array2D<Vector3>(m_tilingAmount);
			m_pInitialPoints = new Array2D<Vector3>(m_tilingAmount);

			double centroidX, centroidY, centroidZ;
			centroidX = centroidY = centroidZ = 0.0;
			for(int i = 0; i < m_pPoints->getDimensions().x; i++) {
				for(int j = 0; j < m_pPoints->getDimensions().y; j++) {
					(*m_pPoints)(i, j).x = position.x + i*tilingSpacing;
					(*m_pPoints)(i, j).y = position.y;
					(*m_pPoints)(i, j).z = position.z + j*tilingSpacing;
					centroidX += (*m_pPoints)(i, j).x;
					centroidY += (*m_pPoints)(i, j).y;
					centroidZ += (*m_pPoints)(i, j).z;
				}
			}
			m_totalNumPoints = m_pPoints->getDimensions().x*m_pPoints->getDimensions().y;
			m_centroid.x = static_cast<Scalar>(centroidX/m_totalNumPoints);
			m_centroid.y = static_cast<Scalar>(centroidY/m_totalNumPoints);
			m_centroid.z = static_cast<Scalar>(centroidZ/m_totalNumPoints);

			Vector3 zAngleVec = -m_upVector.cross(Vector3(1, 0, 0));
			Vector3 xAngleVec = m_upVector.cross(Vector3(0, 0, 1));
			Scalar angleX_up = angle3D(Vector3(1, 0, 0), xAngleVec);
			Scalar angleZ_up = angle3D(Vector3(0, 0, 1), zAngleVec);
			Quaternion X_up(Vector3(1, 0, 0), RadToDegree(angleZ_up));
			Quaternion Z_up(Vector3(0, 0, 1), RadToDegree(-angleX_up));

			for(int i = 0; i < m_pPoints->getDimensions().x; i++) {
				for(int j = 0; j < m_pPoints->getDimensions().y; j++) {
					Vector3 currPoint = (*m_pPoints)(i, j);
					currPoint -= m_centroid;
					X_up.rotate(&currPoint);
					Z_up.rotate(&currPoint);
					(*m_pPoints)(i, j) = currPoint + m_centroid;
				}
			}
			*m_pInitialPoints = *m_pPoints;
		}
		#pragma endregion Constructors
		

	}
}