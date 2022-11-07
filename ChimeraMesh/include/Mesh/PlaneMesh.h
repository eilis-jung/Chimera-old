#ifndef __CHIMERA_PLANE_MESH_H_
#define __CHIMERA_PLANE_MESH_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraMath.h"

namespace Chimera {
	using namespace Math;

	namespace Data {

		class PlaneMesh {

			#pragma region ClassMembers
			Vector2 m_planeSize;
			Vector3 m_upVector;
			Vector3 m_centroid;
			int m_totalNumPoints;
			dimensions_t m_tilingAmount;
			
			/** Points */
			Array2D<Vector3> *m_pPoints;
			/** Points */
			Array2D<Vector3> *m_pInitialPoints;
			#pragma region ClassMembers

		public:
			#pragma region Constructors
			PlaneMesh(const Vector3 & position, const Vector3 &upVec, const Vector2 &planeSize, Scalar tilingSpacing);
			#pragma endregion Constructors

			#pragma region AccessFunctions
			const Array2D<Vector3> &getPoints() const {
				return *m_pPoints;
			}
			Array2D<Vector3> * getPointsPtr() {
				return m_pPoints;
			}
			const Array2D<Vector3> &getInitialPoints() const {
				return *m_pInitialPoints;
			}
			Array2D<Vector3> * getInitialPointsPtr() {
				return m_pInitialPoints;
			}
			const Vector3 & getOrigin() const {
				return (*m_pPoints)(0, 0);
			}
			const Vector3 & getNormal() const {
				return m_upVector;
			}
			#pragma endregion AccessFunctions
		};
	}
}

#endif