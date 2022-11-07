#ifndef __CHIMERA_PLANE_H_
#define __CHIMERA_PLANE_H_
#pragma once

/** Foundation */
#include "ChimeraCore.h"
#include "Primitives/Object3D.h"
#include "ChimeraData.h"

namespace Chimera {
	using namespace Data;

	namespace Rendering {

		class Plane : public Object3D {

			#pragma region ClassMembers
			PlaneMesh m_mesh;

			/** Drawing */
			bool m_drawPoints;
			#pragma region ClassMembers

		public:
			#pragma region Constructors
			Plane(const Vector3 & position, const Vector3 &upVec, const Vector2 &planeSize, Scalar tilingSpacing);
			#pragma endregion Constructors

			#pragma region Functionalities
			void draw();
			virtual void update(Scalar dt);
			#pragma endregion Functionalities

			#pragma region AccessFunctions
			FORCE_INLINE void drawPoints(bool gDrawPoints) {
				m_drawPoints = gDrawPoints;
			}

			FORCE_INLINE const PlaneMesh &getMesh() const {
				return m_mesh;
			}
			FORCE_INLINE PlaneMesh * getMeshPtr() {
				return &m_mesh;
			}
			#pragma endregion AccessFunctions
		};
	}
}

#endif