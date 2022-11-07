#include "Interpolation/BilinearAPICStaggeredInterpolant2D.h"

namespace Chimera {

	namespace Interpolation {
		#pragma region Functionalities
		/** Velocity-based angular momentum staggered interpolation */
		Vector2 BilinearAPICStaggeredInterpolant2D::angualrMomentemInterpolate(const Vector2 &position) {
			Vector2 gridSpacePosition(position / m_dx);
			Vector2 angularMomentum;

			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y - 0.5f));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			/** Interpolate for X component first*/
			Vector2 x00(i, j + 0.5f);
			Vector2 x01(x00.x, x00.y + 1.0f);
			Vector2 x10(x00.x + 1.0f, x00.y);
			Vector2 x11(x00.x + 1.0f, x00.y + 1.0f);

			Vector2 u00 = (x00 - gridSpacePosition) * m_dx;
			Vector2 u01 = (x01 - gridSpacePosition) * m_dx;
			Vector2 u10 = (x10 - gridSpacePosition) * m_dx;
			Vector2 u11 = (x11 - gridSpacePosition) * m_dx;

			Vector2 v00 = interpolateFaceVelocity(i, j, xComponent); 
			Vector2 v10 = interpolateFaceVelocity(i + 1, j, xComponent);
			Vector2 v01 = interpolateFaceVelocity(i, j + 1, xComponent);
			Vector2 v11 = interpolateFaceVelocity(i + 1, j + 1, xComponent);

			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1)
				angularMomentum.x = u00.cross(v00);
			else if (i == m_values.getDimensions().x - 1)
				angularMomentum.x = (x11.y - gridSpacePosition.y) * u00.cross(v00) + (gridSpacePosition.y - x00.y) * u01.cross(v01);
			else if (j == m_values.getDimensions().y - 1)
				angularMomentum.x = (x11.x - gridSpacePosition.x) * u00.cross(v00) + (gridSpacePosition.x - x00.x) * u10.cross(v10);
			else
				angularMomentum.x = (x11.x - gridSpacePosition.x) * (x11.y - gridSpacePosition.y) * u00.cross(v00) +
									(gridSpacePosition.x - x00.x) * (x11.y - gridSpacePosition.y) * u10.cross(v10) +
									(x11.x - gridSpacePosition.x) * (gridSpacePosition.y - x00.y) * u01.cross(v01) +
									(gridSpacePosition.x - x00.x) * (gridSpacePosition.y - x00.y) * u11.cross(v11);


			/** Interpolate for Y component later*/
			i = static_cast <int> (floor(gridSpacePosition.x - 0.5f));
			j = static_cast <int> (floor(gridSpacePosition.y));

			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			x00 = Vector2(i + 0.5f, j);
			x01 = Vector2(x00.x, x00.y + 1.0f);
			x10 = Vector2(x00.x + 1.0f, x00.y);
			x11 = Vector2(x00.x + 1.0f, x00.y + 1.0f);

			u00 = (x00 - gridSpacePosition) * m_dx;
			u01 = (x01 - gridSpacePosition) * m_dx;
			u10 = (x10 - gridSpacePosition) * m_dx;
			u11 = (x11 - gridSpacePosition) * m_dx;
			
			v00 = interpolateFaceVelocity(i, j, yComponent);
			v10 = interpolateFaceVelocity(i + 1, j, yComponent);
			v01 = interpolateFaceVelocity(i, j + 1, yComponent);
			v11 = interpolateFaceVelocity(i + 1, j + 1, yComponent);
			
			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1)
				angularMomentum.y = u00.cross(v00);
			else if (i == m_values.getDimensions().x - 1)
				angularMomentum.y = (x11.y - gridSpacePosition.y) * u00.cross(v00) + (gridSpacePosition.y - x00.y) * u01.cross(v01);
			else if (j == m_values.getDimensions().y - 1)
				angularMomentum.y = (x11.x - gridSpacePosition.x) * u00.cross(v00) + (gridSpacePosition.x - x00.x) * u10.cross(v10);
			else
				angularMomentum.y = (x11.x - gridSpacePosition.x) * (x11.y - gridSpacePosition.y) * u00.cross(v00) +
									(gridSpacePosition.x - x00.x) * (x11.y - gridSpacePosition.y) * u10.cross(v10) +
									(x11.x - gridSpacePosition.x) * (gridSpacePosition.y - x00.y) * u01.cross(v01) +
									(gridSpacePosition.x - x00.x) * (gridSpacePosition.y - x00.y) * u11.cross(v11);

			return angularMomentum;
		}

		/** Velocity-based inverse of inertia tensor staggered interpolation */
		Matrix3x3 BilinearAPICStaggeredInterpolant2D::inertiaTensorInverseInterpolate(const Vector2 &position, bool yAxis) {
			Matrix3x3 inertiaTensor;

			Vector2 gridSpacePosition(position / m_dx);

			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y - 0.5f));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			Vector2 x00(i, j + 0.5f);
			Vector2 x01(x00.x, x00.y + 1.0f);
			Vector2 x10(x00.x + 1.0f, x00.y);
			Vector2 x11(x00.x + 1.0f, x00.y + 1.0f);

			if (yAxis) {
				i = static_cast <int> (floor(gridSpacePosition.x - 0.5f));
				j = static_cast <int> (floor(gridSpacePosition.y));

				i = clamp(i, 0, m_values.getDimensions().x - 1);
				j = clamp(j, 0, m_values.getDimensions().y - 1);

				x00 = Vector2(i + 0.5f, j);
				x01 = Vector2(x00.x, x00.y + 1.0f);
				x10 = Vector2(x00.x + 1.0f, x00.y);
				x11 = Vector2(x00.x + 1.0f, x00.y + 1.0f);
			}

			Vector2 u00 = (x00 - gridSpacePosition) * m_dx;
			Vector2 u01 = (x01 - gridSpacePosition) * m_dx;
			Vector2 u10 = (x10 - gridSpacePosition) * m_dx;
			Vector2 u11 = (x11 - gridSpacePosition) * m_dx;

			Matrix3x3 k00(
				Vector3(u00.y*u00.y, -u00.x*u00.y, 0), // c0
				Vector3(-u00.x*u00.y, u00.x*u00.x, 0), // c1
				Vector3(0, 0, u00.x*u00.x + u00.y*u00.y)); // c2
			Matrix3x3 k01(
				Vector3(u01.y*u01.y, -u01.x*u01.y, 0),
				Vector3(-u01.x*u01.y, u01.x*u01.x, 0),
				Vector3(0, 0, u01.x*u01.x + u01.y*u01.y));
			Matrix3x3 k10(
				Vector3(u10.y*u10.y, -u10.x*u10.y, 0),
				Vector3(-u10.x*u10.y, u10.x*u10.x, 0),
				Vector3(0, 0, u10.x*u10.x + u10.y*u10.y));
			Matrix3x3 k11(
				Vector3(u11.y*u11.y, -u11.x*u11.y, 0),
				Vector3(-u11.x*u11.y, u11.x*u11.x, 0),
				Vector3(0, 0, u11.x*u11.x + u11.y*u11.y));

			
				inertiaTensor = (x11.x - gridSpacePosition.x) * (x11.y - gridSpacePosition.y) * k00 +
								(gridSpacePosition.x - x00.x) * (x11.y - gridSpacePosition.y) * k10 +
								(x11.x - gridSpacePosition.x) * (gridSpacePosition.y - x00.y) * k01 +
								(gridSpacePosition.x - x00.x) * (gridSpacePosition.y - x00.y) * k11;

			if (inertiaTensor.determinant() < 1e-9f)
				return Matrix3x3();

			inertiaTensor.invert(); 
			return inertiaTensor;
		}

		/** Velocity-based velocity derivative staggered interpolation */
		Matrix2x2 BilinearAPICStaggeredInterpolant2D::velocityDerivativeInterpolate(const Vector2 &position) {
			Matrix2x2 velocityDerivative;

			Vector2 gridSpacePosition(position / m_dx);

			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y - 0.5f));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			/** Interpolate for X component first*/
			Vector2 x00(i, j + 0.5f);
			Vector2 x01(x00.x, x00.y + 1.0f);
			Vector2 x10(x00.x + 1.0f, x00.y);
			Vector2 x11(x00.x + 1.0f, x00.y + 1.0f);

			Vector2 v00 = interpolateFaceVelocity(i, j, xComponent);
			Vector2 v10 = interpolateFaceVelocity(i + 1, j, xComponent);
			Vector2 v01 = interpolateFaceVelocity(i, j + 1, xComponent);
			Vector2 v11 = interpolateFaceVelocity(i + 1, j + 1, xComponent);

			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1)
				velocityDerivative[0].x = 0;
			else if (i == m_values.getDimensions().x - 1)
				velocityDerivative[0].x = 0;
			else if (j == m_values.getDimensions().y - 1)
				velocityDerivative[0].x = -(x11.y - x00.y) * v00.x +
										   (x11.y - x00.y) * v10.x;
			else
				velocityDerivative[0].x = -(x11.y - gridSpacePosition.y) * v00.x +
										   (x11.y - gridSpacePosition.y) * v10.x +
										  -(gridSpacePosition.y - x00.y) * v01.x +
										   (gridSpacePosition.y - x00.y) * v11.x;

			//if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1)
			//	velocityDerivative[0].y = 0;
			//else if (i == m_values.getDimensions().x - 1)
			//	velocityDerivative[0].y = -(x11.x - x00.x) * v00.y +
			//							   (x11.x - x00.x) * v01.y;
			//else if (j == m_values.getDimensions().y - 1)
			//	velocityDerivative[0].y = 0;
			//else
			//	velocityDerivative[0].y = -(x11.x - gridSpacePosition.x) * v00.y +
			//							  -(gridSpacePosition.x - x00.x) * v10.y +
			//							   (x11.x - gridSpacePosition.x) * v01.y +
			//							   (gridSpacePosition.x - x00.x) * v11.y;

			/** Interpolate for Y component later*/
			i = static_cast <int> (floor(gridSpacePosition.x - 0.5f));
			j = static_cast <int> (floor(gridSpacePosition.y));

			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			x00 = Vector2(i + 0.5f, j);
			x01 = Vector2(x00.x, x00.y + 1.0f);
			x10 = Vector2(x00.x + 1.0f, x00.y);
			x11 = Vector2(x00.x + 1.0f, x00.y + 1.0f);

			v00 = interpolateFaceVelocity(i, j, yComponent);
			v10 = interpolateFaceVelocity(i + 1, j, yComponent);
			v01 = interpolateFaceVelocity(i, j + 1, yComponent);
			v11 = interpolateFaceVelocity(i + 1, j + 1, yComponent);
			
			//if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1)
			//	velocityDerivative[1].x = 0;
			//else if (i == m_values.getDimensions().x - 1)
			//	velocityDerivative[1].x = 0;
			//else if (j == m_values.getDimensions().y - 1)
			//	velocityDerivative[1].x = -(x11.y - x00.y) * v00.x +
			//							   (x11.y - x00.y) * v10.x;
			//else
			//	velocityDerivative[1].x = -(x11.y - gridSpacePosition.y) * v00.x +
			//							   (x11.y - gridSpacePosition.y) * v10.x +
			//							  -(gridSpacePosition.y - x00.y) * v01.x +
			//							   (gridSpacePosition.y - x00.y) * v11.x;

			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1)
				velocityDerivative[1].y = 0;
			else if (i == m_values.getDimensions().x - 1)
				velocityDerivative[1].y = -(x11.x - x00.x) * v00.y +
										   (x11.x - x00.x) * v01.y;
			else if (j == m_values.getDimensions().y - 1)
				velocityDerivative[1].y = 0;
			else
				velocityDerivative[1].y = -(x11.x - gridSpacePosition.x) * v00.y +
										  -(gridSpacePosition.x - x00.x) * v10.y +
										   (x11.x - gridSpacePosition.x) * v01.y +
										   (gridSpacePosition.x - x00.x) * v11.y;

			velocityDerivative[0] /= m_dx;
			velocityDerivative[1] /= m_dx;

			return velocityDerivative;
		}
		#pragma endregion
	}
}