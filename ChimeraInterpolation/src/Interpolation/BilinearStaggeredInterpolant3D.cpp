#include "Interpolation/BilinearStaggeredInterpolant3D.h"

namespace Chimera {
	namespace Interpolation {
		
		#pragma region Constructors
		template<class valueType>
		BilinearStaggeredInterpolant3D<valueType>::BilinearStaggeredInterpolant3D(const Array3D<valueType>& values, 
																					Scalar gridDx)
																					: Interpolant(values) {
			m_dx = gridDx;
		}
		#pragma endregion

		#pragma region Functionalities
		template<>
		/** Scalar-based staggered interpolation in regular grids: basically means that scalar-field values are located 
		  * cell-centered positions: offset position vector by (-0.5, -0.5) */
		Scalar BilinearStaggeredInterpolant3D<Scalar>::interpolate(const Vector3 &position) {
			Vector3 gridSpacePosition = position / m_dx;
			Scalar interpolatedValue;

			int i = static_cast <int> (floor(gridSpacePosition.x - 0.5f));
			int j = static_cast <int> (floor(gridSpacePosition.y - 0.5f));
			int k = static_cast <int> (floor(gridSpacePosition.z - 0.5f));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);
			k = clamp(k, 0, m_values.getDimensions().z - 1);

			Vector3 x1Position(i + 0.5f, j + 0.5f, k + 0.5f);
			Vector3 x2Position(i + 1.5f, j + 1.5f, k + 1.5f);

			int nextI = clamp(i + 1, 0, m_values.getDimensions().x - 1);
			int nextJ = clamp(j + 1, 0, m_values.getDimensions().y - 1);
			int nextK = clamp(k + 1, 0, m_values.getDimensions().z - 1);

			Scalar v1 =	m_values(i, j, k) * (x2Position.x - gridSpacePosition.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(nextI, j, k) * (gridSpacePosition.x - x1Position.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i, nextJ, k) * (x2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - x1Position.y) +
						m_values(nextI, nextJ, k) * (gridSpacePosition.x - x1Position.x) * (gridSpacePosition.y - x1Position.y);

			Scalar v2 = m_values(i, j, nextK) * (x2Position.x - gridSpacePosition.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(nextI, j, nextK) * (gridSpacePosition.x - x1Position.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i, nextJ, nextK) * (x2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - x1Position.y) +
						m_values(nextI, nextJ, nextK) * (gridSpacePosition.x - x1Position.x) * (gridSpacePosition.y - x1Position.y);

			Scalar alpha = gridSpacePosition.z - x1Position.z;
			return v1*(1 - alpha) + v2*alpha;
		}
		
		/** Vector-based staggered interpolation in regular grids */
		template<>
		Vector3 BilinearStaggeredInterpolant3D<Vector3>::interpolate(const Vector3 &position) {
			Vector3 gridPosition = position / m_dx;
			Vector3 interpolatedVelocity;
			int i, j, k;

			//X component interpolation
			{
				i = static_cast <int> (floor(gridPosition.x));
				j = static_cast <int> (floor(gridPosition.y - 0.5f));
				k = static_cast <int> (floor(gridPosition.z - 0.5f));

				i = clamp(i, 0, m_values.getDimensions().x - 1);
				j = clamp(j, 0, m_values.getDimensions().y - 1);
				k = clamp(k, 0, m_values.getDimensions().z - 1);

				Vector3 x1Position = Vector3(static_cast <Scalar>(i), j + 0.5f, k + 0.5f);
				Vector3 x2Position = x1Position + Vector3(1, 1, 1);
				interpolateComponent(i, j, k, gridPosition, x1Position, x2Position, xComponent, interpolatedVelocity);
			}

			//Y component interpolation
			{
				i = static_cast <int> (floor(gridPosition.x - 0.5F));
				j = static_cast <int> (floor(gridPosition.y));
				k = static_cast <int> (floor(gridPosition.z - 0.5f));
				i = clamp(i, 0, m_values.getDimensions().x - 1);
				j = clamp(j, 0, m_values.getDimensions().y - 1);
				k = clamp(k, 0, m_values.getDimensions().z - 1);

				Vector3 y1Position = Vector3(i + 0.5f, static_cast <Scalar>(j), k + 0.5f);
				Vector3 y2Position = y1Position + Vector3(1, 1, 1);

				interpolateComponent(i, j, k, gridPosition, y1Position, y2Position, yComponent, interpolatedVelocity);
			}

			//Z component interpolation
			{
				i = static_cast <int> (floor(gridPosition.x - 0.5F));
				j = static_cast <int> (floor(gridPosition.y - 0.5f));
				k = static_cast <int> (floor(gridPosition.z));
				i = clamp(i, 0, m_values.getDimensions().x - 1);
				j = clamp(j, 0, m_values.getDimensions().y - 1);
				k = clamp(k, 0, m_values.getDimensions().z - 1);

				Vector3 z1Position = Vector3(i + 0.5f, j + 0.5f, static_cast <Scalar>(k));
				Vector3 z2Position = z1Position + Vector3(1, 1, 1);
				interpolateComponent(i, j, k, gridPosition, z1Position, z2Position, zComponent, interpolatedVelocity);
			}
			
			return interpolatedVelocity;
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template<>
		Scalar BilinearStaggeredInterpolant3D<Scalar>::interpolateFaceVelocity(int i, int j, int k, velocityComponent_t velocityComponent) {
			return 0;
		}

		template<>
		Vector3 BilinearStaggeredInterpolant3D<Vector3>::interpolateFaceVelocity(int i, int j, int k, velocityComponent_t velocityComponent) {
			Vector3 faceVelocity;
			if (velocityComponent == xComponent) {
				int nextJ = clamp(j + 1, 0, m_values.getDimensions().y - 1);
				int nextK = clamp(k + 1, 0, m_values.getDimensions().z - 1);
				int prevI = clamp(i - 1, 0, m_values.getDimensions().x - 1);
				faceVelocity.x = m_values(i, j, k).x;
				faceVelocity.y = 0.25f*(m_values(i, j, k).y + m_values(prevI, j, k).y + m_values(i, nextJ, k).y + m_values(prevI, nextJ, k).y);
				faceVelocity.z = 0.25f*(m_values(i, j, k).z + m_values(prevI, j, k).z + m_values(i, j, nextK).z + m_values(prevI, j, nextK).z);
			}
			else if (velocityComponent == yComponent) {
				int nextI = clamp(i + 1, 0, m_values.getDimensions().x - 1);
				int nextK = clamp(k + 1, 0, m_values.getDimensions().z - 1);
				int prevJ = clamp(j - 1, 0, m_values.getDimensions().y - 1);
				faceVelocity.y = m_values(i, j, k).y;
				faceVelocity.x = 0.25f*(m_values(i, j, k).x + m_values(i, prevJ, k).x + m_values(nextI, j, k).x + m_values(nextI, prevJ, k).x);
				faceVelocity.z = 0.25f*(m_values(i, j, k).z + m_values(i, prevJ, k).z + m_values(i, j, nextK).z + m_values(i, prevJ, nextK).z);
			}
			else if (velocityComponent == zComponent) {
				int nextI = clamp(i + 1, 0, m_values.getDimensions().x - 1);
				int nextJ = clamp(j + 1, 0, m_values.getDimensions().y - 1);
				int prevK = clamp(k - 1, 0, m_values.getDimensions().z - 1);
				faceVelocity.z = m_values(i, j, k).z;
				faceVelocity.x = 0.25f*(m_values(i, j, k).x + m_values(i, j, prevK).x + m_values(nextI, i, k).x + m_values(nextI, j, prevK).x);
				faceVelocity.y = 0.25f*(m_values(i, j, k).y + m_values(i, j, prevK).y + m_values(i, nextJ, k).y + m_values(i, nextJ, prevK).y);
			}

			return faceVelocity;
		}

		template<>
		void BilinearStaggeredInterpolant3D<Scalar>::interpolateComponent(int i, int j, int k, const Vector3 &position, const Vector3 &cellInitialPosition, const Vector3 & cellFinalPosition, velocityComponent_t velocityComponent, Vector3 &interpolatedVelocity) {
		}

		template<>
		void BilinearStaggeredInterpolant3D<Vector3>::interpolateComponent(int i, int j, int k, const Vector3 &position, const Vector3 &cellInitialPosition, const Vector3 & cellFinalPosition, velocityComponent_t velocityComponent, Vector3 &interpolatedVelocity) {
			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1 && m_values.getDimensions().z - 1) //Last possible cell
				interpolatedVelocity[velocityComponent] = m_values(i, j, k)[velocityComponent];
			else if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1) {
				Scalar alpha = position.z - cellInitialPosition.z;
				interpolatedVelocity[velocityComponent] = m_values(i, j, k)[velocityComponent]*(1 - alpha) + m_values(i, j, k + 1)[velocityComponent]*alpha;
			}
			else if (i == m_values.getDimensions().x - 1 && k == m_values.getDimensions().z - 1) {
				interpolatedVelocity[velocityComponent] = m_values(i, j, k)[velocityComponent] * (cellFinalPosition.y - position.y) +
														  m_values(i, j + 1, k)[velocityComponent] * (position.y - cellInitialPosition.y);
			}
			else if (j == m_values.getDimensions().y - 1 && k == m_values.getDimensions().z - 1) {
				interpolatedVelocity[velocityComponent] = m_values(i, j, k)[velocityComponent] * (cellFinalPosition.x - position.x) +
														  m_values(i + 1, j, k)[velocityComponent] * (position.x - cellInitialPosition.x);
			}
			else if (i == m_values.getDimensions().x - 1) {
				Scalar v1 = m_values(i, j, k)[velocityComponent] * (cellFinalPosition.y - position.y) +
						    m_values(i, j + 1, k)[velocityComponent] * (position.y - cellInitialPosition.y);

				Scalar v2 = m_values(i, j, k + 1)[velocityComponent] * (cellFinalPosition.y - position.y) +
							m_values(i, j + 1, k + 1)[velocityComponent] * (position.y - cellInitialPosition.y);

				Scalar alpha = position.z - cellInitialPosition.z;
				interpolatedVelocity[velocityComponent] = v1*(1 - alpha) + v2*alpha;
			}
			else if (j == m_values.getDimensions().y - 1) {
				Scalar v1 = m_values(i, j, k)[velocityComponent] * (cellFinalPosition.x - position.x) +
							m_values(i + 1, j, k)[velocityComponent] * (position.x - cellInitialPosition.x);
				Scalar v2 = m_values(i, j, k + 1)[velocityComponent] * (cellFinalPosition.x - position.x) +
							m_values(i + 1, j, k + 1)[velocityComponent] * (position.x - cellInitialPosition.x);

				Scalar alpha = position.z - cellInitialPosition.z;
				interpolatedVelocity[velocityComponent] = v1*(1 - alpha) + v2*alpha;
			}
			else if (k == m_values.getDimensions().z - 1) {
				interpolatedVelocity[velocityComponent] = m_values(i, j, k)[velocityComponent] * (cellFinalPosition.x - position.x) * (cellFinalPosition.y - position.y) +
														  m_values(i + 1, j, k)[velocityComponent] * (position.x - cellInitialPosition.x) * (cellFinalPosition.y - position.y) +
														  m_values(i, j + 1, k)[velocityComponent] * (cellFinalPosition.x - position.x) * (position.y - cellInitialPosition.y) +
														  m_values(i + 1, j + 1, k)[velocityComponent] * (position.x - cellInitialPosition.x) * (position.y - cellInitialPosition.y);
			}
			else {
				Scalar v1 = m_values(i, j, k)[velocityComponent] * (cellFinalPosition.x - position.x) * (cellFinalPosition.y - position.y) +
							m_values(i + 1, j, k)[velocityComponent] * (position.x - cellInitialPosition.x) * (cellFinalPosition.y - position.y) +
							m_values(i, j + 1, k)[velocityComponent] * (cellFinalPosition.x - position.x) * (position.y - cellInitialPosition.y) +
							m_values(i + 1, j + 1, k)[velocityComponent] * (position.x - cellInitialPosition.x) * (position.y - cellInitialPosition.y);
				Scalar v2 = m_values(i, j, k + 1)[velocityComponent] * (cellFinalPosition.x - position.x) * (cellFinalPosition.y - position.y) +
							m_values(i + 1, j, k + 1)[velocityComponent] * (position.x - cellInitialPosition.x) * (cellFinalPosition.y - position.y) +
							m_values(i, j + 1, k + 1)[velocityComponent] * (cellFinalPosition.x - position.x) * (position.y - cellInitialPosition.y) +
							m_values(i + 1, j + 1, k + 1)[velocityComponent] * (position.x - cellInitialPosition.x) * (position.y - cellInitialPosition.y);

				Scalar alpha = position.z - cellInitialPosition.z;
				interpolatedVelocity[velocityComponent] = v1*(1 - alpha) + v2*alpha;
			}
		}
		#pragma endregion
		/** Template linker trickerino for templated classes in CPP*/
		template class BilinearStaggeredInterpolant3D<Scalar>;
		template class BilinearStaggeredInterpolant3D<Vector3>;
	}
}