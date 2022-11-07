#include "Interpolation/BilinearStaggeredInterpolant2D.h"

namespace Chimera {
	namespace Interpolation {
		
		#pragma region Constructors
		template<class valueType>
		BilinearStaggeredInterpolant2D<valueType>::BilinearStaggeredInterpolant2D(const Array2D<valueType>& values, 
																					Scalar gridDx)
																					: Interpolant(values) {
			m_dx = gridDx;
		}
		#pragma endregion

		#pragma region Functionalities
		template<>
		/** Scalar-based staggered interpolation in regular grids: basically means that scalar-field values are located 
		  * cell-centered positions: offset position vector by (-0.5, -0.5) */
		Scalar BilinearStaggeredInterpolant2D<Scalar>::interpolate(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			Scalar interpolatedValue;

			int i = static_cast <int> (floor(gridSpacePosition.x - 0.5f));
			int j = static_cast <int> (floor(gridSpacePosition.y - 0.5f));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			Vector2 x1Position(i + 0.5f, j + 0.5f);
			Vector2 x2Position(i + 1.5f, j + 1.5f);

			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1) //Last possible cell
				return m_values(i, j);
			else if (i == m_values.getDimensions().x - 1) //Right boundary
				return m_values(i, j) * (x2Position.y - gridSpacePosition.y) + m_values(i, j + 1) * (gridSpacePosition.y - x1Position.y);
			else if (j == m_values.getDimensions().y - 1) //Top boundary
				return m_values(i, j) * (x2Position.x - gridSpacePosition.x) + m_values(i + 1, j) * (gridSpacePosition.x - x1Position.x);
			else //Normal cell
				return	m_values(i, j) * (x2Position.x - gridSpacePosition.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i + 1, j) * (gridSpacePosition.x - x1Position.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i, j + 1) * (x2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - x1Position.y) +
						m_values(i + 1, j + 1) * (gridSpacePosition.x - x1Position.x) * (gridSpacePosition.y - x1Position.y);
		}
		
		/** Vector-based staggered interpolation in regular grids */
		template<>
		Vector2 BilinearStaggeredInterpolant2D<Vector2>::interpolate(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			Vector2 interpolatedVelocity;

			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y - 0.5f));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);


			/** Interpolate for X component first*/
			Vector2 x1Position(static_cast <Scalar>(i), j + 0.5f);
			Vector2 x2Position(i + 1.0f, j + 1.5f);

			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1)
				interpolatedVelocity.x = m_values(i, j).x;
			else if (i == m_values.getDimensions().x - 1)
				interpolatedVelocity.x = m_values(i, j).x * (x2Position.y - gridSpacePosition.y) + m_values(i, j + 1).x * (gridSpacePosition.y - x1Position.y);
			else if (j == m_values.getDimensions().y - 1)
				interpolatedVelocity.x = m_values(i, j).x * (x2Position.x - gridSpacePosition.x) + m_values(i + 1, j).x * (gridSpacePosition.x - x1Position.x);
			else
				interpolatedVelocity.x = m_values(i, j).x * (x2Position.x - gridSpacePosition.x) * (x2Position.y - gridSpacePosition.y) +
										 m_values(i + 1, j).x * (gridSpacePosition.x - x1Position.x) * (x2Position.y - gridSpacePosition.y) +
										 m_values(i, j + 1).x * (x2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - x1Position.y) +
										 m_values(i + 1, j + 1).x * (gridSpacePosition.x - x1Position.x) * (gridSpacePosition.y - x1Position.y);

			i = static_cast <int> (floor(gridSpacePosition.x - 0.5f));
			j = static_cast <int> (floor(gridSpacePosition.y));

			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			

			/** Interpolate for Y component later*/
			Vector2 y1Position, y2Position;
			y1Position = Vector2(i + 0.5f, static_cast<Scalar>(j));
			y2Position = Vector2(i + 1.5f, j + 1.0f);

			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1)
				interpolatedVelocity.y = m_values(i, j).y;
			else if (i == m_values.getDimensions().x - 1)
				interpolatedVelocity.y = m_values(i, j).y * (x2Position.y - gridSpacePosition.y) + m_values(i, j + 1).y * (gridSpacePosition.y - x1Position.y);
			else if (j == m_values.getDimensions().y - 1)
				interpolatedVelocity.y = m_values(i, j).y * (x2Position.x - gridSpacePosition.x) + m_values(i + 1, j).y * (gridSpacePosition.x - x1Position.x);
			else 
				interpolatedVelocity.y = m_values(i, j).y * (y2Position.x - gridSpacePosition.x) * (y2Position.y - gridSpacePosition.y) +
					m_values(i + 1, j).y * (gridSpacePosition.x - y1Position.x) * (y2Position.y - gridSpacePosition.y) +
					m_values(i, j + 1).y * (y2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - y1Position.y) +
					m_values(i + 1, j + 1).y * (gridSpacePosition.x - y1Position.x) * (gridSpacePosition.y - y1Position.y);
			
			return interpolatedVelocity;
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template<>
		Scalar BilinearStaggeredInterpolant2D<Scalar>::interpolateFaceVelocity(int i, int j, velocityComponent_t velocityComponent) {
			return 0;
		}

		template<>
		Vector2 BilinearStaggeredInterpolant2D<Vector2>::interpolateFaceVelocity(int i, int j, velocityComponent_t velocityComponent) {
			Vector2 faceVelocity;
			if (velocityComponent == xComponent) {
				int nextJ = j + 1;
				if (j == m_values.getDimensions().y - 1) {
					nextJ = j;
				}
				faceVelocity.x = m_values(i, j).x;
				faceVelocity.y = 0.25f*(m_values(i, j).y + m_values(i - 1, j).y + m_values(i, nextJ).y + m_values(i - 1, nextJ).y);
			}
			else if (velocityComponent == yComponent) {
				faceVelocity.y = m_values(i, j).y;
				int nextI = i + 1;
				if (i == m_values.getDimensions().x - 1) {
					nextI = i;
				}
				faceVelocity.y = m_values(i, j).y;
				faceVelocity.x = 0.25f*(m_values(i, j).x + m_values(nextI, j).x + m_values(i, j - 1).x + m_values(nextI, j - 1).x);
			}

			return faceVelocity;
		}
		#pragma endregion
		/** Template linker trickerino for templated classes in CPP*/
		template class BilinearStaggeredInterpolant2D<Scalar>;
		template class BilinearStaggeredInterpolant2D<Vector2>;
	}
}