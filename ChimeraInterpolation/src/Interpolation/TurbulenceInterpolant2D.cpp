#include "Interpolation/TurbulenceInterpolant2D.h"

namespace Chimera {
	namespace Interpolation {
		
		#pragma region Constructors
		template<class valueType>
		TurbulentInterpolant2D<valueType>::TurbulentInterpolant2D(const Array2D<valueType> &values, Array2D<Scalar> *pStreamfunctionValues, 
																	Scalar coarseGridDx, Scalar fineGridDx) : Interpolant<valueType, Array2D, Vector2>() {
			m_pBilinearInterpolant = new BilinearStaggeredInterpolant2D<valueType>(values, coarseGridDx);
			m_pCubicInterpolant = new CubicStreamfunctionInterpolant2D<Vector2>(pStreamfunctionValues, fineGridDx);
			m_dx = coarseGridDx;
			m_fineGridDx = fineGridDx;
		}
		#pragma endregion

		#pragma region Functionalities
		template<>
		Scalar TurbulentInterpolant2D<Scalar>::interpolate(const Vector2 &position) {
		
			return 0;
		}
		/** Vector-based staggered interpolation in regular grids */
		template<>
		Vector2 TurbulentInterpolant2D<Vector2>::interpolate(const Vector2 &position) {
			Vector2 velocity = m_pBilinearInterpolant->interpolate(position);
			Vector2 velocityIncrement = m_pCubicInterpolant->interpolate(position);
			//velocityIncrement.x = velocityIncrement.y = 0;
			return velocity + velocityIncrement;
		}
		#pragma endregion

		/** Template linker trickerino for templated classes in CPP*/
		template class TurbulentInterpolant2D<Scalar>;
		template class TurbulentInterpolant2D<Vector2>;
	}
}