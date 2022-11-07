#include "Integration/ForwardEulerIntegrator.h"
#include "Grids/GridData2D.h"

namespace Chimera {


	
	namespace Advection {

		#pragma region Functionalities
		template<class VectorType, template <class> class ArrayType>
		void ForwardEulerIntegrator<VectorType, ArrayType>::integratePosition(uint particleID, Scalar dt) {
			VectorType &particlePosition = m_pParticlesData->getPositions()[particleID];
			VectorType interpVel = m_pInterpolant->interpolate(particlePosition);

			particlePosition = particlePosition + (interpVel)*dt;
		}

		template<class VectorType, template <class> class ArrayType>
		VectorType ForwardEulerIntegrator<VectorType, ArrayType>::integrate(const VectorType &position, const VectorType &velocity, Scalar dt, Interpolant<VectorType, ArrayType, VectorType> *pCustomInterpolant = nullptr) {
			VectorType integratedPosition = position + (velocity)*dt;

			clampPosition(integratedPosition);			
			if (checkCollision(position, integratedPosition))
				return position;

			return integratedPosition;
		}
		#pragma endregion

		/** Template linker trickerino for templated classes in CPP*/
		template class ForwardEulerIntegrator<Vector2, Array2D>;
		template class ForwardEulerIntegrator<Vector3, Array3D>;
	}


	
}