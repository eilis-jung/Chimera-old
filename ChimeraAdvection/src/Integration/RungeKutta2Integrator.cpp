#include "Integration/RungeKutta2Integrator.h"

namespace Chimera {


	
	namespace Advection {

		#pragma region Functionalities
		template<class VectorType, template <class> class ArrayType>
		void RungeKutta2Integrator<VectorType, ArrayType>::integratePosition(uint particleID, Scalar dt) {
			VectorType &particlePosition = m_pParticlesData->getPositions()[particleID];
			VectorType interpVel = m_pInterpolant->interpolate(particlePosition);

			/** Midpoint step */
			VectorType initialPosition = particlePosition;
			VectorType tempPos = particlePosition + interpVel*dt*0.5;

			clampPosition(tempPos);

			if (checkCollision(particlePosition, tempPos)) {
				return;
			}

			interpVel = m_pInterpolant->interpolate(tempPos);
			particlePosition = particlePosition + interpVel*dt;

			clampPosition(particlePosition);

			if (checkCollision(tempPos, particlePosition)) {
				particlePosition = tempPos;
				return;
			}
		}

		template<class VectorType, template <class> class ArrayType>
		VectorType RungeKutta2Integrator<VectorType, ArrayType>::integrate(const VectorType &position, const VectorType &velocity, 
																			Scalar dt, Interpolant<VectorType, ArrayType, VectorType> *pCustomInterpolant) {
			/** Midpoint step */
			VectorType tempPos = position + velocity*dt*0.5;

			clampPosition(tempPos);

			if (checkCollision(position, tempPos)) {
				return position;
			}

			VectorType interpVel;
			if (pCustomInterpolant) {
				interpVel = pCustomInterpolant->interpolate(tempPos);
				tempPos = position + interpVel*dt;
			}
			else {
				interpVel = m_pInterpolant->interpolate(tempPos);
				tempPos = position + interpVel*dt;
			}

			clampPosition(tempPos);

			if (checkCollision(position, tempPos)) {
				return position;
			}

			return tempPos;
		}
		#pragma endregion

		/** Template linker trickerino for templated classes in CPP*/
		template class RungeKutta2Integrator<Vector2, Array2D>;
		template class RungeKutta2Integrator<Vector3, Array3D>;
		
	}

	
}