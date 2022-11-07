#include "Kernels/BilinearKernel.h"

namespace Chimera {
	
	namespace Advection {
		template <>
		Scalar BilinearKernel<Vector2>::calculateKernel(const Vector2 &position, const Vector2 & destPosition, Scalar r) {
			Vector2 relativeFractions;
			relativeFractions.x = roundClamp<Scalar>(position.x - destPosition.x, 0.f, 1.f);
			relativeFractions.y = roundClamp<Scalar>(position.y - destPosition.y, 0.f, 1.f);
			return ((1 - relativeFractions.x)*(1 - relativeFractions.y));
		}

		template <>
		Scalar BilinearKernel<Vector3>::calculateKernel(const Vector3 &position, const Vector3 & destPosition, Scalar r) {
			Vector3 relativeFractions;
			relativeFractions.x = roundClamp<Scalar>(position.x - destPosition.x, 0.f, 1.f);
			relativeFractions.y = roundClamp<Scalar>(position.y - destPosition.y, 0.f, 1.f);
			relativeFractions.z = roundClamp<Scalar>(position.z - destPosition.z, 0.f, 1.f);
			return (1 - relativeFractions.x)*(1 - relativeFractions.y)*(1 - relativeFractions.z);
		}
	}
	
}