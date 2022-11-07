#include "Kernels/InverseDistanceKernel.h"

namespace Chimera {

	namespace Advection {
		
		template <class VectorType>
		Scalar InverseDistanceKernel<VectorType>::calculateKernel(const VectorType &position, const VectorType & destPosition, Scalar r) {
			Scalar dist2 = (position - destPosition).length();
			dist2 *= dist2;
			return (1.0f/(dist2 + 1e-6));
		}

		template class InverseDistanceKernel<Vector2>;
		template class InverseDistanceKernel<Vector3>;
	}
}