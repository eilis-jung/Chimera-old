#include "Kernels/SPHKernel.h"

namespace Chimera {

	namespace Advection {
		
		template <class VectorType>
		Scalar SPHKernel<VectorType>::calculateKernel(const VectorType &position, const VectorType & destPosition, Scalar r) {
			return (15.0f / (PI * pow(m_kernelSize, 6)))*pow(m_kernelSize*m_kernelSize - r*r, 3);
		}

		template class SPHKernel<Vector2>;
		template class SPHKernel<Vector3>;
	}
}