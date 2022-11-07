#include "TriangleHalfVolume.h"

namespace Chimera {
	namespace CutCells {

		#pragma region Constructors
		template <class VectorType>
		TriangleHalfVolume<VectorType>::TriangleHalfVolume(HalfVolume<VectorType> *pHalfVolume) {
			m_pHalfVolume = pHalfVolume;

			m_pCGALPolyhedron = new CGALWrapper::CgalPolyhedron();

			CGALWrapper::convertToPoly(m_pHalfVolume, m_pCGALPolyhedron);
			m_pCGALPolyhedron->normalize_border();
			CGALWrapper::triangulatePolyhedron(m_pCGALPolyhedron);

			CGALWrapper::Conversion::polyhedron3ToHalfFaces(m_pCGALPolyhedron, m_pHalfVolume->getVerticesMap(), m_elements);
		}

		template class TriangleHalfVolume<Vector3>;
		template class TriangleHalfVolume<Vector3D>;
	}
}