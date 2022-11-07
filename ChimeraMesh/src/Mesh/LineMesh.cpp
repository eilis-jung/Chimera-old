//  Copyright (c) 2013, Vinicius Costa Azevedo
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met: 
//
//1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer. 
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution. 
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//	The views and conclusions contained in the software and documentation are those
//	of the authors and should not be interpreted as representing official policies, 
//	either expressed or implied, of the FreeBSD Project.
//	
#include "Mesh/LineMesh.h"

namespace Chimera {
	
	namespace Meshes {

		#pragma region Functionalities
		template<class ChildT, class VectorType, bool isVector2>
		bool LineMeshBase<ChildT, VectorType, isVector2>::segmentIntersection(const VectorType &v1, const VectorType &v2) {
			for (int i = 0; i < m_vertices.size() - 1; i++) {
				if (DoLinesIntersect(v1, v2, m_vertices[i]->getPosition(), m_vertices[i + 1]->getPosition()))
					return true;
			}
			return false;
		}

		template<class ChildT, class VectorType, bool isVector2>
		bool LineMeshBase<ChildT, VectorType, isVector2>::segmentIntersection(const VectorType &v1, const VectorType &v2, VectorType &outputVec) {
			for (int i = 0; i < m_vertices.size() - 1; i++) {
				if (DoLinesIntersect(v1, v2, m_vertices[i]->getPosition(), m_vertices[i + 1]->getPosition(), outputVec))
					return true;
			}
			return false;
		}		
		#pragma endregion

		#pragma region InitializationFunctions
		template<class ChildT, class VectorType, bool isVector2>
		void LineMeshBase<ChildT, VectorType, isVector2>::initializeEdges() {
			m_elements.clear();
			for (int i = 0; i < m_vertices.size() - 1; i++) {
				m_elements.push_back(new Edge<VectorType>(m_vertices[i], m_vertices[i + 1], geometricEdge));
			}
			if (m_isClosedMesh) {
				m_elements.push_back(new Edge<VectorType>(m_vertices.back(), m_vertices.front(), geometricEdge));
			}
		}
	
		
		template<class ChildT, class VectorType, bool isVector2>
		void LineMeshBase<ChildT, VectorType, isVector2>::initializeRegularGridPatches() {
			for (uint j = 0; j < m_elements.size(); j++) {
				Edge<VectorType> *pCurrEdge = m_elements[j];
				VectorType gridSpacePosition = pCurrEdge->getCentroid() / m_gridDx;
				dimensions_t dimTemp(floor(gridSpacePosition.x), floor(gridSpacePosition.y));
				m_regularGridPatches(floor(gridSpacePosition.x), floor(gridSpacePosition.y)).push_back(j);
			}
		}

		template<class ChildT, class VectorType, bool isVector2>
		void LineMeshBase<ChildT, VectorType, isVector2>::initializeVertexNormals() {
			for (int i = 0; i < m_vertices.size(); i++) {
				int nextI = roundClamp<int>(i + 1, 0, m_vertices.size());
				int prevI = roundClamp<int>(i - 1, 0, m_vertices.size());

				VectorType prevNormal = m_vertices[i]->getPosition() - m_vertices[prevI]->getPosition();
				VectorType nextNormal = m_vertices[nextI]->getPosition() - m_vertices[i]->getPosition();
				prevNormal = prevNormal.perpendicular().normalized();
				nextNormal = nextNormal.perpendicular().normalized();
				m_vertices[i]->setNormal((prevNormal + nextNormal)*0.5);
			}
		}
		#pragma endregion

		template class LineMeshBase<LineMeshT<Vector2, isVector2<Vector2>::value>, Vector2, isVector2<Vector2>::value>;
		template class LineMeshBase<LineMeshT<Vector2D, isVector2<Vector2D>::value>, Vector2D, isVector2<Vector2D>::value>;

		template class LineMeshBase<LineMeshT<Vector3, isVector2<Vector3>::value>, Vector3, isVector2<Vector3>::value>;
		template class LineMeshBase<LineMeshT<Vector3D, isVector2<Vector3D>::value>, Vector3D, isVector2<Vector3D>::value>;

		
		#pragma endregion
	}
}