//#include "ChimeraCGALWrapper.h"
#include "Mesh/Mesh.h"

namespace Chimera {

	namespace Meshes {

		template <class VectorType, template <class> class ElementPrimitive>
		unsigned int Mesh<VectorType, ElementPrimitive>::m_currID = 0;

		#pragma region MeshAuxiliaryStructures

		#pragma region Constructors
		template <class VectorType, template <class> class ElementPrimitive>
		Mesh<VectorType, ElementPrimitive>::Mesh(const vector<Vertex<VectorType> *> &vertices, const vector<ElementPrimitive<VectorType> *> &elementsPrimitive) 
			: m_vertices(vertices), m_elements(elementsPrimitive) {
			m_ID = m_currID++;
			for (int i = 0; i < m_vertices.size(); i++) {
				m_centroid += m_vertices[i]->getPosition();
			}
			m_draw = true;

			m_centroid /= static_cast<DoubleScalar>(m_vertices.size());
		}

		template <class VectorType, template <class> class ElementPrimitive>
		Mesh<VectorType, ElementPrimitive>::Mesh()  {
			m_ID = m_currID++;

			m_draw = true;
		}

		template class Mesh<Vector2,  Edge>;
		template class Mesh<Vector2D, Edge>;
		template class Mesh<Vector3,  Edge>;
		template class Mesh<Vector3D, Edge>;

		template class Mesh<Vector3, Face>;
		template class Mesh<Vector3D,Face>;
		template class Mesh<Vector2, Face>;
		template class Mesh<Vector2D,Face>;

		template class Mesh<Vector3, Volume>;
		template class Mesh<Vector3D, Volume>;

		//Polygonal meshes
		template class Mesh<Vector3, HalfFace>;
		template class Mesh<Vector3D, HalfFace>;
	}
}