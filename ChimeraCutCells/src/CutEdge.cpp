#include "CutCells/CutEdge.h"

namespace Chimera {

	namespace CutCells {

		#pragma region Constructors
		template<>
		CutEdge<Vector2>::CutEdge(int ID, const Vector2 &initialPoint, const Vector2 &finalPoint, Scalar dx, edgeLocation_t edgeLocation, int thinObjectID /* = -1 */)  {
			m_ID = ID;
			m_edgeNeighbors[0] = -1; m_edgeNeighbors[1] = -1;
			m_initialPoint = initialPoint; m_finalPoint = finalPoint;
			m_lengthFraction = (initialPoint - finalPoint).length() / dx;
			m_normal = (initialPoint - finalPoint).normalize().perpendicular();
			m_isValid = true;
			m_thinObjectID = thinObjectID;
			m_velocityWeight = 0.0f;
			m_centroid = (m_initialPoint + m_finalPoint)*0.5;
			m_edgeLocation = edgeLocation;
			m_mergedNode = false;
		}

		template<>
		CutEdge<Vector2D>::CutEdge(int ID, const Vector2D &initialPoint, const Vector2D &finalPoint, Scalar dx, edgeLocation_t edgeLocation, int thinObjectID /* = -1 */)  {
			m_ID = ID;
			m_edgeNeighbors[0] = -1; m_edgeNeighbors[1] = -1;
			m_initialPoint = initialPoint; m_finalPoint = finalPoint;
			m_lengthFraction = (initialPoint - finalPoint).length() / dx;
			m_normal = (initialPoint - finalPoint).normalize().perpendicular();
			m_isValid = true;
			m_thinObjectID = thinObjectID;
			m_velocityWeight = 0.0f;
			m_centroid = (m_initialPoint + m_finalPoint)*0.5;
			m_edgeLocation = edgeLocation;
			m_mergedNode = false;
		}

		template<>
		CutEdge<Vector3>::CutEdge(int ID, const Vector3 &initialPoint, const Vector3 &finalPoint, Scalar dx, edgeLocation_t edgeLocation, int thinObjectID /* = -1 */) {
			m_ID = ID;
			m_edgeNeighbors[0] = -1; m_edgeNeighbors[1] = -1;
			m_initialPoint = initialPoint; m_finalPoint = finalPoint;
			m_lengthFraction = (initialPoint - finalPoint).length() / dx;
			m_isValid = true;
			m_thinObjectID = thinObjectID;
			m_velocityWeight = 0.0f;
			m_centroid = (m_initialPoint + m_finalPoint)*0.5;
			m_edgeLocation = edgeLocation;
			m_mergedNode = true;
		}

		template<>
		CutEdge<Vector3D>::CutEdge(int ID, const Vector3D &initialPoint, const Vector3D &finalPoint, Scalar dx, edgeLocation_t edgeLocation, int thinObjectID /* = -1 */) {
			m_ID = ID;
			m_edgeNeighbors[0] = -1; m_edgeNeighbors[1] = -1;
			m_initialPoint = initialPoint; m_finalPoint = finalPoint;
			m_lengthFraction = (initialPoint - finalPoint).length() / dx;
			m_isValid = true;
			m_thinObjectID = thinObjectID;
			m_velocityWeight = 0.0f;
			m_centroid = (m_initialPoint + m_finalPoint)*0.5;
			m_edgeLocation = edgeLocation;
			m_mergedNode = true;
		}
		#pragma endregion
	}
}