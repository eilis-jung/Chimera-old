#include "Mesh/PolygonMesh.h"


namespace Chimera {

	namespace Data {

		bool PolygonMesh::comparePairs(pair<Vector2, int *> a, pair<Vector2, int *> b) {
			if (a.first == b.first)
				return *a.second < *b.second;
			return a.first < b.first;
		}

		bool PolygonMesh::comparePairs_b(pair<Vector2, int *> a, pair<Vector2, int *> b) {
			return *a.second < *b.second;
		}

		bool PolygonMesh::uniqueVectors(Vector2 a, Vector2 b) {
			return a == b;
		}

		PolygonMesh::PolygonMesh(CutCells2D *pCutCells) {
			//vector<pair<Vector2, int *>> allPairs;
			//vector<polygonPointers_t> tempPolygons;
			//vector<Vector2> tempPoints;
			//vector<nodeType_t> tempNodeTypes;
			//Scalar dx = pCutCells->getGridSpacing();
			//int cellIndex = 0;
			//for (int i = 0; i < pCutCells->getNumberOfCells(); i++) {
			//	CutFace<Vector2> *pFace = pCutCells->getSpecialCellPtr(i);
			//	polygonPointers_t tempPolygon;
			//	for (int j = 0; j < pFace->m_cutEdges.size(); j++) {
			//		Vector2 currPoint = pFace->getEdgeInitialPoint(j);
			//		tempPoints.push_back(currPoint);

			//		int *pointIndex = new int(cellIndex++);
			//		allPairs.push_back(pair<Vector2, int *>(currPoint, pointIndex));
			//		tempPolygon.pointsIndexes.push_back(pointIndex);
			//		
			//		if (isOnGridPoint(currPoint, dx)) {
			//			tempNodeTypes.push_back(gridNode);
			//		}
			//		else {
			//			tempNodeTypes.push_back(geometryNode);
			//		}
			//	}
			//	tempPolygons.push_back(tempPolygon);
			//}

			//sort(allPairs.begin(), allPairs.end(), PolygonMesh::comparePairs);
			////sort(allPairs.begin(), allPairs.end(), PolygonMesh::comparePairs_b);

			//for (int i = 0; i < allPairs.size() - 1; i++) {
			//	while ((allPairs[i].first - allPairs[i + 1].first).length() < 1e-5) {
			//		if (*allPairs[i + 1].second > *allPairs[i].second) {
			//			*allPairs[i + 1].second = *allPairs[i].second;
			//		}
			//		else if (*allPairs[i + 1].second < *allPairs[i].second){
			//			*allPairs[i].second = *allPairs[i + 1].second;
			//		}
			//		i++;
			//		if (i >= allPairs.size() - 1)
			//			break;
			//	}
			//}

			//for (int i = 0; i < allPairs.size() - 1;) {
			//	m_points.push_back(tempPoints[*allPairs[i].second]);
			//	m_nodeTypes.push_back(tempNodeTypes[*allPairs[i].second]);
			//	if (*allPairs[i].second == *allPairs[i + 1].second) {
			//		while (i < allPairs.size() - 1 && *allPairs[i].second == *allPairs[i + 1].second) {
			//			//allPairs[i].second = m_points.size() - 1;
			//			i++;
			//		}
			//		//allPairs[i].second = m_points.size() - 1;
			//		i++;
			//	}
			//	else {
			//		i++;
			//	}
			//}

			//for (int i = 0; i < tempPolygons.size(); i++) {
			//	polygon_t currPolygon;
			//	for (int j = 0; j < tempPolygons[i].pointsIndexes.size(); j++) {
			//		currPolygon.pointsIndexes.push_back(*tempPolygons[i].pointsIndexes[j]);
			//	}
			//	m_polygons.push_back(currPolygon);
			//}
		}

		bool PolygonMesh::isInsideMesh(const Vector2 & position) {
			//for (int i = 0; i < m_triangles.size(); i++) {
			//	Vector3 p1 = m_points[m_triangles[i].pointsIndexes[0]];
			//	Vector3 p2 = m_points[m_triangles[i].pointsIndexes[1]];
			//	Vector3 p3 = m_points[m_triangles[i].pointsIndexes[2]];
			//	Vector3 planeOrigin = p1;
			//	Vector3 planeNormal = (p2 - p1).cross(p3 - p1);
			//	planeNormal.normalize();

			//	Vector3 v1 = position - planeOrigin;
			//	//Testing which side of the plane the point is on
			//	Scalar dprod = planeNormal.dot(v1);

			//	if (abs(dprod) - 1e-5 <= 0)
			//		return true;

			//	if (isTop(planeOrigin, planeNormal, position)) {
			//		return false;
			//	}
			//}
			return true;
		}
	}
}