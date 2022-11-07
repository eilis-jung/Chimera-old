#include "Math/Intersection.h"

namespace Chimera {

	namespace Core {

		using namespace std;
		#pragma region Vector3Functions
		template<class VectorType>
		bool DoLinesIntersectT(const VectorType & p1, const VectorType & p2, const VectorType & p3, const VectorType & p4, 
															VectorType &intersectionPoint, isVector2False) {
			double tolerance = 1e-8;
			VectorType da = p2 - p1;
			VectorType db = p4 - p3;
			VectorType dc = p3 - p1;

			if (abs(dc.dot(da.cross(db))) < tolerance) // lines are not coplanar
				return false;

			DoubleScalar s = (dc.cross(db).dot(da.cross(db))) / da.cross(db).lengthSqr();
			DoubleScalar s2 = (dc.cross(da).dot(da.cross(db))) / da.cross(db).lengthSqr();
			if (s >= tolerance && s <= 1.0 - tolerance && s2 >= tolerance && s2 <= 1.0 - tolerance) {
				intersectionPoint = p1 + da * s;
				return true;
			}

			return false;
		}

		template<class VectorType>
		bool segmentLineIntersectionT(const VectorType & p1, const VectorType & p2, const VectorType & p3, const VectorType & p4, VectorType & intersectionPoint, isVector2False) {
			VectorType da = p2 - p1;
			VectorType db = p4 - p3;
			VectorType dc = p3 - p1;

			if (dc.dot(da.cross(db)) != 0.0) // lines are not coplanar
				return false;

			if (da.cross(db).lengthSqr() == 0) //lines are parallel
				return false;

			DoubleScalar s = (dc.cross(db).dot(da.cross(db))) / da.cross(db).lengthSqr();
			if (s >= 0.0 && s <= 1.0) {
				intersectionPoint = p1 + da * s;
				return true;
			}

			return false;
		}
		#pragma endregion

		#pragma region Vector2Functions
		template<class VectorType>
		bool DoLinesIntersectT(const VectorType & p1, const VectorType & p2, const VectorType & p3, const VectorType & p4,
								VectorType &intersectionPoint, isVector2True) {
			
			double tolerance = 1e-8;
			double x1 = p1.x, x2 = p2.x, x3 = p3.x, x4 = p4.x;
			double y1 = p1.y, y2 = p2.y, y3 = p3.y, y4 = p4.y;

			double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

			// If d is zero, there is no intersection
			if (d == 0) return false;

			// Get the x and y
			double pre = (x1*y2 - y1*x2), post = (x3*y4 - y3*x4);
			double x = (pre * (x3 - x4) - (x1 - x2) * post) / d;
			double y = (pre * (y3 - y4) - (y1 - y2) * post) / d;

			// Check if the x and y coordinates are within both lines
			if (x < min(x1, x2) - tolerance || x > max(x1, x2) + tolerance ||
				x < min(x3, x4) - tolerance || x > max(x3, x4) + tolerance) return false;
			if (y < min(y1, y2) - tolerance || y > max(y1, y2) + tolerance ||
				y < min(y3, y4) - tolerance || y > max(y3, y4) + tolerance) return false;

			intersectionPoint.x = x; intersectionPoint.y = y;
			return true;
		}
		template<class VectorType>
		bool segmentLineIntersectionT(const VectorType & p1, const VectorType & p2, const VectorType & p3, const VectorType & p4, VectorType & intersectionPoint, isVector2True) {
			double x1 = p1.x, x2 = p2.x, x3 = p3.x, x4 = p4.x;
			double y1 = p1.y, y2 = p2.y, y3 = p3.y, y4 = p4.y;

			double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

			// If d is zero, there is no intersection
			if (d == 0) return false;

			// Get the x and y
			double pre = (x1*y2 - y1*x2), post = (x3*y4 - y3*x4);
			double x = (pre * (x3 - x4) - (x1 - x2) * post) / d;
			double y = (pre * (y3 - y4) - (y1 - y2) * post) / d;

			// Check if the x and y coordinates are within both lines
			if (x < min(x1, x2) - g_epsilon || x > max(x1, x2) + g_epsilon)  return false;
			if (y < min(y1, y2) - g_epsilon || y > max(y1, y2) + g_epsilon) return false;

			intersectionPoint.x = x; intersectionPoint.y = y;
			return true;
		}
		#pragma endregion

		template bool DoLinesIntersectT<Vector2>(const Vector2 & p1, const Vector2 & p2, const Vector2& p3, const Vector2& p4, Vector2&intersectionPoint, isVector2True);
		template bool DoLinesIntersectT<Vector2D>(const Vector2D & p1, const Vector2D & p2, const Vector2D& p3, const Vector2D& p4, Vector2D &intersectionPoint, isVector2True);

		template bool DoLinesIntersectT<Vector3>(const Vector3 & p1, const Vector3 & p2, const Vector3& p3, const Vector3& p4, Vector3&intersectionPoint, isVector2False);
		template bool DoLinesIntersectT<Vector3D>(const Vector3D & p1, const Vector3D & p2, const Vector3D& p3, const Vector3D& p4, Vector3D &intersectionPoint, isVector2False);

		template bool segmentLineIntersectionT<Vector2>(const Vector2 & p1, const Vector2 & p2, const Vector2& p3, const Vector2& p4, Vector2&intersectionPoint, isVector2True);
		template bool segmentLineIntersectionT<Vector2D>(const Vector2D & p1, const Vector2D & p2, const Vector2D& p3, const Vector2D& p4, Vector2D &intersectionPoint, isVector2True);
		template bool segmentLineIntersectionT<Vector3>(const Vector3 & p1, const Vector3 & p2, const Vector3& p3, const Vector3& p4, Vector3&intersectionPoint, isVector2False);
		template bool segmentLineIntersectionT<Vector3D>(const Vector3D & p1, const Vector3D & p2, const Vector3D& p3, const Vector3D& p4, Vector3D &intersectionPoint, isVector2False);
	}


	
}