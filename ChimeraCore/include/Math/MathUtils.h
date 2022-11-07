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

#ifndef _MATH_UTILS_
#define _MATH_UTILS_
#pragma once


#include "Config/ChimeraConfig.h"
#include "Math/MathUtilsCore.h"
#include "Math/Matrix/Matrix3x3.h"


using namespace std;
namespace Chimera {
	namespace Core {
		/************************************************************************/
		/* Geometry utils                                                       */
		/************************************************************************/
		template <class VectorType>
		DoubleScalar angle2D(const VectorType &v1, const VectorType &v2) {
			DoubleScalar dtheta, theta1, theta2;

			theta1 = atan2(v1.y, v1.x);
			theta2 = atan2(v2.y, v2.x);
			dtheta = theta2 - theta1;

			while (dtheta > PI)
				dtheta -= static_cast<Scalar>(PI * 2);
			while (dtheta < -PI)
				dtheta += static_cast<Scalar>(PI * 2);

			return(dtheta);
		}

		template <class VectorType>
		DoubleScalar angle3D(const VectorType &v1, const VectorType &v2) {
			DoubleScalar currAngle = atan2(v1.cross(v2).length(), v1.dot(v2));
			/*DoubleScalar dotVec = v1.dot(v2);
			DoubleScalar l1 = v1.lengthSqr();
			DoubleScalar l2 = v2.lengthSqr();
			DoubleScalar currAngle = acos(dotVec / sqrt(l1 * l2));*/

			/*while (currAngle > PI)
				currAngle -= static_cast<Scalar>(PI * 2);
			while (currAngle < -PI)
				currAngle += static_cast<Scalar>(PI * 2);*/

			return(currAngle);
		}

		Vector3D barycentricWeights(const Vector3D &point, const Vector3D &t1, const Vector3D &t2, const Vector3D &t3);

		/************************************************************************/
		/* Inside Polygons                                                      */
		/************************************************************************/
		bool isInsideTriangle(Vector2 point, Vector2 trianglePoints[3]);
		bool isInsideTriangle(Vector3 point, Vector3 trianglePoints[3]);

		template <class VectorType>
		bool isInsidePolygon(const VectorType &point, const vector<VectorType> &polygon);

		bool isInsideConvexPolygon(const Vector2 &point, const vector<Vector2> &polygon);
		bool isInsideTetrahedra(Vector3 point, Vector3 *tetraPoints);

		template <class VectorType>
		bool isOnEdge(const VectorType &p1, const VectorType &p2, const VectorType& point, Scalar proximityThreshold);

		bool isComplexPolygon(const Vector2 &p1, const Vector2 &p2, const Vector2 &p3, const Vector2 &p4);
		bool linePlaneIntersection(const Vector3 &p1, const Vector3 &p2, const Vector3 &planeOrigin, const Vector3 &planeNormal, Vector3 &intersectionPoint);
		
		template<class VectorType>
		DoubleScalar CalcAngleSum(VectorType point, const vector<VectorType> &points) {
			double m1, m2;
			double anglesum = 0, costheta;
			VectorType p1, p2;

			for (unsigned i = 0; i < points.size(); i++) {
				p1 = points[i] - point;
				p2 = points[(i + 1) % points.size()] - point;

				m1 = p1.length();
				m2 = p2.length();
				if (m1*m2 <= 0.0000001)
					return (PI * 2); /* We are on a node, consider this inside */
				else
					costheta = (p1.x*p2.x + p1.y*p2.y + p1.z*p2.z) / (m1*m2);

				anglesum += acos(costheta);
			}
			return anglesum;
		}


		template <class VectorType>
		DoubleScalar calculateSignedPolygonArea(const vector<VectorType> &polygonPoints, const VectorType &polygonNormal) {
			DoubleScalar areaSum = 0.0f;
			for (int i = 0; i < polygonPoints.size(); i++) {
				int nextI = i + 1;
				if (i == polygonPoints.size() - 1) {
					nextI = 0;
				}
				areaSum += polygonNormal.dot(polygonPoints[i].cross(polygonPoints[nextI]));
			}
			return areaSum*0.5;
		}

		template <class VectorType>
		DoubleScalar calculatePolygonArea(const vector<VectorType> &polygonPoints, const VectorType &polygonNormal) {
			return abs(calculateSignedPolygonArea(polygonPoints, polygonNormal));
		}


		template <class VectorType>
		DoubleScalar calculatePolygonArea(const vector<VectorType> &polygonPoints) {
			DoubleScalar areaSum = 0.0f;
			for (int i = 0; i < polygonPoints.size(); i++) {
				int nextI = i + 1;
				if (i == polygonPoints.size() - 1) {
					nextI = 0;
				}
				areaSum += (polygonPoints[i].cross(polygonPoints[nextI]));
			}
			return areaSum*0.5;
		}

		
		
		
		template <class VectorType>
		DoubleScalar calculateTriangleArea(const VectorType &p1, const VectorType &p2, const VectorType &p3);
		
		/************************************************************************/
		/* Distance functions                                                   */
		/************************************************************************/
		FORCE_INLINE Scalar distancePointToLineSegment(const Vector2 &point, const Vector2 &v0, const Vector2 &v1) {
			float vx = v0.x - point.x, vy = v0.y - point.y, ux = v1.x - v0.x, uy = v1.y - v0.y;
			float length = ux * ux + uy * uy;

			float det = (-vx * ux) + (-vy * uy); //if this is < 0 or > length then its outside the line segment
			if (det < 0)
				return (v0.x - point.x) * (v0.x - point.x) + (v0.y - point.y) * (v0.y - point.y);
			if (det > length)
				return (v1.x - point.x) * (v1.x - point.x) + (v1.y - point.y) * (v1.y - point.y);

			det = ux * vy - uy * vx;
			return (det * det) / length;
		}

		FORCE_INLINE Scalar distanceToLine(const Vector2 &point, const Vector2 &v0, const Vector2 &v1) {
			float vx = v0.x - point.x, vy = v0.y - point.y, ux = v1.x - v0.x, uy = v1.y - v0.y;
			float length = ux * ux + uy * uy;
			float det = ux * vy - uy * vx;
			return (det * det) / length;
		}

		template <class VectorType>
		FORCE_INLINE DoubleScalar distanceToLine(const VectorType &point, const VectorType &v0, const VectorType &v1) {
			VectorType numerator = (point - v0).cross(point - v1);
			VectorType denominator = (v1 - v0);
			DoubleScalar denominatorLenght = denominator.length();
			if (denominatorLenght == 0.0f)
				return 0.0f;

			return numerator.length() / denominatorLenght;
		}

		template <class VectorType>
		FORCE_INLINE DoubleScalar distanceToLineSegment(const VectorType &point, const VectorType &v0, const VectorType &v1) {
			
			VectorType v = v1 - v0;
			VectorType w = point - v0;

			DoubleScalar c1 = v.dot(w);
			if (c1 <= 0)
				return (point - v0).length(); //d(P, S.P0);

			DoubleScalar c2 = v.dot(v);
			if (c2 <= c1)
				return (point - v1).length(); //d(P, S.P1);

			double b = c1 / c2;
			VectorType pointB = v0 + v*b;
			return (point - pointB).length(); //d(P, Pb);
		}

		template<class VectorType>
		VectorType closesPointOnTriangle(const VectorType &position, const VectorType &t1, const VectorType &t2, const VectorType &t3) {
			VectorType edge0 = t2 - t1;
			VectorType edge1 = t3 - t1;
			VectorType v0 = t1 - position;

			//Verify distance to vertices first 
			if ((position - t1).length() < 1e-10) {
				return t1;
			}
			if ((position - t2).length() < 1e-10) {
				return t2;
			}
			if ((position - t3).length() < 1e-10) {
				return t3;
			}

			double a = edge0.dot(edge0);
			double b = edge0.dot(edge1);
			double c = edge1.dot(edge1);
			double d = edge0.dot(v0);
			double e = edge1.dot(v0);

			double det = a*c - b*b;
			double s = b*e - c*d;
			double t = b*d - a*e;

			if (s + t < det)
			{
				if (s < 0.f)
				{
					if (t < 0.f)
					{
						if (d < 0.f)
						{
							s = clamp<double>(-d / a, 0.f, 1.f);
							t = 0.f;
						}
						else
						{
							s = 0.f;
							t = clamp<double>(-e / c, 0.f, 1.f);
						}
					}
					else
					{
						s = 0.f;
						t = clamp<double>(-e / c, 0.f, 1.f);
					}
				}
				else if (t < 0.f)
				{
					s = clamp<double>(-d / a, 0.f, 1.f);
					t = 0.f;
				}
				else
				{
					double invDet = 1.f / det;
					s *= invDet;
					t *= invDet;
				}
			}
			else
			{
				if (s < 0.f)
				{
					double tmp0 = b + d;
					double tmp1 = c + e;
					if (tmp1 > tmp0)
					{
						double numer = tmp1 - tmp0;
						double denom = a - 2 * b + c;
						s = clamp<double>(numer / denom, 0.f, 1.f);
						t = 1 - s;
					}
					else
					{
						t = clamp<double>(-e / c, 0.f, 1.f);
						s = 0.f;
					}
				}
				else if (t < 0.f)
				{
					if (a + d > b + e)
					{
						double numer = c + e - b - d;
						double denom = a - 2 * b + c;
						s = clamp<double>(numer / denom, 0.f, 1.f);
						t = 1 - s;
					}
					else
					{
						s = clamp<double>(-e / c, 0.f, 1.f);
						t = 0.f;
					}
				}
				else
				{
					double numer = c + e - b - d;
					double denom = a - 2 * b + c;
					s = clamp<double>(numer / denom, 0.f, 1.f);
					t = 1.f - s;
				}
			}

			
			return t1 + edge0 * s + edge1 * t;
		}
		
		template<class VectorType>
		DoubleScalar distanceToTriangle(const VectorType &point, const VectorType &t1, const VectorType &t2, const VectorType &t3) {
			return (point - closesPointOnTriangle(point, t1, t2, t3)).length();
		}

		template <class VectorType>
		DoubleScalar distanceToPlane(const VectorType &point, const VectorType &planePoint, const VectorType &planeNormal) {
			VectorType projectionOnPlane = point - planeNormal*point.dot(planeNormal);
			return (point - projectionOnPlane).length();
		}
		Scalar averageDistanceToPoints(const Vector3 &point, const vector<Vector3> &points);

		/************************************************************************/
		/* Boundaries calculation                                               */
		/************************************************************************/
		template <class VectorType>
		void calculateBoundaries(vector<VectorType> *m_pPoints, VectorType &minBoundaries, VectorType &maxBoundaries);


		/************************************************************************/
		/* Scalar field                                                         */
		/************************************************************************/
		Scalar calculateCurvature(int i, int j, const Core::Array2D<Scalar> &lsArray, Scalar dx);
		/************************************************************************/
		/* Filtering                                                            */
		/************************************************************************/
		void filterIsoline(vector<Vector2> &isoline, int numFilteringIterations, bool closedLine = false);

		/************************************************************************/
		/* Rotation                                                             */
		/************************************************************************/
		void rotatePoints(const vector<Vector2> & points, vector<Vector2> & dstPoints, Vector2 rotationPoint, Scalar orientation);

		/************************************************************************/
		/* Ray-collision                                                        */
		/************************************************************************/
		
		//Returns 0 if theres no intersection, 1 if theres a valid intersection and -1 
		//if the intersection crossed a vertex or edge 
		template <class VectorT>
		bool rayTriangleIntersect(const VectorT &point, const VectorT &rayDirection, const VectorT trianglePoints[3], const VectorT &triangleNormal) {
			//Barycentric weights 
			DoubleScalar b[2];

			// Edge vectors
			const VectorT& e_1 = trianglePoints[1] - trianglePoints[0];
			const VectorT& e_2 = trianglePoints[2] - trianglePoints[0];

			VectorT q = rayDirection.cross(e_2);
			DoubleScalar a = e_1.dot(q);

			if ((abs(a) <= 1e-10))
				return false;

			DoubleScalar invA = 1 / a;
			const VectorT& s = (point - trianglePoints[0]);
			b[0] = s.dot(q)*invA;

			if ((b[0] < 0.0) || (b[0] > 1.0))
				return false;

			const VectorT& r = s.cross(e_1);
			b[1] = r.dot(rayDirection)*invA;
			if ((b[1] < 0.0) || (b[0] + b[1] > 1.0))
				return false;

			DoubleScalar t = invA*e_2.dot(r);
			if (t >= 0)
				return true;

			return false;
		}


		template <class VectorT>
		bool segmentTriangleIntersect(const VectorT &v1, const VectorT &v2, const VectorT trianglePoints[3]) {
			VectorT point = v1, rayDirection = (v2 - v1).normalized();
			//Barycentric weights 
			DoubleScalar b[2];

			// Edge vectors
			const VectorT& e_1 = trianglePoints[1] - trianglePoints[0];
			const VectorT& e_2 = trianglePoints[2] - trianglePoints[0];

			VectorT q = rayDirection.cross(e_2);
			DoubleScalar a = e_1.dot(q);

			if ((abs(a) <= 1e-10))
				return false;

			DoubleScalar invA = 1 / a;
			const VectorT& s = (point - trianglePoints[0]);
			b[0] = s.dot(q)*invA;

			if ((b[0] < 0.0) || (b[0] > 1.0))
				return false;

			const VectorT& r = s.cross(e_1);
			b[1] = r.dot(rayDirection)*invA;
			if ((b[1] < 0.0) || (b[0] + b[1] > 1.0))
				return false;

			DoubleScalar t = invA*e_2.dot(r);
			if (t >= 0 && t <= 1) {
				return true;
			}

			return false;
		}

		template <class VectorT>
		bool rayTriangleIntersect(const VectorT &point, const VectorT &rayDirection, const VectorT trianglePoints[3], const VectorT &triangleNormal, VectorT &intersectedPoint) {
			//Barycentric weights 
			DoubleScalar b[2];

			// Edge vectors
			const VectorT& e_1 = trianglePoints[1] - trianglePoints[0];
			const VectorT& e_2 = trianglePoints[2] - trianglePoints[0];

			VectorT q = rayDirection.cross(e_2);
			DoubleScalar a = e_1.dot(q);

			if ((abs(a) <= 1e-10))
				return false;

			DoubleScalar invA = 1 / a;
			const VectorT& s = (point - trianglePoints[0]);
			b[0] = s.dot(q)*invA;

			if ((b[0] < 0.0) || (b[0] > 1.0))
				return false;

			const VectorT& r = s.cross(e_1);
			b[1] = r.dot(rayDirection)*invA;
			if ((b[1] < 0.0) || (b[0] + b[1] > 1.0))
				return false;

			DoubleScalar t = invA*e_2.dot(r);
			if (t >= 0) {
				intersectedPoint = point + rayDirection*t;
				return true;
			}

			return false;
		}
		/**Computes if a ray intersects a plane. Assumes that rayDirection and planeNormal are normalized*/
		template<class VectorType>
		bool rayPlaneIntersect(const Vector3D &rayPoint, const VectorType &rayDirection, const Vector3D &planePoint, const Vector3D &planeNormal, Vector3D &intersectionPoint) {
			DoubleScalar dotNormals = rayDirection.dot(planeNormal);
			if (abs(dotNormals) < doublePrecisionThreshold) {
				return false;
			}
			DoubleScalar d = -planePoint.dot(planeNormal);
			DoubleScalar t = -(rayPoint.dot(planeNormal) + d) / dotNormals;
			intersectionPoint = rayPoint + rayDirection*t;
			return true;
		}
		/************************************************************************/
		/* Dumping utils                                                        */
		/************************************************************************/
		template<class VectorType>
		string vector2ToStr(const VectorType &vec) {
			string vecStr("(" + scalarToStr(vec.x) + ", " + scalarToStr(vec.y) + ")");
			return vecStr;
		}

		template<class VectorType>
		string vector3ToStr(const VectorType &vec) {
			string vecStr("(" + scalarToStr(vec.x) + ", " + scalarToStr(vec.y) + ", " + scalarToStr(vec.z) + ")");
			return vecStr;
		}
	}

}



#endif