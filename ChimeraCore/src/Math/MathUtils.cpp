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

#include "Math/MathUtils.h"

namespace Chimera {
	namespace Core {
		template<>
		bool isOnEdge(const Vector2 &p1, const Vector2 &p2, const Vector2& point, Scalar proximityThreshold) {
			if (isInBetween(p1, p2, point)) {
				if (abs(p2.x - p1.x) < proximityThreshold)
					return true;

				float a = (p2.y - p1.y) / (p2.x - p1.x);
				float b = p1.y - a * p1.x;
				Scalar temp = point.y - (a*point.x + b);
				Scalar temp2 = abs((p2.x - p1.x)*(point.y - p1.y) - (p2.y - p1.y)*(point.x - p1.x));
				if (abs(point.y - (a*point.x + b)) < proximityThreshold)
					return true;
			}

			return false;
		}

		template<>
		bool isOnEdge(const Vector2D &p1, const Vector2D &p2, const Vector2D& point, Scalar proximityThreshold) {
			if (isInBetween(p1, p2, point)) {
				if (abs(p2.x - p1.x) < proximityThreshold)
					return true;

				float a = (p2.y - p1.y) / (p2.x - p1.x);
				float b = p1.y - a * p1.x;
				DoubleScalar temp = point.y - (a*point.x + b);
				DoubleScalar temp2 = abs((p2.x - p1.x)*(point.y - p1.y) - (p2.y - p1.y)*(point.x - p1.x));
				if (abs(point.y - (a*point.x + b)) < proximityThreshold)
					return true;
			}

			return false;
		}

		template<>
		bool isOnEdge(const Vector3 &p1, const Vector3 &p2, const Vector3 &point, Scalar proximityThreshold) {
			if (isInBetween(p1, p2, point)) {
				if (distanceToLine(point, p1, p2) < proximityThreshold) {
					return true;
				}
			}
			return false;
		}

		template<>
		bool isOnEdge(const Vector3D &p1, const Vector3D &p2, const Vector3D &point, Scalar proximityThreshold) {
			if (isInBetween(p1, p2, point)) {
				if (distanceToLine(point, p1, p2) < proximityThreshold) {
					return true;
				}
			}
			return false;
		}

		bool isInsideTriangle(Vector2 point, Vector2 trianglePoints[3]) {
			if(isOnEdge(trianglePoints[0], trianglePoints[1], point, 1e-4))
				return true;
			if (isOnEdge(trianglePoints[1], trianglePoints[2], point, 1e-4))
				return true;
			if (isOnEdge(trianglePoints[2], trianglePoints[0], point, 1e-4))
				return true;
			
			// Compute vectors        
			Vector2 v0 = trianglePoints[2] - trianglePoints[0];
			Vector2 v1 = trianglePoints[1] - trianglePoints[0];
			Vector2  v2 = point - trianglePoints[0];

			DoubleScalar dot00 = v0.x*v0.x + v0.y*v0.y;
			DoubleScalar dot01 = v0.x*v1.x + v0.y*v1.y;
			DoubleScalar dot02 = v0.x*v2.x + v0.y*v2.y;
			DoubleScalar dot11 = v1.x*v1.x + v1.y*v1.y;
			DoubleScalar dot12 = v1.x*v2.x + v1.y*v2.y;

			// Compute barycentric coordinates
			DoubleScalar invDenom = 1.0/ (dot00 * dot11 - dot01 * dot01);
			DoubleScalar u = (dot11 * dot02 - dot01 * dot12) * invDenom;
			DoubleScalar v = (dot00 * dot12 - dot01 * dot02) * invDenom;

			// Check if point is in triangle
			return (u >= 0) && (v >= 0) && (u + v <= 1);
		}
		
		bool isInsideTriangle(Vector3 point, Vector3 trianglePoints[3]) {
			return true;
		}

		template<>
		bool isInsidePolygon(const Vector2 &point, const vector<Vector2> &polygon) {
			/** Checking if points are livin on the edge */
			if (isOnEdge(polygon[polygon.size() - 1], polygon[0], point, 1e-4))
				return true;
			for (unsigned i = 0; i < polygon.size() - 1; i++) {
				if (isOnEdge(polygon[i], polygon[i + 1], point, 1e-4))
					return true;
			}

			//http://alienryderflex.com/polygon/ polygon function
			int i = 0, j = static_cast<int>(polygon.size() - 1);
			bool insidePolygon = false;
			
			for (unsigned i = 0; i < polygon.size(); i++) {
				if (	(polygon[i].y < point.y && polygon[j].y >= point.y)
						||  (polygon[j].y < point.y && polygon[i].y >= point.y) ) {
							if (polygon[i].x + (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y)*(polygon[j].x - polygon[i].x) < point.x) {
								insidePolygon = !insidePolygon; 
							}
				}
				j = i; 
			}

			return insidePolygon;
		}

		template<>
		bool isInsidePolygon(const Vector2D &point, const vector<Vector2D> &polygon) {
			/** Checking if points are livin on the edge */
			if (isOnEdge(polygon[polygon.size() - 1], polygon[0], point, 1e-10))
				return true;
			for (unsigned i = 0; i < polygon.size() - 1; i++) {
				if (isOnEdge(polygon[i], polygon[i + 1], point, 1e-10))
					return true;
			}

			//http://alienryderflex.com/polygon/ polygon function
			int i = 0, j = static_cast<int>(polygon.size() - 1);
			bool insidePolygon = false;

			for (unsigned i = 0; i < polygon.size(); i++) {
				if ((polygon[i].y < point.y && polygon[j].y >= point.y)
					|| (polygon[j].y < point.y && polygon[i].y >= point.y)) {
					if (polygon[i].x + (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y)*(polygon[j].x - polygon[i].x) < point.x) {
						insidePolygon = !insidePolygon;
					}
				}
				j = i;
			}

			return insidePolygon;
		}

		template<>
		bool isInsidePolygon(const Vector3 &point, const vector<Vector3> &polygon) {
			Scalar totalAngle = CalcAngleSum(point, polygon);
			if (abs(abs(totalAngle) - PI * 2) < 0.0) {
				return true;
			}
			else {
				return false;
			}
		}

		template<>
		bool isInsidePolygon(const Vector3D &point, const vector<Vector3D> &polygon) {
			Scalar totalAngle = CalcAngleSum(point, polygon);
			if (abs(abs(totalAngle) - PI * 2) < 0.0) {
				return true;
			}
			else {
				return false;
			}
		}
		

		bool isInsideConvexPolygon(const Vector2 &point, const vector<Vector2> &polygon) {
			if (isOnEdge(polygon[polygon.size() - 1], polygon[0], point, 1e-4))
				return true;

			bool isLeftCheck = isLeft(polygon[polygon.size() - 1], polygon[0], point);
			for (int i = 0; i < polygon.size() - 1; i++)  {
				if (isOnEdge(polygon[i], polygon[i + 1], point, 1e-4))
					return true;
				if (isLeftCheck != isLeft(polygon[i], polygon[i + 1], point) ) {
					return false;
				}
			}

			return true;
		}

		bool isInsideTetrahedra(Vector3 point, Vector3 *tetraPoints) {
			Matrix3x3 mTransformMatrix;
			Vector3 tempPoint = tetraPoints[0] - tetraPoints[3]; 
			mTransformMatrix.column[0] = tetraPoints[0] - tetraPoints[3]; 
			mTransformMatrix.column[1] = tetraPoints[1] - tetraPoints[3];
			mTransformMatrix.column[2] = tetraPoints[2] - tetraPoints[3];
			
			mTransformMatrix.invert();
			Vector3 localR = mTransformMatrix*(point - tetraPoints[3]);

			if(localR.x > 1 || localR.x < 0 || localR.y > 1 || localR.y < 0 || localR.z > 1 || localR.z < 0)
				return false;

			return true;
		}		

		/************************************************************************/
		/* Polygons utils                                                       */
		/************************************************************************/
		Scalar perpDot(const Vector2 &p1, const Vector2 &p2) {
			return p1.y*p2.x - p1.x*p2.y;
		}

		bool DoLinesIntersect(const Vector2 &p1, const Vector2 &p2, const Vector2 &p3, const Vector2 &p4) {
			double x1 = p1.x, x2 = p2.x, x3 = p3.x, x4 = p4.x;
			double y1 = p1.y, y2 = p2.y, y3 = p3.y, y4 = p4.y;

			double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

			// If d is zero, there is no intersection
			if (d == 0) return false;

			// Get the x and y
			double pre = (x1*y2 - y1*x2), post = (x3*y4 - y3*x4);
			double x = (pre * (x3 - x4) - (x1 - x2) * post) / d;
			double y = ( pre * (y3 - y4) - (y1 - y2) * post ) / d;

			// Check if the x and y coordinates are within both lines
			if ( x < min(x1, x2) - g_epsilon || x > max(x1, x2) + g_epsilon ||
				 x < min(x3, x4) - g_epsilon || x > max(x3, x4) + g_epsilon) return false;
			if ( y < min(y1, y2) - g_epsilon || y > max(y1, y2) + g_epsilon ||
				 y < min(y3, y4) - g_epsilon || y > max(y3, y4) + g_epsilon) return false;
				
			return true;
		}

		bool DoLinesIntersect(const Vector2 &p1, const Vector2 &p2, const Vector2 &p3, const Vector2 &p4, Vector2 &intersectionPoint, Scalar tolerance) {
			double x1 = p1.x, x2 = p2.x, x3 = p3.x, x4 = p4.x;
			double y1 = p1.y, y2 = p2.y, y3 = p3.y, y4 = p4.y;

			double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

			// If d is zero, there is no intersection
			if (d == 0) return false;

			// Get the x and y
			double pre = (x1*y2 - y1*x2), post = (x3*y4 - y3*x4);
			double x = ( pre * (x3 - x4) - (x1 - x2) * post ) / d;
			double y = ( pre * (y3 - y4) - (y1 - y2) * post ) / d;

			intersectionPoint.x = x; intersectionPoint.y = y;
			// Check if the x and y coordinates are within both lines
			if (x < min(x1, x2) - tolerance || x > max(x1, x2) + tolerance ||
				x < min(x3, x4) - tolerance || x > max(x3, x4) + tolerance) return false;
			if (y < min(y1, y2) - tolerance || y > max(y1, y2) + tolerance ||
				y < min(y3, y4) - tolerance || y > max(y3, y4) + tolerance) return false;

			intersectionPoint.x = x; intersectionPoint.y = y;
			return true;
		}

		Vector2 linesIntersection(const Vector2 &p1, const Vector2 &p2, const Vector2 &p3, const Vector2 &p4) {
			float x1 = p1.x, x2 = p2.x, x3 = p3.x, x4 = p4.x;
			float y1 = p1.y, y2 = p2.y, y3 = p3.y, y4 = p4.y;

			float d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

			// If d is zero, there is no intersection
			if (d == 0) return Vector2(FLT_MAX, FLT_MAX);

			// Get the x and y
			float pre = (x1*y2 - y1*x2), post = (x3*y4 - y3*x4);
			float x = ( pre * (x3 - x4) - (x1 - x2) * post ) / d;
			float y = ( pre * (y3 - y4) - (y1 - y2) * post ) / d;

			return Vector2(x, y);
		}

		bool isComplexPolygon(const Vector2 &p1, const Vector2 &p2, const Vector2 &p3, const Vector2 &p4) {
			return DoLinesIntersect(p1, p2, p3, p4) || DoLinesIntersect(p2, p3, p4, p1); 
		}

		bool linePlaneIntersection(const Vector3 &p1, const Vector3 &p2, const Vector3 &planeOrigin, 
			const Vector3 &planeNormal, Vector3 &intersectionPoint) {
			Vector3 u = p2 - p1;
			Vector3 w = p1 - planeOrigin;

			double     D = planeNormal.x*u.x + planeNormal.y*u.y  + planeNormal.z*u.z; //planeNormal.dot(u);
			double     N = -(planeNormal.x*w.x + planeNormal.y*w.y + planeNormal.z*w.z); //planeNormal.dot(w);

			if (abs(D) < 0.00000001) {           // segment is parallel to plane
				if (N == 0)                      // segment lies in plane
					return false;
				else
					return false;                    // no intersection
			}
			// they are not parallel
			// compute intersect param
			double sI = N / D;
			if (sI < 0 || sI >= 1)
				return false;                        // no intersection

			intersectionPoint = p1 + u*static_cast<Scalar>(sI);                  // compute segment intersect point
			return true;
		}

		template<>
		DoubleScalar calculateTriangleArea(const Vector2 &p1, const Vector2 &p2, const Vector2 &p3) {
			Matrix3x3 tempMat;
			tempMat.column[0].x = p1.x;
			tempMat.column[1].x = p2.x;
			tempMat.column[2].x = p3.x;
			tempMat.column[0].y = p1.y;
			tempMat.column[1].y = p2.y;
			tempMat.column[2].y = p3.y;
			tempMat.column[0].z = 1;
			tempMat.column[1].z = 1;
			tempMat.column[2].z = 1;
			return tempMat.determinant()*0.5;
		}

		template<>
		DoubleScalar calculateTriangleArea(const Vector2D &p1, const Vector2D &p2, const Vector2D &p3) {
			Matrix3x3 tempMat;
			tempMat.column[0].x = p1.x;
			tempMat.column[1].x = p2.x;
			tempMat.column[2].x = p3.x;
			tempMat.column[0].y = p1.y;
			tempMat.column[1].y = p2.y;
			tempMat.column[2].y = p3.y;
			tempMat.column[0].z = 1;
			tempMat.column[1].z = 1;
			tempMat.column[2].z = 1;
			return tempMat.determinant()*0.5;
		}

		template<>
		DoubleScalar calculateTriangleArea(const Vector3 &p1, const Vector3 &p2, const Vector3 &p3) {
			/*Matrix3x3 tempMat;
			tempMat.column[0].x = p1.x;
			tempMat.column[1].x = p2.x;
			tempMat.column[2].x = p3.x;
			tempMat.column[0].y = p1.y;
			tempMat.column[1].y = p2.y;
			tempMat.column[2].y = p3.y;
			tempMat.column[0].z = p1.z;
			tempMat.column[1].z = p2.z;
			tempMat.column[2].z = p3.z;
			return tempMat.determinant()*0.5;*/
			DoubleScalar l1 = (p1 - p2).length();
			DoubleScalar l2 = (p2 - p3).length();
			DoubleScalar l3 = (p3 - p1).length();
			DoubleScalar lMean = (l1 + l2 + l3) / 2;
			return sqrt(lMean*(lMean - l1)*(lMean - l2)*(lMean - l3));
		}

		template<>
		DoubleScalar calculateTriangleArea(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3) {
			/*Matrix3x3 tempMat;
			tempMat.column[0].x = p1.x;
			tempMat.column[1].x = p2.x;
			tempMat.column[2].x = p3.x;
			tempMat.column[0].y = p1.y;
			tempMat.column[1].y = p2.y;
			tempMat.column[2].y = p3.y;
			tempMat.column[0].z = p1.z;
			tempMat.column[1].z = p2.z;
			tempMat.column[2].z = p3.z;

			return tempMat.determinant()*0.5;*/

			DoubleScalar l1 = (p1 - p2).length();
			DoubleScalar l2 = (p2 - p3).length();
			DoubleScalar l3 = (p3 - p1).length();
			DoubleScalar lMean = (l1 + l2 + l3) / 2;
			return sqrt(lMean*(lMean - l1)*(lMean - l2)*(lMean - l3));
		}

		Scalar averageDistanceToPoints(const Vector3 &point, const vector<Vector3> &points) {
			double totalDistance = 0;
			for(int i = 0; i < points.size(); i++) {
				totalDistance += (points[i] - point).length();
			}
			totalDistance /= points.size();
			return static_cast<Scalar>(totalDistance);
		}

		/************************************************************************/
		/* Boundaries calculation                                               */
		/************************************************************************/
		template<>
		void calculateBoundaries(vector<Vector2> *m_pPoints, Vector2 &minBoundaries, Vector2  &maxBoundaries) {
			minBoundaries.x = minBoundaries.y = FLT_MAX;
			maxBoundaries.x = maxBoundaries.y = -FLT_MAX;
			for(int i = 0; i < m_pPoints->size(); i++) {
				if((*m_pPoints)[i].x < minBoundaries.x) {
					minBoundaries.x = (*m_pPoints)[i].x; 
				}
				if((*m_pPoints)[i].y < minBoundaries.y) {
					minBoundaries.y = (*m_pPoints)[i].y; 
				}

				if((*m_pPoints)[i].x > maxBoundaries.x) {
					maxBoundaries.x = (*m_pPoints)[i].x; 
				}
				if((*m_pPoints)[i].y > maxBoundaries.y) {
					maxBoundaries.y = (*m_pPoints)[i].y; 
				}
			}
		}

		template<>
		void calculateBoundaries(vector<Vector3> *m_pPoints, Vector3 &minBoundaries, Vector3  &maxBoundaries) {
			minBoundaries.x = minBoundaries.y = minBoundaries.z = FLT_MAX;
			maxBoundaries.x = maxBoundaries.y = maxBoundaries.z = -FLT_MAX;
			for(int i = 0; i < m_pPoints->size(); i++) {
				if((*m_pPoints)[i].x < minBoundaries.x) {
					minBoundaries.x = (*m_pPoints)[i].x; 
				}
				if((*m_pPoints)[i].y < minBoundaries.y) {
					minBoundaries.y = (*m_pPoints)[i].y; 
				}
				if((*m_pPoints)[i].z < minBoundaries.z) {
					minBoundaries.z = (*m_pPoints)[i].z; 
				}

				if((*m_pPoints)[i].x > maxBoundaries.x) {
					maxBoundaries.x = (*m_pPoints)[i].x; 
				}
				if((*m_pPoints)[i].y > maxBoundaries.y) {
					maxBoundaries.y = (*m_pPoints)[i].y; 
				}
				if((*m_pPoints)[i].z > maxBoundaries.z) {
					maxBoundaries.z = (*m_pPoints)[i].z; 
				}
			}
		}

		Scalar calculateCurvature(int i, int j, const Core::Array2D<Scalar> &lsArray, Scalar dx) {
			if(lsArray(i, j) < 0)
				return 0.0f;

			//Second derivatives
			Scalar phi_xx = lsArray(i + 1, j) - 2*lsArray(i, j) + lsArray(i - 1, j);
			phi_xx /= dx*dx;
			Scalar phi_yy = lsArray(i, j + 1) - 2*lsArray(i, j) + lsArray(i, j - 1);
			phi_yy /= dx*dx;

			//First derivatives
			Scalar phi_x = lsArray(i + 1, j) - lsArray(i - 1, j);
			phi_x /= dx*2;

			Scalar phi_y = lsArray(i, j + 1) - lsArray(i, j - 1);
			phi_y /= dx*2;

			//Cross derivatives
			Scalar phi_xy_dy1 = lsArray(i + 1, j + 1) - lsArray(i + 1, j - 1);
			Scalar phi_xy_dy2 = lsArray(i - 1, j + 1) - lsArray(i - 1, j - 1);
			Scalar phi_xy = (phi_xy_dy1/(2*dx) - phi_xy_dy2/(2*dx))/2*dx;
			//phi_xy = lsArray(i + 1, j + 1) - lsArray(i + 1, j - 1) - lsArray(i - 1, j + 1) + lsArray(i - 1, j - 1);
			//phi_xy /= 4*dx*dx;

			Scalar kCurvature = phi_xx*phi_y*phi_y - 2*phi_x*phi_y*phi_xy + phi_yy*phi_x*phi_x;
			kCurvature /= pow((phi_x*phi_x + phi_y*phi_y), 1.5f);

			
			return kCurvature;
		}


		void filterIsoline(vector<Vector2> &isoline, int numFilteringIterations, bool closedLine) {
			vector<Vector2> tempIsoline = isoline;
			for(int k = 0; k < numFilteringIterations; k++) {
				for(int i = 1; i < isoline.size() - 1; i++) {
					isoline[i] = (tempIsoline[i-1] + tempIsoline[i + 1])*0.5;
				}
				if(closedLine) {
					isoline[0] = (tempIsoline[1] + tempIsoline[isoline.size() - 1])*0.5;
					isoline[isoline.size() - 1] = (tempIsoline[0] + tempIsoline[isoline.size() - 2])*0.5;
				}
				tempIsoline = isoline;
			}
		}


		void rotatePoints(const vector<Vector2> & points, vector<Vector2> &dstPoints, Vector2 rotationPoint, Scalar orientation) {
			for (int i = 0; i < points.size(); i++) {
				Vector2 tempPoint = points[i] - rotationPoint;
				tempPoint.rotate(orientation);
				dstPoints[i] = tempPoint + rotationPoint;
			}
		}

		/************************************************************************/
		/* Ray-collision                                                        */
		/************************************************************************/

		Vector3D barycentricWeights(const Vector3D &point, const Vector3D &t1, const Vector3D &t2, const Vector3D &t3) {
			Vector3D barycentricWeigts;

			if ((point - t1).length() < 1e-6) {
				return Vector3D(1, 0, 0);
			} else if ((point - t2).length() < 1e-6) {
				return Vector3D(0, 1, 0);
			} else if ((point - t3).length() < 1e-6) {
				return Vector3D(0, 0, 1);
			}
			Vector3D v0 = t2 - t1, v1 = t3 - t1, v2 = point - t1;

			DoubleScalar d00 = v0.dot(v0);
			DoubleScalar d01 = v0.dot(v1);
			DoubleScalar d11 = v1.dot(v1);
			DoubleScalar d20 = v2.dot(v0);
			DoubleScalar d21 = v2.dot(v1);
			DoubleScalar denom = d00 * d11 - d01 * d01;
			barycentricWeigts.x = (d11 * d20 - d01 * d21) / denom;
			barycentricWeigts.y = (d00 * d21 - d01 * d20) / denom;
			barycentricWeigts.z = 1.0f - barycentricWeigts.x - barycentricWeigts.y;

			return barycentricWeigts;
		}
		


		//int rayTriangleIntersect(const Vector3D &point, const Vector3D &rayDirection, const Vector3D trianglePoints[3], const Vector3D &triangleNormal) {
		//	DoubleScalar doublePrecisionThreshold = 1e-10;
		//	//Barycentric weights 
		//	DoubleScalar b[3];

		//	// Edge vectors
		//	const Vector3D& e_1 = trianglePoints[1] - trianglePoints[0];
		//	const Vector3D& e_2 = trianglePoints[2] - trianglePoints[0];

		//	Vector3D pvec = rayDirection.cross(e_2);
		//	DoubleScalar det = e_1.dot(pvec);

		//	if (det <= doublePrecisionThreshold) {
		//		/*if (det == 0)
		//			return -1;
		//		else*/
		//			return 0;
		//	}
		//	const Vector3D &tVec = point - trianglePoints[0];

		//	b[0] = tVec.dot(pvec);

		//	//Check if the ray passes on top of an edge or vertex
		//	if (b[0] < doublePrecisionThreshold) {
		//		if (b[0] >= 0) 
		//			return -1;
		//		return 0;
		//	} else if(b[0] >= det - doublePrecisionThreshold) {
		//		if (b[0] - det <= doublePrecisionThreshold)
		//			return -1;
		//		return 0;
		//	}

		//	const Vector3D& qVec = tVec.cross(e_1);

		//	b[1] = rayDirection.dot(qVec);

		//	//Check if the ray passes on top of an edge or vertex
		//	if (b[1] < doublePrecisionThreshold) {
		//		if (b[1] >= 0) 
		//			return -1;
		//		return 0;
		//	} else if(b[0] + b[1] >= det - doublePrecisionThreshold) {
		//		if (b[0] + b[1] - det <= doublePrecisionThreshold)
		//			return -1;
		//		return 0;
		//	}

		//	return 1;
		//}



	}
}