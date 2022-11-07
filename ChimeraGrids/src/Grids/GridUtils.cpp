#include "Grids/GridUtils.h"

namespace Chimera {

	namespace Grids {
		
		#pragma region CheckingFunctions
		
		/************************************************************************/
		/* isOnGridPoint(const VectorT &point, Scalar dx);						*/
		/************************************************************************/
		template<>
		bool isOnGridPoint(const Vector2 &point, Scalar dx) {
			Vector2 tempNodalPoint = (point)/dx;
			Scalar tempPrecision = singlePrecisionThreshold / dx;
			return 
				(abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision || 
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision) /* vx*/
				&&
				(abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision) /* vy */;
		}

		template<>
		bool isOnGridPoint(const Vector2D &point, Scalar dx) {
			Vector2D tempNodalPoint = (point) / dx;
			Scalar tempPrecision = doublePrecisionThreshold / dx;
			return
				(abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision ||
					abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision) /* vx*/
				&&
				(abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
					abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision) /* vy */;
		}
		template<>
		bool isOnGridPoint(const Vector3 &point, Scalar dx) {
			Vector3 tempNodalPoint = (point) / dx;
			Scalar tempPrecision = singlePrecisionThreshold / dx;
			return
				(abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision ||
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision) /* vx*/
				&&
				(abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision) /* vy */
				&&
				(abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision ||
				abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision) /* vz */;
		}
		template<>
		bool isOnGridPoint(const Vector3D &point, Scalar dx) {
			Vector3D tempNodalPoint = (point) / dx;
			DoubleScalar tempPrecision = doublePrecisionThreshold / dx;
			DoubleScalar vx[2], vy[2];
			vx[0] = abs(tempNodalPoint.x - floor(tempNodalPoint.x));
			vx[1] = abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1);
			vy[0] = abs(tempNodalPoint.y - floor(tempNodalPoint.y));
			vy[1] = abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1);

			return
				(abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision ||
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision) /* vx*/
				&&
				(abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision) /* vy */
				&&
				(abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision ||
				abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision) /* vz */;
		}

		bool isOnGridPoint(const Vector3D &point, DoubleScalar dx, DoubleScalar tolerance) {
			Vector3D tempNodalPoint = (point) / dx;
			return
				(abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tolerance ||
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tolerance) /* vx*/
				&&
				(abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tolerance ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tolerance) /* vy */
				&&
				(abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tolerance ||
				abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tolerance) /* vz */;
		}

		/********************************************************************************/
		/* isOnGridPoint(const VectorT &point, Scalar dx, dimensions_t &pointLocation); */
		/********************************************************************************/
		template<>
		bool isOnGridPoint(const Vector2 &point, Scalar dx, dimensions_t &pointLocation) {
			Vector2 tempNodalPoint = (point)/dx;
			bool vx[2], vy[2];
			Scalar tempPrecision = singlePrecisionThreshold / dx;
			vx[0] = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision;
			vx[1] = abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			vy[0] = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision;
			vy[1] = abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			if(vx[0]) {
				pointLocation.x = static_cast<int>(floor(tempNodalPoint.x));
			} else if(vx[1]) {
				pointLocation.x = static_cast<int>(floor(tempNodalPoint.x)) + 1;
			}
			if(vy[0]) {
				pointLocation.y = static_cast<int>(floor(tempNodalPoint.y));
			} else if(vy[1]) {
				pointLocation.y = static_cast<int>(floor(tempNodalPoint.y)) + 1;
			}
			return (vx[0] || vx[1]) && (vy[0] || vy[1]);
		}

		template<>
		bool isOnGridPoint(const Vector3 &point, Scalar dx, dimensions_t &pointLocation) {
			Vector3 tempNodalPoint = (point)/dx;
			bool vx[2], vy[2], vz[2];
			Scalar tempPrecision = singlePrecisionThreshold / dx;
			vx[0] = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision;
			vx[1] = abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			vy[0] = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision;
			vy[1] = abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			vz[0] = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision;
			vz[1] = abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			if(vx[0]) {
				pointLocation.x = static_cast<int>(floor(tempNodalPoint.x));
			} else if(vx[1]) {
				pointLocation.x = static_cast<int>(floor(tempNodalPoint.x)) + 1;
			}
			if(vy[0]) {
				pointLocation.y = static_cast<int>(floor(tempNodalPoint.y));
			} else if(vy[1]) {
				pointLocation.y = static_cast<int>(floor(tempNodalPoint.y)) + 1;
			}
			if(vz[0]) {
				pointLocation.z = static_cast<int>(floor(tempNodalPoint.z));
			} else if(vz[1]) {
				pointLocation.z = static_cast<int>(floor(tempNodalPoint.z)) + 1;
			}

			return (vx[0] || vx[1]) && (vy[0] || vy[1]) && (vz[0] || vz[1]);
		}

		template<>
		bool isOnGridPoint(const Vector3D &point, Scalar dx, dimensions_t &pointLocation) {
			Vector3D tempNodalPoint = (point) / dx;
			bool vx[2], vy[2], vz[2];
			DoubleScalar tempPrecision = doublePrecisionThreshold / dx;
			vx[0] = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision;
			vx[1] = abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			vy[0] = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision;
			vy[1] = abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			vz[0] = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision;
			vz[1] = abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			if (vx[0]) {
				pointLocation.x = static_cast<int>(floor(tempNodalPoint.x));
			}
			else if (vx[1]) {
				pointLocation.x = static_cast<int>(floor(tempNodalPoint.x)) + 1;
			}
			if (vy[0]) {
				pointLocation.y = static_cast<int>(floor(tempNodalPoint.y));
			}
			else if (vy[1]) {
				pointLocation.y = static_cast<int>(floor(tempNodalPoint.y)) + 1;
			}
			if (vz[0]) {
				pointLocation.z = static_cast<int>(floor(tempNodalPoint.z));
			}
			else if (vz[1]) {
				pointLocation.z = static_cast<int>(floor(tempNodalPoint.z)) + 1;
			}

			return (vx[0] || vx[1]) && (vy[0] || vy[1]) && (vz[0] || vz[1]);
		}

		template<>
		int isOnGridEdge(const Vector2 &point, Scalar dx) {
			Vector2 tempNodalPoint = (point)/dx;
			Scalar tempPrecision = singlePrecisionThreshold / dx;
			bool vx = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision ||
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			bool vy = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;

			if(vx && vy)
				return -1;
			else if(vx)
				return 1;
			else if(vy)
				return 2;
			return 0;
		}

		template<>
		int isOnGridEdge(const Vector2D &point, Scalar dx) {
			Vector2D tempNodalPoint = (point) / dx;
			DoubleScalar tempPrecision = doublePrecisionThreshold / dx;
			bool vx = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision ||
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			bool vy = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;

			if (vx && vy)
				return -1;
			else if (vx)
				return 1;
			else if (vy)
				return 2;
			return 0;
		}

		template<>
		int isOnGridEdge(const Vector3 &point, Scalar dx) {
			Vector3 tempNodalPoint = (point)/ dx;
			Scalar tempPrecision = singlePrecisionThreshold / dx;;
			bool vx = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision ||
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			bool vy = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			bool vz = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision ||
				abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			if (vx && vy && vz)
				return -1;
			else if (vy && vz)
				return 1;
			else if (vx && vz)
				return 2;
			else if (vx && vy)
				return 3;
			return 0;
		}

		template<>
		int isOnGridEdge(const Vector3D &point, Scalar dx) {
			Vector3D tempNodalPoint = (point) / dx;
			DoubleScalar tempPrecision = doublePrecisionThreshold / dx;
			bool vx = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision ||
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			bool vy = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			bool vz = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision ||
				abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			if (vx && vy && vz)
				return -1;
			else if (vy && vz)
				return 1;
			else if (vx && vz)
				return 2;
			else if (vx && vy)
				return 3;
			return 0;
		}

		template<>
		int isOnGridEdge(const Vector3 &point, Scalar dx, dimensions_t &faceLocation) {
			Vector3 tempNodalPoint = (point) / dx;
			Scalar tempPrecision = singlePrecisionThreshold / dx;

			faceLocation.x = static_cast<int>(floor(tempNodalPoint.x));
			faceLocation.y = static_cast<int>(floor(tempNodalPoint.y));
			faceLocation.z = static_cast<int>(floor(tempNodalPoint.z));

			bool vx[2], vy[2], vz[2];
			vx[0] = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision;
			vx[1] = abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			vy[0] = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision;
			vy[1] = abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			vz[0] = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision;
			vz[1] = abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			bool vxx = vx[0] || vx[1];
			bool vyy = vy[0] || vy[1];
			bool vzz = vz[0] || vz[1];

			if (vxx && vyy && vzz) {
				return -1;
			}
			if (vx[1]) {
				faceLocation.x = static_cast<int>(floor(tempNodalPoint.x)) + 1;
			}
			if (vy[1]) {
				faceLocation.y = static_cast<int>(floor(tempNodalPoint.y)) + 1;
			}
			if (vz[1]) {
				faceLocation.z = static_cast<int>(floor(tempNodalPoint.z)) + 1;
			}

			if (vyy && vzz)
				return 1;
			else if (vxx && vzz)
				return 2;
			else if (vxx && vyy)
				return 3;

			return 0;
		}

		template<>
		int isOnGridEdge(const Vector3D &point, Scalar dx, dimensions_t &faceLocation) {
			Vector3D tempNodalPoint = (point) / dx;
			DoubleScalar tempPrecision = doublePrecisionThreshold / dx;

			faceLocation.x = static_cast<int>(floor(tempNodalPoint.x));
			faceLocation.y = static_cast<int>(floor(tempNodalPoint.y));
			faceLocation.z = static_cast<int>(floor(tempNodalPoint.z));

			bool vx[2], vy[2], vz[2];
			vx[0] = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision;
			vx[1] = abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			vy[0] = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision;
			vy[1] = abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			vz[0] = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision;
			vz[1] = abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			bool vxx = vx[0] || vx[1];
			bool vyy = vy[0] || vy[1];
			bool vzz = vz[0] || vz[1];

			if (vxx && vyy && vzz) {
				return -1;
			}
			if (vx[1]) {
				faceLocation.x = static_cast<int>(floor(tempNodalPoint.x)) + 1;
			}
			if (vy[1]) {
				faceLocation.y = static_cast<int>(floor(tempNodalPoint.y)) + 1;
			}
			if (vz[1]) {
				faceLocation.z = static_cast<int>(floor(tempNodalPoint.z)) + 1;
			}

			if (vyy && vzz)
				return 1;
			else if (vxx && vzz)
				return 2;
			else if (vxx && vyy)
				return 3;

			return 0;
		}


		template<>
		int isOnGridFace(const Vector3 &point, Scalar dx) {
			Vector3 tempNodalPoint = (point) / dx;
			Scalar tempPrecision = singlePrecisionThreshold / dx;
			bool vx = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision ||
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			bool vy = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			bool vz = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision ||
				abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			if (vx && vy && vz)
				return -1;
			else if (vz)
				return 1;
			else if (vy)
				return 2;
			else if (vx)
				return 3;
			return 0;
		}

		template<>
		int isOnGridFace(const Vector3D &point, Scalar dx) {
			Vector3D tempNodalPoint = (point) / dx;
			DoubleScalar tempPrecision = doublePrecisionThreshold / dx;
			bool vx = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision ||
				abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			bool vy = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision ||
				abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			bool vz = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision ||
				abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			if (vx && vy && vz)
				return -1;
			else if (vz)
				return 1;
			else if (vy)
				return 2;
			else if (vx)
				return 3;
			return 0;
		}


		template<>
		int isOnGridFace(const Vector3 &point, Scalar dx, dimensions_t &faceLocation) {
			Vector3 tempNodalPoint = (point) / dx;
			Scalar tempPrecision = singlePrecisionThreshold / dx;

			faceLocation.x = static_cast<int>(floor(tempNodalPoint.x));
			faceLocation.y = static_cast<int>(floor(tempNodalPoint.y));
			faceLocation.z = static_cast<int>(floor(tempNodalPoint.z));

			bool vx[2], vy[2], vz[2];
			vx[0] = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision;
			vx[1] = abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			vy[0] = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision;
			vy[1] = abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			vz[0] = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision;
			vz[1] = abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			bool vxx = vx[0] || vx[1];
			bool vyy = vy[0] || vy[1];
			bool vzz = vz[0] || vz[1];

			if (vxx && vyy && vzz) {
				return -1;
			}
			else if (vx[0]) {
				return 3;
			} else if (vx[1]) {
				faceLocation.x = static_cast<int>(floor(tempNodalPoint.x)) + 1;
				return 3;
			} else if (vy[0]) {
				return 2;
			} else if (vy[1]) {
				faceLocation.y = static_cast<int>(floor(tempNodalPoint.y)) + 1;
				return 2;
			} else if (vz[0]) {
				return 1;
			} else if (vz[1]) {
				faceLocation.z = static_cast<int>(floor(tempNodalPoint.z)) + 1;
				return 1;
			}
			return 0;
		}

		template<>
		int isOnGridFace(const Vector3D &point, Scalar dx, dimensions_t &faceLocation) {
			Vector3D tempNodalPoint = (point) / dx;
			DoubleScalar tempPrecision = doublePrecisionThreshold/ dx;

			faceLocation.x = static_cast<int>(floor(tempNodalPoint.x));
			faceLocation.y = static_cast<int>(floor(tempNodalPoint.y));
			faceLocation.z = static_cast<int>(floor(tempNodalPoint.z));

			bool vx[2], vy[2], vz[2];
			vx[0] = abs(tempNodalPoint.x - floor(tempNodalPoint.x)) <= tempPrecision;
			vx[1] = abs(tempNodalPoint.x - floor(tempNodalPoint.x) - 1) <= tempPrecision;
			vy[0] = abs(tempNodalPoint.y - floor(tempNodalPoint.y)) <= tempPrecision;
			vy[1] = abs(tempNodalPoint.y - floor(tempNodalPoint.y) - 1) <= tempPrecision;
			vz[0] = abs(tempNodalPoint.z - floor(tempNodalPoint.z)) <= tempPrecision;
			vz[1] = abs(tempNodalPoint.z - floor(tempNodalPoint.z) - 1) <= tempPrecision;

			bool vxx = vx[0] || vx[1];
			bool vyy = vy[0] || vy[1];
			bool vzz = vz[0] || vz[1];

			if (vxx && vyy && vzz) {
				return -1;
			}
			else if (vx[0]) {
				return 3;
			}
			else if (vx[1]) {
				faceLocation.x = static_cast<int>(floor(tempNodalPoint.x)) + 1;
				return 3;
			}
			else if (vy[0]) {
				return 2;
			}
			else if (vy[1]) {
				faceLocation.y = static_cast<int>(floor(tempNodalPoint.y)) + 1;
				return 2;
			}
			else if (vz[0]) {
				return 1;
			}
			else if (vz[1]) {
				faceLocation.z = static_cast<int>(floor(tempNodalPoint.z)) + 1;
				return 1;
			}
			return 0;
		}

		bool crossedGridEdge(const Vector2 &v1, const Vector2 &v2, Vector2 &crossedPoint, Scalar dx) {
			//Assuming that v1 is inside a grid cell
			int i = static_cast<int>(floor(v1.x / dx));
			int j = static_cast<int>(floor(v1.y / dx));

			Vector2 gridPoints[4];
			gridPoints[0].x = i*dx;			gridPoints[0].y = j*dx;
			gridPoints[1].x = (i + 1)*dx;	gridPoints[1].y = j*dx;
			gridPoints[2].x = (i + 1)*dx;	gridPoints[2].y = (j + 1)*dx;
			gridPoints[3].x = i*dx;			gridPoints[3].y = (j + 1)*dx;
			
			//Left face
			if(DoLinesIntersect(gridPoints[0], gridPoints[3], v1, v2)) {
				crossedPoint = linesIntersection(gridPoints[0], gridPoints[3], v1, v2);
				return true;
			}
			//Right face
			if(DoLinesIntersect(gridPoints[1], gridPoints[2], v1, v2)) {
				crossedPoint = linesIntersection(gridPoints[1], gridPoints[2], v1, v2);
				return true;
			}
			//Bottom face
			if(DoLinesIntersect(gridPoints[0], gridPoints[1], v1, v2)) {
				crossedPoint = linesIntersection(gridPoints[0], gridPoints[1], v1, v2);
				return true;
			}
			//Top face
			if(DoLinesIntersect(gridPoints[2], gridPoints[3], v1, v2)) {
				crossedPoint = linesIntersection(gridPoints[2], gridPoints[3], v1, v2);
				return true;
			}
			return false;
		}
		bool isInsideCell(const Vector2 &point, dimensions_t cellIndex, Scalar dx) {
			if( (point.x - cellIndex.x*dx >= -singlePrecisionThreshold) &&
				((cellIndex.x + 1)*dx - point.x >= -singlePrecisionThreshold) &&
				(point.y - cellIndex.y*dx >= -singlePrecisionThreshold) &&
				((cellIndex.y + 1)*dx - point.y >= -singlePrecisionThreshold)) {
				return true;
			}
			return false;
		}

		bool isInsideGrid(const Vector2 &point, dimensions_t gridDimensions, Scalar dx) {
			Vector2 transformedPoint = point/dx;
			if(floor(transformedPoint.x) < 0.0f || floor(transformedPoint.x) > gridDimensions.x)
				return false;
			if(floor(transformedPoint.y) < 0.0f || floor(transformedPoint.y) > gridDimensions.y)
				return false;
			return true;
		}
		bool isInsideGrid(const Vector3 &point, dimensions_t gridDimensions, Scalar dx) {
			Vector3 transformedPoint = point/dx;
			if(floor(transformedPoint.x) < 0.0f || floor(transformedPoint.x) > gridDimensions.x)
				return false;
			if(floor(transformedPoint.y) < 0.0f || floor(transformedPoint.y) > gridDimensions.y)
				return false;
			if(floor(transformedPoint.z) < 0.0f || floor(transformedPoint.z) > gridDimensions.z)
				return false;
			return true;
		}
		#pragma endregion CheckingFunction

		#pragma region GeometryUtilities
		void snapLinePointsToEdges(vector<Vector2> &thinObjectPoints, Scalar dx) {
			for(unsigned int i = 0; i < thinObjectPoints.size(); ++i) {
				dimensions_t gridPoint;
				if(isOnGridPoint(thinObjectPoints[i], dx, gridPoint)) {
					thinObjectPoints[i].x = gridPoint.x*dx; thinObjectPoints[i].y = gridPoint.y*dx;
				}
			}
		}
		
		void perturbLinePoints(vector<Vector2> &thinObjectPoints, Scalar dx) {
			for(int i = 0; i < thinObjectPoints.size() ; i++) {
				Vector2 transformedThinObjectPoint = thinObjectPoints[i]/dx;
				bool vx = abs(transformedThinObjectPoint.x - floor(transformedThinObjectPoint.x)) <= singlePrecisionThreshold
					|| abs(transformedThinObjectPoint.x - floor(transformedThinObjectPoint.x) - 1) <= singlePrecisionThreshold;
				bool vy = abs(transformedThinObjectPoint.y - floor(transformedThinObjectPoint.y)) <= singlePrecisionThreshold
					|| abs(transformedThinObjectPoint.y - floor(transformedThinObjectPoint.y) - 1) <= singlePrecisionThreshold;
				Vector2 increment;
				if(vx && vy) {
					increment.x = transformedThinObjectPoint.x - floor(transformedThinObjectPoint.x) <= singlePrecisionThreshold ? transformedThinObjectPoint.x - floor(transformedThinObjectPoint.x) :
						transformedThinObjectPoint.x - (floor(transformedThinObjectPoint.x) + 1);
					increment.y = transformedThinObjectPoint.y - floor(transformedThinObjectPoint.y) <= singlePrecisionThreshold ? transformedThinObjectPoint.y - floor(transformedThinObjectPoint.y) :
						transformedThinObjectPoint.y - (floor(transformedThinObjectPoint.y) + 1);
					increment.normalize();
					if(increment.length() == 0) { //Believe me, it does happen
						increment.x = 1;
						increment.y = 0;
					}
					increment.x = 1; increment.y = 0;
					Logger::getInstance()->get() << "Perturbing thinObject point at " << transformedThinObjectPoint.x << " " << transformedThinObjectPoint.y << endl;
					Logger::getInstance()->get() << "Increment is " << increment.x << " " << increment.y << endl;
					//} else if(vx) {
					//	increment.x = transformedThinObjectPoint.x - floor(transformedThinObjectPoint.x) <= currThreshold ? transformedThinObjectPoint.x - floor(transformedThinObjectPoint.x) :
					//		transformedThinObjectPoint.x - (floor(transformedThinObjectPoint.x) + 1);
					//	increment.y = 0;
					//	increment.normalize();
					//	if(increment.length() == 0) { //Believe me, it does happen
					//		increment.x = 1;
					//		increment.y = 0;
					//	}
					//} else if(vy) {
					//	increment.x = 0;
					//	increment.y = transformedThinObjectPoint.y - floor(transformedThinObjectPoint.y) <= currThreshold ? transformedThinObjectPoint.y - floor(transformedThinObjectPoint.y) :
					//		transformedThinObjectPoint.y - (floor(transformedThinObjectPoint.y) + 1);
					//	increment.normalize();
					//	if(increment.length() == 0) { //Believe me, it does happen
					//		increment.x = 0;
					//		increment.y = 1;
					//	}
				} else {
					increment.x = increment.y = 0;
				}
				thinObjectPoints[i] += increment*dx*singlePrecisionThreshold * 2;
				if(increment.length() > 0)
					Logger::getInstance()->get() << "New thinObject point is " << thinObjectPoints[i].x << " " << thinObjectPoints[i].y << endl;
			}
		}
		void extendLocalPointsToGridEdges(vector<Vector2> &thinObjectPoints, Scalar dx) {
			Vector2 v1, v2, crossedPoint;
			v1 = thinObjectPoints[1];
			v2 = thinObjectPoints[0];
			if(isOnGridEdge(v2, dx))
				return;
			Vector2 direction = (v2 - v1).normalized();
			Scalar stepSize = (v2 - v1).length();			
			do { //Now we are going to step on the direction of v1v2 until we find a grid edge
				direction = (v2 - v1).normalized();
				stepSize = (v2 - v1).length();
				v1 = v2;
				v2 += direction*stepSize;
				thinObjectPoints.insert(thinObjectPoints.begin(), v2);
			} while(!crossedGridEdge(v1, v2, crossedPoint, dx));
			thinObjectPoints.erase(thinObjectPoints.begin());
			thinObjectPoints.insert(thinObjectPoints.begin(), crossedPoint + direction*stepSize*0.01);

			v1 = thinObjectPoints[thinObjectPoints.size() - 2];
			v2 = thinObjectPoints[thinObjectPoints.size() - 1];

			do { //Now we are going to step on the direction of v1v2 until we find a grid edge
				direction = (v2 - v1).normalized();
				stepSize = (v2 - v1).length();
				v1 = v2;
				v2 += direction*stepSize;
				thinObjectPoints.push_back(v2);
			} while(!crossedGridEdge(v1, v2, crossedPoint, dx));
			thinObjectPoints.pop_back();
			thinObjectPoints.push_back(crossedPoint + direction*stepSize*0.01);
		}

		#pragma endregion GeometryUtilities
	}
}