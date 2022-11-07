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

#include "Math/MathUtilsCore.h"

namespace Chimera {
	namespace Core {

		template<>
		bool isInBetween(const Vector2 &p1, const Vector2 &p2, const Vector2 &position) {
			Vector2 pMin = Vector2(p1.x < p2.x ? p1.x : p2.x, p1.y < p2.y ? p1.y : p2.y);
			Vector2 pMax = Vector2(p1.x > p2.x ? p1.x : p2.x, p1.y > p2.y ? p1.y : p2.y);
			if (position.x >= pMin.x && position.y >= pMin.y && position.x <= pMax.x && position.y <= pMax.y)
				return true;
			return false;
		}

		template<>
		bool isInBetween(const Vector2D &p1, const Vector2D &p2, const Vector2D &position) {
			Vector2D pMin = Vector2D(p1.x < p2.x ? p1.x : p2.x, p1.y < p2.y ? p1.y : p2.y);
			Vector2D pMax = Vector2D(p1.x > p2.x ? p1.x : p2.x, p1.y > p2.y ? p1.y : p2.y);
			if (position.x >= pMin.x && position.y >= pMin.y && position.x <= pMax.x && position.y <= pMax.y)
				return true;
			return false;
		}

		template <>
		bool isInBetween(const Vector3 &p1, const Vector3 &p2, const Vector3 &position) {
			Vector3 pMin = Vector3(p1.x < p2.x ? p1.x : p2.x, p1.y < p2.y ? p1.y : p2.y, p1.z < p2.z ? p1.z : p2.z);
			Vector3 pMax = Vector3(p1.x > p2.x ? p1.x : p2.x, p1.y > p2.y ? p1.y : p2.y, p1.z > p2.z ? p1.z : p2.z);
			if (position.x >= pMin.x && position.y >= pMin.y && position.z >= pMin.z &&
				position.x <= pMax.x && position.y <= pMax.y && position.z <= pMax.z)
				return true;
			return false;
		}
		template <>
		bool isInBetween(const Vector3D &p1, const Vector3D &p2, const Vector3D &position) {
			Vector3D pMin = Vector3D(p1.x < p2.x ? p1.x : p2.x, p1.y < p2.y ? p1.y : p2.y, p1.z < p2.z ? p1.z : p2.z);
			Vector3D pMax = Vector3D(p1.x > p2.x ? p1.x : p2.x, p1.y > p2.y ? p1.y : p2.y, p1.z > p2.z ? p1.z : p2.z);
			if (position.x >= pMin.x && position.y >= pMin.y && position.z >= pMin.z &&
				position.x <= pMax.x && position.y <= pMax.y && position.z <= pMax.z)
				return true;
			return false;
		}

		/************************************************************************/
		/* Limiters  2D                                                         */
		/************************************************************************/
		/** Scalar quantities 2D*/
		Scalar getMinLimiter(const Vector2 &position, Core::Array2D<Scalar> &scalarField) {
			Scalar limiter;
			dimensions_t gridDimensions = scalarField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x);
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y);

			limiter = std::min(scalarField(i, j), scalarField(i + 1, j));
			limiter = std::min(limiter, scalarField(i, j + 1));
			limiter = std::min(limiter, scalarField(i + 1, j + 1));
			
			return limiter;
		}

		Scalar getMaxLimiter(const Vector2 &position, Core::Array2D<Scalar> &scalarField) {
			Scalar limiter;
			dimensions_t gridDimensions = scalarField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x);
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y);

			limiter = std::max(scalarField(i, j), scalarField(i + 1, j));
			limiter = std::max(limiter, scalarField(i, j + 1));
			limiter = std::max(limiter, scalarField(i + 1, j + 1));
			
			return limiter;
		}

		Scalar getMinLimiterX(const Vector2 &position, Core::Array2D<Vector2> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y - 1);

			limiter = std::min(velocityField(i, j).x, velocityField(i + 1, j).x);
			limiter = std::min(limiter, velocityField(i, j + 1).x);
			limiter = std::min(limiter, velocityField(i + 1, j + 1).x);

			return limiter;
		}

		Scalar getMaxLimiterX(const Vector2 &position, Core::Array2D<Vector2> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y - 1);

			limiter = std::max(velocityField(i, j).x, velocityField(i + 1, j).x);
			limiter = std::max(limiter, velocityField(i, j + 1).x);
			limiter = std::max(limiter, velocityField(i + 1, j + 1).x);

			return limiter;
		}
		

		Scalar getMinLimiterY(const Vector2 &position, Core::Array2D<Vector2> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y));
			j = clamp(j, 0, gridDimensions.y - 1);

			limiter = std::min(velocityField(i, j).y, velocityField(i + 1, j).y);
			limiter = std::min(limiter, velocityField(i, j + 1).y);
			limiter = std::min(limiter, velocityField(i + 1, j + 1).y);

			return limiter;
		}

		Scalar getMaxLimiterY(const Vector2 &position, Core::Array2D<Vector2> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y));
			j = clamp(j, 0, gridDimensions.y - 1);

			limiter = std::max(velocityField(i, j).y, velocityField(i + 1, j).y);
			limiter = std::max(limiter, velocityField(i, j + 1).y);
			limiter = std::max(limiter, velocityField(i + 1, j + 1).y);

			return limiter;
		}
			

		/************************************************************************/
		/* Limiters  3D                                                         */
		/************************************************************************/

		Scalar getMinLimiter(const Vector3 &position, Core::Array3D<Scalar> &scalarField) {
			Scalar limiter;
			dimensions_t gridDimensions = scalarField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x );
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y);
			int k = static_cast <int> (floor(position.z - 0.5));
			k = clamp(k, 0, gridDimensions.z);


			limiter = std::min(scalarField(i, j, k), scalarField(i + 1, j, k));
			limiter = std::min(limiter, scalarField(i, j + 1, k));
			limiter = std::min(limiter, scalarField(i + 1, j + 1, k));
			limiter = std::min(limiter, scalarField(i, j, k + 1));
			limiter = std::min(limiter, scalarField(i + 1, j, k + 1));
			limiter = std::min(limiter, scalarField(i, j + 1, k + 1));
			limiter = std::min(limiter, scalarField(i + 1, j + 1, k + 1));

			return limiter;
		}

		Scalar getMaxLimiter(const Vector3 &position, Core::Array3D<Scalar> &scalarField) {
			Scalar limiter;
			dimensions_t gridDimensions = scalarField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x );
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y);
			int k = static_cast <int> (floor(position.z - 0.5));
			k = clamp(k, 0, gridDimensions.z);


			limiter = std::max(scalarField(i, j, k), scalarField(i + 1, j, k));
			limiter = std::max(limiter, scalarField(i, j + 1, k));
			limiter = std::max(limiter, scalarField(i + 1, j + 1, k));
			limiter = std::max(limiter, scalarField(i, j, k + 1));
			limiter = std::max(limiter, scalarField(i + 1, j, k + 1));
			limiter = std::max(limiter, scalarField(i, j + 1, k + 1));
			limiter = std::max(limiter, scalarField(i + 1, j + 1, k + 1));

			return limiter;

		}
		Scalar getMinLimiterX(const Vector3 &position, Core::Array3D<Vector3> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y - 1);
			int k = static_cast <int> (floor(position.z - 0.5));
			k = clamp(k, 0, gridDimensions.z - 1);

			limiter = std::min(velocityField(i, j, k).x, velocityField(i + 1, j, k).x);
			limiter = std::min(limiter, velocityField(i, j + 1, k).x);
			limiter = std::min(limiter, velocityField(i + 1, j + 1, k).x);

			limiter = std::min(limiter, velocityField(i, j, k + 1).x);
			limiter = std::min(limiter, velocityField(i + 1, j, k + 1).x);
			limiter = std::min(limiter, velocityField(i, j + 1, k + 1).x);
			limiter = std::min(limiter, velocityField(i + 1, j + 1, k + 1).x);

			return limiter;
		}

		Scalar getMaxLimiterX(const Vector3 &position, Core::Array3D<Vector3> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y - 1);
			int k = static_cast <int> (floor(position.z - 0.5));
			k = clamp(k, 0, gridDimensions.z - 1);

			limiter = std::max(velocityField(i, j, k).x, velocityField(i + 1, j, k).x);
			limiter = std::max(limiter, velocityField(i, j + 1, k).x);
			limiter = std::max(limiter, velocityField(i + 1, j + 1, k).x);

			limiter = std::max(limiter, velocityField(i, j, k + 1).x);
			limiter = std::max(limiter, velocityField(i + 1, j, k + 1).x);
			limiter = std::max(limiter, velocityField(i, j + 1, k + 1).x);
			limiter = std::max(limiter, velocityField(i + 1, j + 1, k + 1).x);

			return limiter;
		}


		Scalar getMinLimiterY(const Vector3 &position, Core::Array3D<Vector3> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y));
			j = clamp(j, 0, gridDimensions.y - 1);
			int k = static_cast <int> (floor(position.z - 0.5));
			k = clamp(k, 0, gridDimensions.z - 1);

			limiter = std::min(velocityField(i, j, k).y, velocityField(i + 1, j, k).y);
			limiter = std::min(limiter, velocityField(i, j + 1, k).y);
			limiter = std::min(limiter, velocityField(i + 1, j + 1, k).y);

			limiter = std::min(limiter, velocityField(i, j, k + 1).y);
			limiter = std::min(limiter, velocityField(i + 1, j, k + 1).y);
			limiter = std::min(limiter, velocityField(i, j + 1, k + 1).y);
			limiter = std::min(limiter, velocityField(i + 1, j + 1, k + 1).y);

			return limiter;
		}

		Scalar getMaxLimiterY(const Vector3 &position, Core::Array3D<Vector3> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y));
			j = clamp(j, 0, gridDimensions.y - 1);
			int k = static_cast <int> (floor(position.z - 0.5));
			k = clamp(k, 0, gridDimensions.z - 1);

			limiter = std::max(velocityField(i, j, k).y, velocityField(i + 1, j, k).y);
			limiter = std::max(limiter, velocityField(i, j + 1, k).y);
			limiter = std::max(limiter, velocityField(i + 1, j + 1, k).y);

			limiter = std::max(limiter, velocityField(i, j, k + 1).y);
			limiter = std::max(limiter, velocityField(i + 1, j, k + 1).y);
			limiter = std::max(limiter, velocityField(i, j + 1, k + 1).y);
			limiter = std::max(limiter, velocityField(i + 1, j + 1, k + 1).y);

			return limiter;
		}

		Scalar getMinLimiterZ(const Vector3 &position, Core::Array3D<Vector3> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y - 1);
			int k = static_cast <int> (floor(position.z));
			k = clamp(k, 0, gridDimensions.z - 1);

			limiter = std::min(velocityField(i, j, k).z, velocityField(i + 1, j, k).z);
			limiter = std::min(limiter, velocityField(i, j + 1, k).z);
			limiter = std::min(limiter, velocityField(i + 1, j + 1, k).z);

			limiter = std::min(limiter, velocityField(i, j, k + 1).z);
			limiter = std::min(limiter, velocityField(i + 1, j, k + 1).z);
			limiter = std::min(limiter, velocityField(i, j + 1, k + 1).z);
			limiter = std::min(limiter, velocityField(i + 1, j + 1, k + 1).z);

			return limiter;
		}

		Scalar getMaxLimiterZ(const Vector3 &position, Core::Array3D<Vector3> &velocityField) {
			Scalar limiter;
			dimensions_t gridDimensions = velocityField.getDimensions();

			int i = static_cast <int> (floor(position.x - 0.5));
			i = clamp(i, 0, gridDimensions.x - 1);
			int j = static_cast <int> (floor(position.y - 0.5));
			j = clamp(j, 0, gridDimensions.y - 1);
			int k = static_cast <int> (floor(position.z));
			k = clamp(k, 0, gridDimensions.z - 1);

			limiter = std::max(velocityField(i, j, k).z, velocityField(i + 1, j, k).z);
			limiter = std::max(limiter, velocityField(i, j + 1, k).z);
			limiter = std::max(limiter, velocityField(i + 1, j + 1, k).z);

			limiter = std::max(limiter, velocityField(i, j, k + 1).z);
			limiter = std::max(limiter, velocityField(i + 1, j, k + 1).z);
			limiter = std::max(limiter, velocityField(i, j + 1, k + 1).z);
			limiter = std::max(limiter, velocityField(i + 1, j + 1, k + 1).z);

			return limiter;
		}



	}
}