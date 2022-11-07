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

#ifndef _MATH_CUBIC_STREAMFUNCTION_INTERPOLANT_2D_H_
#define _MATH_CUBIC_STREAMFUNCTION_INTERPOLANT_2D_H_

#pragma once

#include "Interpolation/Interpolant.h"
#include "Interpolation/BilinearStreamfunctionInterpolant2D.h"
#include "Interpolation/CubicMVCs.h"

namespace Chimera {

	using namespace Core;

	namespace Interpolation {

		/** Cubic Streamfunction Interpolant: uses the analytic form of the cubic interpolation gradient to recover vectors 
		  * from nodal streamfunction values. It does not make sense to use a template different then Vector2 with this 
		  * class, but since CHEWBACCA COMES FROM ENDOR AND THAT DOES NOT MAKE SENSE, I WILL LEAVE THE TEMPLATE HERE. 
		  * (Chewbacca Defense, thanks Johnnie Cochran) */
		template <class valueType>
		class CubicStreamfunctionInterpolant2D : public BilinearStreamfunctionInterpolant2D<valueType> {

		public:

		#pragma region Constructors
			CubicStreamfunctionInterpolant2D(const Array2D<valueType> &values, Scalar gridDx);

			/** Initialization for discontinuous streamfunction per grid cell. */
			CubicStreamfunctionInterpolant2D(Array2D<vector<Scalar>> *pStreamfunctionValues, Scalar gridDx);

			/** Initialization for a single unified streamfunction per domain. */
			CubicStreamfunctionInterpolant2D(Array2D<Scalar> *pStreamfunctionValues, Scalar gridDx);
		#pragma endregion


		#pragma region Functionalities
			/* Basic interpolation function */
			virtual valueType interpolate(const Vector2 &position);

			void saveCellInfoToFile(int i, int j, int numSubdivisions, const string &filename);
		#pragma endregion

		protected:
		
		#pragma region PrivateFunctionalities
			/** Interpolates an scalar inside a cut-cell using cubic mvcs */
			Scalar interpolateScalar(const Vector2 &position, int currCellIndex);
			Scalar interpolateScalar(const Vector2 &position, const vector<Scalar> &streamfunctionValues);

			Scalar getAdjacentCellVelocity(const HalfFace<Vector2> &cutCell, halfEdgeLocation_t cutEdgeLocation, const Vector2 &matchingPoint);
			pair<Scalar, Scalar> getTangentialDerivatives(int currCellIndex, int edgeIndex);
			pair<Scalar, Scalar> getNormalDerivatives(int currCellIndex, int edgeIndex);
			Scalar getFaceVelocity(const dimensions_t &cellLocation, halfEdgeLocation_t currEdgeLocation, HalfFace<Vector2> *pNextCell, const Vector2 &initialPoint);


			Vector2 interpolateCutCell(const Vector2 &position) override;

			/** Partial derivative in respect to X of the bilinear interpolation function. */
			Scalar cubicPartialDerivativeX(const Vector2 &position, const vector<Scalar> &streamfunctionValues);

			/** Partial derivative in respect to Y of the bilinear interpolation function. */
			Scalar cubicPartialDerivativeY(const Vector2 &position, const vector<Scalar> &streamfunctionValues);

			void calculateDiscontinuousDerivatives(int i, int j, Scalar *gradientsX, Scalar *gradientsY, Scalar *gradientsXY);
			void calculateContinuousDerivatives(int i, int j, Scalar *gradientsX, Scalar *gradientsY, Scalar *gradientsXY);
			void generateCubicTable(const Scalar origFunction[4], const Scalar gradientsX[4], const Scalar gradientsY[4],
										const Scalar crossDerivatives[4], Vector2 scaleFactor, Scalar *cubicTable);
		#pragma endregion
		#pragma region ClassMembers
			static const int coefficientsMatrix[256];
		#pragma endregion
		};
	}


}
#endif