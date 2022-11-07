#pragma once
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

#ifndef _MATH_SUBDIVISION_STREAMFUNCTION_INTERPOLANT_2D_H_
#define _MATH_SUBDIVISION_STREAMFUNCTION_INTERPOLANT_2D_H_

#pragma once

#include "Interpolation/Interpolant.h"
#include "Interpolation/BilinearStreamfunctionInterpolant2D.h"

namespace Chimera {

	using namespace Core;
	using namespace CutCells;
	namespace Interpolation {

		/** Subdivision Streamfunction Interpolant: constructed on top of the BilinearStreamfunction interpolant, it
		  * subdivides a cell with more points, but uses a higher order interpolation scheme to find these values.
		  * Then, it uses a standard mean value interpolant to interpolate values inside cells. */
		template <class valueType>
		class SubdivisionStreamfunctionInterpolant2D : public BilinearStreamfunctionInterpolant2D<valueType> {

		public:

		#pragma region Constructors
			/** Standard constructor. Use this when the update of the streamfunction values is responsibility of this 
			  * class. */
			SubdivisionStreamfunctionInterpolant2D(const Array2D<valueType> &values, Scalar gridDx, Scalar subdivisionDx);
		#pragma endregion

		#pragma region AccessFunctions
			virtual void setCutCellsVelocities(CutCellsVelocities2D *pCutCellsVelocities) {
				BilinearStreamfunctionInterpolant2D::setCutCellsVelocities(pCutCellsVelocities);

			}
		#pragma endregion

		#pragma region Functionalities
			/* Basic interpolation function */
			virtual valueType interpolate(const Vector2 &position);

			/** Computes the streamfunction values based on the fluxes stored on the original values passed on the
			* constructor*/
			virtual void computeStreamfunctions();

			void saveCellInfoToFile(int i, int j, int numSubdivisions, const string &filename);
		#pragma endregion

		protected:
		#pragma region PrivateStructures
			typedef struct subdividedEdge_t {
				vector<Vector2> points;
				vector<Scalar> streamfunctionValues;

				subdividedEdge_t() {

				}
				subdividedEdge_t(const Vector2 &initialPoint, const Vector2 &finalPoint, Scalar subdivisionDx) {
					initialize(initialPoint, finalPoint, subdivisionDx);
				}

				void initialize(const Vector2 &initialPoint, const Vector2 &finalPoint, Scalar subdivisionDx) {
					points.push_back(initialPoint);
					streamfunctionValues.push_back(0);
					//Fill in points 
					Scalar edgeLength = (finalPoint - initialPoint).length();
					Vector2 edgeVec = (finalPoint - initialPoint).normalized();
					int numSubdivis = floor(edgeLength / subdivisionDx);
					if (numSubdivis > 0) {
						Scalar newDx = edgeLength / numSubdivis;
						for (int i = 0; i < numSubdivis - 1; i++) {
							points.push_back(initialPoint + edgeVec*(i + 1)*newDx);
							streamfunctionValues.push_back(0);
						}
					}
					streamfunctionValues.push_back(0);
					points.push_back(finalPoint);
				}
			} subdividedEdge_t;
		#pragma endregion

		#pragma region PrivateFunctionalities
			FORCE_INLINE Scalar calculateADet(const Vector2 &v1, const Vector2 &v2) {
				Matrix2x2 mat;
				mat.column[0] = v1;
				mat.column[1] = v2;
				return mat.determinant();
			}
			Scalar interpolateScalar(const Vector2 &position, const vector<subdividedEdge_t> edges);

			void initializeSubdividedEdges();

			void computeCellStreamfunctionsLinear(int i, int j);
			void computeCellStreamfunctionsCubic(int i, int j);


			Scalar cubicInterpolation(Scalar x, Scalar valueIni, Scalar valueFinal, Scalar derivIni, Scalar derivFinal);
		#pragma endregion


		
		#pragma region ClassMembers
			Scalar m_subdivisionDx;
			Array2D<vector<subdividedEdge_t>> m_regularEdges;
			vector<vector<subdividedEdge_t>> m_subDividedEdges;
		#pragma endregion
		};
	}


}
#endif