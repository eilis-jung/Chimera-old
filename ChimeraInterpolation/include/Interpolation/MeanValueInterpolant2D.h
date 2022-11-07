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

#ifndef _MATH_MEAN_VALUE_INTERPOLANT_2D_H_
#define _MATH_MEAN_VALUE_INTERPOLANT_2D_H_

#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraCutCells.h"
#include "CutCells/CutCellsVelocities2D.h"

#include "Interpolation/Interpolant.h"
#include "Interpolation/BilinearNodalInterpolant2D.h"

namespace Chimera {

	using namespace Core;
	using namespace Grids;
	using namespace CutCells;

	namespace Interpolation {

		/** Bilinear nodal interpolation for regular grids*/
		template <class valueType>
		class MeanValueInterpolant2D : public Interpolant<valueType, Array2D, Vector2> {

		public:

		#pragma region Constructors


			/** Standard constructor - responsible for the velocity interpolation on all the grid */
			MeanValueInterpolant2D(const Array2D<valueType> &values, CutCellsVelocities2D *m_pCutCellsVelocities, Scalar gridDx, bool useAuxiliaryVelocities = false);
			
			/** Reduced constructor - will interpolate values only on cut-cells */
			MeanValueInterpolant2D(CutCellsVelocities2D *m_pCutCellsVelocities, Scalar gridDx, bool useAuxiliaryVelocities = false);

			/** Alternative constructor without cut-cells velocities: usually used for scalar-field functions, no need 
			    for auxiliary velocities. */
			MeanValueInterpolant2D(const Array2D<valueType> &values, CutCells2D<Vector2> *pCutCells2D, Scalar gridDx);
		#pragma endregion


		#pragma region Functionalities
			/* Basic interpolation function */
			virtual valueType interpolate(const Vector2 &position);

			/** Special function for calculating weights - does not needs a class to be called */
			//static virtual vector<DoubleScalar> calculateWeights(const Vector2 &position, const vector<Vector2> &polygonPoints);

			/** Same as staggeredToNodeCentered, but considers cut-cells adjacency to return nodal-based velocities. */
			virtual void updateNodalVelocities(const Array2D<Vector2> &sourceStaggered, Array2D<Vector2> &targetNodal, bool useAuxVels = false);
		#pragma endregion

		#pragma region AccessFunctions
		void setCutCells2D(CutCells2D<Vector2> *pCutCells) {
			m_pCutCells2D = pCutCells;
		}

		CutCells2D<Vector2> * getCutCells2D() {
			return m_pCutCells2D;
		}
		#pragma endregion
		protected:
		#pragma region PrivateFunctionalities
			virtual valueType interpolateCutCell(int ithCutCell, const Vector2 &position);

			FORCE_INLINE Scalar calculateADet(const Vector2 &v1, const Vector2 &v2) {
				Matrix2x2 mat;
				mat.column[0] = v1;
				mat.column[1] = v2;
				return mat.determinant();
			}
		#pragma endregion
		#pragma region ClassMembers
			BilinearNodalInterpolant2D<valueType> *m_pNodalInterpolant;
			CutCells2D<Vector2> *m_pCutCells2D;
			CutCellsVelocities2D *m_pCutCellsVelocities2D;
			Scalar m_dx;
			bool m_useAuxiliaryVelocities;
		#pragma endregion
		};
	}


}
#endif