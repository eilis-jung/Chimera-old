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

#ifndef _MATH_MEAN_VALUE_INTERPOLANT_3D_H_
#define _MATH_MEAN_VALUE_INTERPOLANT_3D_H_

#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraCutCells.h"

#include "Interpolation/Interpolant.h"
#include "Interpolation/BilinearNodalInterpolant3D.h"

namespace Chimera {

	using namespace Core;
	using namespace Grids;
	using namespace CutCells;

	namespace Interpolation {

		/** Bilinear nodal interpolation for regular grids*/
		template <class valueType>
		class MeanValueInterpolant3D : public Interpolant<valueType, Array3D, Vector3> {

		public:

		#pragma region Constructors
			/** Standard constructor - responsible for the velocity interpolation on all the grid */
			MeanValueInterpolant3D(const Array3D<valueType> &values, CutVoxels3D<Vector3> *pCutVoxels, CutVoxelsVelocities3D *pCutVoxelsVelocities, Scalar gridDx, bool useAuxVels = false);
		#pragma endregion


		#pragma region Functionalities
			/* Basic interpolation function */
			virtual valueType interpolate(const Vector3 &position);

			/** Same as staggeredToNodeCentered, but considers cut-cells adjacency to return nodal-based velocities. */
			virtual void updateNodalVelocities(const Array3D<Vector3> &sourceStaggered, Array3D<Vector3> &targetNodal, bool useAuxVels = false);
		#pragma endregion

		#pragma region AccessFunctions


		#pragma endregion
		protected:
		#pragma region PrivateFunctionalities
			virtual valueType interpolateCutVoxel(int ithCutCell, const Vector3 &position);

			/*FORCE_INLINE Scalar calculateADet(const Vector3 &v1, const Vector3 &v2) {
				Matrix2x2 mat;
				mat.column[0] = v1;
				mat.column[1] = v2;
				return mat.determinant();
			}*/
		#pragma endregion
		#pragma region ClassMembers
			BilinearNodalInterpolant3D<valueType> *m_pNodalInterpolant;
			CutVoxels3D<Vector3> *m_pCutVoxels;
			CutVoxelsVelocities3D *m_pCutVoxelsVelocity;
			Scalar m_dx;
			bool m_useAuxiliaryVelocities;
		#pragma endregion
		};
	}


}
#endif