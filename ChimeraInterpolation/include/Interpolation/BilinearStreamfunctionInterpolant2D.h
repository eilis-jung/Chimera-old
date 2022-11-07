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

#ifndef _MATH_BILINEAR_STREAMFUNCTION_INTERPOLANT_2D_H_
#define _MATH_BILINEAR_STREAMFUNCTION_INTERPOLANT_2D_H_

#pragma once

#include "Interpolation/Interpolant.h"
#include "Interpolation/MeanValueInterpolant2D.h"

namespace Chimera {

	using namespace Core;
	using namespace CutCells;
	namespace Interpolation {

		/** Bilinear Streamfunction Interpolant: uses the analytic form of the bilinear gradient to recover vectors from
		  * nodal streamfunction values. It does not make sense to use a template different then Vector2 with this class,
		  * but since CHEWBACCA COMES FROM ENDOR AND THAT DOES NOT MAKE SENSE, I WILL LEAVE THE TEMPLATE HERE. 
		  * (Chewbacca Defense, thanks Johnnie Cochran) */
		template <class valueType>
		class BilinearStreamfunctionInterpolant2D : public Interpolant<valueType, Array2D, Vector2> {

		public:

		#pragma region Constructors
			/** Standard constructor. Use this when the update of the streamfunction values is responsibility of this 
			  * class. */
			BilinearStreamfunctionInterpolant2D(const Array2D<valueType> &values, Scalar gridDx);

			/** External update of the streamfunctions constructor. One has to update streamfunctions EXPLICITLY to use
			  * this constructor. This constructor assumes discontinuous streamfunction per grid cells. */
			BilinearStreamfunctionInterpolant2D(Array2D<vector<Scalar>> *pStreamfunctionValues, Scalar gridDx);

			/** External update of the streamfunctions constructor. One has to update streamfunctions EXPLICITLY to use
			* this constructor. This constructor assumes for a single unified streamfunction per domain. */
			BilinearStreamfunctionInterpolant2D(Array2D<Scalar> *pStreamfunctionValues, Scalar gridDx);

		#pragma endregion


		#pragma region Functionalities
			/* Basic interpolation function */
			virtual valueType interpolate(const Vector2 &position);

			/** Computes the streamfunction values based on the fluxes stored on the original values passed on the 
			  * constructor*/
			virtual void computeStreamfunctions();

			/** Compute continuous streamfunctions for all the grid*/
			virtual void computeContinuousStreamfunctions();
		#pragma endregion

		#pragma region AccessFunctions
			Array<vector<Scalar>> * getStreamfunctionValuesArrayPtr() {
				return m_pStreamDiscontinuousValues;
			}

			virtual void setCutCellsVelocities(CutCellsVelocities2D *pCutCellsVelocities) {
				if (m_pMeanvalueInterpolant)
					delete m_pMeanvalueInterpolant;

				m_pCutCellsVelocities = pCutCellsVelocities;
				m_pCutCells = dynamic_cast<CutCells2D<Vector2> *>(pCutCellsVelocities->getMesh());

				//Use reduced interpolant: only neeed for cut-cells interpolation, no nodal-based regular grid interpolation
				m_pMeanvalueInterpolant = new MeanValueInterpolant2D<Scalar>(pCutCellsVelocities, m_dx);
				m_pMeanTestInterpolant = new MeanValueInterpolant2D<Vector2>(m_values, pCutCellsVelocities, m_dx);

				buildCutCellPoints();
			}

		#pragma endregion

		protected:
		
		#pragma region PrivateFunctionalities
			virtual Vector2 interpolateCutCell(const Vector2 &position);

			/** Partial derivative in respect to X of the bilinear interpolation function. */
			Scalar bilinearPartialDerivativeX(const Vector2 &position, const vector<Scalar> &streamfunctionValues);

			/** Partial derivative in respect to Y of the bilinear interpolation function. */
			Scalar bilinearPartialDerivativeY(const Vector2 &position, const vector<Scalar> &streamfunctionValues);

			/** Initialized m_cutCellsPoints acceleration function */
			void buildCutCellPoints();
		#pragma endregion
		#pragma region ClassMembers
			Scalar m_dx;

			/** Discontinuous streamfunction values per grid cell. */
			Array2D<vector<Scalar>> *m_pStreamDiscontinuousValues;

			/** Continuous streamfunction values for the whole grid. */
			Array2D<Scalar> *m_pStreamContinuousValues;

			/** Mean value interpolant used for cut-cells. */
			MeanValueInterpolant2D<Scalar> *m_pMeanvalueInterpolant;

			MeanValueInterpolant2D<Vector2> *m_pMeanTestInterpolant;

			CutCellsVelocities2D *m_pCutCellsVelocities;
			CutCells2D<Vector2> *m_pCutCells;

			//Class used for wrapper acceleration
			vector<vector<Vector2>> m_cutCellsPoints;

			#pragma endregion
		};
	}


}
#endif