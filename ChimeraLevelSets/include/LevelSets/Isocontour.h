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

#ifndef __CHIMERA_Isocontour__
#define __CHIMERA_Isocontour__
#pragma once


#include "ChimeraCore.h"
#include "ChimeraGrids.h"

namespace Chimera {
	namespace LevelSets {

		using namespace Core;

		using namespace Grids;
		namespace Isocontour {
			/** Defines the bounds of the maximum number of isocontour points.*/
			const static int maxIsoContourPoints = (int) 1e4; //10k


			/************************************************************************/
			/* Gradient stepping                                                    */
			/************************************************************************/
			/** Creates an Isocontour based on the perpendicular gradient stepping method 
			 ** @input horizontalStep: Sets the "time-step" for the stepping method;
			 ** Warning: This method modifies the pGridData velocity fields. Therefore do not
			 ** create isocontours with this pGridData, if the GridData2D is being used for any kind 
			 ** of simulation. */
			void gradientStepping(vector<Vector2> &isocontourPoints, GridData2D *pGridData, Scalar isoValue, Scalar timestep);
			void gradientStepping(vector<Vector2> &isocontourPoints, const Array2D<Scalar> & scalarField, GridData2D *pGridData, Scalar isoValue, Scalar timestep);

			/** Creates an Isocontour based on the perpendicular gradient stepping method.
			 ** The isocontour begins at starting point and will end if it reaches a determined
			 ** distance to the end point. 
			 ** @input horizontalStep: Sets the "time-step" for the stepping method;
			 ** Warning: This method modifies the pGridData velocity fields. Therefore do not
			 ** create isocontours with this pGridData, if the GridData2D is being used for any kind 
			 ** of simulation. */
			void gradientStepping(vector<Vector2> &isocontourPoints, GridData2D *pGridData, Scalar timestep, const Vector2 &startingPoint, const Vector2 &endPoint);

			/************************************************************************/
			/* Marching squares                                                     */
			/************************************************************************/
			/**Marching squares auxiliary functions */
			dimensions_t goToNextCell(const Array2D<int> &gridMask, const dimensions_t &currentCell);
			Vector2 calculatePoint(const Array2D<int> &gridMask, const Array2D<Scalar> &scalarField, GridData2D *pGridData, const dimensions_t &currentCell, Scalar isoValue);

			/** Creates an isocontour based on the marching squares method*/
			void marchingSquares(vector<Vector2> *pIsocontourPoints, const Array2D<Scalar> &scalarField, GridData2D *pGridData, Scalar isoValue);

			/************************************************************************/
			/* Utils                                                                */
			/************************************************************************/
			Scalar calculateCurvature(int i, const vector<Vector2> & points);
		}

	}
}

#endif