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

#ifndef _CHIMERA_GRID_UTILS_
#define _CHIMERA_GRID_UTILS_
#pragma once

#include "ChimeraCore.h"

namespace Chimera {
	using namespace Core;
	namespace Grids {

		static const Scalar singlePrecisionThreshold = 1e-6;
		static const DoubleScalar doublePrecisionThreshold = 1e-13;

		#pragma region CheckingFunctions
		
		/** Finds if a point lies on top of a regular grid point (or nearly enough) */
		template <class VectorT>
		bool isOnGridPoint(const VectorT &point, Scalar dx);

		bool isOnGridPoint(const Vector3D &point, DoubleScalar dx, DoubleScalar tolerance);

		/** Finds if a point lies on top of a regular grid point (or nearly enough). Also returns on which grid point (integer 
		 ** dimensions) the point is. */
		template <class VectorT>
		bool isOnGridPoint(const VectorT &point, Scalar dx, dimensions_t &pointLocation);
		
		/** Finds if a point lies on a regular grid edge (or nearly enough). If lies on a grid point, returns -1. 
		 ** If it lies on horizontal edge, returns 1. If lies on a vertical edge, returns 2. Otherwise returns 0. */
		template <class VectorT>
		int isOnGridEdge(const VectorT &point, Scalar dx);

		/** Finds if a point lies on a regular grid edge (or nearly enough). If lies on a grid point, returns -1.
		** If it lies on horizontal edge, returns 1. If lies on a vertical edge, returns 2. Otherwise returns 0. 
		** Also returns the cell edge dimension where the point lies. */
		template <class VectorT>
		int isOnGridEdge(const VectorT &point, Scalar dx, dimensions_t &edgeLocation);
		
		/** If the point lies on a back-face, returns 1. If the point lies on a bottom-face returns 2. If the point lies on a
		left-face, returns 3. If the point lies on a grid point, returns -1. Otherwise returns 0. */
		template<class VectorT>
		int isOnGridFace(const VectorT &point, Scalar dx);

		/** If the point lies on a back-face, returns 1. If the point lies on a bottom-face returns 2. If the point lies on a
		left-face, returns 3. If the point lies on a grid point, returns -1. Otherwise returns 0. Also returns the cell face 
		dimension where the point lies.*/
		template <class VectorT>
		int isOnGridFace(const VectorT &point, Scalar dx, dimensions_t &faceLocation);

		/** Finds if a line segment crossed a grid edge. Returns the location of the grid edge crossed */
		bool crossedGridEdge(const Vector2 &v1, const Vector2 &v2, Vector2 &crossedPoint, Scalar dx);
		/** Finds if a point is inside a cell boundary. ProximityThreshold is used to "grow" the cell a little bit to robustly
		 ** handle points very close to edges*/ 
		bool isInsideCell(const Vector2 &point, dimensions_t cellIndex, Scalar dx);
		
		/** Checks if point after transformation is inside grid boundaries.*/
		template <class VectorT>
		bool isInsideGrid(const VectorT &point, dimensions_t gridDimensions, Scalar dx);
		#pragma endregion CheckingFunctions

		#pragma region GeometryUtilities
		/** Snap thinObject points to grid edges */
		void snapLinePointsToEdges(vector<Vector2> &thinObjectPoints, Scalar dx);
		/** Treat singularities such as crossings exactly on top of grid nodes*/
		void perturbLinePoints(vector<Vector2> &thinObjectPoints, Scalar dx);
		/** Extends thinObjectPoints to match grid-edges. */
		void extendLocalPointsToGridEdges(vector<Vector2> &thinObjectPoints, Scalar dx);
		#pragma endregion GeometryUtilities
	}
}
#endif

