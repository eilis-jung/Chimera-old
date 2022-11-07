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

#ifndef _MATH_INTERSECTION_CORE_
#define _MATH_INTERSECTION_CORE_
#pragma once

#include "Config/ChimeraConfig.h"
#include "Math/MathUtilsCore.h"

namespace Chimera {

	namespace Core {

		/************************************************************************/
		/* Polygons utils                                                       */
		/************************************************************************/
		
		/** Compute lines segment intersection - 2D and 3-D declaration*/
		template<class VectorType>
		bool DoLinesIntersectT(const VectorType &p1, const VectorType &p2, const VectorType &p3, const VectorType &p4, VectorType &intersectionPoint, isVector2True);
		
		template<class VectorType>
		bool DoLinesIntersectT(const VectorType &p1, const VectorType &p2, const VectorType &p3, const VectorType &p4, VectorType &intersectionPoint, isVector2False);
		
		/** Function wrappers: external libs should call these: */
		template<class VectorType>
		bool DoLinesIntersect(const VectorType &p1, const VectorType &p2, const VectorType &p3, const VectorType &p4) {
			VectorType intersectionPoint;
			return DoLinesIntersectT(p1, p2, p3, p4, intersectionPoint, isVector2<VectorType>());
		}

		template<class VectorType>
		bool DoLinesIntersect(const VectorType &p1, const VectorType &p2, const VectorType &p3, const VectorType &p4, VectorType &intersectionPoint) {
			return DoLinesIntersectT(p1, p2, p3, p4, intersectionPoint, isVector2<VectorType>());
		}

		template <class VectorType>
		/** Checks if a segment intersects a line. If true, returns intersection on outvec */
		bool segmentLineIntersectionT(const VectorType &p1, const VectorType &p2, const VectorType &p3, const VectorType &p4, VectorType &outVec, isVector2True);

		template <class VectorType>
		/** Checks if a segment intersects a line. If true, returns intersection on outvec */
		bool segmentLineIntersectionT(const VectorType &p1, const VectorType &p2, const VectorType &p3, const VectorType &p4, VectorType &outVec, isVector2False);

		template<class VectorType>
		bool segmentLineIntersection(const VectorType &p1, const VectorType &p2, const VectorType &p3, const VectorType &p4, VectorType &outVec) {
			return segmentLineIntersectionT(p1, p2, p3, p4, outVec, isVector2<VectorType>());
		}

		Vector2 linesIntersection(const Vector2 &p1, const Vector2 &p2, const Vector2 &p3, const Vector2 &p4);
	}
}

#endif