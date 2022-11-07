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

#ifndef __CHIMERA_CROSSING_H__
#define __CHIMERA_CROSSING_H__
#pragma once

#include "ChimeraCore.h"


namespace Chimera {
	namespace CutCells {
		using namespace Core;
		enum crossingType_t {
			horizontalCrossing,
			verticalCrossing,
			transversalCrossing,
			hybridCrossing // when a points lies exactly on a horizontal and vertical edge
		};


		template <class VectorT>
		class Crossing {
		public:
			//TO-DO Create access functions and protect class members
			#pragma region ClassMembers
			crossingType_t m_crossType;
			VectorT m_crossPoint;
			Scalar m_alpha;
			dimensions_t m_regularGridIndex; //Regular grid index
			int m_crossingsIndex; //Index inside all crossings vector
			int m_thinObjectPointsIndex; //Index of thin object line (2-D)
			int m_thinObjectID; //Thin object ID: used in multiple objects 
			/* Crossings when first found usually start new cut-cells. However there are special cases
				when this should not happen; e.g. when the line connecting the current and the next 
				(or previous crossing) aligns with the grid edge. In this case, invalid faces are added to 
				the other side of the crossing. Therefore the algorithm should this considering adding
				new cut-cells if this flag is set when this crossing is found. */
			bool m_startNewCutCell; 
			#pragma endregion
		
			#pragma region Constructors
			Crossing() {
				m_crossingsIndex = -1;
				m_thinObjectPointsIndex = -1;
				m_startNewCutCell = true;
			}
			Crossing(int gThinObjectID, crossingType_t gFaceType, int thinObjectIndex, VectorT c, float a, dimensions_t index): 
				m_crossType(gFaceType), m_thinObjectPointsIndex(thinObjectIndex), m_crossPoint(c), m_alpha(a), m_regularGridIndex(index),
				m_thinObjectID(gThinObjectID) {
					m_crossingsIndex = thinObjectIndex;
					m_startNewCutCell = true;
			}
			#pragma endregion

			

			#pragma region Operators
			bool operator==(const Crossing& a) const {
				return m_crossType == a.m_crossType && 
					m_alpha == a.m_alpha && 
					m_crossPoint == a.m_crossPoint &&
					m_crossingsIndex == a.m_crossingsIndex &&
					m_thinObjectPointsIndex == a.m_thinObjectPointsIndex &&
					m_thinObjectID == a.m_thinObjectID;
			}
			#pragma endregion

			static bool compareCrossing(Crossing<VectorT> a, Crossing<VectorT> b){
				if(a.m_crossingsIndex != b.m_crossingsIndex)
					return a.m_crossingsIndex < b.m_crossingsIndex;
				return a.m_alpha < b.m_alpha; 
			}

			static bool compareCrossingX (Crossing<VectorT> a, Crossing<VectorT> b) { return a.m_crossPoint.x < b.m_crossPoint.x; }
			static bool compareCrossingY (Crossing<VectorT> a, Crossing<VectorT> b) { return a.m_crossPoint.y < b.m_crossPoint.y; }
			static bool compareCrossingZ (Crossing<VectorT> a, Crossing<VectorT> b) { return a.m_crossPoint.z < b.m_crossPoint.z; }
		};
		
	}
}

#endif