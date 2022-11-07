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

#ifndef __CHIMERA_BV_LEVELSET_2D__
#define __CHIMERA_BV_LEVELSET_2D__
#pragma once

#include "LevelSets/LevelSet2D.h"

namespace Chimera {
	namespace LevelSets {

		/** Boundary Value Level-Set. It is solved by the fast marching method (FMM). */

		class BVLevelSet2D : public LevelSet2D {

		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			BVLevelSet2D(const params_t &params, QuadGrid *pGrid = NULL);

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			void solveFMM();
			/** Updates level set according new points */
			void update();

		private:
			
			/************************************************************************/
			/* Initialization                                                       */
			/************************************************************************/
			void initializeArbitraryPoints();


			/************************************************************************/
			/* Internal structs                                                     */
			/************************************************************************/
			typedef struct minHeapElement_t {
				dimensions_t cellID;
				Scalar time;

				minHeapElement_t(Scalar gTime, const dimensions_t &gCellID) :
					time(gTime), cellID(gCellID) {

				}
			} minHeapElement_t;

			struct minHeapCompare : public std::binary_function<minHeapElement_t, minHeapElement_t, bool>
			{
				bool operator()(minHeapElement_t *lhs, minHeapElement_t *rhs) const
				{
					return lhs->time > rhs->time;
				}

			};

			/************************************************************************/
			/* NarrowBand class                                                     */
			/************************************************************************/
			class NarrowBand {
			public:
				/** Ctor */
				NarrowBand(StructuredGrid<Vector2> *pGrid);
				
				/** Functionalities */
				void addCell(Scalar timeValue, int i, int j);

				/** Access functions*/
				FORCE_INLINE bool isEmpty() const {
					return m_multiMap.empty();
				}

				FORCE_INLINE minHeapElement_t * getTopElement() const {
					return m_multiMap.begin()->second;
				}

				FORCE_INLINE void popTopElement()  {
					m_multiMap.erase(m_multiMap.begin());
				}

				FORCE_INLINE Scalar getTopValue() const {
					return m_multiMap.begin()->second->time;
				}

				/** Class members*/
				dimensions_t m_dimensions;
				multimap<Scalar, minHeapElement_t*> m_multiMap;
				Array2D<char> *pUsedMap;
				StructuredGrid<Vector2> *m_pGrid;
			};

			/************************************************************************/
			/* Internal functionalities                                             */
			/************************************************************************/
			Scalar calculateTime(int x, int y, bool useFrozen = true);
			Scalar calculateTimeCurvature(int x, int y);
			Scalar selectUpwindNeighbor(int x, int y, const velocityComponent_t &direction);

			FORCE_INLINE bool isInsideSearchRange(int i, int j) {
				if(i < 1 || i > m_pGrid->getDimensions().x - 2 || j < 1 || j > m_pGrid->getDimensions().y - 2)
					return false;
				return true;
			}

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			NarrowBand *pNarrowBand;
		};
	}
}

#endif
