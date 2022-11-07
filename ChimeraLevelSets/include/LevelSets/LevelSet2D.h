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

#ifndef __CHIMERA_LEVELSET_2D__
#define __CHIMERA_LEVELSET_2D__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"

#include "LevelSets/Isocontour.h"

namespace Chimera {
	namespace LevelSets {

		using namespace Core;
		using namespace Grids;
		class LevelSet2D {

		public:

			/************************************************************************/
			/* Parameter structure                                                  */
			/************************************************************************/
			typedef struct params_t {
				Scalar dx;
				Vector2 initialBoundary;
				Vector2 finalBoundary;

				vector<Vector2> *pPolygonPoints;

				bool convexPolygon;

				params_t() {
					dx = 0;
					initialBoundary = finalBoundary = Vector2(0, 0);
					convexPolygon = false;
				}
			} params_t;

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			LevelSet2D(const params_t &params, QuadGrid *pGrid = NULL) : m_params(params) {
				if(pGrid == NULL)
					initializeGrid();
				else
					m_pGrid = pGrid;
			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			/** Updates all level set values accordingly with polygon points. If polygon is convex
			 ** (params.convexPolygon), then the negative distance is stored for points inside
			 ** the polygon.*/
			void updateDistanceField();
			/** Updates the distance field in a narrow band around isocontour points */
			void updateDistanceField(int bandSize);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			const dimensions_t & getDimensions() const {
				return m_pGrid->getDimensions();
			}

			const Array2D<Scalar> & getArray() const {
				return m_pGrid->getGridData2D()->getLevelSetArray();
			}  

			const params_t & getParams() const {
				return m_params;
			}
			params_t & getParams() {
				return m_params;
			}

			Scalar getValue(int i, int j) const {
				return m_pGrid->getGridData2D()->getLevelSetValue(i, j);
			}

			void setValue(Scalar value, int i, int j) {
				m_pGrid->getGridData2D()->setLevelSetValue(value, i, j);
			}

			QuadGrid * getGrid() {
				return m_pGrid;
			}
			protected:
			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			params_t m_params;
			//Temporary: Change quadgrid by a scalargrid or something
			QuadGrid *m_pGrid;

			const static int maxIsocontourPoints;

			/************************************************************************/
			/* Initialization                                                       */
			/************************************************************************/
			virtual void initializeGrid();
		};
	}
}

#endif
