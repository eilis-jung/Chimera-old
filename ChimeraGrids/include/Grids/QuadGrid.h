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

#ifndef __CHIMERA_QUADGRID__
#define __CHIMERA_QUADGRID__
#pragma once


#include "ChimeraCore.h"
#include "Grids/StructuredGrid.h"


using namespace std;
namespace Chimera {

	using namespace Core;

	namespace Grids {

		/** QuadGrid Class. Represents quadrilateral structured 2D grids. */
		class QuadGrid : public StructuredGrid<Vector2> {
			/** Grid points initialized by loadGrid function */
			Array2D<Vector2> *m_pGridPoints;

			/************************************************************************/
			/* Metrics & grid functionalities                                       */
			/************************************************************************/
			/** Initialize grid data using forward differencing scheme. */ 
			void initializeGridMetrics();

			/** Auxiliary grid point function: returns the points of internal temporary gridPoints, not yet transferred
			 ** to grid data */
			FORCE_INLINE const Vector2 getPoint(int i, int j) const {
				return (*m_pGridPoints)(i, j);
			}
			
			/************************************************************************/
			/* Grid loading                                                         */
			/************************************************************************/
			void loadGrid(const string &gridFilename);
			void loadPeriodicGrid(const string &gridFilename);

		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			/** Filename: grid file to be loaded
			 ** initializeGraphics: only false if the graphics interface is not used */
			QuadGrid(const string &gridFilename, bool periodicBCs = false, bool subGrid = false);

			/** Create a regular grid with pre-specified dimensions. The grid is generated according with initial and final
			 ** points. This type of grid is always non-periodic. */
			QuadGrid(const Vector2 &initialPoint, const Vector2 &finalPoint, Scalar gridSpacing, bool subgrid = false);

			/** Creates a grid with the pre-specified set of points. */
			QuadGrid(Array2D<Vector2> *pGridPoints, bool subGrid = false);

			~QuadGrid() {
			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			bool isInsideCell(Vector2 position, int x, int y) const;

			/************************************************************************/
			/* Grid I/O			                                                    */
			/************************************************************************/
			void loadSolidCircle(const Vector2 &centerPoint, Scalar circleSize);
			void loadSolidRectangle(const Vector2 &recPosition, const Vector2 &recSize);
			void loadObject(const vector<Vector2> &objectPoints);
			void exportToFile(const string &gridExportname);
		};
	}
}

#endif