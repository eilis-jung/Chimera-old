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


#ifndef __CHIMERA_MARCHING_SQUARES_2D__
#define __CHIMERA_MARCHING_SQUARES_2D__

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraMesh.h"

namespace Chimera {

	using namespace Core;
	using namespace Grids;
	using namespace Meshes;
	namespace LevelSets {


		class MarchingSquares {


		public:

			typedef enum cellTypes { internalCell, externalCell, visitedCell };

			MarchingSquares(const Array2D<Scalar> &levelSet, Scalar gridSpacing);

			/* Extracts a polygon, starting with the cell indicated by initialCell */
			vector<LineMesh<Vector2> *> extract(Scalar isoValue);

			/** Returns current state of cell types. Used after extract() to get a list of visited cells */
			const Array2D<cellTypes> & getCellTypes() const {
				return m_cellTypes;
			}

			const vector<dimensions_t> & getVisitedCellsList() const {
				return m_visitedCellsList;
			}

		private:
			Array2D<Scalar> m_levelSet;
			Array2D<cellTypes> m_cellTypes;
			Scalar m_gridSpacing;
			vector<dimensions_t> m_visitedCellsList;

			dimensions_t goToNextCell(const dimensions_t &currentCell);
			Vector2 calculatePoint(Scalar isovalue, const dimensions_t &cell);
			//Checks cells with index cellIndex has a cellType neighbor in a 4 neighborhood around it
			bool hasNeighbor(const dimensions_t &cellIndex, cellTypes cellType);
		};
	}
	
}

#endif