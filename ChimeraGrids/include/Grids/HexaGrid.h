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

#ifndef __CHIMERA_HEXA_GRID__
#define __CHIMERA_HEXA_GRID__
#pragma once

#include "ChimeraCore.h"
#include "Grids/StructuredGrid.h"

using namespace std;
namespace Chimera {

	typedef struct cellFace_t{
		int m_vertexIndex[4];
	} cellFace_t;

	using namespace Core;

	namespace Grids {

		/** Hexagrid Class. Represents hexaedral structured 3D grids. */
		class HexaGrid : public StructuredGrid<Vector3> {	
			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			/** Grid points initialized by loadGrid function */
			Array3D<Vector3> *m_pGridPoints;

			/************************************************************************/
			/* Metrics & grid functionalities                                       */
			/************************************************************************/
			/** Different differencing schemes implemented by hexa grid class. */ 
			void initializeGridMetrics();

			/************************************************************************/
			/* Grid Loading															*/
			/************************************************************************/
			void loadGrid(const string &filename);
			void loadPeriodicGrid(const string &gridFilename);

			inline const Vector3 & getPoint(int i, int j, int k) const {
				return (*m_pGridPoints)(i, j, k);
			}


		public:
			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			HexaGrid(const string &gridFilename, bool periodicBCs = false);
			HexaGrid(const Vector3 &initialBoundary, const Vector3 & finalBoundary, Scalar dx);
			~HexaGrid();

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			bool isInsideCell(Vector3 position, int x, int y, int z) const;

			/************************************************************************/
			/* Grid I/O			                                                    */
			/************************************************************************/
			/** Grid loading.	*/
			void loadSolidCircle(const Vector3 &centerPoint, Scalar circleSize);

			/** Grid exporting. */
			void exportToFile(const string &gridExportname);

		};
	}
	
}

#endif