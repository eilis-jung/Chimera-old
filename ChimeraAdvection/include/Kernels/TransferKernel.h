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

#ifndef __CHIMERA_TRANSFER_KERNEL_H_
#define __CHIMERA_TRANSFER_KERNEL_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraCutCells.h"


namespace Chimera {

	using namespace Core;
	using namespace Grids;
	using namespace CutCells;

	namespace Advection {

		/** Transfer kernel from a single particle to the grid. Its two main functions are getCellList() and
		  *  calculateKernel(). GetCellList() function depends on the size and support of the kernel, but its implemented
	      * in this class. Sub-classes implement the calculateKernel(), which actually provides a radial basis function
		  * around a point */
		template<class VectorType>
		class TransferKernel {

		public:
			//Will only reserve sizes inside vectors, explicit position/velocity initialization is on user-side
			TransferKernel(GridData<VectorType> *pGridData, Scalar kernelSize) {
				m_pGridData = pGridData;
				m_kernelSize = kernelSize;
			}

			#pragma region Functionalities
			/** Actual kernel calculation. All subclasses must implement this. */
			virtual Scalar calculateKernel(const VectorType &position, const VectorType & destPosition, Scalar r) = 0;
			/** Retrieves the list of cells that have their centroids in reach of the current kernel size 
			 ** and the position passed as the  argument. The list is passed as an argument, since this function is 
			 ** usually called several times in a accumulation loop. This method disregards cut-cells. 
				Position is on WORLD SPACE. */
			virtual void getCellList(const VectorType &position, vector<dimensions_t> &cellsList);
		
			/** Same as above, but this method considers only cut-cells. Position is on WORLD SPACE. 
			  * TODO: implement this*/
			virtual void getCutCellList(const VectorType &position, vector<int>  &cutCellsIDList);
			#pragma endregion
			
			#pragma region AcessFunctions 
			/*void setCutCells(CutCellsBase<VectorType, ArrayType> *pCutCells) {
				m_pCutCells = pCutCells;
			}*/
			#pragma endregion

		protected:
			Scalar m_kernelSize;
			GridData<VectorType> *m_pGridData;

			//CutCellsBase<VectorType, ArrayType> *m_pCutCells;
		};
	}
}

#endif
