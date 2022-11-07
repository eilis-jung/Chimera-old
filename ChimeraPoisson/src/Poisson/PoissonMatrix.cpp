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

#include "Poisson/PoissonMatrix.h"

/************************************************************************/
/* CUSP                                                                 */
/************************************************************************/

namespace Chimera {
	namespace Poisson {

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		PoissonMatrix::PoissonMatrix(dimensions_t dimensions, bool supportGPU, bool isPeridioc) { 
			m_dimensions = dimensions;
			m_supportGPU = supportGPU;
			m_pCuspDiagonalMatrix = NULL;
			m_pCuspHybMatrix = NULL;
			m_pDeviceCuspHybMatrix = NULL;

			m_periodicBDs = isPeridioc;
			m_diagonalMatrix = true;
			m_numAdditionalCells = 0;

			if(dimensions.z == 0) {
				if(m_periodicBDs)
					m_matrixShape = sevenPointLaplace;
				else
					m_matrixShape = fivePointLaplace;

				init2D();
			} else {
				if(m_periodicBDs)
					m_matrixShape = ninePointLaplace;
				else
					m_matrixShape = sevenPointLaplace;

				init3D();
			}
			
			if(m_supportGPU)
				initDeviceMatrix();
			
		}

		PoissonMatrix::PoissonMatrix(dimensions_t dimensions, uint additionalCells, bool supportGPU) { 
			m_dimensions = dimensions;
			m_supportGPU = supportGPU;
			m_pDeviceCuspDiagonalMatrix = NULL;
			m_pCuspHybMatrix = NULL;
			m_pDeviceCuspHybMatrix = NULL;
			m_periodicBDs = false;
			m_diagonalMatrix = false;
			m_numAdditionalCells = additionalCells;

			if(dimensions.z == 0) {
				if(m_periodicBDs)
					m_matrixShape = sevenPointLaplace;
				else
					m_matrixShape = fivePointLaplace;

				init2D();
			} else {
				if(m_periodicBDs)
					m_matrixShape = ninePointLaplace;
				else
					m_matrixShape = sevenPointLaplace;

				init3D();
			}

			if(m_supportGPU)
				initDeviceMatrix();

		}

		PoissonMatrix::PoissonMatrix(dimensions_t dimensions, matrixShape_t matrixShape, bool supportGPU /* = true */, bool isPeridioc /* = false */) {
				m_dimensions = dimensions;
				m_supportGPU = supportGPU;
				m_pDeviceCuspDiagonalMatrix = NULL;
				m_pCuspHybMatrix = NULL;
				m_pDeviceCuspHybMatrix = NULL;
				m_periodicBDs = isPeridioc;
				m_numAdditionalCells = 0;

				m_matrixShape = matrixShape;

				if(m_dimensions.z == 0) { //2D
					init2D();
				} else { //3D
					init3D();
				}
				if(m_supportGPU)
					initDeviceMatrix();
		}
		
	}
}