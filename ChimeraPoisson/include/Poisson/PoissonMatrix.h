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

#ifndef _MATH_LAPLACIAN_MATRIX_H_
#define _MATH_LAPLACIAN_MATRIX_H_
#pragma once


#include "ChimeraCore.h"

using namespace std;

namespace Chimera {
	using namespace Core;

	namespace Poisson {

		typedef struct stencil2D_t {
			int offsetX;
			int offsetY;

			stencil2D_t(int offX, int offY) {
				offsetX = offX;
				offsetY = offY;
			}
		} stencil2D_t;

		typedef struct stencil3D_t {
			int offsets[3];

			stencil3D_t(int offX, int offY, int offZ) {
				offsets[0] = offX;
				offsets[1] = offY;
				offsets[2] = offZ;
			}
		} stencil3D_t;

		

		/** Poisson matrix, used for solving the Poisson Equation. The matrix corresponds to the fluid cells - and the 
		 ** bottom-most cells correspond to the lower indices. This way was chosen to remain consistent with 
		 ** the cell drawing since we spatially access cells according with regular index orientation. 
		 ** Boundary conditions - different BDs will result in different matrix sizes. The basic rule is:
		 **		Neumann: Will remove the cells corresponding to this boundary, for better efficiency in CUDA algorithms
		 **			Ex: Inflow, NoSlip, FreeSlip, Jet
		 **		Dirichlet: Will maintain the cells corresponding to this boundary.
		 **			Ex: Outflow
		 ** */
		class PoissonMatrix {

		public:
			/************************************************************************/
			/* Internal strucs                                                      */
			/************************************************************************/
			typedef enum matrixShape_t {
				fivePointLaplace,				/** Regular 2D approach */
				sevenPointLaplace,				/** Periodic 2D approach or Regular 3D approach */
				ninePointLaplace,				/** Extended Kernel for 2D approach or Periodic 3D approach */
				fifteenPointLaplace				/** Extended Kernel for 3D approach*/
			} matrixShape_t;

		private:

			/************************************************************************/
			/* Member variables                                                     */
			/************************************************************************/
			cusp::dia_matrix<Integer, Scalar, cusp::host_memory> *m_pCuspDiagonalMatrix;
			cusp::dia_matrix<Integer, Scalar, cusp::device_memory> *m_pDeviceCuspDiagonalMatrix;

			cusp::coo_matrix<Integer, Scalar, cusp::host_memory> *m_pCuspCOOMatrix;

			cusp::hyb_matrix<Integer, Scalar, cusp::host_memory> *m_pCuspHybMatrix;
			cusp::hyb_matrix<Integer, Scalar, cusp::device_memory> *m_pDeviceCuspHybMatrix;

			dimensions_t m_dimensions;
			int m_numAdditionalCells;
			bool m_diagonalMatrix;
			bool m_supportGPU;
			bool m_periodicBDs;
			unsigned int m_matrixSize;

			//Central offset
			int mOffset;
			//Five point 2D offsets
			int nOffset, sOffset, wOffset, eOffset;
			//Periodic 2D offsets
			int pEOffset, pWOffset;
			//Nine point 2D offsets
			int neOffset, nwOffset, seOffset, swOffset;
			//Seven point 3D additional offsets
			int bOffset, fOffset;
			//Periodic 3D offsets;
			int pBOffset, pFOffset;

			vector<int> m_transformedOffsets;
			vector<stencil2D_t> m_stencils;
			matrixShape_t m_matrixShape;

			
			/************************************************************************/
			/* Initialization	                                                    */
			/************************************************************************/
			
			/** Initialize the configuration for 2D Poisson matrix */
			void init2D();

			/** Initialize the configuration for 3D Poisson matrix */
			void init3D();

			/** Different type of matrix shapes initialization */
			void initFivePointLaplace();
			void initSevenPointLaplace();
			void initNinePointLaplace();

			/************************************************************************/
			/* Private functionalities                                              */
			/************************************************************************/

			FORCE_INLINE int transformStencil(const stencil2D_t &stencil) {
				return getRowIndex(stencil.offsetX, stencil.offsetY);
			}

			/************************************************************************/
			/* Cuda functions                                                       */
			/************************************************************************/
			/** CUDA side function - will be implemented in the .cu file */
			void initDeviceMatrix();

		

		public:
			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			/** 2D and 3D Poisson Matrix Solver constructor */
			PoissonMatrix(dimensions_t dimensions, bool supportGPU = true, bool isPeridioc = false);

			PoissonMatrix(dimensions_t dimensions, uint additionalCells, bool supportGPU = true);

			PoissonMatrix(dimensions_t dimensions, matrixShape_t matrixShape, bool supportGPU = true, bool isPeridioc = false);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			/** Returns raw CUDA matrix*/
			cusp::dia_matrix<Integer, Scalar, cusp::device_memory> * getGPUData() const;
			cusp::hyb_matrix<Integer, Scalar, cusp::device_memory> * getGPUDataHyb() const;
			cusp::dia_matrix<Integer, Scalar, cusp::host_memory> * getCPUData() const;

			cusp::coo_matrix<Integer, Scalar, cusp::host_memory> * getCPUDataCOO() const;

			cusp::ell_matrix<Integer, Scalar, cusp::host_memory> * getCPUDataELL() const;
			cusp::hyb_matrix<Integer, Scalar, cusp::host_memory> * getCPUDataHyb() const;

			FORCE_INLINE bool supportGPU() const {
				return m_supportGPU;
			}

			FORCE_INLINE dimensions_t getDimensions() const {
				return m_dimensions;
			}

			FORCE_INLINE unsigned int getMatrixSize() const {
				return m_matrixSize;
			}
			

			FORCE_INLINE const vector<stencil2D_t> & getStencils() const {
				return m_stencils;
			}

			FORCE_INLINE const matrixShape_t & getMatrixShape() const {
				return m_matrixShape;
			}

			FORCE_INLINE bool isPeriodic() const {
				return m_periodicBDs;
			}

			FORCE_INLINE void setNumberAdditionalCells(int additionalCells) {
				m_numAdditionalCells = additionalCells;
			}

			FORCE_INLINE int getNumberAdditionalCells() const {
				return m_numAdditionalCells;
			}
			/************************************************************************/
			/* Row access functions                                                 */
			/************************************************************************/
			/** Since it only access host-side cusp functions, it can be compiled by visual studio CL compiler */
			/**Setters*/
			void setRow(int row, Scalar pn, Scalar pw, Scalar pc, Scalar pe, Scalar ps);
			void setRow(int row, Scalar pn, Scalar pw, Scalar pb, Scalar pc, Scalar pe, Scalar ps, Scalar pf);
			
			void setCentralValue(int i, Scalar pc);

			void setNorthValue(int i, Scalar pn);
			void setSouthValue(int i, Scalar ps);
			void setWestValue(int i, Scalar pw);
			void setEastValue(int i, Scalar pe);
			
			void setBackValue(int i , Scalar pb);
			void setFrontValue(int i, Scalar pf);
			
			void setPeriodicWestValue(int i, Scalar pWF);
			void setPeriodicEastValue(int i, Scalar pEF);

			void setPeriodicBackValue(int i, Scalar pBV);
			void setPeriodicFrontValue(int i, Scalar pFV);

			void setNorthWestValue(int i, Scalar pNW);
			void setNorthEastValue(int i, Scalar pNE);
			
			void setSouthWestValue(int i, Scalar pSW);
			void setSouthEastValue(int i, Scalar pSE);

			/** Only works for non-diagonal matrix formulation */
			void setValue(int ithElement, int row, int column, Scalar pValue);
			Scalar getValue(int row, int column) const;

			/**Getters*/
			Scalar getCentralValue(int i) const;

			Scalar getNorthValue(int i) const;
			Scalar getSouthValue(int i) const;
			Scalar getWestValue(int i) const;
			Scalar getEastValue(int i) const;
			
			Scalar getBackValue(int i) const;
			Scalar getFrontValue(int i) const;
			
			Scalar getPeriodicWestValue(int i) const;
			Scalar getPeriodicEastValue(int i) const;
			
			Scalar getNorthWestValue(int i) const;
			Scalar getNorthEastValue(int i) const;
			Scalar getSouthWestValue(int i) const;
			Scalar getSouthEastValue(int i) const;
			
			Scalar getValue(int row, const stencil2D_t &elementOffset);
			

			/************************************************************************/
			/* Offsets                                                              */
			/************************************************************************/
			int getNorthOffset() const;
			int getSouthOffset() const;
			int getWestOffset() const;
			int getEastOffset() const;

			FORCE_INLINE int getNumRows() const {
				return m_matrixSize;
			}

			FORCE_INLINE int getRowIndex(int i, int j) const {
				return m_dimensions.x*j + i;
			}

			FORCE_INLINE int getRowIndex(int i, int j, int k) const {
				return m_dimensions.x*m_dimensions.y*k + m_dimensions.x*j + i; 
			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			/** CUDA side function - will be implemented in the .cu file */
			void updateCudaData();
			void printGPUData();
			/** Checks if matrix is singular */
			bool isSingular() const;
			/** Checks if matrix is singular */
			bool isSingularCOO() const;
			/** Applies correction to remove matrix singularity.*/
			void applyCorrection(Scalar delta = 1e-5);
			
			void enablePeriodicBoundaryConditions(bool periodic) {
				m_periodicBDs = periodic;
			}

			void copyDIAtoCOO();
			int getNumberOfEntriesCOO();
			vector <pair <uint, Scalar>> getRowCOOMatrix(int row);
			void resizeToFitCOO();
			void copyCOOtoHyb();
			/************************************************************************/
			/* Printing                                                             */
			/************************************************************************/
			void cuspPrint() const;
		};
	}


}
#endif