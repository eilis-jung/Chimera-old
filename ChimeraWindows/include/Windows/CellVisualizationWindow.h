
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

#ifndef _RENDERING_CELL_VISUALIZATION_WINDOW_H
#define _RENDERING_CELL_VISUALIZATION_WINDOW_H

#pragma  once
#include "ChimeraCore.h"
#include "ChimeraSolvers.h"

/************************************************************************/
/* Rendering                                                            */
/************************************************************************/
#include "Windows/BaseWindow.h"
#include "Windows/CutCellsWindow.h"

namespace Chimera {
	using namespace Solvers;
	namespace Windows {

		template <class VectorT, template <class> class ArrayType>
		class CellVisualizationWindow : public BaseWindow {

		public:
			#pragma region ExternalStructures
			typedef enum cellAttributesView_t {
				scalarField,
				velocityField,
				geometry,
				poissonMatrix
			} cellAttributesView_t;
			#pragma endregion

		private:
			#pragma region InternalStructures
			typedef struct poissonMatrixAttributes_t {
				//Five point laplace
				Scalar centerValue, westValue, eastValue, northValue, southValue;
				//Seven point laplace
				Scalar periodicWest, periodicEast;
				//Balance is the sum of the elements of the row of the matrix
				Scalar cellBalance;

				PoissonMatrix::matrixShape_t matrixShape;

				poissonMatrixAttributes_t() {
					centerValue = westValue = eastValue = northValue = southValue = 0.0f;
					periodicWest = periodicEast = 0.0f;
					cellBalance = 0.0f;
					matrixShape = PoissonMatrix::fivePointLaplace;
				}
			};

			typedef struct scalarFieldAttributes_t {
				//Different scalar field variables
				Scalar pressureValue;
				Scalar divergentValue;
				Scalar levelSetValue;
				Scalar streamfunctionValue;
				Scalar kineticEnergy;
				Scalar kineticEnergyChange;
				scalarFieldAttributes_t() {
					pressureValue = divergentValue = levelSetValue = streamfunctionValue = kineticEnergy = kineticEnergyChange = 0.0f;
				}

			};
			#pragma endregion
		
		public:
			#pragma region Constructors
			CellVisualizationWindow(FlowSolver<VectorT, ArrayType> *pFlowSolver);
			#pragma endregion

			#pragma region AccessFunctions
			FORCE_INLINE void setCutCellSolver2D(CutCellSolver2D *pCutCellSolver) {
				bool initializeCutCellsAttrib = m_pCutCellSolver == nullptr;
				m_pCutCellSolver = pCutCellSolver;
				if (initializeCutCellsAttrib)
					initializeCutCellsAttributes();
			}

			FORCE_INLINE StructuredGrid<VectorT> * getGrid() const {
				return m_pGrid;
			}

			FORCE_INLINE void switchCellAttribute(cellAttributesView_t cellAtribute) {
				m_currentCellAttribute = cellAtribute;
				TwRemoveAllVars(m_pBaseBar);
			}

			FORCE_INLINE void switchCellIndex(dimensions_t cellIndex) {
				m_selectedCellIndex = cellIndex;
				updatePoissonMatrixAttributes();
			}

			FORCE_INLINE poissonMatrixAttributes_t & getPoissonMatrixAttributes() {
				return m_poissonMatrixAttributes;
			}


			FORCE_INLINE void setStreamfunctionGrid(Interpolant<Scalar, Array2D, Vector2> *pInterpolant, Array2D<Scalar> *pStreamfunctionGrid, Scalar streamfunctionGridDx) {
				m_pStreamFunctionInterpolant = pInterpolant;
				m_pStreamfunctionGrid = pStreamfunctionGrid;
				m_streamfunctionGridDx = streamfunctionGridDx;
			}
			#pragma endregion

			#pragma region Functionalities
			int selectCell(const Vector2 &ray);
			#pragma endregion

		private:
			#pragma region ClassMembers
			//Grid
			StructuredGrid<VectorT> *m_pGrid;
			
			//Flow solver
			FlowSolver<VectorT, ArrayType> *m_pFlowSolver;
			
			//Cell attributes view type
			cellAttributesView_t m_currentCellAttribute;

			//Scalar field attributes;
			scalarFieldAttributes_t m_scalarFieldAttributes;

			//Poisson matrix attributes 
			poissonMatrixAttributes_t m_poissonMatrixAttributes;

			/** Cut cells */
			CutCellSolver2D *m_pCutCellSolver;
			int m_selectedCutCell;

			//Indicates the current cell index to display the cell information
			dimensions_t m_selectedCellIndex;
			dimensions_t m_lastSelectedCellIndex;

			//Cell area/volume
			Scalar m_cellVolume;

			/** External streamfunction grid */
			Array2D<Scalar> *m_pStreamfunctionGrid;
			Scalar m_streamfunctionGridDx;
			Interpolant<Scalar, Array2D, Vector2> *m_pStreamFunctionInterpolant;
			#pragma endregion

			#pragma region InitializationFunctions
			void initialize();
			void initializeCutCellsAttributes();
			void initScalarFieldAttributes();
			void initVelocityFieldAttributes();
			void initGeometryAttributes();
			void initPoissonMatrixAttributes();
			#pragma endregion

			#pragma region UpdateFunctions
			void update();
			void updatePoissonMatrixAttributes();
			void updateScalarFieldAttributes();
			#pragma endregion

		};
	}
}

#endif