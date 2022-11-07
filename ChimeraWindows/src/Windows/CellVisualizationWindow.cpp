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

#include "Windows/CellVisualizationWindow.h"
#include "BaseGLRenderer.h"

namespace Chimera {
	namespace Windows {

		#pragma region Constructors
		template <class VectorT, template <class> class ArrayType>
		CellVisualizationWindow<VectorT, ArrayType>::CellVisualizationWindow(FlowSolver<VectorT, ArrayType> *pFlowSolver)
			: BaseWindow(Vector2(16, 700), Vector2(300, 320), "Cell Visualization") {
			m_pFlowSolver = pFlowSolver;
			m_pGrid = m_pFlowSolver->getGrid();
			
			m_currentCellAttribute = poissonMatrix;
			m_selectedCellIndex = dimensions_t(0, 0, 0);
			m_lastSelectedCellIndex = dimensions_t(-1, -1, -1); //For an initial update
			m_pCutCellSolver = nullptr;
			m_selectedCutCell = -1;

			string defStr = "'Cell Visualization' text='light' refresh=0.05"; 
			TwDefine(defStr.c_str());

			m_pStreamfunctionGrid = NULL;
			m_pStreamFunctionInterpolant = NULL;
			
			initialize();
		}
		#pragma endregion

		#pragma region Functionalities
		template <class VectorT, template <class> class ArrayType>
		int CellVisualizationWindow<VectorT, ArrayType>::selectCell(const Vector2 &ray) {
			dimensions_t selectedCell = dimensions_t(floor(ray.x), floor(ray.y));
			if(m_pCutCellSolver) {
				CutCells2D<Vector2> *pSpecialCells = m_pCutCellSolver->getCutCells();
				dimensions_t rayLocation(ray.x, ray.y, 0);
				pSpecialCells->getNumberCutCells();

				if (pSpecialCells->isCutCell(rayLocation)) {
					m_selectedCutCell = pSpecialCells->getCutCellIndex(ray);
				}
				else {
					m_selectedCutCell = -1;
				}
				
				/*if((m_selectedCellIndex = pSpecialCells->getCutCellIndex(ray)) != -1){
					m_pSpecialCellsWindow->setSelectedCell(selectedIndex);
					m_scalarFieldAttributes.pressureValue = m_pSpecialCellsWindow->getPressure();
					m_scalarFieldAttributes.divergentValue = m_pSpecialCellsWindow->getDivergent();
				}
				else {
					m_pSpecialCellsWindow->setSelectedCell(-1);
				}*/
			}
			switchCellIndex(selectedCell);

			if (m_pStreamfunctionGrid) {
				Scalar invSubdvisionFactor = m_streamfunctionGridDx / m_pGrid->getGridData2D()->getScaleFactor(0, 0).x;
				m_scalarFieldAttributes.streamfunctionValue = m_pStreamFunctionInterpolant->interpolate(ray*m_pGrid->getGridData2D()->getGridSpacing());
				//m_scalarFieldAttributes.streamfunctionValue = interpolateNodeBasedScalar(ray*invSubdvisionFactor, *m_pStreamfunctionGrid);
				//m_scalarFieldAttributes.streamfunctionValue *= 100000000;
			}

			return m_selectedCutCell;
		}
		#pragma endregion

		#pragma region InitializationFunctions
		template <class VectorT, template <class> class ArrayType>
		void CellVisualizationWindow<VectorT, ArrayType>::initialize() {
			if(m_pFlowSolver) {
				m_poissonMatrixAttributes.matrixShape = m_pFlowSolver->getPoissonMatrix()->getMatrixShape();
			}

			TwAddVarRO(m_pBaseBar, "cellIndex_i", TW_TYPE_INT32, &m_selectedCellIndex.x, "label='Cell index i' group='General'");
			TwAddVarRO(m_pBaseBar, "cellIndex_j", TW_TYPE_INT32, &m_selectedCellIndex.y, "label='Cell index j' group='General'");
			TwAddVarRO(m_pBaseBar, "cellVolume", TW_TYPE_FLOAT, &m_cellVolume, "label='Cell Volume' group='General'");

			update();
			initScalarFieldAttributes();
			initPoissonMatrixAttributes();
		}

		template <class VectorT, template <class> class ArrayType>
		void CellVisualizationWindow<VectorT, ArrayType>::initializeCutCellsAttributes() {
			
		}

		template <class VectorT, template <class> class ArrayType>
		void CellVisualizationWindow<VectorT, ArrayType>::initScalarFieldAttributes() {
			TwAddVarRO(m_pBaseBar, "pressureValue", TW_TYPE_FLOAT, &m_scalarFieldAttributes.pressureValue, "label='Pressure' group='Scalar field'");
			TwAddVarRO(m_pBaseBar, "levelSetValue", TW_TYPE_FLOAT, &m_scalarFieldAttributes.levelSetValue, "label='Level set' group='Scalar field'");
			TwAddVarRO(m_pBaseBar, "divergent", TW_TYPE_FLOAT, &m_scalarFieldAttributes.divergentValue, "label='Divergent' group='Scalar field'");
			TwAddVarRO(m_pBaseBar, "streamfunction", TW_TYPE_FLOAT, &m_scalarFieldAttributes.streamfunctionValue, "label='Streamfunction' group='Scalar field'");
			TwAddVarRO(m_pBaseBar, "kineticEnergy", TW_TYPE_FLOAT, &m_scalarFieldAttributes.kineticEnergy, "label='Kinetic Energy' group='Scalar field'");
			TwAddVarRO(m_pBaseBar, "kineticEnergyChange", TW_TYPE_FLOAT, &m_scalarFieldAttributes.kineticEnergyChange, "label='Kinetic Energy Change' group='Scalar field'");
		}

		template <class VectorT, template <class> class ArrayType>
		void CellVisualizationWindow<VectorT, ArrayType>::initGeometryAttributes() {
			
		}

		template <class VectorT, template <class> class ArrayType>
		void CellVisualizationWindow<VectorT, ArrayType>::initVelocityFieldAttributes() {
			
		}

		template <class VectorT, template <class> class ArrayType>
		void CellVisualizationWindow<VectorT, ArrayType>::initPoissonMatrixAttributes() {
			/**Poisson matrix values display*/
			TwAddVarRO(m_pBaseBar, "pc", TW_TYPE_FLOAT, &m_poissonMatrixAttributes.centerValue, "label='Central value' group='Poisson Matrix'");
			TwAddVarRO(m_pBaseBar, "pw", TW_TYPE_FLOAT, &m_poissonMatrixAttributes.westValue, "label='West value' group='Poisson Matrix'");
			TwAddVarRO(m_pBaseBar, "pe", TW_TYPE_FLOAT, &m_poissonMatrixAttributes.eastValue, "label='East value' group='Poisson Matrix'");
			TwAddVarRO(m_pBaseBar, "pn", TW_TYPE_FLOAT, &m_poissonMatrixAttributes.northValue, "label='North value' group='Poisson Matrix'");
			TwAddVarRO(m_pBaseBar, "ps", TW_TYPE_FLOAT, &m_poissonMatrixAttributes.southValue, "label='South value' group='Poisson Matrix'");
			if(m_poissonMatrixAttributes.matrixShape == PoissonMatrix::sevenPointLaplace) {
				TwAddVarRO(m_pBaseBar, "ppw", TW_TYPE_FLOAT, &m_poissonMatrixAttributes.periodicWest, "label='Periodic west value' group='Poisson Matrix'");
				TwAddVarRO(m_pBaseBar, "ppe", TW_TYPE_FLOAT, &m_poissonMatrixAttributes.periodicEast, "label='Periodic east value' group='Poisson Matrix'");
			}
			TwAddVarRO(m_pBaseBar, "centralBalance", TW_TYPE_FLOAT, &m_poissonMatrixAttributes.cellBalance, "label='Central balance' group='Poisson Matrix'");
		}
		#pragma endregion

		#pragma region UpdateFunctions
		template <class VectorT, template <class> class ArrayType>
		void CellVisualizationWindow<VectorT, ArrayType>::update() {
			if(m_lastSelectedCellIndex.x != m_selectedCellIndex.x || m_lastSelectedCellIndex.y != m_selectedCellIndex.y) {
				m_lastSelectedCellIndex = m_selectedCellIndex;
				updatePoissonMatrixAttributes();
			}

			if(m_pGrid)
				updateScalarFieldAttributes();
		}

		template <class VectorT, template <class> class ArrayType>
		void CellVisualizationWindow<VectorT, ArrayType>::updatePoissonMatrixAttributes() {
			if(m_pFlowSolver && m_selectedCellIndex.x > 0 && m_selectedCellIndex.y > 0) {
				PoissonMatrix *pPoissonMatrix = m_pFlowSolver->getPoissonMatrix();
				dimensions_t tempDim = m_selectedCellIndex;
				if(m_pFlowSolver->getParams().pPoissonSolverParams->solverCategory == Krylov) {
					if(!m_pGrid->isPeriodic())
						tempDim.x -= 1;
					tempDim.y -= 1;

					tempDim.x = clamp(tempDim.x, 0, m_pGrid->getDimensions().x);
					tempDim.y = clamp(tempDim.y, 0, m_pGrid->getDimensions().y);
				}
				m_poissonMatrixAttributes.centerValue = pPoissonMatrix->getCentralValue(pPoissonMatrix->getRowIndex(tempDim.x, tempDim.y));
				m_poissonMatrixAttributes.westValue = pPoissonMatrix->getWestValue(pPoissonMatrix->getRowIndex(tempDim.x, tempDim.y));
				m_poissonMatrixAttributes.eastValue = pPoissonMatrix->getEastValue(pPoissonMatrix->getRowIndex(tempDim.x, tempDim.y));
				m_poissonMatrixAttributes.northValue = pPoissonMatrix->getNorthValue(pPoissonMatrix->getRowIndex(tempDim.x, tempDim.y));
				m_poissonMatrixAttributes.southValue = pPoissonMatrix->getSouthValue(pPoissonMatrix->getRowIndex(tempDim.x, tempDim.y));
				if(m_pGrid->isPeriodic()) {
					m_poissonMatrixAttributes.periodicWest = pPoissonMatrix->getPeriodicWestValue(pPoissonMatrix->getRowIndex(tempDim.x, tempDim.y));
					m_poissonMatrixAttributes.periodicEast = pPoissonMatrix->getPeriodicEastValue(pPoissonMatrix->getRowIndex(tempDim.x, tempDim.y));
				}

				m_poissonMatrixAttributes.cellBalance = 0;
				m_poissonMatrixAttributes.cellBalance += m_poissonMatrixAttributes.centerValue;
				m_poissonMatrixAttributes.cellBalance += m_poissonMatrixAttributes.westValue;
				m_poissonMatrixAttributes.cellBalance += m_poissonMatrixAttributes.eastValue;
				m_poissonMatrixAttributes.cellBalance += m_poissonMatrixAttributes.northValue;
				m_poissonMatrixAttributes.cellBalance += m_poissonMatrixAttributes.southValue;
				m_poissonMatrixAttributes.cellBalance += m_poissonMatrixAttributes.periodicWest;
				m_poissonMatrixAttributes.cellBalance += m_poissonMatrixAttributes.periodicEast;
			}
		}

		template <>
		void CellVisualizationWindow<Vector2, Array2D>::updateScalarFieldAttributes() {
			if(m_pCutCellSolver) {
				CutCells2D<Vector2> *pCutCells = m_pCutCellSolver->getCutCells();
				if (m_selectedCutCell >= 0 && m_selectedCutCell < pCutCells->getNumberCutCells()) {
					m_scalarFieldAttributes.pressureValue = m_pCutCellSolver->getCutCellPressure(m_selectedCutCell);
					m_scalarFieldAttributes.divergentValue = m_pCutCellSolver->getCutCellDivergence(m_selectedCutCell);
					m_cellVolume = m_pGrid->getGridData2D()->getVolume(m_selectedCellIndex.x, m_selectedCellIndex.y);
				}
				else {
					m_scalarFieldAttributes.pressureValue = m_pGrid->getGridData2D()->getPressure(m_selectedCellIndex.x, m_selectedCellIndex.y);
					m_scalarFieldAttributes.divergentValue = m_pGrid->getGridData2D()->getDivergent(m_selectedCellIndex.x, m_selectedCellIndex.y);
					m_scalarFieldAttributes.levelSetValue = m_pGrid->getGridData2D()->getLevelSetValue(m_selectedCellIndex.x, m_selectedCellIndex.y);
					m_scalarFieldAttributes.kineticEnergy = m_pGrid->getGridData2D()->getKineticEnergyValue(m_selectedCellIndex.x, m_selectedCellIndex.y);
					m_scalarFieldAttributes.kineticEnergyChange = m_pGrid->getGridData2D()->getKineticEnergyChangeValue(m_selectedCellIndex.x, m_selectedCellIndex.y);
					m_cellVolume = m_pGrid->getGridData2D()->getVolume(m_selectedCellIndex.x, m_selectedCellIndex.y);
				}
			}
			else if (m_selectedCellIndex.x > 0 && m_selectedCellIndex.y > 0) {
				m_scalarFieldAttributes.pressureValue = m_pGrid->getGridData2D()->getPressure(m_selectedCellIndex.x, m_selectedCellIndex.y);
				m_scalarFieldAttributes.divergentValue = m_pGrid->getGridData2D()->getDivergent(m_selectedCellIndex.x, m_selectedCellIndex.y);
				m_scalarFieldAttributes.levelSetValue = m_pGrid->getGridData2D()->getLevelSetValue(m_selectedCellIndex.x, m_selectedCellIndex.y);
				m_scalarFieldAttributes.kineticEnergy = m_pGrid->getGridData2D()->getKineticEnergyValue(m_selectedCellIndex.x, m_selectedCellIndex.y);
				m_scalarFieldAttributes.kineticEnergyChange = m_pGrid->getGridData2D()->getKineticEnergyChangeValue(m_selectedCellIndex.x, m_selectedCellIndex.y);
				m_cellVolume = m_pGrid->getGridData2D()->getVolume(m_selectedCellIndex.x, m_selectedCellIndex.y);
			}
		}

		template <>
		void CellVisualizationWindow<Vector3, Array3D>::updateScalarFieldAttributes() {
			if (m_selectedCellIndex.x > 0 && m_selectedCellIndex.y > 0){
				m_scalarFieldAttributes.pressureValue = m_pGrid->getGridData3D()->getPressure(m_selectedCellIndex.x, m_selectedCellIndex.y, m_selectedCellIndex.z);
				m_scalarFieldAttributes.divergentValue = m_pGrid->getGridData3D()->getDivergent(m_selectedCellIndex.x, m_selectedCellIndex.y, m_selectedCellIndex.z);
				m_scalarFieldAttributes.levelSetValue = m_pGrid->getGridData3D()->getLevelSetValue(m_selectedCellIndex.x, m_selectedCellIndex.y, m_selectedCellIndex.z);
				m_scalarFieldAttributes.kineticEnergy = m_pGrid->getGridData3D()->getKineticEnergyValue(m_selectedCellIndex.x, m_selectedCellIndex.y, m_selectedCellIndex.z);
				m_scalarFieldAttributes.kineticEnergyChange = m_pGrid->getGridData3D()->getKineticEnergyChangeValue(m_selectedCellIndex.x, m_selectedCellIndex.y, m_selectedCellIndex.z);
				m_cellVolume = m_pGrid->getGridData3D()->getVolume(m_selectedCellIndex.x, m_selectedCellIndex.y, m_selectedCellIndex.z);
			}
		}
		#pragma endregion
		
		/* Linking trick*/
		template CellVisualizationWindow<Vector2, Array2D>;
		template CellVisualizationWindow<Vector3, Array3D>;
	}
}