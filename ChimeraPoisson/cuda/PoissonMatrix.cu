#include "Poisson/PoissonMatrix.h"
#include <cusp/gallery/poisson.h>

namespace Chimera {
	namespace Poisson {
		/************************************************************************/
		/* Cuda functions                                                       */
		/************************************************************************/
		void PoissonMatrix::initDeviceMatrix() {
			m_pDeviceCuspDiagonalMatrix = NULL;
			int nonZeroEntriesPerRow, nonZeroElements = 0;
			switch(m_matrixShape) {
				case fivePointLaplace:
					nonZeroEntriesPerRow = 5;
				break;
				
				case sevenPointLaplace:
					nonZeroEntriesPerRow = 7;
				break;

				case ninePointLaplace:
					nonZeroEntriesPerRow = 9;
				break;
			}

			nonZeroElements = m_matrixSize*nonZeroEntriesPerRow;

			m_pDeviceCuspDiagonalMatrix = new cusp::dia_matrix<Integer, Scalar, cusp::device_memory>(m_matrixSize, 
																									m_matrixSize, 
																									nonZeroElements, nonZeroEntriesPerRow);
			*m_pDeviceCuspDiagonalMatrix = *m_pCuspDiagonalMatrix;
			
		}
		void PoissonMatrix::updateCudaData() {
			if(supportGPU() && m_pDeviceCuspDiagonalMatrix != NULL)
				*m_pDeviceCuspDiagonalMatrix = *m_pCuspDiagonalMatrix;
		}
		
		void PoissonMatrix::printGPUData() {
			cusp::dia_matrix<Integer, Scalar, cusp::host_memory> *pTempMatrix;
			pTempMatrix = new cusp::dia_matrix<Integer, Scalar, cusp::host_memory>(m_matrixSize, m_matrixSize, 
																									m_matrixSize*5, 5);
			*pTempMatrix = *m_pDeviceCuspDiagonalMatrix;
			cusp::print_matrix(*pTempMatrix);
			delete pTempMatrix;
		}

		/************************************************************************/
		/* Access functions                                                     */
		/************************************************************************/
		cusp::dia_matrix<Integer, Scalar, cusp::device_memory> * PoissonMatrix::getGPUData() const {
			return m_pDeviceCuspDiagonalMatrix;
		}

		cusp::dia_matrix<Integer, Scalar, cusp::host_memory> * PoissonMatrix::getCPUData() const {
			return m_pCuspDiagonalMatrix;
		}
		cusp::coo_matrix<Integer, Scalar, cusp::host_memory> *  PoissonMatrix::getCPUDataCOO() const {
			return m_pCuspCOOMatrix;
		}

		cusp::hyb_matrix<Integer, Scalar, cusp::host_memory> *  PoissonMatrix::getCPUDataHyb() const {
			return m_pCuspHybMatrix;
		}

		cusp::hyb_matrix<Integer, Scalar, cusp::device_memory> *  PoissonMatrix::getGPUDataHyb() const {
			return m_pDeviceCuspHybMatrix;
		}


		/************************************************************************/
		/* Initialization	                                                    */
		/************************************************************************/
		void PoissonMatrix::init2D() {
			int nonZeroElements;
			m_matrixSize = m_dimensions.x*m_dimensions.y;
			int nonZeroEntriesPerRow = 0;

			switch(m_matrixShape) {
				case fivePointLaplace:
					nonZeroEntriesPerRow = 5;
				break;

				case sevenPointLaplace:
					nonZeroEntriesPerRow = 7;
				break;

				case ninePointLaplace:
					nonZeroEntriesPerRow = 9;
				break;
			}
			nonZeroElements = m_matrixSize*nonZeroEntriesPerRow;
			
			Logger::get() << "PoissonMatrix: Creating Poisson matrix 2D with " << m_matrixSize << " rows and " << 
								nonZeroElements << " elements." << endl;

			if(m_diagonalMatrix) {
				m_pCuspDiagonalMatrix = new cusp::dia_matrix<Integer, Scalar, cusp::host_memory>(m_matrixSize, m_matrixSize, 
																								nonZeroElements, nonZeroEntriesPerRow);
				m_pCuspCOOMatrix = new cusp::coo_matrix<Integer, Scalar, cusp::host_memory>(m_matrixSize, m_matrixSize, nonZeroElements);
			}
			else{
				m_pCuspDiagonalMatrix = new cusp::dia_matrix<Integer, Scalar, cusp::host_memory>(m_matrixSize, m_matrixSize, 
																									nonZeroElements, nonZeroEntriesPerRow);
				m_pCuspCOOMatrix = new cusp::coo_matrix<Integer, Scalar, cusp::host_memory>(m_matrixSize, m_matrixSize, nonZeroElements);
			}

			switch(m_matrixShape) {
				case fivePointLaplace:
					initFivePointLaplace();
				break;

				case sevenPointLaplace:
					initSevenPointLaplace();
				break;

				case ninePointLaplace:
					initNinePointLaplace();
				break;
			}
			
			
			for(int i = 0; i < m_pCuspDiagonalMatrix->diagonal_offsets.size(); i++) {
				Logger::get() << i  << " " << m_pCuspDiagonalMatrix->diagonal_offsets[i] << endl;
			}
		}

		void PoissonMatrix::init3D() {
			int nonZeroElements;
			m_matrixSize = m_dimensions.x*m_dimensions.y*m_dimensions.z;
			if(m_periodicBDs) {
				nonZeroElements = m_matrixSize*9;
				Logger::get() << "PoissonMatrix: Creating Poisson matrix 3D with " << m_matrixSize << " rows and " <<
									nonZeroElements << " elements." << endl;

				m_pCuspDiagonalMatrix = new cusp::dia_matrix<Integer, Scalar, cusp::host_memory>(m_matrixSize, m_matrixSize, 
																									nonZeroElements, 9);
				//Offset configuration - used as accelerators to access Poisson matrix rows.
				pFOffset = 0;
				bOffset = 1;
				sOffset = 2;
				eOffset = 3;
				mOffset = 4;
				wOffset = 5;
				nOffset = 6;
				fOffset = 7;
				pBOffset = 8;

				//	There is no correspondent cusp function that creates a 9 point 3D periodic matrix. Initialize 
				//	coefficients manually.
				m_pCuspDiagonalMatrix->diagonal_offsets[pFOffset] = -(m_dimensions.z - 1)*m_dimensions.x* m_dimensions.y;
				m_pCuspDiagonalMatrix->diagonal_offsets[bOffset] = -m_dimensions.x*m_dimensions.y;
				m_pCuspDiagonalMatrix->diagonal_offsets[sOffset] = -m_dimensions.x;
				m_pCuspDiagonalMatrix->diagonal_offsets[eOffset] = -1;
				m_pCuspDiagonalMatrix->diagonal_offsets[mOffset] = 0;
				m_pCuspDiagonalMatrix->diagonal_offsets[wOffset] = 1;
				m_pCuspDiagonalMatrix->diagonal_offsets[nOffset] = m_dimensions.x;
				m_pCuspDiagonalMatrix->diagonal_offsets[fOffset] = m_dimensions.x*m_dimensions.y;
				m_pCuspDiagonalMatrix->diagonal_offsets[pBOffset] = (m_dimensions.z - 1)*m_dimensions.x* m_dimensions.y;

				for(int i = 0; i < m_pCuspDiagonalMatrix->diagonal_offsets.size(); i++) {
					Logger::get() << i  << " " << m_pCuspDiagonalMatrix->diagonal_offsets[i] << endl;
				}
			} else {
				nonZeroElements = m_matrixSize*7;
				Logger::get() << "PoissonMatrix: Creating Poisson matrix 3D with " << m_matrixSize << " rows and " <<
					nonZeroElements << " elements." << endl;

				m_pCuspDiagonalMatrix = new cusp::dia_matrix<Integer, Scalar, cusp::host_memory>(m_matrixSize, m_matrixSize, 
					nonZeroElements, 7);
				if(!m_diagonalMatrix) {

					m_pCuspCOOMatrix = new cusp::coo_matrix<Integer, Scalar, cusp::host_memory>(m_matrixSize, m_matrixSize, nonZeroElements);
				}

				cusp::gallery::poisson7pt(*m_pCuspDiagonalMatrix, m_dimensions.x, m_dimensions.y, m_dimensions.z);
				/*m_pCuspDiagonalMatrix->diagonal_offsets[0] = (-m_dimensions.x*m_dimensions.y);
				m_pCuspDiagonalMatrix->diagonal_offsets[1] = (-m_dimensions.x);
				m_pCuspDiagonalMatrix->diagonal_offsets[2] = (-1);
				m_pCuspDiagonalMatrix->diagonal_offsets[3] = (0);
				m_pCuspDiagonalMatrix->diagonal_offsets[4] = (1);
				m_pCuspDiagonalMatrix->diagonal_offsets[5] = (m_dimensions.x);
				m_pCuspDiagonalMatrix->diagonal_offsets[6] = (m_dimensions.x*m_dimensions.y);*/
				for(int i = 0; i < m_pCuspDiagonalMatrix->diagonal_offsets.size(); i++) {
					Logger::get() << i  << " " << m_pCuspDiagonalMatrix->diagonal_offsets[i] << endl;
				}

				//Offset configuration
				bOffset = 0;
				sOffset = 1;
				eOffset = 2;
				mOffset = 3;
				wOffset = 4;
				nOffset = 5;
				fOffset = 6;
			}
			

		}

		void PoissonMatrix::initFivePointLaplace() {
			cusp::gallery::poisson5pt(*m_pCuspDiagonalMatrix, m_dimensions.x, m_dimensions.y);
			if(!m_diagonalMatrix)
				cusp::gallery::poisson5pt(*m_pCuspCOOMatrix, m_dimensions.x, m_dimensions.y);

			/** Configuring stencils and transformed offsets used in accessing elements in a row */
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(-1, 0)));
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(0, -1)));
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(0, 0)));
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(1, 0)));
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(0, 1)));

			m_stencils.push_back(stencil2D_t(-1, 0));
			m_stencils.push_back(stencil2D_t(0, -1));
			m_stencils.push_back(stencil2D_t(0, 0));
			m_stencils.push_back(stencil2D_t(1, 0));
			m_stencils.push_back(stencil2D_t(0, 1));

			//Transformed offsets align
			sort(m_transformedOffsets.begin(), m_transformedOffsets.end());

			//Offset configuration - used as accelerators to access Poisson matrix rows.
			sOffset = 0;
			wOffset = 1;
			mOffset = 2;
			eOffset = 3;
			nOffset = 4;
			pEOffset = pWOffset = neOffset = nwOffset = seOffset = swOffset = bOffset = fOffset = -1; //Not used
		}

		void PoissonMatrix::initSevenPointLaplace() {
			//Offset configuration - used as accelerators to access Poisson matrix rows.
			sOffset = 0;
			pEOffset = 1;
			wOffset = 2;
			mOffset = 3;
			eOffset = 4;
			pWOffset = 5;
			nOffset = 6;

			neOffset = nwOffset = seOffset = swOffset = bOffset = fOffset = -1; //Not used

			//	There is no correspondent cusp function that creates a 7 point 2D periodic matrix. Initialize 
			//	coefficients manually.
			m_pCuspDiagonalMatrix->diagonal_offsets[sOffset] = -m_dimensions.x;
			m_pCuspDiagonalMatrix->diagonal_offsets[pEOffset] = -(m_dimensions.x - 1);
			m_pCuspDiagonalMatrix->diagonal_offsets[wOffset] = -1;
			m_pCuspDiagonalMatrix->diagonal_offsets[mOffset] = 0;
			m_pCuspDiagonalMatrix->diagonal_offsets[eOffset] = 1;
			m_pCuspDiagonalMatrix->diagonal_offsets[pWOffset] = (m_dimensions.x - 1);
			m_pCuspDiagonalMatrix->diagonal_offsets[nOffset] = m_dimensions.x;

			/** Configuring stencils and transformed offsets used in accessing elements in a row */
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(-1, 0)));
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(0, -1)));
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(0, 0)));
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(1, 0)));
			m_transformedOffsets.push_back(transformStencil(stencil2D_t(0, 1)));

			m_stencils.push_back(stencil2D_t(-1, 0));
			m_stencils.push_back(stencil2D_t(0, -1));
			m_stencils.push_back(stencil2D_t(0, 0));
			m_stencils.push_back(stencil2D_t(1, 0));
			m_stencils.push_back(stencil2D_t(0, 1));

			//Transformed offsets align
			sort(m_transformedOffsets.begin(), m_transformedOffsets.end());

		}

		void PoissonMatrix::initNinePointLaplace() {
			/** Configuring stencils and transformed offsets used in accessing elements in a row */
			for(int i = -1; i < 2; i++) {
				for(int j = -1; j < 2; j++) {
					m_transformedOffsets.push_back(transformStencil(stencil2D_t(i, j)));
					m_stencils.push_back(stencil2D_t(i, j));
				}
			}
			
			//Transformed offsets align
			sort(m_transformedOffsets.begin(), m_transformedOffsets.end());

			//Offset configuration - used as accelerators to access Poisson matrix rows.
			swOffset = 0;
			sOffset = 1;
			seOffset = 2;
			wOffset = 3;
			mOffset = 4;
			eOffset = 5;
			nwOffset = 6;
			nOffset = 7;
			neOffset = 8;

			if(m_periodicBDs) { 
				// By the nature of the Nine point Laplace matrix, coincidentally we have that the periodic offsets
				// at boundaries are the same values as the given offsets
				pEOffset = seOffset;
				pWOffset = nwOffset;
			}

			cusp::gallery::poisson9pt(*m_pCuspDiagonalMatrix, m_dimensions.x, m_dimensions.y);
		}

		/************************************************************************/
		/* Row access functions                                                 */
		/************************************************************************/
		void PoissonMatrix::setRow(int row, Scalar pn, Scalar pw, Scalar pc, Scalar pe, Scalar ps) {
			m_pCuspDiagonalMatrix->values(row, nOffset) = pn;
			m_pCuspDiagonalMatrix->values(row, wOffset) = pw;
			m_pCuspDiagonalMatrix->values(row, mOffset) = pc;
			m_pCuspDiagonalMatrix->values(row, eOffset) = pe;
			m_pCuspDiagonalMatrix->values(row, sOffset) = ps;
		}

		void PoissonMatrix::setRow(int row, Scalar pn, Scalar pw, Scalar pb, Scalar pc, Scalar pe, Scalar ps, Scalar pf) {
			m_pCuspDiagonalMatrix->values(row, nOffset) = pn;
			m_pCuspDiagonalMatrix->values(row, wOffset) = pw;
			m_pCuspDiagonalMatrix->values(row, bOffset) = pb;
			m_pCuspDiagonalMatrix->values(row, mOffset) = pc;
			m_pCuspDiagonalMatrix->values(row, eOffset) = pe;
			m_pCuspDiagonalMatrix->values(row, sOffset) = ps;
			m_pCuspDiagonalMatrix->values(row, fOffset) = pf;
		}
		void PoissonMatrix::setNorthValue(int i, Scalar pn) {
			m_pCuspDiagonalMatrix->values(i, nOffset) = pn;
		}
		void PoissonMatrix::setSouthValue(int i, Scalar ps) {
			m_pCuspDiagonalMatrix->values(i, sOffset) = ps;
		}
		void PoissonMatrix::setCentralValue(int i, Scalar pc) {
			m_pCuspDiagonalMatrix->values(i, mOffset) = pc;
		}
		void PoissonMatrix::setWestValue(int i, Scalar pw) {
			m_pCuspDiagonalMatrix->values(i, wOffset) = pw;
		}
		void PoissonMatrix::setEastValue(int i, Scalar pe) {
			m_pCuspDiagonalMatrix->values(i, eOffset) = pe;
		}
		void PoissonMatrix::setBackValue(int i, Scalar pb) {
			m_pCuspDiagonalMatrix->values(i, bOffset) = pb;
		}
		void PoissonMatrix::setFrontValue(int i, Scalar pf) {
			m_pCuspDiagonalMatrix->values(i, fOffset) = pf;
		}
		void PoissonMatrix::setPeriodicWestValue(int i, Scalar pWF) {
			m_pCuspDiagonalMatrix->values(i, pWOffset) = pWF;
		}
		void PoissonMatrix::setPeriodicEastValue(int i, Scalar pEF) {
			m_pCuspDiagonalMatrix->values(i, pEOffset) = pEF;
		}

		void PoissonMatrix::setPeriodicBackValue(int i, Scalar pBV) {
			m_pCuspDiagonalMatrix->values(i, pBOffset) = pBV;
		}
		void PoissonMatrix::setPeriodicFrontValue(int i, Scalar pFV) {
			m_pCuspDiagonalMatrix->values(i, pFOffset) = pFV;
		}

		void PoissonMatrix::setNorthWestValue(int i, Scalar pNW) {
			m_pCuspDiagonalMatrix->values(i, nwOffset) = pNW;
		}
		void PoissonMatrix::setNorthEastValue(int i, Scalar pNE) {
			m_pCuspDiagonalMatrix->values(i, neOffset) = pNE;
		}
		void PoissonMatrix::setSouthWestValue(int i, Scalar pSW) {
			m_pCuspDiagonalMatrix->values(i, swOffset) = pSW;
		}
		void PoissonMatrix::setSouthEastValue(int i, Scalar pSE) {
			m_pCuspDiagonalMatrix->values(i, seOffset) = pSE;
		}
	
		void PoissonMatrix::setValue(int ithElement, int row, int column, Scalar pValue) {
			m_pCuspCOOMatrix->row_indices[ithElement] = row;
			m_pCuspCOOMatrix->column_indices[ithElement] = column;
			m_pCuspCOOMatrix->values[ithElement] = pValue;
		}

		Scalar PoissonMatrix::getValue(int row, int column) const {
			for(int i = 0; i < m_pCuspCOOMatrix->num_entries; i++) {
				if(m_pCuspCOOMatrix->row_indices[i] == row &&
					m_pCuspCOOMatrix->column_indices[i] == column)
					return m_pCuspCOOMatrix->values[i];
			}
			return -1;
		}
		int PoissonMatrix::getNumberOfEntriesCOO() {
			return m_pCuspCOOMatrix->num_entries;
		}
		vector <pair <uint, Scalar>> PoissonMatrix::getRowCOOMatrix(int row) {
			vector <pair <uint, Scalar>> res;
			for (int i = 0; i < m_pCuspCOOMatrix->num_entries; i++) {
				if (m_pCuspCOOMatrix->row_indices[i] == row && m_pCuspCOOMatrix->values[i] != 0)
					res.push_back(pair<uint, Scalar>(m_pCuspCOOMatrix->column_indices[i], m_pCuspCOOMatrix->values[i]));
				}
			return res;
		}

		Scalar PoissonMatrix::getNorthValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, nOffset);
		}
		Scalar PoissonMatrix::getSouthValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, sOffset);
		}
		Scalar PoissonMatrix::getCentralValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, mOffset);
		}
		Scalar PoissonMatrix::getWestValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, wOffset);
		}
		Scalar PoissonMatrix::getEastValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, eOffset);
		}
		Scalar PoissonMatrix::getBackValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, bOffset);
		}
		Scalar PoissonMatrix::getFrontValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, fOffset);
		}
		Scalar PoissonMatrix::getPeriodicWestValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, pWOffset);
		}
		Scalar PoissonMatrix::getPeriodicEastValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, pEOffset);
		}

		Scalar PoissonMatrix::getNorthWestValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, nwOffset);
		}
		Scalar PoissonMatrix::getNorthEastValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, neOffset);
		}
		
		Scalar PoissonMatrix::getSouthWestValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, swOffset);
		}
		Scalar PoissonMatrix::getSouthEastValue(int i) const {
			return m_pCuspDiagonalMatrix->values(i, seOffset);
		}
		

		Scalar PoissonMatrix::getValue(int row, const stencil2D_t &elementOffset) {
			int transformedIndex = transformStencil(elementOffset);
			 int realIndex = lower_bound(m_transformedOffsets.begin(), m_transformedOffsets.end(), transformedIndex)
								- m_transformedOffsets.begin();
			return m_pCuspDiagonalMatrix->values(row, realIndex);
		}

		int PoissonMatrix::getNorthOffset() const {
			return m_pCuspDiagonalMatrix->diagonal_offsets[nOffset];
		}
		int PoissonMatrix::getSouthOffset() const {
			return m_pCuspDiagonalMatrix->diagonal_offsets[sOffset];
		}
		int PoissonMatrix::getWestOffset() const {
			return  m_pCuspDiagonalMatrix->diagonal_offsets[wOffset];
		}
		int PoissonMatrix::getEastOffset() const {
			return m_pCuspDiagonalMatrix->diagonal_offsets[eOffset];
		}


		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		bool PoissonMatrix::isSingular() const {
			DoubleScalar totalSum = 0;
			int finalIndex = 0;
			if(m_dimensions.z == 0 && !m_periodicBDs) {
				finalIndex = 5;
			} else if(m_dimensions.z == 0){
				finalIndex = 7;
			} else if(!m_periodicBDs) {
				finalIndex = 7;
			} else {
				finalIndex = 9;
			}
			for(int i = 0; i < m_matrixSize; i++) {
				for(int k = 0; k < finalIndex; k++) {
					totalSum += m_pCuspDiagonalMatrix->values(i, k);
				}
			}
			if(abs(totalSum) < 1e-4)
				return true;

			return false;
		}

		bool PoissonMatrix::isSingularCOO() const {
			DoubleScalar totalSum = 0;
			for (int i = 0; i < m_pCuspCOOMatrix->num_entries; i++) {
				totalSum += m_pCuspCOOMatrix->values[i];
			}

			if (abs(totalSum) < 1e-4)
				return true;

			return false;
		}

		


		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void PoissonMatrix::applyCorrection(Scalar delta) {
			logLevel_t lastLogLevel = Logger::get().getDefaultLogLevel();
			//Logger::get().setDefaultLogLevel(Log_HighPriority);
			//Logger::get() << "PoissonMatrix: Applying correction of " << delta << endl;
			//Logger::get().setDefaultLogLevel(lastLogLevel);
			for(int i = 0; i < m_matrixSize; i++) {
				m_pCuspDiagonalMatrix->values(i, mOffset) = m_pCuspDiagonalMatrix->values(i, mOffset)*(1 + delta);
			}
		}
		void PoissonMatrix::copyDIAtoCOO() {
			*m_pCuspCOOMatrix = *m_pCuspDiagonalMatrix;
		}
		void PoissonMatrix::resizeToFitCOO() {
			int nonZeroElements = (m_matrixSize + m_numAdditionalCells)*5;
			if(m_dimensions.z != 0)
				nonZeroElements = (m_matrixSize + m_numAdditionalCells)*7;

			m_pCuspCOOMatrix->resize(m_matrixSize + m_numAdditionalCells, m_matrixSize + m_numAdditionalCells, nonZeroElements);
		}

		void PoissonMatrix::copyCOOtoHyb() {
			if(m_pCuspHybMatrix != NULL)
				delete m_pCuspHybMatrix;
			m_pCuspHybMatrix = new cusp::hyb_matrix<int,float,cusp::host_memory>(*m_pCuspCOOMatrix);
			if (m_pDeviceCuspDiagonalMatrix) {
				if (m_pDeviceCuspHybMatrix != NULL)
					delete m_pDeviceCuspHybMatrix;
				m_pDeviceCuspHybMatrix = new cusp::hyb_matrix<int, float, cusp::device_memory>(*m_pCuspHybMatrix);
			}
			
		}


		/************************************************************************/
		/* Printing                                                             */
		/************************************************************************/
		void PoissonMatrix::cuspPrint() const {
			cusp::print_matrix(*m_pCuspDiagonalMatrix);
			cout << "PRINTING COO MATRIX" << endl;
			cusp::print_matrix(*m_pCuspCOOMatrix);
		}

		

	}
}