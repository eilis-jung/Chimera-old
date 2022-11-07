#include "Interpolation/BilinearStreamfunctionInterpolant2D.h"

namespace Chimera {
	namespace Interpolation {
		
		#pragma region Constructors
		template<class valueType>
		BilinearStreamfunctionInterpolant2D<valueType>::BilinearStreamfunctionInterpolant2D(const Array2D<valueType>& values, Scalar gridDx)
			: Interpolant(values) {
			m_pStreamDiscontinuousValues = new Array2D <vector<Scalar>> (values.getDimensions());
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for(int j = 0; j < m_gridDimensions.y; j++) {
					(*m_pStreamDiscontinuousValues)(i, j).resize(4);
				}
			}
			m_dx = gridDx;
			m_pMeanvalueInterpolant = nullptr;
			m_pCutCells = nullptr;
		}

		template<class valueType>
		BilinearStreamfunctionInterpolant2D<valueType>::BilinearStreamfunctionInterpolant2D(Array2D<vector<Scalar>> *pStreamfunctionValues, Scalar gridDx)
			: Interpolant() {
			m_pStreamDiscontinuousValues = pStreamfunctionValues;
			m_pStreamContinuousValues = nullptr;
			m_gridDimensions = pStreamfunctionValues->getDimensions();
			m_dx = gridDx;
			m_pMeanvalueInterpolant = nullptr;
			m_pCutCells = nullptr;
		}

		template<class valueType>
		BilinearStreamfunctionInterpolant2D<valueType>::BilinearStreamfunctionInterpolant2D(Array2D<Scalar> *pStreamfunctionValues, Scalar gridDx)
			: Interpolant() {
			m_pStreamContinuousValues = pStreamfunctionValues;
			m_pStreamDiscontinuousValues = nullptr;
			m_gridDimensions = pStreamfunctionValues->getDimensions();
			m_dx = gridDx;
			m_pMeanvalueInterpolant = nullptr;
			m_pCutCells = nullptr;
		}
		#pragma endregion

		#pragma region Functionalities
		/** Templated interpolation is the only one needed */
		template<>
		Vector2 BilinearStreamfunctionInterpolant2D<Vector2>::interpolate(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));

			i = clamp(i, 0, m_values.getDimensions().x - 2);
			j = clamp(j, 0, m_values.getDimensions().y - 2);

			if (m_pCutCells) {
				if (m_pCutCells->isCutCellAt(i, j)) { //Actually implementing MVC
					return interpolateCutCell(position);
				}
			}
			

			if (i < 0 || j < 0 || i > m_gridDimensions.x - 1 || j > m_gridDimensions.y - 1)
				return Vector2(0, 0);

			Scalar partialX = 0, partialY = 0;
			if (m_pStreamDiscontinuousValues) {
				partialX = bilinearPartialDerivativeX(gridSpacePosition, (*m_pStreamDiscontinuousValues)(i, j));
				partialY = bilinearPartialDerivativeY(gridSpacePosition,(*m_pStreamDiscontinuousValues)(i, j));
			}
			else {
				vector<Scalar> streamfunctionValues(4);
				streamfunctionValues[0] = (*m_pStreamContinuousValues)(i, j);
				streamfunctionValues[1] = (*m_pStreamContinuousValues)(i + 1, j);
				streamfunctionValues[2] = (*m_pStreamContinuousValues)(i + 1, j + 1);
				streamfunctionValues[3] = (*m_pStreamContinuousValues)(i, j + 1);
				partialX = bilinearPartialDerivativeX(gridSpacePosition, streamfunctionValues);
				partialY = bilinearPartialDerivativeY(gridSpacePosition, streamfunctionValues);
			}

			partialX /= m_dx;
			partialY /= m_dx;

			return Vector2(partialY, -partialX);
		}


		template<>
		Vector2 BilinearStreamfunctionInterpolant2D<Vector2>::interpolateCutCell(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));

			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			Scalar streamInterp = m_pMeanvalueInterpolant->interpolate(position);

			Scalar dxS = 0.0001;

			Scalar signX = 1.0f, signY = 1.0f;
			Scalar streamInterpX, streamInterpY;

			uint currCellIndex = m_pCutCells->getCutCellIndex(gridSpacePosition);
			Vector2 gridSpaceNextX = gridSpacePosition + Vector2(dxS, 0) / m_dx;
			Vector2 gridSpaceNextY = gridSpacePosition + Vector2(0, dxS) / m_dx;

			dimensions_t gridSpaceNextXDim(gridSpaceNextX.x, gridSpaceNextX.y);
			dimensions_t gridSpaceNextYDim(gridSpaceNextY.x, gridSpaceNextY.y);
			if ((!m_pCutCells->isCutCell(gridSpaceNextXDim)) || m_pCutCells->getCutCellIndex(gridSpaceNextX) != currCellIndex) {
				streamInterpX = m_pMeanvalueInterpolant->interpolate(position - Vector2(dxS, 0));
				signX = -1.0f;
			}
			else {
				streamInterpX = m_pMeanvalueInterpolant->interpolate(position + Vector2(dxS, 0));
			}
			if ((!m_pCutCells->isCutCell(gridSpaceNextYDim)) || m_pCutCells->getCutCellIndex(gridSpaceNextY) != currCellIndex) {
				streamInterpY = m_pMeanvalueInterpolant->interpolate(position - Vector2(0, dxS));
				signY = -1.0f;
			}
			else {
				streamInterpY = m_pMeanvalueInterpolant->interpolate(position + Vector2(0, dxS));
			}

			//return Vector2(0, 0);

			Vector2 tempVelocity = Vector2((streamInterpY - streamInterp) / dxS, -(streamInterpX - streamInterp) / dxS);

			return Vector2(signY*(streamInterpY - streamInterp) / dxS, -signX*(streamInterpX - streamInterp) / dxS);
		}

		template<class valueType>
		void BilinearStreamfunctionInterpolant2D<valueType>::computeStreamfunctions() {
			for (int i = 0; i < (*m_pStreamDiscontinuousValues).getDimensions().x - 1; i++) {
				for (int j = 0; j < (*m_pStreamDiscontinuousValues).getDimensions().y - 1; j++) {
					Scalar cellFluxes[3];
					
					if (m_pCutCells && m_pCutCells->isCutCellAt(i, j)) {
						continue;
					} 
					
					if (m_pCutCells && m_pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge).size() > 0) {
						auto currEdge = m_pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge).front();
						cellFluxes[0] = -currEdge->getVelocity().y*m_dx;
					}
					else {
						cellFluxes[0] = (-m_values(i, j).y)*m_dx;
					}

					if (m_pCutCells && m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), yAlignedEdge).size() > 0) {
						auto currEdge = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), yAlignedEdge).front();
						cellFluxes[1] = currEdge->getVelocity().x*m_dx;
					}
					else {
						cellFluxes[1] = m_values(i + 1, j).x*m_dx;
					}

					if (m_pCutCells && m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), xAlignedEdge).size() > 0) {
						auto currEdge = m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), xAlignedEdge).front();
						cellFluxes[2] = currEdge->getVelocity().y*m_dx;
					}
					else {
						cellFluxes[2] = m_values(i, j + 1).y*m_dx;
					}

					Scalar s1, s2, s3, s4;
					(*m_pStreamDiscontinuousValues)(i, j)[0] = 0;
					s1 = 0; 
					(*m_pStreamDiscontinuousValues)(i, j)[1] = cellFluxes[0] + (*m_pStreamDiscontinuousValues)(i, j)[0];
					s2 = (*m_pStreamDiscontinuousValues)(i, j)[1];
					(*m_pStreamDiscontinuousValues)(i, j)[2] = cellFluxes[1] + (*m_pStreamDiscontinuousValues)(i, j)[1];
					s3 = (*m_pStreamDiscontinuousValues)(i, j)[2];
					(*m_pStreamDiscontinuousValues)(i, j)[3] = cellFluxes[2] + (*m_pStreamDiscontinuousValues)(i, j)[2];
					s4 = (*m_pStreamDiscontinuousValues)(i, j)[3];

					s4 = -1;
				}
			}
			if (m_pMeanvalueInterpolant != nullptr && m_pCutCells != nullptr) {
				CutCellsVelocities2D *pCutCellsVelocities2D = dynamic_cast<CutCellsVelocities2D *>(m_pCutCellsVelocities);

				for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
					HalfFace<Vector2> & curCutCell = m_pCutCells->getCutCell(i);
					DoubleScalar prevStreamfunction;
					for (int j = 0; j < curCutCell.getHalfEdges().size(); j++) {
						auto currVertex = curCutCell.getHalfEdges()[j]->getVertices().first;
						if (j == 0) {
							curCutCell.setStreamfunction(Vector2(0, 0), j);
							prevStreamfunction = 0;
						}
						else {
							Vector2 edgeNormal = curCutCell.getHalfEdges()[j - 1]->getNormal();
							auto currEdge = curCutCell.getHalfEdges()[j - 1]->getEdge();
							//Since the poisson matrix is normalized on regular cells, we multiply the fluxes by m_gridSpacing everywhere. 
							//LengthFraction gives us the un-obstructed cell fraction
							Scalar currFlux = edgeNormal.dot(currEdge->getVelocity())*currEdge->getRelativeFraction()*m_dx;

							curCutCell.setStreamfunction(Vector2(currFlux + prevStreamfunction, currFlux + prevStreamfunction), j);
							Scalar ss = currFlux + prevStreamfunction;
							ss = 0;
							prevStreamfunction += currFlux;
							//m_pCutCellsVelocities->getStreamfunctions()[i][j] = currFlux + m_pCutCellsVelocities->getStreamfunctions()[i][j - 1];
						}
					}
				}
			}
		}


		template<class valueType>
		void BilinearStreamfunctionInterpolant2D<valueType>::computeContinuousStreamfunctions() {
			/*(*m_pStreamContinuousValues)(0, 0) = 0;
			for (int i = 1; i < m_gridDimensions.x; i++) {
				(*m_pStreamContinuousValues)(i, 0) = m_values(i, 0).y*m_dx + (*m_pStreamContinuousValues)(i - 1, 0);
			}
			
			for (int i = 1; i < m_gridDimensions.x; i++) {
				for (int j = 1; j < m_gridDimensions.y; j++) {
					(*m_pStreamContinuousValues)(i, j) = m_values(i, j).x*m_dx + (*m_pStreamContinuousValues)(i, j - 1);
				}
			}
			m_streamfunctionGrid(i + 1, j + 1) = m_fineGridVelocities(i + 1, j).x*fineGridDx + m_streamfunctionGrid(i + 1, j);
*/

			/*for (int j = 1; j < m_gridDimensions.y; j++) {
			(*m_pStreamContinuousValues)(0, j) = m_values(0, j).x*m_dx + (*m_pStreamContinuousValues)(0, j - 1);
			}*/
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		/** Partial derivative in respect to X of the bilinear interpolation function. */
		template<class valueType>
		Scalar BilinearStreamfunctionInterpolant2D<valueType>::bilinearPartialDerivativeX(const Vector2 &position, const vector<Scalar> &streamfunctionValues) {
			int i = static_cast <int> (floor(position.x));
			int j = static_cast <int> (floor(position.y));

			Vector2 cellStart(i + 0.0f, j + 0.0f);
			Vector2 cellEnd(i + 1.0f, j + 1.0f);

			Scalar normalizedY = position.y - cellStart.y;

			return  streamfunctionValues[0] * (normalizedY - 1) +	//F(0, 0)
					streamfunctionValues[1] * (1 - normalizedY) +	//F(1, 0)
					streamfunctionValues[2] * (normalizedY)+		//F(1, 1)
					streamfunctionValues[3] * (-normalizedY);		//F(0, 1)
		}

		/** Partial derivative in respect to Y of the bilinear interpolation function. */
		template<class valueType>
		Scalar BilinearStreamfunctionInterpolant2D<valueType>::bilinearPartialDerivativeY(const Vector2 &position, const vector<Scalar> &streamfunctionValues) {
			int i = static_cast <int> (floor(position.x));
			int j = static_cast <int> (floor(position.y));

			Vector2 cellStart(i + 0.0f, j + 0.0f);
			Vector2 cellEnd(i + 1.0f, j + 1.0f);

			Scalar normalizedX = position.x - cellStart.x;

			return  streamfunctionValues[0] * (normalizedX - 1) +	//F(0, 0)
					streamfunctionValues[1] * (-normalizedX) +	//F(1, 0)
					streamfunctionValues[2] * (normalizedX)+		//F(1, 1)
					streamfunctionValues[3] * (1 - normalizedX);	//F(0, 1)
		}

		template<class valueType>
		void BilinearStreamfunctionInterpolant2D<valueType>::buildCutCellPoints() {
			m_cutCellsPoints.clear();
			for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
				vector<Vector2> currCellPoints;
				auto currCell = m_pCutCells->getCutCell(i);
				for (int j = 0; j < currCell.getHalfEdges().size(); j++) {
					currCellPoints.push_back(currCell.getHalfEdges()[j]->getVertices().first->getPosition());
				}
				m_cutCellsPoints.push_back(currCellPoints);
			}
		}

		#pragma endregion
		/** Template linker trickerino for templated classes in CPP*/
		template class BilinearStreamfunctionInterpolant2D<Vector2>;
	}
}