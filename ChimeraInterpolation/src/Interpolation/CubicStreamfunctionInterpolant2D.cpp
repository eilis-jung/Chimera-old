#include "Interpolation/CubicStreamfunctionInterpolant2D.h"

namespace Chimera {
	namespace Interpolation {

		template<class valueType>
		const int CubicStreamfunctionInterpolant2D<valueType>::coefficientsMatrix[256] = {
			1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			-3, 0, 0, 3, 0, 0, 0, 0,-2, 0, 0,-1, 0, 0, 0, 0,
			2, 0, 0,-2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0,-3, 0, 0, 3, 0, 0, 0, 0,-2, 0, 0,-1,
			0, 0, 0, 0, 2, 0, 0,-2, 0, 0, 0, 0, 1, 0, 0, 1,
			-3, 3, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0,
			9,-9, 9,-9, 6, 3,-3,-6, 6,-6,-3, 3, 4, 2, 1, 2,
			-6, 6,-6, 6,-4,-2, 2, 4,-3, 3, 3,-3,-2,-1,-1,-2,
			2,-2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0,
			-6, 6,-6, 6,-3,-3, 3, 3,-4, 4, 2,-2,-2,-2,-1,-1,
			4,-4, 4,-4, 2, 2,-2,-2, 2,-2,-2, 2, 1, 1, 1, 1
		};

		#pragma region Constructors
		template<>
		CubicStreamfunctionInterpolant2D<Vector2>::CubicStreamfunctionInterpolant2D(const Array2D<Vector2> &values, Scalar gridDx)
			: BilinearStreamfunctionInterpolant2D(values, gridDx) {

		}

		template<>
		CubicStreamfunctionInterpolant2D<Vector2>::CubicStreamfunctionInterpolant2D(Array2D<vector<Scalar>> *pStreamfunctionValues, Scalar gridDx)
			: BilinearStreamfunctionInterpolant2D(pStreamfunctionValues, gridDx) {

		}
		CubicStreamfunctionInterpolant2D<Vector2>::CubicStreamfunctionInterpolant2D(Array2D<Scalar> *pStreamfunctionValues, Scalar gridDx) 
			: BilinearStreamfunctionInterpolant2D(pStreamfunctionValues, gridDx) {
			
		}
		#pragma endregion

		#pragma region Functionalities
		template<>
		Vector2 CubicStreamfunctionInterpolant2D<Vector2>::interpolate(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));

			if (m_pCutCells && i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
				if (m_pCutCells->isCutCellAt(i, j)) { //Actually implementing MVC
					return interpolateCutCell(position);
				}
			}

			Scalar partialX, partialY;
			if (i <= 1 || j <= 1 || i >= m_gridDimensions.x - 2 || j >= m_gridDimensions.y - 2) {
				return BilinearStreamfunctionInterpolant2D<Vector2>::interpolate(position);
			}
			else {
				if (m_pStreamDiscontinuousValues) {
					Scalar incrementDx = 0.0001;
					Scalar interpScalar = interpolateScalar(gridSpacePosition, (*m_pStreamDiscontinuousValues)(i, j));
					Scalar interpX = interpolateScalar(gridSpacePosition + Vector2(1, 0)*incrementDx, (*m_pStreamDiscontinuousValues)(i, j));
					Scalar interpY = interpolateScalar(gridSpacePosition + Vector2(0, 1)*incrementDx, (*m_pStreamDiscontinuousValues)(i, j));

					partialX = cubicPartialDerivativeX(gridSpacePosition, (*m_pStreamDiscontinuousValues)(i, j));
					partialY = cubicPartialDerivativeY(gridSpacePosition, (*m_pStreamDiscontinuousValues)(i, j));
				}
				else {
					vector<Scalar> streamfunctionValues(4);
					streamfunctionValues[0] = (*m_pStreamContinuousValues)(i, j);
					streamfunctionValues[1] = (*m_pStreamContinuousValues)(i + 1, j);
					streamfunctionValues[2] = (*m_pStreamContinuousValues)(i + 1, j + 1);
					streamfunctionValues[3] = (*m_pStreamContinuousValues)(i, j + 1);
					partialX = cubicPartialDerivativeX(gridSpacePosition, streamfunctionValues);
					partialY = cubicPartialDerivativeY(gridSpacePosition, streamfunctionValues);
				}
			}
			partialX /= m_dx;
			partialY /= m_dx;

			return Vector2(partialY, -partialX);
		}

		template<>
		void CubicStreamfunctionInterpolant2D<Vector2>::saveCellInfoToFile(int i, int j, int numSubdivisions, const string &filename) {
			string densityExportName("Flow Logs/" + filename);
			auto_ptr<ofstream> fileStream(new ofstream(densityExportName.c_str()));

			//(*fileStream) << intToStr(scalarFieldArray.getDimensions().x) << " " << intToStr(scalarFieldArray.getDimensions().y) << endl;

			Scalar newDx = m_dx / numSubdivisions;
			Vector2 perturbation(1e-4, 1e-4);
			Vector2 cellStart(i*m_dx, j*m_dx);
			for (int k = 0; k <= numSubdivisions; k++) {
				for (int l = 0; l <= numSubdivisions; l++) {
					Vector2 interpolationPoint = cellStart + Vector2(k, l)*newDx;
					if (k == 0)
						interpolationPoint.x += perturbation.x;
					if (l == 0)
						interpolationPoint.y += perturbation.y;

					if (k == numSubdivisions)
						interpolationPoint.x -= perturbation.x;
					if (l == numSubdivisions)
						interpolationPoint.y -= perturbation.y;

					(*fileStream) << scalarToStr(interpolateScalar(interpolationPoint/m_dx, (*m_pStreamDiscontinuousValues)(i, j))) << " ";
				}
				(*fileStream) << endl;
			}
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template<class valueType>
		Scalar CubicStreamfunctionInterpolant2D<valueType>::interpolateScalar(const Vector2 &position, int currCellIndex) {
			auto cutCell = m_pCutCells->getCutCell(currCellIndex);

			vector<DoubleScalar> weights(m_cutCellsPoints[currCellIndex].size(), 0.0f);
			vector<DoubleScalar> tangentialWeights(m_cutCellsPoints[currCellIndex].size() * 2, 0.0f);
			vector<DoubleScalar> normalWeights(m_cutCellsPoints[currCellIndex].size() * 2, 0.0f);

			CubicMVC::cubicMVCs(m_cutCellsPoints[currCellIndex], position, weights, normalWeights, tangentialWeights);

			Scalar interpolatedValue = 0.0f;
			for (int i = 0; i < weights.size(); i++) {
				pair<Scalar, Scalar> tangentialGrad = getTangentialDerivatives(currCellIndex, i);
				pair<Scalar, Scalar> normalGrad = getNormalDerivatives(currCellIndex, i);
				int nextI = roundClamp<int>(i + 1, 0, weights.size());

				Scalar tangentContribution = 0;
				Scalar normalContribution = 0;

				tangentContribution = tangentialWeights[i * 2] * tangentialGrad.first + tangentialWeights[i * 2 + 1] * tangentialGrad.second;
				normalContribution = normalWeights[i * 2] * normalGrad.first + normalWeights[i * 2 + 1] * normalGrad.second;
				
				auto currVertex = cutCell.getHalfEdges()[i]->getVertices().first;
				interpolatedValue += cutCell.getStreamfunction(i).x*weights[i] + tangentContribution + normalContribution;
			}
			return interpolatedValue;
		}

		template<class valueType>
		Scalar CubicStreamfunctionInterpolant2D<valueType>::interpolateScalar(const Vector2 &position, const vector<Scalar> &streamfunctionValues) {
			int i = static_cast<int> (floor(position.x));
			int j = static_cast<int> (floor(position.y));

			Vector2 densityP1, densityP2;

			densityP1 = Vector2(i, j);
			densityP2 = Vector2(i + 1.f, j + 1.f);

			Scalar cubicTable[16];
			Scalar origFunction[4];

			origFunction[0] = streamfunctionValues[0];
			origFunction[1] = streamfunctionValues[1];
			origFunction[2] = streamfunctionValues[2];
			origFunction[3] = streamfunctionValues[3];

			Scalar gradientsX[4];
			Scalar gradientsY[4];
			Scalar crossDerivatives[4];

			if (m_pStreamDiscontinuousValues)
				calculateDiscontinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);
			else
				calculateContinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);

			generateCubicTable(origFunction, gradientsX, gradientsY, crossDerivatives, Vector2(m_dx, m_dx), cubicTable);

			Scalar t, u;
			t = position.x - densityP1.x;
			u = position.y - densityP1.y;

			Scalar ansy = 0.0f;

			for (i = 3; i >= 0; i--) {
				ansy = t*ansy + ((cubicTable[i * 4 + 3] * u + cubicTable[i * 4 + 2])*u + cubicTable[i * 4 + 1])*u + cubicTable[i * 4];
			}

			return ansy;
		}

		template<class valueType>
		Scalar CubicStreamfunctionInterpolant2D<valueType>::getAdjacentCellVelocity(const HalfFace<Vector2> &cutCell, halfEdgeLocation_t cutEdgeLocation, const Vector2 &matchingPoint) {
			Scalar dx = m_pCutCells->getGridSpacing();

			for (int i = 0; i < cutCell.getHalfEdges().size(); i++) {
				auto currEdge = cutCell.getHalfEdges()[i];
				if (currEdge->getLocation() == cutEdgeLocation &&
					(currEdge->getVertices().first->getPosition() == matchingPoint || currEdge->getVertices().second->getPosition() == matchingPoint)) {

					return currEdge->getNormal().dot(currEdge->getEdge()->getVelocity());
				}
			}

			return 0;
		}
		template<class valueType>
		pair<Scalar, Scalar> CubicStreamfunctionInterpolant2D<valueType>::getTangentialDerivatives(int currCellIndex, int edgeIndex) {
			pair<Scalar, Scalar> tangentialDerivatives (0, 0);

			auto cutCell = m_pCutCells->getCutCell(currCellIndex);
			auto currHalfedge = cutCell.getHalfEdges()[edgeIndex];

			tangentialDerivatives.first = currHalfedge->getVertices().first->getVelocity().dot(currHalfedge->getNormal());
			tangentialDerivatives.second = -currHalfedge->getVertices().second->getVelocity().dot(currHalfedge->getNormal());
			return tangentialDerivatives;
		}

		template<class valueType>
		pair<Scalar, Scalar> CubicStreamfunctionInterpolant2D<valueType>::getNormalDerivatives(int currCellIndex, int edgeIndex) {
			pair<Scalar, Scalar> normalDerivatives(0, 0);
			
			auto cutCell = m_pCutCells->getCutCell(currCellIndex);
			auto currHalfedge = cutCell.getHalfEdges()[edgeIndex];

			int prevJ = roundClamp<int>(edgeIndex - 1, 0, cutCell.getHalfEdges().size());
			int nextJ = roundClamp<int>(edgeIndex + 1, 0, cutCell.getHalfEdges().size());
			auto prevEdge = cutCell.getHalfEdges()[prevJ];
			auto nextEdge = cutCell.getHalfEdges()[nextJ];

			normalDerivatives.first = -currHalfedge->getVertices().first->getVelocity().dot(prevEdge->getNormal());
			normalDerivatives.second = currHalfedge->getVertices().second->getVelocity().dot(nextEdge->getNormal());

			if (currHalfedge->getVertices().first->getVertexType() == geometryHalfEdge) {
				normalDerivatives.first = 0;
			}
			if (currHalfedge->getVertices().second->getVertexType() == geometryHalfEdge) {
				normalDerivatives.second = 0;
			}
			return normalDerivatives;
		}

		template<class valueType>
		Scalar CubicStreamfunctionInterpolant2D<valueType>::getFaceVelocity(const dimensions_t &cellLocation, halfEdgeLocation_t currEdgeLocation, HalfFace<Vector2> *pNextCell, const Vector2 &initialPoint) {
			int i = cellLocation.x; int j = cellLocation.y;
			switch (currEdgeLocation) {
			case rightHalfEdge:
				if (pNextCell != nullptr) {
					return getAdjacentCellVelocity(*pNextCell, rightHalfEdge, initialPoint);
				}
				else {
					dimensions_t gridPointDim;
					if (isOnGridPoint(initialPoint, m_dx, gridPointDim)) {
						if (gridPointDim.y == j) {
							if (m_pCutCells->getEdgeVector(dimensions_t(i + 1, j - 1), yAlignedEdge).size() > 0) {
								auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j - 1), yAlignedEdge).front();
								return cutEdge->getVelocity().x;
							}
							else {
								return m_values(i + 1, j - 1).x;
							}
						}
						else {
							if (m_pCutCells->getEdgeVector(dimensions_t(i + 1, j + 1), yAlignedEdge).size() > 0) {
								auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j + 1), yAlignedEdge).front();
								return cutEdge->getVelocity().x;
							}
							else {
								return m_values(i + 1, j + 1).x;
							}
						}
					}
					else {

						//Geometry tangential derivative, should be zero for no slip
						return 0;
					}
				}
				break;

			case bottomHalfEdge:
				if (pNextCell != nullptr) {
					return getAdjacentCellVelocity(*pNextCell, bottomHalfEdge, initialPoint);
				}
				else {
					dimensions_t gridPointDim;
					if (isOnGridPoint(initialPoint, m_dx, gridPointDim)) {
						if (gridPointDim.x == i) {
							if (m_pCutCells->getEdgeVector(dimensions_t(i - 1, j), xAlignedEdge).size() > 0) {
								auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i - 1, j), xAlignedEdge).front();
								return -cutEdge->getVelocity().y;
							}
							else {
								return -m_values(i - 1, j).y;
							}
						}
						else {
							if (m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), xAlignedEdge).size() > 0) {
								auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), xAlignedEdge).front();
								return -cutEdge->getVelocity().y;
							}
							else {
								return -m_values(i + 1, j).y;
							}
						}
					}
					else {
						//Geometry tangential derivative, should be zero for no slip
						return 0;
					}
				}
				break;
			case leftHalfEdge:
				if (pNextCell != nullptr) {
					return getAdjacentCellVelocity(*pNextCell, leftHalfEdge, initialPoint);
				}
				else {
					dimensions_t gridPointDim;
					if (isOnGridPoint(initialPoint, m_dx, gridPointDim)) {
						if (gridPointDim.y == j + 1) {
							if (m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), yAlignedEdge).size() > 0) {
								auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), yAlignedEdge).front();
								return -cutEdge->getVelocity().x;
							}
							else {
								return -m_values(i, j + 1).x;
							}
						}
						else {
							if (m_pCutCells->getEdgeVector(dimensions_t(i, j - 1), yAlignedEdge).size() > 0) {
								auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i, j - 1), yAlignedEdge).front();
								return -cutEdge->getVelocity().x;
							}
							else {
								return -m_values(i, j - 1).x;
							}
						}

					}
					else {
						//Geometry tangential derivative, should be zero for no slip
						return 0;
					}

				}
				break;

			case topHalfEdge:
				if (pNextCell != nullptr) {
					return getAdjacentCellVelocity(*pNextCell, topHalfEdge, initialPoint);
				}
				else {
					dimensions_t gridPointDim;
					if (isOnGridPoint(initialPoint, m_dx, gridPointDim)) {
						if (gridPointDim.x == i + 1) {
							if (m_pCutCells->getEdgeVector(dimensions_t(i + 1, j + 1), xAlignedEdge).size() > 0) {
								auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j + 1), xAlignedEdge).front();
								return cutEdge->getVelocity().y;
							}
							else {
								return m_values(i + 1, j + 1).y;
							}
						}
						else {
							if (m_pCutCells->getEdgeVector(dimensions_t(i - 1, j + 1), xAlignedEdge).size() > 0) {
								auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i - 1, j + 1), xAlignedEdge).front();
								return cutEdge->getVelocity().y;
							}
							else {
								return m_values(i - 1, j + 1).y;
							}
						}
					}
					else {
						//Geometry tangential derivative, should be zero for no slip
						return 0;
					}
				}
			case geometryHalfEdge:
				if (pNextCell != nullptr) {
					return getAdjacentCellVelocity(*pNextCell, geometryHalfEdge, initialPoint);
				}
				/*else {
					throw("Invalid Tangential derivative location information");
				}*/
				break;
			default:
				return 0;
				break;
			}
		}


		template<class valueType>
		Vector2 CubicStreamfunctionInterpolant2D<valueType>::interpolateCutCell(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			int currCellIndex = m_pCutCells->getCutCellIndex(gridSpacePosition);

			Scalar streamInterp = interpolateScalar(position, currCellIndex);
			
			Scalar dxS = 0.0001;
			Scalar signX = 1.0f, signY = 1.0f;
			Scalar streamInterpX = 0.0f, streamInterpY = 0.0f;

			Vector2 gridSpaceNextX = gridSpacePosition + Vector2(dxS, 0) / m_dx;
			Vector2 gridSpaceNextY = gridSpacePosition + Vector2(0, dxS) / m_dx;

			dimensions_t gridSpaceNextXDim(gridSpaceNextX.x, gridSpaceNextX.y);
			dimensions_t gridSpaceNextYDim(gridSpaceNextY.x, gridSpaceNextY.y);
			if ((!m_pCutCells->isCutCell(gridSpaceNextXDim)) || m_pCutCells->getCutCellIndex(gridSpaceNextX) != currCellIndex) {
				streamInterpX = interpolateScalar(position - Vector2(dxS, 0), currCellIndex);
				//streamInterpX = m_pMeanvalueInterpolant->interpolate(position - Vector2(dxS, 0));
				signX = -1.0f;
			}
			else {
				streamInterpX = interpolateScalar(position + Vector2(dxS, 0), currCellIndex);
				//streamInterpX = m_pMeanvalueInterpolant->interpolate(position + Vector2(dxS, 0));
			}
			if ((!m_pCutCells->isCutCell(gridSpaceNextYDim)) || m_pCutCells->getCutCellIndex(gridSpaceNextY) != currCellIndex) {
				streamInterpY = interpolateScalar(position - Vector2(0, dxS), currCellIndex);
				//streamInterpY = m_pMeanvalueInterpolant->interpolate(position - Vector2(0, dxS));
				signY = -1.0f;
			}
			else {
				streamInterpY = interpolateScalar(position + Vector2(0, dxS), currCellIndex);
				//streamInterpY = m_pMeanvalueInterpolant->interpolate(position + Vector2(0, dxS));
			}
			return Vector2(signY*(streamInterpY - streamInterp)/ dxS, -signX*(streamInterpX - streamInterp) / dxS);


		}

		/** Partial derivative in respect to X of the bilinear interpolation function. */
		template<class valueType>
		Scalar CubicStreamfunctionInterpolant2D<valueType>::cubicPartialDerivativeX(const Vector2 &position, const vector<Scalar> &streamfunctionValues) {
			int i = static_cast<int> (floor(position.x));
			int j = static_cast<int> (floor(position.y));

			Vector2 densityP1, densityP2;

			densityP1 = Vector2(i, j);
			densityP2 = Vector2(i + 1.f, j + 1.f);

			Scalar cubicTable[16];
			Scalar origFunction[4];
			origFunction[0] = streamfunctionValues[0];
			origFunction[1] = streamfunctionValues[1];
			origFunction[2] = streamfunctionValues[2];
			origFunction[3] = streamfunctionValues[3];

			Scalar gradientsX[4];
			Scalar gradientsY[4];
			Scalar crossDerivatives[4];

			if (m_pStreamDiscontinuousValues)
				calculateDiscontinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);
			else
				calculateContinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);

			generateCubicTable(origFunction, gradientsX, gradientsY, crossDerivatives, Vector2(m_dx, m_dx), cubicTable);

			Scalar t, u;
			t = position.x - densityP1.x;
			u = position.y - densityP1.y;

			Scalar ansy = 0.0f;

			for (i = 3; i >= 0; i--) {
				ansy = u*ansy + (3 * cubicTable[3 * 4 + i] * t + 2 * cubicTable[2 * 4 + i])*t + cubicTable[1 * 4 + i];
			}

			return ansy;

		}

		/** Partial derivative in respect to Y of the bilinear interpolation function. */
		template<class valueType>
		Scalar CubicStreamfunctionInterpolant2D<valueType>::cubicPartialDerivativeY(const Vector2 &position, const vector<Scalar> &streamfunctionValues) {
			int i = static_cast<int> (floor(position.x));
			int j = static_cast<int> (floor(position.y));

			Vector2 densityP1, densityP2;

			densityP1 = Vector2(i, j);
			densityP2 = Vector2(i + 1.f, j + 1.f);

			Scalar cubicTable[16];
			Scalar origFunction[4];
			origFunction[0] = streamfunctionValues[0];
			origFunction[1] = streamfunctionValues[1];
			origFunction[2] = streamfunctionValues[2];
			origFunction[3] = streamfunctionValues[3];

			Scalar gradientsX[4];
			Scalar gradientsY[4];
			Scalar crossDerivatives[4];

			if (m_pStreamDiscontinuousValues)
				calculateDiscontinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);
			else
				calculateContinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);

			generateCubicTable(origFunction, gradientsX, gradientsY, crossDerivatives, Vector2(m_dx, m_dx), cubicTable);

			Scalar t, u;
			t = position.x - densityP1.x;
			u = position.y - densityP1.y;

			Scalar ansy = 0.0f;

			for (i = 3; i >= 0; i--) {
				ansy = t*ansy + (3 * cubicTable[i * 4 + 3] * u + 2 * cubicTable[i * 4 + 2])*u + cubicTable[i * 4 + 1];
			}

			return ansy;
		}

		template<class valueType>
		void CubicStreamfunctionInterpolant2D<valueType>::calculateDiscontinuousDerivatives(int i, int j, Scalar *gradientsX, Scalar *gradientsY, Scalar *gradientsXY) {
			//F(0, 0)
			Scalar forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[1] - (*m_pStreamDiscontinuousValues)(i, j)[0];
			Scalar backDiff = (*m_pStreamDiscontinuousValues)(i - 1, j)[1] - (*m_pStreamDiscontinuousValues)(i - 1, j)[0];
			forwardDiff /= m_dx;
			backDiff /= m_dx;
			if (m_pCutCells && m_pCutCells->isCutCellAt(i - 1, j)) {
				auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j), yAlignedEdge).back()->getConnectedHalfFaces().front();
				backDiff = getAdjacentCellVelocity(*cutCellNeighbor, bottomHalfEdge, Vector2(i, j) * m_dx);
			}
			gradientsX[0] = (forwardDiff + backDiff)*0.5;

			//F(1, 0)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i + 1, j)[1] - (*m_pStreamDiscontinuousValues)(i + 1, j)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j)[1] - (*m_pStreamDiscontinuousValues)(i, j)[0];
			forwardDiff /= m_dx;
			backDiff /= m_dx;
			if (m_pCutCells && m_pCutCells->isCutCellAt(i + 1, j)) {
				auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), yAlignedEdge).back()->getConnectedHalfFaces().front();
				forwardDiff = getAdjacentCellVelocity(*cutCellNeighbor, bottomHalfEdge, Vector2(i + 1, j) * m_dx);
			}
			gradientsX[1] = (forwardDiff + backDiff)*0.5;

			//F(1, 1)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i + 1, j)[2] - (*m_pStreamDiscontinuousValues)(i + 1, j)[3];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[3];
			forwardDiff /= m_dx;
			backDiff /= m_dx;
			if (m_pCutCells && m_pCutCells->isCutCellAt(i + 1, j)) {
				auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), yAlignedEdge).back()->getConnectedHalfFaces().front();
				forwardDiff = -getAdjacentCellVelocity(*cutCellNeighbor, topHalfEdge, Vector2(i + 1, j + 1) * m_dx);
			}
			gradientsX[2] = (forwardDiff + backDiff)*0.5;

			//F(0, 1)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[3];
			backDiff = (*m_pStreamDiscontinuousValues)(i - 1, j)[2] - (*m_pStreamDiscontinuousValues)(i - 1, j)[3];
			forwardDiff /= m_dx;
			backDiff /= m_dx;
			if (m_pCutCells && m_pCutCells->isCutCellAt(i - 1, j)) {
				auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j), yAlignedEdge).back()->getConnectedHalfFaces().front();
				backDiff = -getAdjacentCellVelocity(*cutCellNeighbor, topHalfEdge, Vector2(i, j + 1) * m_dx);
			}
			gradientsX[3] = (forwardDiff + backDiff)*0.5;

			//Central differencing - Y gradients
			//F(0, 0)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j - 1)[3] - (*m_pStreamDiscontinuousValues)(i, j - 1)[0];
			forwardDiff /= m_dx;
			backDiff /= m_dx;
			if (m_pCutCells && m_pCutCells->isCutCellAt(i, j - 1)) {
				auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge).back()->getConnectedHalfFaces().front();
				backDiff = -getAdjacentCellVelocity(*cutCellNeighbor, leftHalfEdge, Vector2(i, j) * m_dx);
			}
			gradientsY[0] = (forwardDiff + backDiff)*0.5;

			//F(1, 0)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[1];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j - 1)[2] - (*m_pStreamDiscontinuousValues)(i, j - 1)[1];
			forwardDiff /= m_dx;
			backDiff /= m_dx;
			if (m_pCutCells && m_pCutCells->isCutCellAt(i, j - 1)) {
				auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge).back()->getConnectedHalfFaces().front();
				backDiff = getAdjacentCellVelocity(*cutCellNeighbor, rightHalfEdge, Vector2(i + 1, j) * m_dx);
			}
			gradientsY[1] = (forwardDiff + backDiff)*0.5;

			//F(1, 1)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i, j + 1)[2] - (*m_pStreamDiscontinuousValues)(i, j + 1)[1];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[1];
			forwardDiff /= m_dx;
			backDiff /= m_dx;
			if (m_pCutCells && m_pCutCells->isCutCellAt(i, j + 1)) {
				auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), xAlignedEdge).back()->getConnectedHalfFaces().front();
				forwardDiff = getAdjacentCellVelocity(*cutCellNeighbor, rightHalfEdge, Vector2(i + 1, j + 1) * m_dx);
			}
			gradientsY[2] = (forwardDiff + backDiff)*0.5;

			//F(0, 1)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i, j + 1)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
			forwardDiff /= m_dx;
			backDiff /= m_dx;
			if (m_pCutCells && m_pCutCells->isCutCellAt(i, j + 1)) {
				auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), xAlignedEdge).back()->getConnectedHalfFaces().front();
				forwardDiff = -getAdjacentCellVelocity(*cutCellNeighbor, leftHalfEdge , Vector2(i, j + 1) * m_dx);
			}
			gradientsY[3] = (forwardDiff + backDiff)*0.5;

			//Central differencing - Cross Derivatives
			Scalar sqrOfTwo = sqrt(2.0);
			//F(0, 0)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i - 1, j - 1)[2] - (*m_pStreamDiscontinuousValues)(i - 1, j - 1)[0];
			gradientsXY[0] = (forwardDiff - backDiff);
			forwardDiff = (*m_pStreamDiscontinuousValues)(i - 1, j)[3] - (*m_pStreamDiscontinuousValues)(i - 1, j)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j - 1)[3] - (*m_pStreamDiscontinuousValues)(i, j - 1)[0];
			gradientsXY[0] += (forwardDiff - backDiff);
			gradientsXY[0] *= 0.5 / sqrOfTwo*m_dx;

			//F(1, 0)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i + 1, j)[2] - (*m_pStreamDiscontinuousValues)(i + 1, j)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j - 1)[2] - (*m_pStreamDiscontinuousValues)(i, j - 1)[0];
			gradientsXY[1] = (forwardDiff - backDiff);
			forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i + 1, j - 1)[3] - (*m_pStreamDiscontinuousValues)(i + 1, j - 1)[0];
			gradientsXY[1] += (forwardDiff - backDiff);
			gradientsXY[1] *= 0.5 / sqrOfTwo*m_dx;

			//F(1, 1)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i + 1, j + 1)[2] - (*m_pStreamDiscontinuousValues)(i + 1, j + 1)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[0];
			gradientsXY[2] = (forwardDiff - backDiff);
			forwardDiff = (*m_pStreamDiscontinuousValues)(i, j + 1)[3] - (*m_pStreamDiscontinuousValues)(i, j + 1)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i + 1, j)[3] - (*m_pStreamDiscontinuousValues)(i + 1, j)[0];
			gradientsXY[2] += (forwardDiff - backDiff);
			gradientsXY[2] *= 0.5 / sqrOfTwo*m_dx;

			//F(0, 1)
			forwardDiff = (*m_pStreamDiscontinuousValues)(i, j + 1)[2] - (*m_pStreamDiscontinuousValues)(i, j + 1)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i - 1, j)[2] - (*m_pStreamDiscontinuousValues)(i - 1, j)[0];
			gradientsXY[3] = (forwardDiff - backDiff);
			forwardDiff = (*m_pStreamDiscontinuousValues)(i - 1, j + 1)[3] - (*m_pStreamDiscontinuousValues)(i - 1, j + 1)[0];
			backDiff = (*m_pStreamDiscontinuousValues)(i, j)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
			gradientsXY[3] += (forwardDiff - backDiff);
			gradientsXY[3] *= 0.5 / sqrOfTwo*m_dx;


			gradientsXY[0] = gradientsXY[1] = gradientsXY[2] = gradientsXY[3] = 0;
		}

		template<class valueType>
		void CubicStreamfunctionInterpolant2D<valueType>::calculateContinuousDerivatives(int i, int j, Scalar *gradientsX, Scalar *gradientsY, Scalar *gradientsXY) {
			//Central differencing - X gradients
			gradientsX[0] = (*m_pStreamContinuousValues)(i + 1, j) - 2*(*m_pStreamContinuousValues)(i, j) + (*m_pStreamContinuousValues)(i - 1, j);
			gradientsX[1] = (*m_pStreamContinuousValues)(i + 2, j) - 2 * (*m_pStreamContinuousValues)(i + 1, j) + (*m_pStreamContinuousValues)(i, j);
			gradientsX[2] = (*m_pStreamContinuousValues)(i + 2, j + 1) - 2 * (*m_pStreamContinuousValues)(i + 1, j + 1) + (*m_pStreamContinuousValues)(i, j + 1);
			gradientsX[3] = (*m_pStreamContinuousValues)(i + 1, j + 1) - 2 * (*m_pStreamContinuousValues)(i, j + 1) + (*m_pStreamContinuousValues)(i - 1, j + 1);

			//Central differencing - Y gradients
			//F(0, 0)
			gradientsY[0] = (*m_pStreamContinuousValues)(i, j + 1) - 2 * (*m_pStreamContinuousValues)(i, j) + (*m_pStreamContinuousValues)(i, j - 1);
			gradientsY[1] = (*m_pStreamContinuousValues)(i + 1, j + 1) - 2 * (*m_pStreamContinuousValues)(i + 1, j) + (*m_pStreamContinuousValues)(i + 1, j - 1);
			gradientsY[2] = (*m_pStreamContinuousValues)(i + 1, j + 2) - 2 * (*m_pStreamContinuousValues)(i + 1, j + 1) + (*m_pStreamContinuousValues)(i + 1, j);
			gradientsY[3] = (*m_pStreamContinuousValues)(i, j + 2) - 2 * (*m_pStreamContinuousValues)(i, j + 1) + (*m_pStreamContinuousValues)(i, j);

			gradientsXY[0] = gradientsXY[1] = gradientsXY[2] = gradientsXY[3] = 0;
		}

		template<class valueType>
		void CubicStreamfunctionInterpolant2D<valueType>::generateCubicTable(const Scalar origFunction[4], const Scalar gradientsX[4],
																				const Scalar gradientsY[4], const Scalar crossDerivatives[4],
																				Vector2 scaleFactor, Scalar *cubicTable) {
			int l, k, j, i;

			Scalar xx, volume = scaleFactor.x*scaleFactor.y;

			Scalar cl[16], x[16];

			for (i = 0; i < 4; i++) { //Pack into the x matrix the 4 
				x[i] = origFunction[i];
				x[i + 4] = gradientsX[i] * scaleFactor.x;
				x[i + 8] = gradientsY[i] * scaleFactor.y;
				x[i + 12] = crossDerivatives[i] * volume;
			}

			for (i = 0; i < 16; i++) { //Matrix-multiply by the stored table.
				xx = 0.0;
				for (k = 0; k < 16; k++) {
					xx += coefficientsMatrix[i * 16 + k] * x[k];
				}
				cl[i] = xx;
			}

			l = 0;
			for (i = 0; i < 4; i++) { // Unpack the result into the output table.
				for (j = 0; j < 4; j++) {
					cubicTable[i * 4 + j] = cl[l++];
				}
			}

		}
		#pragma endregion
		/** Template linker trickerino for templated classes in CPP*/
		template class CubicStreamfunctionInterpolant2D<Vector2>;
	}
}