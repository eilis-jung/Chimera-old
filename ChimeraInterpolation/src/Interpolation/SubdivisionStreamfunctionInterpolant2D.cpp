#include "Interpolation/SubdivisionStreamfunctionInterpolant2D.h"

namespace Chimera {
	namespace Interpolation {

		#pragma region Constructors
		template<>
		SubdivisionStreamfunctionInterpolant2D<Vector2>::SubdivisionStreamfunctionInterpolant2D(const Array2D<Vector2> &values, Scalar gridDx, Scalar subdivisionDx)
			: BilinearStreamfunctionInterpolant2D(values, gridDx), m_regularEdges(values.getDimensions()) {
			m_subdivisionDx = subdivisionDx;
			initializeSubdividedEdges();
		}
		#pragma endregion

		#pragma region Functionalities
		template<>
		Vector2 SubdivisionStreamfunctionInterpolant2D<Vector2>::interpolate(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));

			if (m_pCutCells && i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
				if (m_pCutCells->isCutCellAt(i, j)) { //Actually implementing MVC
					return interpolateCutCell(position);
				}
			}

			Scalar streamInterp = interpolateScalar(position, m_regularEdges(i, j));

			Scalar dxS = 0.0001;
			Scalar signX = 1.0f, signY = 1.0f;
			Scalar streamInterpX, streamInterpY;

			Vector2 gridSpaceNextX = gridSpacePosition + Vector2(dxS, 0) / m_dx;
			Vector2 gridSpaceNextY = gridSpacePosition + Vector2(0, dxS) / m_dx;

			if (floor(gridSpaceNextX.x) != i) {
				streamInterpX = interpolateScalar(position - Vector2(dxS, 0), m_regularEdges(i, j));
				signX = -1.0f;
			} else {
				streamInterpX = interpolateScalar(position + Vector2(dxS, 0), m_regularEdges(i, j));
			}

			if (floor(gridSpaceNextY.y) != j) {
				streamInterpY = interpolateScalar(position - Vector2(0, dxS), m_regularEdges(i, j));
				signY = -1.0f;
			} else {
				streamInterpY = interpolateScalar(position + Vector2(0, dxS), m_regularEdges(i, j));
			}

			Vector2 tempVelocity = Vector2((streamInterpY - streamInterp) / dxS, -(streamInterpX - streamInterp) / dxS);

			return Vector2(signY*(streamInterpY - streamInterp) / dxS, -signX*(streamInterpX - streamInterp) / dxS);
		}

		template<>
		void SubdivisionStreamfunctionInterpolant2D<Vector2>::computeStreamfunctions() {
			BilinearStreamfunctionInterpolant2D::computeStreamfunctions();
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					if (i == 0 || j == 0 || i == m_gridDimensions.x - 1|| j == m_gridDimensions.y - 1) {
						computeCellStreamfunctionsLinear(i, j);
					}
					else {
						computeCellStreamfunctionsCubic(i, j);
						//computeCellStreamfunctionsLinear(i, j); // 
					}
				}
			}
		}

		template<>
		void SubdivisionStreamfunctionInterpolant2D<Vector2>::saveCellInfoToFile(int i, int j, int numSubdivisions, const string &filename) {
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

					(*fileStream) << scalarToStr(interpolateScalar(interpolationPoint, m_regularEdges(i, j))) << " ";
				}
				(*fileStream) << endl;
			}
		}

		//template<>
		//Vector2 SubdivisionStreamfunctionInterpolant2D<Vector2>::interpolate(const Vector2 &position) {
		//	Vector2 gridSpacePosition = position / m_dx;
		//	int i = static_cast <int> (floor(gridSpacePosition.x));
		//	int j = static_cast <int> (floor(gridSpacePosition.y));

		//	if (m_pCutCells && i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
		//		if (m_pCutCells->isCutCell(i, j)) { //Actually implementing MVC
		//			return interpolateCutCell(position);
		//		}
		//	}

		//	Scalar partialX, partialY;
		//	if (i <= 1 || j <= 1 || i >= m_gridDimensions.x - 2 || j >= m_gridDimensions.y - 2) {
		//		return BilinearStreamfunctionInterpolant2D<Vector2>::interpolate(position);
		//	}
		//	else {
		//		if (m_pStreamDiscontinuousValues) {
		//			Scalar incrementDx = 0.0001;
		//			Scalar interpScalar = interpolateScalar(gridSpacePosition, static_cast<const Scalar *>((*m_pStreamDiscontinuousValues)(i, j)));
		//			Scalar interpX = interpolateScalar(gridSpacePosition + Vector2(1, 0)*incrementDx, static_cast<const Scalar *>((*m_pStreamDiscontinuousValues)(i, j)));
		//			Scalar interpY = interpolateScalar(gridSpacePosition + Vector2(0, 1)*incrementDx, static_cast<const Scalar *>((*m_pStreamDiscontinuousValues)(i, j)));

		//			partialX = cubicPartialDerivativeX(gridSpacePosition, static_cast<const Scalar *>((*m_pStreamDiscontinuousValues)(i, j)));
		//			partialY = cubicPartialDerivativeY(gridSpacePosition, static_cast<const Scalar *>((*m_pStreamDiscontinuousValues)(i, j)));
		//		}
		//		else {
		//			Scalar streamfunctionValues[4];
		//			streamfunctionValues[0] = (*m_pStreamContinuousValues)(i, j);
		//			streamfunctionValues[1] = (*m_pStreamContinuousValues)(i + 1, j);
		//			streamfunctionValues[2] = (*m_pStreamContinuousValues)(i + 1, j + 1);
		//			streamfunctionValues[3] = (*m_pStreamContinuousValues)(i, j + 1);
		//			partialX = cubicPartialDerivativeX(gridSpacePosition, streamfunctionValues);
		//			partialY = cubicPartialDerivativeY(gridSpacePosition, streamfunctionValues);
		//		}
		//	}
		//	partialX /= m_dx;
		//	partialY /= m_dx;

		//	return Vector2(partialY, -partialX);
		//}

		#pragma endregion
		
		#pragma region PrivateFunctionalities
		template<class valueType>
		Scalar SubdivisionStreamfunctionInterpolant2D<valueType>::interpolateScalar(const Vector2 & position, const vector<subdividedEdge_t> edges) {
			Vector2 rCurr, rPrev, rNext;
			Scalar rCurrLength;

			int totalNumberOfPoints = 0;
			for (int i = 0; i < edges.size(); i++) {
				totalNumberOfPoints += edges[i].points.size() - 1;
			}

			vector<Scalar> weights(totalNumberOfPoints, 0);
			int weightIndex = 0;
			for (int i = 0; i < edges.size(); i++) {
				int initialIndex = 0;
				for (int j = 1; j < edges[i].points.size(); j++) {
					rCurr = edges[i].points[j] - position;
					rCurrLength = rCurr.length();
					if (rCurrLength <= g_pointProximityLenght)
						return edges[i].streamfunctionValues[j];

					rPrev = edges[i].points[j - 1] - position;
					if (j == edges[i].points.size() - 1) {
						int nextI = roundClamp<int>(i + 1, 0, edges.size());
						rNext = edges[nextI].points[1] - position;
					}
					else {
						rNext = edges[i].points[j + 1] - position;
					}

					Scalar aVal = calculateADet(rPrev, rCurr) / 2;
					if (aVal != 0)
						weights[weightIndex] += (rPrev.length() - rPrev.dot(rCurr) / rCurrLength) / aVal;
					aVal = calculateADet(rCurr, rNext) / 2;
					if (aVal != 0)
						weights[weightIndex] += (rNext.length() - rNext.dot(rCurr) / rCurrLength) / aVal;

					weightIndex++;
				}
			}
			
			Scalar totalWeight = 0;
			for (int i = 0; i < weights.size(); i++) {
				totalWeight += weights[i];
			}
			for (int i = 0; i < weights.size(); i++) {
				weights[i] /= totalWeight;
			}

			Scalar result = 0;
			weightIndex = 0;
			for (int i = 0; i < edges.size(); i++) {
				for (int j = 1; j < edges[i].points.size(); j++) {
					result += edges[i].streamfunctionValues[j]*weights[weightIndex++];
				}
			}
			return result;
		}
		template<class valueType>
		void SubdivisionStreamfunctionInterpolant2D<valueType>::initializeSubdividedEdges() {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					Vector2 gridCorners[4];
					gridCorners[0] = Vector2(i, j)*m_dx;
					gridCorners[1] = Vector2(i + 1, j)*m_dx;
					gridCorners[2] = Vector2(i + 1, j + 1)*m_dx;
					gridCorners[3] = Vector2(i, j + 1)*m_dx;

					m_regularEdges(i, j).push_back(subdividedEdge_t(gridCorners[0], gridCorners[1], m_subdivisionDx));
					m_regularEdges(i, j).push_back(subdividedEdge_t(gridCorners[1], gridCorners[2], m_subdivisionDx));
					m_regularEdges(i, j).push_back(subdividedEdge_t(gridCorners[2], gridCorners[3], m_subdivisionDx));
					m_regularEdges(i, j).push_back(subdividedEdge_t(gridCorners[3], gridCorners[0], m_subdivisionDx));
				}
			}
		}

		template <class valueType>
		void SubdivisionStreamfunctionInterpolant2D<valueType>::computeCellStreamfunctionsLinear(int i, int j) {
			
			//Bottom edge
			{
				subdividedEdge_t *pSubEdge = &m_regularEdges(i, j)[0];
				pSubEdge->streamfunctionValues.front() = 0;
				pSubEdge->streamfunctionValues.back() = (*m_pStreamDiscontinuousValues)(i, j)[1];
				Scalar edgeLength = (pSubEdge->points.back() - pSubEdge->points.front()).length();
				for (int k = 1; k < pSubEdge->points.size() - 1; k++) {
					Scalar alpha = abs(pSubEdge->points[k].x - pSubEdge->points[0].x) / edgeLength;
					pSubEdge->streamfunctionValues[k] = (1 - alpha)*pSubEdge->streamfunctionValues.front() + alpha*pSubEdge->streamfunctionValues.back();
				}
			}

			//Right edge
			{
				subdividedEdge_t *pSubEdge = &m_regularEdges(i, j)[1];
				pSubEdge->streamfunctionValues.front() = (*m_pStreamDiscontinuousValues)(i, j)[1];
				pSubEdge->streamfunctionValues.back() = (*m_pStreamDiscontinuousValues)(i, j)[2];
				Scalar edgeLength = (pSubEdge->points.back() - pSubEdge->points.front()).length();
				for (int k = 1; k < pSubEdge->points.size() - 1; k++) {
					Scalar alpha = abs(pSubEdge->points[k].y - pSubEdge->points[0].y) / edgeLength;
					pSubEdge->streamfunctionValues[k] = (1 - alpha)*pSubEdge->streamfunctionValues.front() + alpha*pSubEdge->streamfunctionValues.back();
				}
			}

			//Top edge
			{
				subdividedEdge_t *pSubEdge = &m_regularEdges(i, j)[2];
				pSubEdge->streamfunctionValues.front() = (*m_pStreamDiscontinuousValues)(i, j)[2];;
				pSubEdge->streamfunctionValues.back() = (*m_pStreamDiscontinuousValues)(i, j)[3];
				Scalar edgeLength = (pSubEdge->points.back() - pSubEdge->points.front()).length();
				for (int k = 1; k < pSubEdge->points.size() - 1; k++) {
					Scalar alpha = abs(pSubEdge->points[k].x - pSubEdge->points[0].x) / edgeLength;
					pSubEdge->streamfunctionValues[k] = (1 - alpha)*pSubEdge->streamfunctionValues.front() + alpha*pSubEdge->streamfunctionValues.back();
				}
			}

			//Left edge
			{
				subdividedEdge_t *pSubEdge = &m_regularEdges(i, j)[3];
				pSubEdge->streamfunctionValues.front() = (*m_pStreamDiscontinuousValues)(i, j)[3];
				pSubEdge->streamfunctionValues.back() = (*m_pStreamDiscontinuousValues)(i, j)[0];
				Scalar edgeLength = (pSubEdge->points.back() - pSubEdge->points.front()).length();
				for (int k = 1; k < pSubEdge->points.size() - 1; k++) {
					Scalar alpha = abs(pSubEdge->points[k].y - pSubEdge->points[0].y) / edgeLength;
					pSubEdge->streamfunctionValues[k] = (1 - alpha)*pSubEdge->streamfunctionValues.front() + alpha*pSubEdge->streamfunctionValues.back();
				}
			}
		}


		template <class valueType>
		void SubdivisionStreamfunctionInterpolant2D<valueType>::computeCellStreamfunctionsCubic(int i, int j) {

			//Bottom edge
			{
				subdividedEdge_t *pSubEdge = &m_regularEdges(i, j)[0];
				pSubEdge->streamfunctionValues.front() = 0;
				pSubEdge->streamfunctionValues.back() = (*m_pStreamDiscontinuousValues)(i, j)[1];
				Scalar initialDerivative = -(m_values(i - 1, j).y + m_values(i, j).y)*0.5*m_dx;
				Scalar finalDerivative = -(m_values(i + 1, j).y + m_values(i, j).y)*0.5*m_dx;
				Scalar edgeLength = (pSubEdge->points.back() - pSubEdge->points.front()).length();
				for (int k = 1; k < pSubEdge->points.size() - 1; k++) {
					Scalar alpha = abs(pSubEdge->points[k].x - pSubEdge->points[0].x) / edgeLength;
					pSubEdge->streamfunctionValues[k] = cubicInterpolation(alpha, 
																		   pSubEdge->streamfunctionValues.front(), 
																		   pSubEdge->streamfunctionValues.back(),
																		   initialDerivative, finalDerivative);
				}
			}

			//Right edge
			{
				subdividedEdge_t *pSubEdge = &m_regularEdges(i, j)[1];
				pSubEdge->streamfunctionValues.front() = (*m_pStreamDiscontinuousValues)(i, j)[1];
				pSubEdge->streamfunctionValues.back() = (*m_pStreamDiscontinuousValues)(i, j)[2];
				Scalar initialDerivative = (m_values(i + 1, j - 1).x + m_values(i + 1, j).x)*0.5*m_dx;
				Scalar finalDerivative = (m_values(i + 1, j + 1).x + m_values(i + 1, j).x)*0.5*m_dx;
				Scalar edgeLength = (pSubEdge->points.back() - pSubEdge->points.front()).length();
				for (int k = 1; k < pSubEdge->points.size() - 1; k++) {
					Scalar alpha = abs(pSubEdge->points[k].y - pSubEdge->points[0].y) / edgeLength;
					pSubEdge->streamfunctionValues[k] = cubicInterpolation(alpha, 
																		   pSubEdge->streamfunctionValues.front(), 
																		   pSubEdge->streamfunctionValues.back(),
																		   initialDerivative, finalDerivative);
				}
			}

			//Top edge
			{
				subdividedEdge_t *pSubEdge = &m_regularEdges(i, j)[2];
				pSubEdge->streamfunctionValues.front() = (*m_pStreamDiscontinuousValues)(i, j)[2];
				pSubEdge->streamfunctionValues.back() = (*m_pStreamDiscontinuousValues)(i, j)[3];
				Scalar initialDerivative = (m_values(i + 1, j + 1).y + m_values(i, j + 1).y)*0.5*m_dx;
				Scalar finalDerivative = (m_values(i - 1, j + 1).y + m_values(i, j + 1).y)*0.5*m_dx;
				Scalar edgeLength = (pSubEdge->points.back() - pSubEdge->points.front()).length();
				for (int k = 1; k < pSubEdge->points.size() - 1; k++) {
					Scalar alpha = abs(pSubEdge->points[k].x - pSubEdge->points[0].x) / edgeLength;
					pSubEdge->streamfunctionValues[k] = cubicInterpolation(alpha, 
																		   pSubEdge->streamfunctionValues.front(), 
																		   pSubEdge->streamfunctionValues.back(),
																		   initialDerivative, finalDerivative);	
				}
			}

			//Left edge
			{
				subdividedEdge_t *pSubEdge = &m_regularEdges(i, j)[3];
				pSubEdge->streamfunctionValues.front() = (*m_pStreamDiscontinuousValues)(i, j)[3];
				pSubEdge->streamfunctionValues.back() = (*m_pStreamDiscontinuousValues)(i, j)[0];
				Scalar initialDerivative = -(m_values(i, j + 1).x + m_values(i, j).x)*0.5*m_dx;
				Scalar finalDerivative = -(m_values(i, j - 1).x + m_values(i, j).x)*0.5*m_dx;
				Scalar edgeLength = (pSubEdge->points.back() - pSubEdge->points.front()).length();
				for (int k = 1; k < pSubEdge->points.size() - 1; k++) {
					Scalar alpha = abs(pSubEdge->points[k].y - pSubEdge->points[0].y) / edgeLength;
					pSubEdge->streamfunctionValues[k] = cubicInterpolation(alpha, 
																		   pSubEdge->streamfunctionValues.front(), 
																		   pSubEdge->streamfunctionValues.back(),
																		   initialDerivative, finalDerivative);
				}
			}
		}

		
		template<class valueType>
		Scalar SubdivisionStreamfunctionInterpolant2D<valueType>::cubicInterpolation(Scalar x, Scalar valueIni, Scalar valueFinal, Scalar derivIni, Scalar derivFinal) {
			Scalar iValue = 0;
			Scalar d = valueIni;
			Scalar c = derivIni;

			Scalar a = 2 * valueIni + derivIni - 2 * valueFinal + derivFinal;
			Scalar b = valueFinal - a - c - d;

			a = 2 * valueIni - 2 * valueFinal + derivIni + derivFinal;
			b = 3 * valueFinal - 3 * valueIni - 2 * derivIni - derivFinal;

			return a*x*x*x + b*x*x + c*x + d;

			//Derivation:
			//	e: ax3 + bx2 + cx + d
			//	ed : 3ax2 + 2bx + c

			//	d = valueIni;
			//  c = derivIni;

			//  x == 1
			//	  a + b + c + d = valueFinal
			//	  b = -((a + derivIni + valueIni) - valueFinal);

			//  3a + 2b + c = derivFinal
			//	-3a = 2(-((a + derivIni + valueIni) - valueFinal)) + derivIni - derivFinal;
			//	-a = -2 * derivIni - 2 * valueIni + valueFinal + derivIni - derivFinal;
			//	a = derivIni + 2valueIni - valueFinal + derivFinal

		}

		//template<class valueType>
		//Scalar CubicStreamfunctionInterpolant2D<valueType>::interpolateScalar(const Vector2 &position, int currCellIndex) {
		//	auto cutCell = m_pCutCells->getCutCell(currCellIndex);

		//	vector<DoubleScalar> weights(m_cutCellsPoints[currCellIndex].size(), 0.0f);
		//	vector<DoubleScalar> tangentialWeights(m_cutCellsPoints[currCellIndex].size() * 2, 0.0f);
		//	vector<DoubleScalar> normalWeights(m_cutCellsPoints[currCellIndex].size() * 2, 0.0f);

		//	CubicMVC::cubicMVCs(m_cutCellsPoints[currCellIndex], position, weights, normalWeights, tangentialWeights);

		//	Scalar interpolatedValue = 0.0f;
		//	for (int i = 0; i < weights.size(); i++) {
		//		pair<Scalar, Scalar> tangentialGrad = getTangentialDerivatives(currCellIndex, i);
		//		pair<Scalar, Scalar> normalGrad = getNormalDerivatives(currCellIndex, i);
		//		int nextI = roundClamp<int>(i + 1, 0, weights.size());

		//		Scalar tangentContribution = 0;
		//		Scalar normalContribution = 0;

		//		tangentContribution = tangentialWeights[i * 2] * tangentialGrad.first + tangentialWeights[i * 2 + 1] * tangentialGrad.second;
		//		normalContribution = normalWeights[i * 2] * normalGrad.first + normalWeights[i * 2 + 1] * normalGrad.second;
		//		
		//		auto currVertex = cutCell.getHalfEdges()[i]->getVertices().first;
		//		interpolatedValue += cutCell.getStreamfunction(i).x*weights[i] + tangentContribution + normalContribution;
		//	}
		//	return interpolatedValue;
		//}

		//template<class valueType>
		//Scalar CubicStreamfunctionInterpolant2D<valueType>::interpolateScalar(const Vector2 &position, const Scalar *pStreamfunctionValues) {
		//	int i = static_cast<int> (floor(position.x));
		//	int j = static_cast<int> (floor(position.y));

		//	Vector2 densityP1, densityP2;

		//	densityP1 = Vector2(i, j);
		//	densityP2 = Vector2(i + 1.f, j + 1.f);

		//	Scalar cubicTable[16];
		//	Scalar origFunction[4];

		//	origFunction[0] = pStreamfunctionValues[0];
		//	origFunction[1] = pStreamfunctionValues[1];
		//	origFunction[2] = pStreamfunctionValues[2];
		//	origFunction[3] = pStreamfunctionValues[3];

		//	Scalar gradientsX[4];
		//	Scalar gradientsY[4];
		//	Scalar crossDerivatives[4];

		//	if (m_pStreamDiscontinuousValues)
		//		calculateDiscontinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);
		//	else
		//		calculateContinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);

		//	generateCubicTable(origFunction, gradientsX, gradientsY, crossDerivatives, Vector2(m_dx, m_dx), cubicTable);

		//	Scalar t, u;
		//	t = position.x - densityP1.x;
		//	u = position.y - densityP1.y;

		//	Scalar ansy = 0.0f;

		//	for (i = 3; i >= 0; i--) {
		//		ansy = t*ansy + ((cubicTable[i * 4 + 3] * u + cubicTable[i * 4 + 2])*u + cubicTable[i * 4 + 1])*u + cubicTable[i * 4];
		//	}

		//	return ansy;
		//}

		//template<class valueType>
		//Scalar CubicStreamfunctionInterpolant2D<valueType>::getAdjacentCellVelocity(const HalfFace<Vector2> &cutCell, halfEdgeLocation_t cutEdgeLocation, const Vector2 &matchingPoint) {
		//	Scalar dx = m_pCutCells->getGridSpacing();

		//	for (int i = 0; i < cutCell.getHalfEdges().size(); i++) {
		//		auto currEdge = cutCell.getHalfEdges()[i];
		//		if (currEdge->getLocation() == cutEdgeLocation &&
		//			(currEdge->getVertices().first->getPosition() == matchingPoint || currEdge->getVertices().second->getPosition() == matchingPoint)) {

		//			return currEdge->getNormal().dot(currEdge->getEdge()->getVelocity());
		//		}
		//	}

		//	return 0;
		//}
		//template<class valueType>
		//pair<Scalar, Scalar> CubicStreamfunctionInterpolant2D<valueType>::getTangentialDerivatives(int currCellIndex, int edgeIndex) {
		//	pair<Scalar, Scalar> tangentialDerivatives (0, 0);

		//	auto cutCell = m_pCutCells->getCutCell(currCellIndex);
		//	auto currHalfedge = cutCell.getHalfEdges()[edgeIndex];

		//	tangentialDerivatives.first = currHalfedge->getVertices().first->getVelocity().dot(currHalfedge->getNormal());
		//	tangentialDerivatives.second = -currHalfedge->getVertices().second->getVelocity().dot(currHalfedge->getNormal());
		//	return tangentialDerivatives;
		//}

		//template<class valueType>
		//pair<Scalar, Scalar> CubicStreamfunctionInterpolant2D<valueType>::getNormalDerivatives(int currCellIndex, int edgeIndex) {
		//	pair<Scalar, Scalar> normalDerivatives(0, 0);
		//	
		//	auto cutCell = m_pCutCells->getCutCell(currCellIndex);
		//	auto currHalfedge = cutCell.getHalfEdges()[edgeIndex];

		//	int prevJ = roundClamp<int>(edgeIndex - 1, 0, cutCell.getHalfEdges().size());
		//	int nextJ = roundClamp<int>(edgeIndex + 1, 0, cutCell.getHalfEdges().size());
		//	auto prevEdge = cutCell.getHalfEdges()[prevJ];
		//	auto nextEdge = cutCell.getHalfEdges()[nextJ];

		//	normalDerivatives.first = -currHalfedge->getVertices().first->getVelocity().dot(prevEdge->getNormal());
		//	normalDerivatives.second = currHalfedge->getVertices().second->getVelocity().dot(nextEdge->getNormal());

		//	if (currHalfedge->getVertices().first->getVertexType() == geometryHalfEdge) {
		//		normalDerivatives.first = 0;
		//	}
		//	if (currHalfedge->getVertices().second->getVertexType() == geometryHalfEdge) {
		//		normalDerivatives.second = 0;
		//	}
		//	return normalDerivatives;
		//}

		//template<class valueType>
		//Scalar CubicStreamfunctionInterpolant2D<valueType>::getFaceVelocity(const dimensions_t &cellLocation, halfEdgeLocation_t currEdgeLocation, HalfFace<Vector2> *pNextCell, const Vector2 &initialPoint) {
		//	int i = cellLocation.x; int j = cellLocation.y;
		//	switch (currEdgeLocation) {
		//	case rightHalfEdge:
		//		if (pNextCell != nullptr) {
		//			return getAdjacentCellVelocity(*pNextCell, rightHalfEdge, initialPoint);
		//		}
		//		else {
		//			dimensions_t gridPointDim;
		//			if (isOnGridPoint(initialPoint, m_dx, gridPointDim)) {
		//				if (gridPointDim.y == j) {
		//					if (m_pCutCells->getEdgeVector(dimensions_t(i + 1, j - 1), leftRightEdge).size() > 0) {
		//						auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j - 1), leftRightEdge).front();
		//						return cutEdge->getVelocity().x;
		//					}
		//					else {
		//						return m_values(i + 1, j - 1).x;
		//					}
		//				}
		//				else {
		//					if (m_pCutCells->getEdgeVector(dimensions_t(i + 1, j + 1), leftRightEdge).size() > 0) {
		//						auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j + 1), leftRightEdge).front();
		//						return cutEdge->getVelocity().x;
		//					}
		//					else {
		//						return m_values(i + 1, j + 1).x;
		//					}
		//				}
		//			}
		//			else {

		//				//Geometry tangential derivative, should be zero for no slip
		//				return 0;
		//			}
		//		}
		//		break;

		//	case bottomHalfEdge:
		//		if (pNextCell != nullptr) {
		//			return getAdjacentCellVelocity(*pNextCell, bottomHalfEdge, initialPoint);
		//		}
		//		else {
		//			dimensions_t gridPointDim;
		//			if (isOnGridPoint(initialPoint, m_dx, gridPointDim)) {
		//				if (gridPointDim.x == i) {
		//					if (m_pCutCells->getEdgeVector(dimensions_t(i - 1, j), bottomTopEdge).size() > 0) {
		//						auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i - 1, j), bottomTopEdge).front();
		//						return -cutEdge->getVelocity().y;
		//					}
		//					else {
		//						return -m_values(i - 1, j).y;
		//					}
		//				}
		//				else {
		//					if (m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), bottomTopEdge).size() > 0) {
		//						auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), bottomTopEdge).front();
		//						return -cutEdge->getVelocity().y;
		//					}
		//					else {
		//						return -m_values(i + 1, j).y;
		//					}
		//				}
		//			}
		//			else {
		//				//Geometry tangential derivative, should be zero for no slip
		//				return 0;
		//			}
		//		}
		//		break;
		//	case leftHalfEdge:
		//		if (pNextCell != nullptr) {
		//			return getAdjacentCellVelocity(*pNextCell, leftHalfEdge, initialPoint);
		//		}
		//		else {
		//			dimensions_t gridPointDim;
		//			if (isOnGridPoint(initialPoint, m_dx, gridPointDim)) {
		//				if (gridPointDim.y == j + 1) {
		//					if (m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), leftRightEdge).size() > 0) {
		//						auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), leftRightEdge).front();
		//						return -cutEdge->getVelocity().x;
		//					}
		//					else {
		//						return -m_values(i, j + 1).x;
		//					}
		//				}
		//				else {
		//					if (m_pCutCells->getEdgeVector(dimensions_t(i, j - 1), leftRightEdge).size() > 0) {
		//						auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i, j - 1), leftRightEdge).front();
		//						return -cutEdge->getVelocity().x;
		//					}
		//					else {
		//						return -m_values(i, j - 1).x;
		//					}
		//				}

		//			}
		//			else {
		//				//Geometry tangential derivative, should be zero for no slip
		//				return 0;
		//			}

		//		}
		//		break;

		//	case topHalfEdge:
		//		if (pNextCell != nullptr) {
		//			return getAdjacentCellVelocity(*pNextCell, topHalfEdge, initialPoint);
		//		}
		//		else {
		//			dimensions_t gridPointDim;
		//			if (isOnGridPoint(initialPoint, m_dx, gridPointDim)) {
		//				if (gridPointDim.x == i + 1) {
		//					if (m_pCutCells->getEdgeVector(dimensions_t(i + 1, j + 1), bottomTopEdge).size() > 0) {
		//						auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j + 1), bottomTopEdge).front();
		//						return cutEdge->getVelocity().y;
		//					}
		//					else {
		//						return m_values(i + 1, j + 1).y;
		//					}
		//				}
		//				else {
		//					if (m_pCutCells->getEdgeVector(dimensions_t(i - 1, j + 1), bottomTopEdge).size() > 0) {
		//						auto cutEdge = m_pCutCells->getEdgeVector(dimensions_t(i - 1, j + 1), bottomTopEdge).front();
		//						return cutEdge->getVelocity().y;
		//					}
		//					else {
		//						return m_values(i - 1, j + 1).y;
		//					}
		//				}
		//			}
		//			else {
		//				//Geometry tangential derivative, should be zero for no slip
		//				return 0;
		//			}
		//		}
		//	case geometryHalfEdge:
		//		if (pNextCell != nullptr) {
		//			return getAdjacentCellVelocity(*pNextCell, geometryHalfEdge, initialPoint);
		//		}
		//		/*else {
		//			throw("Invalid Tangential derivative location information");
		//		}*/
		//		break;
		//	default:
		//		return 0;
		//		break;
		//	}
		//}


		//template<class valueType>
		//Vector2 CubicStreamfunctionInterpolant2D<valueType>::interpolateCutCell(const Vector2 &position) {
		//	Vector2 gridSpacePosition = position / m_dx;
		//	int i = static_cast <int> (floor(gridSpacePosition.x));
		//	int j = static_cast <int> (floor(gridSpacePosition.y));
		//	i = clamp(i, 0, m_values.getDimensions().x - 1);
		//	j = clamp(j, 0, m_values.getDimensions().y - 1);

		//	int currCellIndex = m_pCutCells->getCutCellIndex(gridSpacePosition);

		//	Scalar streamInterp = interpolateScalar(position, currCellIndex);
		//	
		//	Scalar dxS = 0.0001;
		//	Scalar signX = 1.0f, signY = 1.0f;
		//	Scalar streamInterpX = 0.0f, streamInterpY = 0.0f;

		//	Vector2 gridSpaceNextX = gridSpacePosition + Vector2(dxS, 0) / m_dx;
		//	Vector2 gridSpaceNextY = gridSpacePosition + Vector2(0, dxS) / m_dx;

		//	if ((!m_pCutCells->isCutCell(gridSpaceNextX)) || m_pCutCells->getCutCellIndex(gridSpaceNextX) != currCellIndex) {
		//		streamInterpX = interpolateScalar(position - Vector2(dxS, 0), currCellIndex);
		//		//streamInterpX = m_pMeanvalueInterpolant->interpolate(position - Vector2(dxS, 0));
		//		signX = -1.0f;
		//	}
		//	else {
		//		streamInterpX = interpolateScalar(position + Vector2(dxS, 0), currCellIndex);
		//		//streamInterpX = m_pMeanvalueInterpolant->interpolate(position + Vector2(dxS, 0));
		//	}
		//	if ((!m_pCutCells->isCutCell(gridSpaceNextY)) || m_pCutCells->getCutCellIndex(gridSpaceNextY) != currCellIndex) {
		//		streamInterpY = interpolateScalar(position - Vector2(0, dxS), currCellIndex);
		//		//streamInterpY = m_pMeanvalueInterpolant->interpolate(position - Vector2(0, dxS));
		//		signY = -1.0f;
		//	}
		//	else {
		//		streamInterpY = interpolateScalar(position + Vector2(0, dxS), currCellIndex);
		//		//streamInterpY = m_pMeanvalueInterpolant->interpolate(position + Vector2(0, dxS));
		//	}
		//	return Vector2(signY*(streamInterpY - streamInterp)/ dxS, -signX*(streamInterpX - streamInterp) / dxS);


		//}

		///** Partial derivative in respect to X of the bilinear interpolation function. */
		//template<class valueType>
		//Scalar CubicStreamfunctionInterpolant2D<valueType>::cubicPartialDerivativeX(const Vector2 &position, const Scalar *pStreamfunctionValues) {
		//	int i = static_cast<int> (floor(position.x));
		//	int j = static_cast<int> (floor(position.y));

		//	Vector2 densityP1, densityP2;

		//	densityP1 = Vector2(i, j);
		//	densityP2 = Vector2(i + 1.f, j + 1.f);

		//	Scalar cubicTable[16];
		//	Scalar origFunction[4];
		//	origFunction[0] = pStreamfunctionValues[0];
		//	origFunction[1] = pStreamfunctionValues[1];
		//	origFunction[2] = pStreamfunctionValues[2];
		//	origFunction[3] = pStreamfunctionValues[3];

		//	Scalar gradientsX[4];
		//	Scalar gradientsY[4];
		//	Scalar crossDerivatives[4];

		//	if (m_pStreamDiscontinuousValues)
		//		calculateDiscontinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);
		//	else
		//		calculateContinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);

		//	generateCubicTable(origFunction, gradientsX, gradientsY, crossDerivatives, Vector2(m_dx, m_dx), cubicTable);

		//	Scalar t, u;
		//	t = position.x - densityP1.x;
		//	u = position.y - densityP1.y;

		//	Scalar ansy = 0.0f;

		//	for (i = 3; i >= 0; i--) {
		//		ansy = u*ansy + (3 * cubicTable[3 * 4 + i] * t + 2 * cubicTable[2 * 4 + i])*t + cubicTable[1 * 4 + i];
		//	}

		//	return ansy;

		//}

		///** Partial derivative in respect to Y of the bilinear interpolation function. */
		//template<class valueType>
		//Scalar CubicStreamfunctionInterpolant2D<valueType>::cubicPartialDerivativeY(const Vector2 &position, const Scalar *pStreamfunctionValues) {
		//	int i = static_cast<int> (floor(position.x));
		//	int j = static_cast<int> (floor(position.y));

		//	Vector2 densityP1, densityP2;

		//	densityP1 = Vector2(i, j);
		//	densityP2 = Vector2(i + 1.f, j + 1.f);

		//	Scalar cubicTable[16];
		//	Scalar origFunction[4];
		//	origFunction[0] = pStreamfunctionValues[0];
		//	origFunction[1] = pStreamfunctionValues[1];
		//	origFunction[2] = pStreamfunctionValues[2];
		//	origFunction[3] = pStreamfunctionValues[3];

		//	Scalar gradientsX[4];
		//	Scalar gradientsY[4];
		//	Scalar crossDerivatives[4];

		//	if (m_pStreamDiscontinuousValues)
		//		calculateDiscontinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);
		//	else
		//		calculateContinuousDerivatives(i, j, gradientsX, gradientsY, crossDerivatives);

		//	generateCubicTable(origFunction, gradientsX, gradientsY, crossDerivatives, Vector2(m_dx, m_dx), cubicTable);

		//	Scalar t, u;
		//	t = position.x - densityP1.x;
		//	u = position.y - densityP1.y;

		//	Scalar ansy = 0.0f;

		//	for (i = 3; i >= 0; i--) {
		//		ansy = t*ansy + (3 * cubicTable[i * 4 + 3] * u + 2 * cubicTable[i * 4 + 2])*u + cubicTable[i * 4 + 1];
		//	}

		//	return ansy;
		//}

		//template<class valueType>
		//void CubicStreamfunctionInterpolant2D<valueType>::calculateDiscontinuousDerivatives(int i, int j, Scalar *gradientsX, Scalar *gradientsY, Scalar *gradientsXY) {
		//	//F(0, 0)
		//	Scalar forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[1] - (*m_pStreamDiscontinuousValues)(i, j)[0];
		//	Scalar backDiff = (*m_pStreamDiscontinuousValues)(i - 1, j)[1] - (*m_pStreamDiscontinuousValues)(i - 1, j)[0];
		//	forwardDiff /= m_dx;
		//	backDiff /= m_dx;
		//	if (m_pCutCells && m_pCutCells->isCutCell(i - 1, j)) {
		//		auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j), leftRightEdge).back()->getConnectedHalfFaces().front();
		//		backDiff = getAdjacentCellVelocity(*cutCellNeighbor, bottomHalfEdge, Vector2(i, j) * m_dx);
		//	}
		//	gradientsX[0] = (forwardDiff + backDiff)*0.5;

		//	//F(1, 0)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i + 1, j)[1] - (*m_pStreamDiscontinuousValues)(i + 1, j)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j)[1] - (*m_pStreamDiscontinuousValues)(i, j)[0];
		//	forwardDiff /= m_dx;
		//	backDiff /= m_dx;
		//	if (m_pCutCells && m_pCutCells->isCutCell(i + 1, j)) {
		//		auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), leftRightEdge).back()->getConnectedHalfFaces().front();
		//		forwardDiff = getAdjacentCellVelocity(*cutCellNeighbor, bottomHalfEdge, Vector2(i + 1, j) * m_dx);
		//	}
		//	gradientsX[1] = (forwardDiff + backDiff)*0.5;

		//	//F(1, 1)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i + 1, j)[2] - (*m_pStreamDiscontinuousValues)(i + 1, j)[3];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[3];
		//	forwardDiff /= m_dx;
		//	backDiff /= m_dx;
		//	if (m_pCutCells && m_pCutCells->isCutCell(i + 1, j)) {
		//		auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i + 1, j), leftRightEdge).back()->getConnectedHalfFaces().front();
		//		forwardDiff = -getAdjacentCellVelocity(*cutCellNeighbor, topHalfEdge, Vector2(i + 1, j + 1) * m_dx);
		//	}
		//	gradientsX[2] = (forwardDiff + backDiff)*0.5;

		//	//F(0, 1)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[3];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i - 1, j)[2] - (*m_pStreamDiscontinuousValues)(i - 1, j)[3];
		//	forwardDiff /= m_dx;
		//	backDiff /= m_dx;
		//	if (m_pCutCells && m_pCutCells->isCutCell(i - 1, j)) {
		//		auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j), leftRightEdge).back()->getConnectedHalfFaces().front();
		//		backDiff = -getAdjacentCellVelocity(*cutCellNeighbor, topHalfEdge, Vector2(i, j + 1) * m_dx);
		//	}
		//	gradientsX[3] = (forwardDiff + backDiff)*0.5;

		//	//Central differencing - Y gradients
		//	//F(0, 0)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j - 1)[3] - (*m_pStreamDiscontinuousValues)(i, j - 1)[0];
		//	forwardDiff /= m_dx;
		//	backDiff /= m_dx;
		//	if (m_pCutCells && m_pCutCells->isCutCell(i, j - 1)) {
		//		auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j), bottomTopEdge).back()->getConnectedHalfFaces().front();
		//		backDiff = -getAdjacentCellVelocity(*cutCellNeighbor, leftHalfEdge, Vector2(i, j) * m_dx);
		//	}
		//	gradientsY[0] = (forwardDiff + backDiff)*0.5;

		//	//F(1, 0)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[1];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j - 1)[2] - (*m_pStreamDiscontinuousValues)(i, j - 1)[1];
		//	forwardDiff /= m_dx;
		//	backDiff /= m_dx;
		//	if (m_pCutCells && m_pCutCells->isCutCell(i, j - 1)) {
		//		auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j), bottomTopEdge).back()->getConnectedHalfFaces().front();
		//		backDiff = getAdjacentCellVelocity(*cutCellNeighbor, rightHalfEdge, Vector2(i + 1, j) * m_dx);
		//	}
		//	gradientsY[1] = (forwardDiff + backDiff)*0.5;

		//	//F(1, 1)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i, j + 1)[2] - (*m_pStreamDiscontinuousValues)(i, j + 1)[1];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[1];
		//	forwardDiff /= m_dx;
		//	backDiff /= m_dx;
		//	if (m_pCutCells && m_pCutCells->isCutCell(i, j + 1)) {
		//		auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), bottomTopEdge).back()->getConnectedHalfFaces().front();
		//		forwardDiff = getAdjacentCellVelocity(*cutCellNeighbor, rightHalfEdge, Vector2(i + 1, j + 1) * m_dx);
		//	}
		//	gradientsY[2] = (forwardDiff + backDiff)*0.5;

		//	//F(0, 1)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i, j + 1)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
		//	forwardDiff /= m_dx;
		//	backDiff /= m_dx;
		//	if (m_pCutCells && m_pCutCells->isCutCell(i, j + 1)) {
		//		auto cutCellNeighbor = m_pCutCells->getEdgeVector(dimensions_t(i, j + 1), bottomTopEdge).back()->getConnectedHalfFaces().front();
		//		forwardDiff = -getAdjacentCellVelocity(*cutCellNeighbor, leftHalfEdge , Vector2(i, j + 1) * m_dx);
		//	}
		//	gradientsY[3] = (forwardDiff + backDiff)*0.5;

		//	//Central differencing - Cross Derivatives
		//	Scalar sqrOfTwo = sqrt(2.0);
		//	//F(0, 0)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i - 1, j - 1)[2] - (*m_pStreamDiscontinuousValues)(i - 1, j - 1)[0];
		//	gradientsXY[0] = (forwardDiff - backDiff);
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i - 1, j)[3] - (*m_pStreamDiscontinuousValues)(i - 1, j)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j - 1)[3] - (*m_pStreamDiscontinuousValues)(i, j - 1)[0];
		//	gradientsXY[0] += (forwardDiff - backDiff);
		//	gradientsXY[0] *= 0.5 / sqrOfTwo*m_dx;

		//	//F(1, 0)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i + 1, j)[2] - (*m_pStreamDiscontinuousValues)(i + 1, j)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j - 1)[2] - (*m_pStreamDiscontinuousValues)(i, j - 1)[0];
		//	gradientsXY[1] = (forwardDiff - backDiff);
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i, j)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i + 1, j - 1)[3] - (*m_pStreamDiscontinuousValues)(i + 1, j - 1)[0];
		//	gradientsXY[1] += (forwardDiff - backDiff);
		//	gradientsXY[1] *= 0.5 / sqrOfTwo*m_dx;

		//	//F(1, 1)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i + 1, j + 1)[2] - (*m_pStreamDiscontinuousValues)(i + 1, j + 1)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j)[2] - (*m_pStreamDiscontinuousValues)(i, j)[0];
		//	gradientsXY[2] = (forwardDiff - backDiff);
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i, j + 1)[3] - (*m_pStreamDiscontinuousValues)(i, j + 1)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i + 1, j)[3] - (*m_pStreamDiscontinuousValues)(i + 1, j)[0];
		//	gradientsXY[2] += (forwardDiff - backDiff);
		//	gradientsXY[2] *= 0.5 / sqrOfTwo*m_dx;

		//	//F(0, 1)
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i, j + 1)[2] - (*m_pStreamDiscontinuousValues)(i, j + 1)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i - 1, j)[2] - (*m_pStreamDiscontinuousValues)(i - 1, j)[0];
		//	gradientsXY[3] = (forwardDiff - backDiff);
		//	forwardDiff = (*m_pStreamDiscontinuousValues)(i - 1, j + 1)[3] - (*m_pStreamDiscontinuousValues)(i - 1, j + 1)[0];
		//	backDiff = (*m_pStreamDiscontinuousValues)(i, j)[3] - (*m_pStreamDiscontinuousValues)(i, j)[0];
		//	gradientsXY[3] += (forwardDiff - backDiff);
		//	gradientsXY[3] *= 0.5 / sqrOfTwo*m_dx;


		//	gradientsXY[0] = gradientsXY[1] = gradientsXY[2] = gradientsXY[3] = 0;
		//}

		//template<class valueType>
		//void CubicStreamfunctionInterpolant2D<valueType>::calculateContinuousDerivatives(int i, int j, Scalar *gradientsX, Scalar *gradientsY, Scalar *gradientsXY) {
		//	//Central differencing - X gradients
		//	gradientsX[0] = (*m_pStreamContinuousValues)(i + 1, j) - 2*(*m_pStreamContinuousValues)(i, j) + (*m_pStreamContinuousValues)(i - 1, j);
		//	gradientsX[1] = (*m_pStreamContinuousValues)(i + 2, j) - 2 * (*m_pStreamContinuousValues)(i + 1, j) + (*m_pStreamContinuousValues)(i, j);
		//	gradientsX[2] = (*m_pStreamContinuousValues)(i + 2, j + 1) - 2 * (*m_pStreamContinuousValues)(i + 1, j + 1) + (*m_pStreamContinuousValues)(i, j + 1);
		//	gradientsX[3] = (*m_pStreamContinuousValues)(i + 1, j + 1) - 2 * (*m_pStreamContinuousValues)(i, j + 1) + (*m_pStreamContinuousValues)(i - 1, j + 1);

		//	//Central differencing - Y gradients
		//	//F(0, 0)
		//	gradientsY[0] = (*m_pStreamContinuousValues)(i, j + 1) - 2 * (*m_pStreamContinuousValues)(i, j) + (*m_pStreamContinuousValues)(i, j - 1);
		//	gradientsY[1] = (*m_pStreamContinuousValues)(i + 1, j + 1) - 2 * (*m_pStreamContinuousValues)(i + 1, j) + (*m_pStreamContinuousValues)(i + 1, j - 1);
		//	gradientsY[2] = (*m_pStreamContinuousValues)(i + 1, j + 2) - 2 * (*m_pStreamContinuousValues)(i + 1, j + 1) + (*m_pStreamContinuousValues)(i + 1, j);
		//	gradientsY[3] = (*m_pStreamContinuousValues)(i, j + 2) - 2 * (*m_pStreamContinuousValues)(i, j + 1) + (*m_pStreamContinuousValues)(i, j);

		//	gradientsXY[0] = gradientsXY[1] = gradientsXY[2] = gradientsXY[3] = 0;
		//}

		//template<class valueType>
		//void CubicStreamfunctionInterpolant2D<valueType>::generateCubicTable(const Scalar origFunction[4], const Scalar gradientsX[4],
		//																		const Scalar gradientsY[4], const Scalar crossDerivatives[4],
		//																		Vector2 scaleFactor, Scalar *cubicTable) {
		//	int l, k, j, i;

		//	Scalar xx, volume = scaleFactor.x*scaleFactor.y;

		//	Scalar cl[16], x[16];

		//	for (i = 0; i < 4; i++) { //Pack into the x matrix the 4 
		//		x[i] = origFunction[i];
		//		x[i + 4] = gradientsX[i] * scaleFactor.x;
		//		x[i + 8] = gradientsY[i] * scaleFactor.y;
		//		x[i + 12] = crossDerivatives[i] * volume;
		//	}

		//	for (i = 0; i < 16; i++) { //Matrix-multiply by the stored table.
		//		xx = 0.0;
		//		for (k = 0; k < 16; k++) {
		//			xx += coefficientsMatrix[i * 16 + k] * x[k];
		//		}
		//		cl[i] = xx;
		//	}

		//	l = 0;
		//	for (i = 0; i < 4; i++) { // Unpack the result into the output table.
		//		for (j = 0; j < 4; j++) {
		//			cubicTable[i * 4 + j] = cl[l++];
		//		}
		//	}

		//}
		#pragma endregion

		/** Template linker trickerino for templated classes in CPP*/
		template class SubdivisionStreamfunctionInterpolant2D<Vector2>;
	}
}