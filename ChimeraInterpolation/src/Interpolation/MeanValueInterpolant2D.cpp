#include "Interpolation/MeanValueInterpolant2D.h"
#include "CutCells/CutCells2D.h"
#include "CutCells/CutCellsVelocities2D.h"

namespace Chimera {
	namespace Interpolation {
		
		#pragma region Constructors
		template<class valueType>
		MeanValueInterpolant2D<valueType>::MeanValueInterpolant2D(const Array2D<valueType> &values, CutCellsVelocities2D *pCutCellsVelocities2D, Scalar gridDx, bool useAuxiliaryVelocities) : 
																	Interpolant(values){
			m_useAuxiliaryVelocities = useAuxiliaryVelocities;
			//Initialize nodal interpolant
			m_pNodalInterpolant = new BilinearNodalInterpolant2D<valueType>(values, gridDx);
			
			//Initialize cut-cells and velocities
			m_pCutCellsVelocities2D = pCutCellsVelocities2D;
			m_pCutCells2D = dynamic_cast<CutCells2D<Vector2> *>(m_pCutCellsVelocities2D->getMesh());

			m_dx = gridDx;
		}
		template<class valueType>
		MeanValueInterpolant2D<valueType>::MeanValueInterpolant2D(CutCellsVelocities2D *pCutCellsVelocities2D, Scalar gridDx, bool useAuxiliaryVelocities) :
			Interpolant() {
			m_useAuxiliaryVelocities = useAuxiliaryVelocities;
			//No nodal interpolant
			m_pNodalInterpolant = NULL;

			//Initialize cut-cells and velocities
			m_pCutCellsVelocities2D = pCutCellsVelocities2D;
			m_pCutCells2D = dynamic_cast<CutCells2D<Vector2> *>(m_pCutCellsVelocities2D->getMesh());

			m_dx = gridDx;
		}

		template<class valueType>
		MeanValueInterpolant2D<valueType>::MeanValueInterpolant2D(const Array2D<valueType> &values, CutCells2D<Vector2> *pCutCells2D, Scalar gridDx) :
			Interpolant(values) {
			//No nodal interpolant
			m_pNodalInterpolant = new BilinearNodalInterpolant2D<valueType>(values, gridDx);

			//Initialize cut-cells and velocities
			m_pCutCellsVelocities2D = NULL;
			m_pCutCells2D = pCutCells2D;

			m_dx = gridDx;
		}

		#pragma endregion

		#pragma region Functionalities
		template<>
		/** Scalar-based nodal interpolation in regular grids: assume that the values on the Array2D are stored in nodal locations.
		  * This means no off-set from current position. */
		Scalar MeanValueInterpolant2D<Scalar>::interpolate(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			int i = floor(gridSpacePosition.x); int j = floor(gridSpacePosition.y);
			if (m_pCutCells2D->isCutCellAt(i, j)) { //Actually implementing MVC
				return interpolateCutCell(m_pCutCells2D->getCutCellIndex(gridSpacePosition), position);
			}
			else {
				return m_pNodalInterpolant->interpolate(position);
			}
		}
		
		template<>
		Vector2 MeanValueInterpolant2D<Vector2>::interpolate(const Vector2 &position) {
			Vector2 gridSpacePosition = position / m_dx;
			int i = floor(gridSpacePosition.x); int j = floor(gridSpacePosition.y);
			if (m_pCutCells2D && m_pCutCells2D->isCutCellAt(i, j)) { //Actually implementing MVC
				return interpolateCutCell(m_pCutCells2D->getCutCellIndex(gridSpacePosition), position);
			} else {
				return m_pNodalInterpolant->interpolate(position);
			}
		}
		

		//template<class valueType>
		//vector<DoubleScalar> MeanValueInterpolant2D<valueType>::calculateWeights(const Vector2 &position, const vector<Vector2> &polygonPoints) {
		//	Vector2 rCurr, rPrev, rNext;
		//	Scalar rCurrLength;
		//	vector<DoubleScalar> weights(polygonPoints.size(), 0);
		//	for (int i = 0; i < polygonPoints.size(); i++) {
		//		rCurr = polygonPoints[i] - position;
		//		rCurrLength = rCurr.length();
		//		if (rCurrLength <= g_pointProximityLenght) {
		//			weights.assign(weights.size(), 0);
		//			weights[i] = 1.0;
		//			return weights;
		//		}

		//		int prevI = roundClamp<int>(i - 1, 0, polygonPoints.size());
		//		rPrev = polygonPoints[prevI] - position;
		//		int nextI = roundClamp<int>(i + 1, 0, polygonPoints.size());
		//		rNext = polygonPoints[nextI] - position;

		//		DoubleScalar aVal = calculateADet(rPrev, rCurr) / 2;
		//		if (aVal != 0)
		//			weights[i] += (rPrev.length() - rPrev.dot(rCurr) / rCurrLength) / aVal;
		//		aVal = calculateADet(rCurr, rNext) / 2;
		//		if (aVal != 0)
		//			weights[i] += (rNext.length() - rNext.dot(rCurr) / rCurrLength) / aVal;
		//	}

		//	DoubleScalar totalWeight = 0;
		//	for (int i = 0; i < weights.size(); i++) {
		//		totalWeight += weights[i];
		//	}

		//	for (int i = 0; i < weights.size(); i++) {
		//		weights[i] /= totalWeight;
		//	}
		//	return weights;
		//}

		template<>
		void MeanValueInterpolant2D<Scalar>::updateNodalVelocities(const Array2D<Vector2> &sourceStaggered, Array2D<Vector2> &targetNodal, bool useAuxVels) {

		}
		template<>
		void MeanValueInterpolant2D<Vector2>::updateNodalVelocities(const Array2D<Vector2> &sourceStaggered, Array2D<Vector2> &targetNodal, bool useAuxVels) {
			//Update regular grid nodal velocities first
			m_pNodalInterpolant->staggeredToNodeCentered(sourceStaggered, targetNodal, m_pCutCells2D, useAuxVels);
			
			m_pCutCellsVelocities2D->update(targetNodal, useAuxVels);
		}

		#pragma endregion
		#pragma region PrivateFunctionalities		
		template<>
		Scalar MeanValueInterpolant2D<Scalar>::interpolateCutCell(int ithCutCell, const Vector2 &position) {
			Vector2 rCurr, rPrev, rNext;
			Scalar rCurrLength;
			auto cutCellEdges = m_pCutCells2D->getCutCell(ithCutCell).getHalfEdges();
			vector<Scalar> weights(cutCellEdges.size(), 0);
			for (int i = 0; i < cutCellEdges.size(); i++) {
				auto currVertex = cutCellEdges[i]->getVertices().first;
				rCurr = currVertex->getPosition() - position;
				rCurrLength = rCurr.length();
				if (rCurrLength <= g_pointProximityLenght)
					return m_pCutCells2D->getCutCell(ithCutCell).getStreamfunction(i).x;

				int prevI = roundClamp<int>(i - 1, 0, cutCellEdges.size());
				rPrev = cutCellEdges[prevI]->getVertices().first->getPosition() - position;
				int nextI = roundClamp<int>(i + 1, 0, cutCellEdges.size());
				rNext = cutCellEdges[nextI]->getVertices().first->getPosition() - position;

				Scalar aVal = calculateADet(rPrev, rCurr) / 2;
				if (aVal != 0)
					weights[i] += (rPrev.length() - rPrev.dot(rCurr) / rCurrLength) / aVal;
				aVal = calculateADet(rCurr, rNext) / 2;
				if (aVal != 0)
					weights[i] += (rNext.length() - rNext.dot(rCurr) / rCurrLength) / aVal;
			}

			Scalar totalWeight = 0;
			for (int i = 0; i < weights.size(); i++) {
				totalWeight += weights[i];
			}
			Scalar result = 0;
			//Use streamfunction.x for 2-D
			for (int i = 0; i < weights.size(); i++) {
				result += m_pCutCells2D->getCutCell(ithCutCell).getStreamfunction(i).x * (weights[i] / totalWeight);
			}

			for (int i = 0; i < weights.size(); i++) {
				weights[i] /= totalWeight;
			}

			return result;
		}

		/** Vector-based nodal interpolation in regular grids */
		template<>
		Vector2 MeanValueInterpolant2D<Vector2>::interpolateCutCell(int ithCutCell, const Vector2 &position) {
			auto cutCellEdges = m_pCutCells2D->getCutCell(ithCutCell).getHalfEdges();
			vector<Scalar> weights(cutCellEdges.size(), 0);
			Vector2 rCurr, rPrev, rNext;
			Scalar rCurrLength;
			for (int i = 0; i < cutCellEdges.size(); i++) {
				auto currVertex = cutCellEdges[i]->getVertices().first;
				rCurr = currVertex->getPosition() - position;

				rCurrLength = rCurr.length();
				if (rCurrLength <= g_pointProximityLenght) {
					if(m_useAuxiliaryVelocities)
						return currVertex->getAuxiliaryVelocity();
					else
						return currVertex->getVelocity();
				}	
				
				int prevI = roundClamp<int>(i - 1, 0, cutCellEdges.size());
				rPrev = cutCellEdges[prevI]->getVertices().first->getPosition() - position;
				int nextI = roundClamp<int>(i + 1, 0, cutCellEdges.size());
				rNext = cutCellEdges[nextI]->getVertices().first->getPosition() - position;

				Scalar aVal = calculateADet(rPrev, rCurr) / 2;
				if (aVal != 0)
					weights[i] += (rPrev.length() - rPrev.dot(rCurr) / rCurrLength) / aVal;
				aVal = calculateADet(rCurr, rNext) / 2;
				if (aVal != 0)
					weights[i] += (rNext.length() - rNext.dot(rCurr) / rCurrLength) / aVal;
			}

			Scalar totalWeight = 0;
			for (int i = 0; i < weights.size(); i++) {
				totalWeight += weights[i];
			}
			Vector2 result(0, 0);
			for (int i = 0; i < weights.size(); i++) {
				if(m_useAuxiliaryVelocities) 
					result += cutCellEdges[i]->getVertices().first->getAuxiliaryVelocity() * (weights[i] / totalWeight);
				else 
					result += cutCellEdges[i]->getVertices().first->getVelocity() * (weights[i] / totalWeight);
			}
			return result;
		}
		#pragma endregion

		/** Template linker trickerino for templated classes in CPP*/
		template class MeanValueInterpolant2D<Scalar>;
		template class MeanValueInterpolant2D<Vector2>;
	}
}