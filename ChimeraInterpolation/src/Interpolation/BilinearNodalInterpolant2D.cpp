#include "Interpolation/BilinearNodalInterpolant2D.h"


namespace Chimera {
	namespace Interpolation {
		
		#pragma region Constructors
		template<class valueType>
		BilinearNodalInterpolant2D<valueType>::BilinearNodalInterpolant2D(const Array2D<valueType>& values, Scalar gridDx)
																					: Interpolant(values) {
			m_dx = gridDx;
		}
		#pragma endregion

		#pragma region Functionalities
		template<>
		/** Scalar-based nodal interpolation in regular grids: assume that the values on the Array2D are stored in nodal locations.
		  * This means no off-set from current position. */
		Scalar BilinearNodalInterpolant2D<Scalar>::interpolate(const Vector2 &position) {
			Vector2 gridSpacePosition(position / m_dx);

			Scalar interpolatedValue;

			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);

			Vector2 x1Position(i + 0.0f, j + 0.0f);
			Vector2 x2Position(i + 1.0f, j + 1.0f);

			Scalar tempValue = m_values(i, j) * (x2Position.x - gridSpacePosition.x) * (x2Position.y - gridSpacePosition.y) +
				m_values(i + 1, j) * (gridSpacePosition.x - x1Position.x) * (x2Position.y - gridSpacePosition.y) +
				m_values(i, j + 1) * (x2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - x1Position.y) +
				m_values(i + 1, j + 1) * (gridSpacePosition.x - x1Position.x) * (gridSpacePosition.y - x1Position.y);

			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1) //Last possible cell
				return m_values(i, j);
			else if (i == m_values.getDimensions().x - 1) //Right boundary
				return m_values(i, j) * (x2Position.y - gridSpacePosition.y) + m_values(i, j + 1) * (gridSpacePosition.y - x1Position.y);
			else if (j == m_values.getDimensions().y - 1) //Top boundary
				return m_values(i, j) * (x2Position.x - gridSpacePosition.x) + m_values(i + 1, j) * (gridSpacePosition.x - x1Position.x);
			else //Normal cell
				return	m_values(i, j) * (x2Position.x - gridSpacePosition.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i + 1, j) * (gridSpacePosition.x - x1Position.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i, j + 1) * (x2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - x1Position.y) +
						m_values(i + 1, j + 1) * (gridSpacePosition.x - x1Position.x) * (gridSpacePosition.y - x1Position.y);

			
		}
		
		/** Vector-based nodal interpolation in regular grids: assume that the values on the Array2D are stored in nodal locations.
		  * This means no off-set from current position. */
		template<>
		Vector2 BilinearNodalInterpolant2D<Vector2>::interpolate(const Vector2 &position) {
			Vector2 gridSpacePosition(position / m_dx);

			Vector2 interpolatedVelocity;

			int i = static_cast <int> (floor(gridSpacePosition.x));
			int j = static_cast <int> (floor(gridSpacePosition.y));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);


			Vector2 x1Position(static_cast <Scalar>(i), static_cast <Scalar>(j));
			Vector2 x2Position(i + 1.0f, j + 1.0f);

			if (i == m_values.getDimensions().x - 1 && j == m_values.getDimensions().y - 1)
				return m_values(i, j) * (x2Position.y - gridSpacePosition.y);
			else if (i == m_values.getDimensions().x - 1)
				return m_values(i, j) * (x2Position.y - gridSpacePosition.y) + m_values(i, j + 1) * (gridSpacePosition.y - x1Position.y);
			else if (j == m_values.getDimensions().y - 1)
				return m_values(i, j) * (x2Position.x - gridSpacePosition.x) + m_values(i + 1, j) * (gridSpacePosition.x - x1Position.x);
			else
				return  m_values(i, j) * (x2Position.x - gridSpacePosition.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i + 1, j) * (gridSpacePosition.x - x1Position.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i, j + 1) * (x2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - x1Position.y) +
						m_values(i + 1, j + 1) * (gridSpacePosition.x - x1Position.x) * (gridSpacePosition.y - x1Position.y);
		}


		template<class valueType>
		void BilinearNodalInterpolant2D<valueType>::staggeredToNodeCentered(const Array2D<Vector2> &sourceStaggered, 
																			Array2D<Vector2> &targetStaggered) {
			Array2D<Vector2> sourceCopy(sourceStaggered);
			for (int i = 0; i < sourceCopy.getDimensions().x; i++) {
				for (int j = 0; j < sourceCopy.getDimensions().y; j++) {
					targetStaggered(i, j).x = j > 0 ? (sourceCopy(i, j - 1).x + sourceCopy(i, j).x)*0.5 : sourceCopy(i, j).x;
					targetStaggered(i, j).y = i > 0 ? (sourceCopy(i - 1, j).y + sourceCopy(i, j).y)*0.5 : sourceCopy(i, j).y;
				}
			}
		}

		template<class valueType>
		void BilinearNodalInterpolant2D<valueType>::staggeredToNodeCentered(const Array2D<Vector2> &sourceStaggered,
																			Array2D<Vector2> &targetStaggered,
																			CutCells2D<Vector2> * pCutCells, bool auxiliaryVelocity) {
			Array2D<Vector2> sourceCopy(sourceStaggered);
			for (int i = 0; i < sourceCopy.getDimensions().x; i++) {
				for (int j = 0; j < sourceCopy.getDimensions().y; j++) {
					Vector2 nodeVelocity;
					if (pCutCells && pCutCells->isCutCellAt(i, j)) {
						auto leftEdges = pCutCells->getEdgeVector(dimensions_t(i, j), yAlignedEdge);
						if (pCutCells->isCutCellAt(i, j - 1)) {
							auto leftBottomEdges = pCutCells->getEdgeVector(dimensions_t(i, j - 1), yAlignedEdge);
							if (leftBottomEdges.size() > 0) {
								Scalar totalAreaFraction = leftEdges.front()->getRelativeFraction() + leftBottomEdges.back()->getRelativeFraction();
								Scalar alfa = leftEdges.front()->getRelativeFraction() / totalAreaFraction;
								if(auxiliaryVelocity)
									targetStaggered(i, j).x = (alfa*leftBottomEdges.back()->getAuxiliaryVelocity().x + (1 - alfa)*leftEdges.front()->getAuxiliaryVelocity().x);
								else
									targetStaggered(i, j).x = (alfa*leftBottomEdges.back()->getVelocity().x + (1 - alfa)*leftEdges.front()->getVelocity().x);
							}
							else { //This can't happen?
								throw("Invalid edges configuration detected on staggered to node centered");
							}
						}
						else {
							Scalar totalAreaFraction = leftEdges.front()->getRelativeFraction() + 1;
							Scalar alfa = leftEdges.front()->getRelativeFraction() / totalAreaFraction;
							if (auxiliaryVelocity)
								targetStaggered(i, j).x = sourceStaggered(i, j - 1).x*alfa + (1 - alfa)*leftEdges.front()->getAuxiliaryVelocity().x;
							else
								targetStaggered(i, j).x = sourceStaggered(i, j - 1).x*alfa + (1 - alfa)*leftEdges.front()->getVelocity().x;
						}

						auto bottomEdges = pCutCells->getEdgeVector(dimensions_t(i, j), xAlignedEdge);
						if (pCutCells->isCutCellAt(i - 1, j)) {
							auto bottomLeftEdges = pCutCells->getEdgeVector(dimensions_t(i - 1, j), xAlignedEdge);
							if (bottomLeftEdges.size() > 0) {
								Scalar totalAreaFraction = bottomEdges.front()->getRelativeFraction() + bottomLeftEdges.back()->getRelativeFraction();
								Scalar alfa = bottomEdges.front()->getRelativeFraction() / totalAreaFraction;
								if (auxiliaryVelocity)
									targetStaggered(i, j).y = (alfa*bottomLeftEdges.back()->getAuxiliaryVelocity().y + (1 - alfa)*bottomEdges.front()->getAuxiliaryVelocity().y);
								else
									targetStaggered(i, j).y = (alfa*bottomLeftEdges.back()->getVelocity().y + (1 - alfa)*bottomEdges.front()->getVelocity().y);
							}
							else { //This can't happen?

							}
						}
						else {
							Scalar totalFaceAreas = 1 + bottomEdges.front()->getRelativeFraction();
							Scalar alfa = bottomEdges.front()->getRelativeFraction() / totalFaceAreas;
							if (auxiliaryVelocity)
								targetStaggered(i, j).y = (1 - alfa)*bottomEdges.front()->getAuxiliaryVelocity().y + alfa*sourceStaggered(i - 1, j).y;
							else
								targetStaggered(i, j).y = (1 - alfa)*bottomEdges.front()->getVelocity().y + alfa*sourceStaggered(i - 1, j).y;
						}
					}
					else {
						targetStaggered(i, j).x = j > 0 ? (sourceCopy(i, j - 1).x + sourceCopy(i, j).x)*0.5 : sourceCopy(i, j).x;
						targetStaggered(i, j).y = i > 0 ? (sourceCopy(i - 1, j).y + sourceCopy(i, j).y)*0.5 : sourceCopy(i, j).y;
						if (j > 0 && pCutCells && pCutCells->isCutCellAt(i, j - 1)) {
							auto leftBottomEdges = pCutCells->getEdgeVector(dimensions_t(i, j - 1), yAlignedEdge);
							if (leftBottomEdges.size() > 0) {
								Scalar totalAreaFraction = 1 + leftBottomEdges.back()->getRelativeFraction();
								Scalar alfa = leftBottomEdges.back()->getRelativeFraction() / totalAreaFraction;
								if (auxiliaryVelocity)
									targetStaggered(i, j).x = (sourceCopy(i, j).x*alfa + (1 - alfa)*leftBottomEdges.back()->getAuxiliaryVelocity().x);
								else
									targetStaggered(i, j).x = (sourceCopy(i, j).x*alfa + (1 - alfa)*leftBottomEdges.back()->getVelocity().x);
							}
						}
						if (i > 0 && pCutCells && pCutCells->isCutCellAt(i - 1, j)) {
							auto bottomLeftEdges = pCutCells->getEdgeVector(dimensions_t(i - 1, j), xAlignedEdge);
							if (bottomLeftEdges.size() > 0) {
								Scalar totalAreaFraction = 1 + bottomLeftEdges.back()->getRelativeFraction();
								Scalar alfa = bottomLeftEdges.back()->getRelativeFraction() / totalAreaFraction;
								if(auxiliaryVelocity)
									targetStaggered(i, j).y = (sourceCopy(i, j).y*alfa + (1 - alfa)*bottomLeftEdges.back()->getAuxiliaryVelocity().y);
								else
									targetStaggered(i, j).y = (sourceCopy(i, j).y*alfa + (1 - alfa)*bottomLeftEdges.back()->getVelocity().y);
							}
						}
					}
				}
			}
		}
		#pragma endregion

		/** Template linker trickerino for templated classes in CPP*/
		template class BilinearNodalInterpolant2D<Scalar>;
		template class BilinearNodalInterpolant2D<Vector2>;
	}
}