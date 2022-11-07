#include "LevelSets/MarchingSquares.h"

namespace Chimera {
	namespace LevelSets {
		

		MarchingSquares::MarchingSquares(const Array2D<Scalar> & levelSet, Scalar gridSpacing) : 
			m_levelSet(levelSet), m_cellTypes(levelSet.getDimensions()) {
			m_gridSpacing = gridSpacing;
		}

		vector<LineMesh<Vector2> *> MarchingSquares::extract(Scalar isoValue) {
			vector<LineMesh<Vector2> *> m_lineMeshes;
			m_visitedCellsList.clear();

			//Classify all cells given their isovalue
			for (int i = 0; i < m_levelSet.getDimensions().x - 0; i++) {
				for (int j = 0; j < m_levelSet.getDimensions().y - 0; j++) {
					if (m_levelSet(i, j) < isoValue) {
						m_cellTypes(i, j) = cellTypes::internalCell;
					}
					else {
						m_cellTypes(i, j) = cellTypes::externalCell;
					}
				}
			}

			int it = 0;
			m_visitedCellsList.reserve(1000);

			for (int i = 1; i < m_levelSet.getDimensions().x - 1; i++) {
				for (int j = 1; j < m_levelSet.getDimensions().y - 1; j++) {
					dimensions_t currCell(i, j);
					if (m_cellTypes(i, j) != cellTypes::visitedCell && m_cellTypes(i, j) == internalCell && 
							hasNeighbor(currCell, cellTypes::externalCell)) {
						
						LineMesh<Vector2>::params_t lineMeshParams;

						while (m_cellTypes(currCell) != cellTypes::visitedCell) {
							m_visitedCellsList.push_back(currCell);

							if (currCell.x <= 1 || currCell.x >= m_levelSet.getDimensions().x - 2 || currCell.y <= 1 ||
								currCell.y >= m_levelSet.getDimensions().y - 1) {
								//Out of bounds
								break;
							}

							lineMeshParams.initialPoints.push_back(calculatePoint(isoValue, currCell));
							dimensions_t tempDimensions(currCell);

							currCell = goToNextCell(currCell);
							m_cellTypes(tempDimensions) = cellTypes::visitedCell;
							it++;
						}

						if (lineMeshParams.initialPoints.size() > 0) {
							lineMeshParams.initialPoints.push_back(lineMeshParams.initialPoints[0]);
							m_lineMeshes.push_back(new LineMesh<Vector2>(lineMeshParams));
						}
					}
				}
			}
			return m_lineMeshes;
		}

		dimensions_t MarchingSquares::goToNextCell(const dimensions_t & currentCell) {
			int mask;
			Scalar ls1 = m_levelSet(currentCell.x, currentCell.y);
			Scalar ls2 = m_levelSet(currentCell.x + 1, currentCell.y);
			Scalar ls3 = m_levelSet(currentCell.x + 1, currentCell.y + 1);
			Scalar ls4 = m_levelSet(currentCell.x, currentCell.y + 1);

			mask = m_levelSet(currentCell.x, currentCell.y) < 0 ;
			mask |= (m_levelSet(currentCell.x + 1, currentCell.y) < 0) << 1;
			mask |= (m_levelSet(currentCell.x + 1, currentCell.y + 1) < 0) << 2;
			mask |= (m_levelSet(currentCell.x, currentCell.y + 1) < 0) << 3;
			//mask = m_cellTypes(currentCell.x, currentCell.y) == cellTypes::internalCell;// pGrid->isSolidCell(currCell.x, currCell.y);
			//mask |= m_cellTypes(currentCell.x + 1, currentCell.y) == cellTypes::internalCell << 1; //pGrid->isSolidCell(currCell.x + 1, currCell.y) << 1;
			//mask |= m_cellTypes(currentCell.x + 1, currentCell.y + 1) == cellTypes::internalCell << 2; //pGrid->isSolidCell(currCell.x + 1, currCell.y + 1) << 2;
			//mask |= m_cellTypes(currentCell.x, currentCell.y + 1) == cellTypes::internalCell << 3; //pGrid->isSolidCell(currCell.x, currCell.y + 1) << 3;
			switch (mask) {
			case 0:
			case 4:
			case 12:
			case 13:
				return currentCell + dimensions_t(1, 0, 0);
				break;

			case 1:
			case 3:
			case 5:
			case 7:
				return currentCell + dimensions_t(-1, 0, 0);
				break;

			case 8:
			case 9:
			case 10:
			case 11:
				return currentCell + dimensions_t(0, 1, 0);
				break;

			case 2:
			case 6:
			case 14:
				return currentCell + dimensions_t(0, -1, 0);
				break;

			default:
				return currentCell;
				break;
			}
		}

		Vector2 MarchingSquares::calculatePoint(Scalar isoValue, const dimensions_t & cell) {
			int mask;
			
			mask = m_levelSet(cell.x, cell.y) < 0;
			mask |= (m_levelSet(cell.x + 1, cell.y) < 0) << 1;
			mask |= (m_levelSet(cell.x + 1, cell.y + 1) < 0) << 2;
			mask |= (m_levelSet(cell.x, cell.y + 1) < 0) << 3;

			//mask = m_cellTypes(cell.x, cell.y) == cellTypes::internalCell;// pGrid->isSolidCell(cell.x, cell.y);
			//mask |= m_cellTypes(cell.x + 1, cell.y) == cellTypes::internalCell << 1; //pGrid->isSolidCell(cell.x + 1, cell.y) << 1;
			//mask |= m_cellTypes(cell.x + 1, cell.y + 1) == cellTypes::internalCell << 2; //pGrid->isSolidCell(cell.x + 1, cell.y + 1) << 2;
			//mask |= m_cellTypes(cell.x, cell.y + 1) == cellTypes::internalCell << 3; //pGrid->isSolidCell(cell.x, cell.y + 1) << 3;

			Scalar isoDistance, alfa;
			switch (mask) {
			case 1:
			case 3:
			case 7:
				isoDistance = abs(m_levelSet(cell.x, cell.y) - m_levelSet(cell.x, cell.y + 1));
				alfa = abs(m_levelSet(cell.x, cell.y) - isoValue) / isoDistance;
				return Vector2(cell.x, cell.y)*m_gridSpacing*(1 - alfa) + Vector2(cell.x, cell.y + 1)*m_gridSpacing*alfa;
				break;

			case 2:
			case 6:
			case 14:
				isoDistance = abs(m_levelSet(cell.x, cell.y) - m_levelSet(cell.x + 1, cell.y));
				alfa = abs(m_levelSet(cell.x, cell.y) - isoValue) / isoDistance;
				return Vector2(cell.x, cell.y)*m_gridSpacing*(1 - alfa) + Vector2(cell.x + 1, cell.y)*m_gridSpacing*alfa;
				break;

			case 4:
			case 12:
			case 13:
				isoDistance = abs(m_levelSet(cell.x + 1, cell.y) - m_levelSet(cell.x + 1, cell.y + 1));
				alfa = abs(m_levelSet(cell.x + 1, cell.y) - isoValue) / isoDistance;
				return Vector2(cell.x + 1, cell.y)*m_gridSpacing*(1 - alfa) + Vector2(cell.x + 1, cell.y + 1)*m_gridSpacing*alfa;
				break;

			case 9:
			case 8:
			case 11:
				isoDistance = abs(m_levelSet(cell.x, cell.y + 1) - m_levelSet(cell.x + 1, cell.y + 1));
				alfa = abs(m_levelSet(cell.x, cell.y + 1) - isoValue) / isoDistance;
				return Vector2(cell.x, cell.y + 1)*m_gridSpacing*(1 - alfa) + Vector2(cell.x + 1, cell.y + 1)*m_gridSpacing*alfa;
				break;
			case 5:
			case 10:
				//treat differently
				return Vector2(-1, -1);
				break;

			default:
				return Vector2(-1, -1);
				break;
			}
		}

		bool MarchingSquares::hasNeighbor(const dimensions_t &cellIndex, cellTypes cellType) {
			if (m_cellTypes(cellIndex.x + 1, cellIndex.y) == cellType ||
				m_cellTypes(cellIndex.x, cellIndex.y + 1) == cellType ||
				m_cellTypes(cellIndex.x + 1, cellIndex.y + 1) == cellType)
				return true;
			return false;
		}
	}
}
