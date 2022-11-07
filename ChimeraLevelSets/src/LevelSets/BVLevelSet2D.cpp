#include "LevelSets/BVLevelSet2D.h"

namespace Chimera {
	namespace LevelSets {

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		BVLevelSet2D::BVLevelSet2D(const params_t &params, QuadGrid *pGrid) : LevelSet2D(params, pGrid) {
			pNarrowBand = new NarrowBand(m_pGrid);
			initializeArbitraryPoints();
			solveFMM();

			for(int i = 0; i < m_pGrid->getDimensions().x; i++) {
				setValue(getValue(i, 1), i, 0);
				setValue(getValue(i, m_pGrid->getDimensions().y - 2), i, m_pGrid->getDimensions().y - 1);
			}
			for(int i = 0; i < m_pGrid->getDimensions().y; i++) {
				setValue(getValue(1, i), 0, i);
				setValue(getValue(m_pGrid->getDimensions().x - 2, i), m_pGrid->getDimensions().x - 1, i);
			}
			for(int i = 0; i < m_pGrid->getDimensions().x; i++) {
				for(int j = 0; j < m_pGrid->getDimensions().y; j++) {
					m_pGrid->setSolidCell(false, i, j);
				}
			}

		}

		/************************************************************************/
		/* Initialization                                                       */
		/************************************************************************/
		void BVLevelSet2D::initializeArbitraryPoints() {
			GridData2D *pGridData2D = m_pGrid->getGridData2D();
			//Initialize alive points 
			for(int i = 0; i < m_pGrid->getDimensions().x; i++) {
				for(int j = 0; j < m_pGrid->getDimensions().y; j++) {
					// All points are initially far-away points
					setValue(FLT_MAX, i, j);
				}
			}
			Vector2 dS = pGridData2D->getScaleFactor(0, 0);

			//Initialize alive points 
			for(unsigned int i = 0; i < m_params.pPolygonPoints->size(); i++) {
				Vector2 gridSpacePoint = (m_params.pPolygonPoints->at(i) - m_pGrid->getGridOrigin())/dS;

				if(gridSpacePoint.x > 0 && gridSpacePoint.x < m_pGrid->getDimensions().x - 1 &&
					gridSpacePoint.y > 0 && gridSpacePoint.y < m_pGrid->getDimensions().y) { //Inside grid boundaries
						m_pGrid->setSolidCell(true, floor(gridSpacePoint.x), floor(gridSpacePoint.y));
				}
			} 

			//Temporary update on the distance field, in order to find the velocity field
			updateDistanceField();

			/*Initializing narrow-band*/
			for(int i = 0; i < m_pGrid->getDimensions().x; i++) {
				for(int j = 0; j < m_pGrid->getDimensions().y; j++) {
					if(m_pGrid->isAdjacentToSolidCell(i, j) && !m_pGrid->isSolidCell(i, j)) {
						m_pGrid->setBoundaryCell(true, i, j);
						//Putting values in the min-heap structure
						Scalar calcTime = calculateTime(i, j);
						pNarrowBand->addCell(calcTime, i, j);
						setValue(calcTime, i, j);
					} else if(!m_pGrid->isSolidCell(i, j)) {
						setValue(FLT_MAX, i, j); //Far away
					}
				}
			}
		}

		/************************************************************************/
		/* Internal functionalities                                             */
		/************************************************************************/
		Scalar BVLevelSet2D::calculateTimeCurvature(int x, int y) {
			Vector2 dS = m_pGrid->getGridData2D()->getScaleFactor(0, 0);
			Vector2 dSquare = dS*dS;

			Scalar time = 0.0f;

			Scalar tx = min(getValue(x + 1, y), getValue(x - 1, y));
			Scalar ty = min(getValue(x, y + 1), getValue(x, y - 1));

			bool validX = tx != FLT_MAX;
			bool validY = ty != FLT_MAX;

			Scalar a = validX/dSquare.x + validY/dSquare.y;
			Scalar b = -(2*tx*validX/dSquare.x + 2*ty*validY/dSquare.x);
			Scalar c = tx*tx*validX/dSquare.x + ty*ty*validY/dSquare.x;
			//c -= 1 - 0.025*(calculateCurvature(x, y, m_pGrid->getGridData2D()->getLevelSetArray(), dS.x));

			time = (-b + (sqrt(b*b -4*a*c)))/(2*a);

			if(b*b -4*a*c < 0) {
				return max(tx, ty) + 1;
			}

			return time;
		}

		Scalar BVLevelSet2D::selectUpwindNeighbor(int x, int y, const velocityComponent_t &direction) {
			Scalar upwindValue = FLT_MAX; //Default non-valid value

			int neigbohrsCases;
			if(direction == velocityComponent_t::xComponent) {
				neigbohrsCases = m_pGrid->isSolidCell(x - 1, y);
				neigbohrsCases |= m_pGrid->isSolidCell(x + 1, y) << 1;
			} else if(direction == velocityComponent_t::yComponent) {
				neigbohrsCases = m_pGrid->isSolidCell(x, y - 1);
				neigbohrsCases |= m_pGrid->isSolidCell(x, y + 1) << 1;
			}

			switch (neigbohrsCases)
			{
			case 0: // No Cell in the X (Y) direction has frozen values
				upwindValue = FLT_MAX;
				break;

			case 1: //Only left (bottom) cell has solid values
				if(direction == velocityComponent_t::xComponent) {
					upwindValue = getValue(x - 1, y);
				} else if(direction == velocityComponent_t::yComponent) {
					upwindValue = getValue(x, y - 1);
				}
				break;

			case 2: //Only right (top) cell has solid values
				if(direction == velocityComponent_t::xComponent) {
					upwindValue = getValue(x + 1, y);
				} else if(direction == velocityComponent_t::yComponent) {
					upwindValue = getValue(x, y + 1);
				}
				break;

			case 3: // Both cells are frozen, select minimum
				if(direction == velocityComponent_t::xComponent) {
					upwindValue = min(getValue(x - 1, y), getValue(x + 1, y));
				} else if(direction == velocityComponent_t::yComponent) {
					upwindValue = min(getValue(x, y - 1), getValue(x, y + 1));
				}
				break;
			}

			return upwindValue;

		}
		Scalar BVLevelSet2D::calculateTime(int x, int y, bool useFrozen) {
			Vector2 dS = m_pGrid->getGridData2D()->getScaleFactor(0, 0);
			Vector2 dSquare = dS*dS;

			Scalar time = 0.0f;

			Scalar tx = selectUpwindNeighbor(x, y, velocityComponent_t::xComponent);
			Scalar ty = selectUpwindNeighbor(x, y, velocityComponent_t::yComponent);

			bool validX = tx != FLT_MAX;
			bool validY = ty != FLT_MAX;

			Scalar a = validX/dSquare.x + validY/dSquare.y;
			Scalar b = -((2*tx*validX)/dSquare.x + (2*ty*validY)/dSquare.x);
			Scalar c = (tx*tx*validX)/dSquare.x + (ty*ty*validY)/dSquare.x;
			c -= 1; //|v| = 1
			Scalar d = (b*b) - (4.0*a*c);

			if(d < 0) {
				return min(tx, ty) + 1;
			} 

			time = (-b + sqrt(d))/(2*a);
			return time;
		}

		/*************************************d***********************************/
		/* NarrowBand class                                                     */
		/************************************************************************/
		/* Ctor */
		BVLevelSet2D::NarrowBand::NarrowBand(StructuredGrid<Vector2> *pGrid)  : m_pGrid(pGrid), m_dimensions(pGrid->getDimensions()) {
			pUsedMap = new Array2D<char>(m_dimensions);
			for(int i = 0; i < m_dimensions.x; i++) {
				for(int j = 0; j < m_dimensions.y; j++) {
					(*pUsedMap)(i, j) = false;
				}
			}
		}
		void BVLevelSet2D::NarrowBand::addCell(Scalar timeValue, int i, int j) {					
			if((*pUsedMap)(i, j)) {
				int cellId = j*m_dimensions.x + i;
				std::multimap<Scalar,minHeapElement_t*>::iterator it;
				Scalar oldTimeValue = m_pGrid->getGridData2D()->getLevelSetValue(i, j);
				minHeapElement_t *pMinElement = NULL;
				for (it = m_multiMap.equal_range(oldTimeValue).first; it != m_multiMap.equal_range(oldTimeValue).second; ++it) {
					if(it->second->cellID.x == i && it->second->cellID.y == j) {
						pMinElement = it->second;
						m_multiMap.erase(it);
						break;
					}
				}
				if(pMinElement != NULL) {
					pMinElement->time = timeValue;
					m_multiMap.insert(pair<Scalar, minHeapElement_t*>(timeValue, pMinElement));
				}
			} else {
				minHeapElement_t *pMinElement = new minHeapElement_t(timeValue, dimensions_t(i, j));
				m_multiMap.insert(pair<Scalar, minHeapElement_t*>(timeValue, pMinElement));
			}
			(*pUsedMap)(i, j) = true;
		}

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void BVLevelSet2D::update() {
			for(int i = 0; i < m_pGrid->getDimensions().x; i++) {
				for(int j = 0; j < m_pGrid->getDimensions().y; j++) {
					(*pNarrowBand->pUsedMap)(i, j) = false;
				}
			}
			initializeArbitraryPoints();
			solveFMM();

			for(int i = 0; i < m_pGrid->getDimensions().x; i++) {
				setValue(getValue(i, 1), i, 0);
				setValue(getValue(i, m_pGrid->getDimensions().y - 2), i, m_pGrid->getDimensions().y - 1);
			}
			for(int i = 0; i < m_pGrid->getDimensions().y; i++) {
				setValue(getValue(1, i), 0, i);
				setValue(getValue(m_pGrid->getDimensions().x - 2, i), m_pGrid->getDimensions().x - 1, i);
			}
			for(int i = 0; i < m_pGrid->getDimensions().x; i++) {
				for(int j = 0; j < m_pGrid->getDimensions().y; j++) {
					m_pGrid->setSolidCell(false, i, j);
				}
			}
		}
		void BVLevelSet2D::solveFMM() {
			int t = 0;
			while(!pNarrowBand->isEmpty()) {
				minHeapElement_t *pCurrElement = pNarrowBand->getTopElement();
				pNarrowBand->popTopElement();
				m_pGrid->setSolidCell(true, pCurrElement->cellID.x, pCurrElement->cellID.y);
				m_pGrid->setBoundaryCell(false, pCurrElement->cellID.x, pCurrElement->cellID.y);

				int i = pCurrElement->cellID.x; int j = pCurrElement->cellID.y;

				//Updating this
				//calculateTimeCurvature(i, j);

				//Recalculate for all neighbors
				if(isInsideSearchRange(i + 1, j) && !m_pGrid->isSolidCell(i + 1, j)) {
					Scalar currTime = calculateTime(i + 1, j, false);
					pNarrowBand->addCell(currTime, i + 1, j);
					if(getValue(i + 1, j) == FLT_MAX) { // Far-away point
						m_pGrid->setBoundaryCell(true, i + 1, j);
					}
					setValue(currTime, i + 1, j);
				}
				if(isInsideSearchRange(i - 1, j) && !m_pGrid->isSolidCell(i - 1, j)) {
					Scalar currTime = calculateTime(i - 1, j, false);
					pNarrowBand->addCell(currTime, i - 1, j);
					if(getValue(i - 1, j) == FLT_MAX) { // Far-away point

						m_pGrid->setBoundaryCell(true, i - 1, j);
					}
					setValue(currTime, i - 1, j);
				}
				if(isInsideSearchRange(i, j + 1) && !m_pGrid->isSolidCell(i, j + 1)) {
					Scalar currTime = calculateTime(i, j + 1, false);
					pNarrowBand->addCell(currTime, i, j + 1);
					if(getValue(i, j + 1) == FLT_MAX) { // Far-away point

						m_pGrid->setBoundaryCell(true, i, j + 1);
					}
					setValue(currTime, i, j + 1);
				}
				if(isInsideSearchRange(i, j - 1) && !m_pGrid->isSolidCell(i, j - 1)) {
					Scalar currTime = calculateTime(i, j - 1, false);
					pNarrowBand->addCell(currTime, i, j - 1);
					if(getValue(i, j - 1) == FLT_MAX) { // Far-away point

						m_pGrid->setBoundaryCell(true, i, j - 1);
					}
					setValue(currTime, i, j - 1);
				}
			}
		}
	}
}