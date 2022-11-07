#include "Liquids/LiquidRepresentation2D.h"



namespace Chimera {

	namespace LevelSets {
		#pragma region ParticlesFunctions

		template <class VectorT>
		void LiquidRepresentation2D<VectorT>::updateMeshes() {
			m_lineMeshes.clear();
			/** First initialize level-set with particles information */
			particlesToLevelSetGrid();

			/** Extract the isosurface 0 of a level set from the grid */
			updateCellTypes();
		}

		template <class VectorT>
		void LiquidRepresentation2D<VectorT>::updateCellTypes() {

			/** Initially all cells are of the air type: no particles */
			m_levelSetCellTypes.assign(levelSetCellType_t::airCell);
		
			/** This is the number of times that each dimension of the high-res grid is higher than the original fluid
			simulation grid. This is needed, since particles of the FLIP solver are usually in the original fluid grid
			space and need to be transformed to the level-set grid space. */
			int subdivisionScaleFactor = pow(2, m_params.levelSetGridSubdivisions);

			//const vector<VectorT> &pParticlesPos = m_params.pParticlesData->getPositions();
			//vector<int> *pParticlesTags = m_params.pParticlesTags;

			///** First-pass: all cells that have particles on it are marked as fluid cells */
			//for (int i = 0; i < pParticlesPos->size(); i++) {
			//	if (pParticlesTags->at(i) == 1) { //Particle's inside the liquid representation
			//		/** Level-set grid index of particle */
			//		dimensions_t levelSetGridIndex(floor(pParticlesPos->at(i).x*subdivisionScaleFactor),
			//			floor(pParticlesPos->at(i).y*subdivisionScaleFactor));

			//		m_levelSetCellTypes(levelSetGridIndex) = levelSetCellType_t::fluidCell;
			//	}
			//}

			///** Second-pass: identify boundary cells: fluid cells that neighbour air cells*/
			//dimensions_t initialCellDimension;
			//bool hasBoundaryCells = false;
			//for (int i = 0; i < pParticlesPos->size(); i++) {
			//	/** Level-set grid index of particle */
			//	dimensions_t levelSetGridIndex(floor(pParticlesPos->at(i).x*subdivisionScaleFactor),
			//									floor(pParticlesPos->at(i).y*subdivisionScaleFactor));
			//	if (	m_levelSetCellTypes(levelSetGridIndex) == levelSetCellType_t::fluidCell &&
			//			((levelSetGridIndex.x > 1 && 
			//			m_levelSetCellTypes(levelSetGridIndex.x - 1, levelSetGridIndex.y) == levelSetCellType_t::airCell) ||
			//			(levelSetGridIndex.x < m_levelSetGrid.getDimensions().x - 1 && 
			//			m_levelSetCellTypes(levelSetGridIndex.x + 1, levelSetGridIndex.y) == levelSetCellType_t::airCell) ||
			//			(levelSetGridIndex.y > 1 && 
			//			m_levelSetCellTypes(levelSetGridIndex.x, levelSetGridIndex.y - 1) == levelSetCellType_t::airCell) ||
			//			(levelSetGridIndex.y < m_levelSetGrid.getDimensions().y - 1 && 
			//			m_levelSetCellTypes(levelSetGridIndex.x - 1, levelSetGridIndex.y + 1) == levelSetCellType_t::airCell))	) {

			//		m_levelSetCellTypes(levelSetGridIndex) = levelSetCellType_t::boundaryCell;
			//		hasBoundaryCells = true;
			//		initialCellDimension = levelSetGridIndex;
			//		m_boundaryCellsMap[m_levelSetGrid.getRawPtrIndex(levelSetGridIndex.x, levelSetGridIndex.y)] = levelSetGridIndex;
			//	}
			//}

			///** Until a boundary cell exists, push-in to the liquid representation meshes a line enclosing a single connected
			//	fluid blob */
		
			///** TODO: Keep a vector (list?) of valid boundary cells to improve performance of this step. */
			//MarchingSquares marchingSquares(m_levelSetGrid, m_levelSetGridSpacing);
			//m_lineMeshes = marchingSquares.extract(0.0f);
			////while (hasBoundaryCells) {
			//	//m_lineMeshes = marchingSquares.extract(0.0f);
			//	//m_lineMeshes.push_back(marchingSquares.extract(0.0f, initialCellDimension));
			//	//const vector<dimensions_t> &visitedCells = marchingSquares.getVisitedCellsList();
			//	//for (int i = 0; i < visitedCells.size(); i++) {
			//	//	removeBoundaryMapEntry(visitedCells[i]);
			//	//	//Removing neighbors from this entry as well
			//	//	removeBoundaryMapEntry(visitedCells[i] + dimensions_t(1, 0));
			//	//	removeBoundaryMapEntry(visitedCells[i] + dimensions_t(-1, 0));
			//	//	removeBoundaryMapEntry(visitedCells[i] + dimensions_t(0, 1));
			//	//	removeBoundaryMapEntry(visitedCells[i] + dimensions_t(0, -1));
			//	//	removeBoundaryMapEntry(visitedCells[i] + dimensions_t(-1, -1));
			//	//	removeBoundaryMapEntry(visitedCells[i] + dimensions_t(1, -1));
			//	//	removeBoundaryMapEntry(visitedCells[i] + dimensions_t(-1, 1));
			//	//	removeBoundaryMapEntry(visitedCells[i] + dimensions_t(1, 1));
			//	//}
			//	//if (m_boundaryCellsMap.size() == 0)
			//	//	hasBoundaryCells = false;
			//	//else {
			//	//	initialCellDimension = m_boundaryCellsMap.begin()->second;
			//	//}
			////}
		
		

		}

		template <class VectorT>
		void LiquidRepresentation2D<VectorT>::particlesToLevelSetGrid() {
			/** Resetting level set values */
			resetLevelSetArrays();

			//vector<Vector2> *pParticlesPos = m_params.pParticlesPositions;
			//vector<int> *pParticlesTags = m_params.pParticlesTags;

			//Scalar coarseGridDx = m_params.pGridData->getScaleFactor(0, 0).x;
			///** This is the number of times that each dimension of the high-res grid is higher than the original fluid 
			//	simulation grid. This is needed, since particles of the FLIP solver are usually in the original fluid grid
			//	space and need to be transformed to the level-set grid space. */
			//Scalar subdivisionScaleFactor = pow(2, m_params.levelSetGridSubdivisions);
		
			//for (int i = 0; i < pParticlesPos->size(); i++) {
			//	//Particle is on air region
			//	if (pParticlesTags->at(i) == 0) {
			//		//Do not splat the function into the level set grid
			//		continue;
			//	}
			//	/** Particle position in world coordinates */
			//	Vector2 transformedPosition = pParticlesPos->at(i);
			//	/** Particle position in grid coordinates*/
			//	Vector2 gridPosition = pParticlesPos->at(i)/coarseGridDx;

			//	/** Level-set grid index of particle */
			//	dimensions_t levelSetGridIndex (floor(gridPosition.x*subdivisionScaleFactor),
			//									floor(gridPosition.y*subdivisionScaleFactor));

			//	/** Splat into a [-2,+2]x[-2,2] level set grid neighborhood */
			//	//Scalar R = coarseGridDx / sqrt(16);
			//	Scalar R = coarseGridDx*sqrt(2.0f)*6;
			//	Scalar particleRadius = R / 200;
			//	for (int j = -subdivisionScaleFactor; j <= subdivisionScaleFactor; j++) {
			//		for (int k = -subdivisionScaleFactor; k <= subdivisionScaleFactor; k++) {
			//			/** Checking the wub-a-lub-bub boundaaaaaaaaaries bitcheeees */
			//			if (levelSetGridIndex.x + j < 0		||	levelSetGridIndex.y + k < 0	|| 
			//				levelSetGridIndex.x + j > m_levelSetGrid.getDimensions().x - 1	||
			//				levelSetGridIndex.y + k > m_levelSetGrid.getDimensions().y - 1) {
			//				continue;
			//			}
			//		
			//			/** Grid point is in world coordinates */
			//			Vector2 levelSetNodePosition((levelSetGridIndex.x + j)*m_levelSetGridSpacing,
			//										 (levelSetGridIndex.y + k)*m_levelSetGridSpacing);

			//			/** Use Bridson's "Animating Sand as a Fluid" kernel */
			//			if((transformedPosition - levelSetNodePosition).length() > R)
			//				continue;
			//			Scalar weight = calculateLiquidKernel((transformedPosition - levelSetNodePosition).length()/R);
			//			m_levelSetGridWeights((levelSetGridIndex.x + j), (levelSetGridIndex.y + k)) += weight;
			//			m_levelSetGrid((levelSetGridIndex.x + j), (levelSetGridIndex.y + k)) += weight*particleRadius;
			//			if (m_averageParticlePositions((levelSetGridIndex.x + j), (levelSetGridIndex.y + k)) == Vector2(0.0f, 0.0f)) {
			//				m_averageParticlePositions((levelSetGridIndex.x + j), (levelSetGridIndex.y + k)) = transformedPosition*weight;
			//			} else {
			//				m_averageParticlePositions((levelSetGridIndex.x + j), (levelSetGridIndex.y + k)) += transformedPosition*weight;
			//			}
			//		}
			//	}
			//}

			///** Normalize splatted kernels from particles by dividing each grid value by its weight */
			//for (int i = 0; i < m_levelSetGrid.getDimensions().x; i++) {
			//	for (int j = 0; j < m_levelSetGrid.getDimensions().y; j++) {
			//		if (m_levelSetGridWeights(i, j) > 0.0f) {
			//			Vector2 levelSetNodePosition((i)*m_levelSetGridSpacing, (j)*m_levelSetGridSpacing);
			//		
			//			Scalar currWeight = m_levelSetGridWeights(i, j);
			//			Vector2 avgParticlePos = m_averageParticlePositions(i, j) / m_levelSetGridWeights(i, j);
			//			m_averageParticlePositions(i, j) /= m_levelSetGridWeights(i, j);

			//			Scalar levelSetF = m_levelSetGrid(i, j);
			//			Scalar finalFu = (levelSetNodePosition - m_averageParticlePositions(i, j)).length() - m_levelSetGrid(i, j);
			//			m_levelSetGrid(i, j) = (levelSetNodePosition - m_averageParticlePositions(i, j)).length() - m_levelSetGrid(i, j);
			//		}
			//	}
			//}
		}

		template <class VectorT>
		void LiquidRepresentation2D<VectorT>::updateParticleTags() {
		/*	Scalar dx = m_params.pGridData->getGridSpacing();
			vector<int> *pParticlesTags = m_params.pParticlesTags;
			for (int i = 0; i < pParticlesTags->size(); i++) {
				for (int j = 0; j < m_params.initialLineMeshes.size(); j++) {
					Vector2 transformedPosition = m_params.pParticlesPositions->at(i);
					if (isInsidePolygon(transformedPosition, m_params.initialLineMeshes[j]->getPoints())) {
						pParticlesTags->at(i) = 1;
					}
					else {
						pParticlesTags->at(i) = 0;
					}
				}
			}*/
		}

		#pragma endregion

		#pragma region PrivateFunctionalities
		
		template <class VectorT>
		void LiquidRepresentation2D<VectorT>::removeBoundaryMapEntry(const dimensions_t &boundaryIndex) {
			int currKey = m_levelSetGrid.getRawPtrIndex(boundaryIndex.x, boundaryIndex.y);
			if (m_boundaryCellsMap.find(currKey) != m_boundaryCellsMap.end()) {
				m_boundaryCellsMap.erase(m_boundaryCellsMap.find(currKey));
			}
		}

		template <class VectorT>
		void LiquidRepresentation2D<VectorT>::resetLevelSetArrays() {
			m_levelSetGrid.assign(0.0f);
			m_levelSetGridWeights.assign(0.0f);
			m_averageParticlePositions.assign(Vector2(0, 0));
		}
		#pragma endregion

		template class LiquidRepresentation2D<Vector2>;
	}

	
}