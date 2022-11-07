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

#ifndef __CHIMERA_STRUCTURED_GRID__
#define __CHIMERA_STRUCTURED_GRID__
#pragma  once

/************************************************************************/
/* Core                                                                 */
/************************************************************************/
#include "ChimeraCore.h"

/************************************************************************/
/* Chimera Data                                                         */
/************************************************************************/
#include "Grids/Grid.h"
#include "Grids/GridData2D.h"
#include "Grids/GridData3D.h"


using namespace std;

namespace Chimera {

	using namespace Core;
	namespace Grids {

		template<class VectorT>
		class StructuredGrid : public Grid<VectorT> {

		public:

			/************************************************************************/
			/* Structs                                                              */
			/************************************************************************/
			typedef enum gridPlanes_t {
				XY_Plane,
				XZ_Plane,
				YZ_Plane
			} gridPlanes_t;

		protected:

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			/** Grid name */
			string m_gridName;

			/** Grid type */
			string m_gridType;

			/** Grid Dimensions */
			dimensions_t m_dimensions;

			/** Grid origin */
			VectorT m_gridOrigin;

			/** Central grid data structure. Here all variables - velocity, pressures, densities, etc - are stored in 
			** one central structure. This is after grid loading and it depends on m_pGridPoints.
			** It also contains the grid metrics, which are set by initializeGridMetrics function. */
			GridData<VectorT> *m_pGridData;

			/** Solid grid markers - used for specifying solid objects immersed on the scene.
			** The map is used for isSolidCell() function. It searches for a entry in the boundary map. 
			** Calculation is the same calculation done in gridData structure to calculate a linearized 
			** grid index: index = y*dimX + x. */
			bool *m_pSolidMarkers;

			/** Boundary grid markers - used for specifying solid boundaries or chimera boundaries. */
			bool *m_pBoundaryMarkers;

			/** Pressure Boundary map: used for chimera phase to mark pressures boundary cell indices on the background 
			** grid. It is a map to improve memory consumption and to avoid duplicated grid indices. */
			map<int, dimensions_t> m_pressureBoundaryCellsIndices;

			/** Velocity Boundary map: used for chimera phase to mark velocity boundary cell indices on the background 
			** grid. It is a map to improve memory consumption and to avoid duplicated grid indices. */
			map<int, dimensions_t> m_velocityBoundaryCellsIndices;

			/** If the grid is loaded in a periodic fashion. */
			bool m_periodicBCs;

			/** If the grid has a singularity point - 
			bool m_hasSingularity/

			/** If it belongs to a multiple grid simulation and its one of the inners (sub) grid. */
			bool m_subGrid;

			/** Grid position and velocity */
			VectorT m_gridPosition;
			VectorT m_gridVelocity;

			/************************************************************************/
			/* Grid metrics                                                         */
			/************************************************************************/
			/** Initialize grid data metrics. 
			** It calculates the metrics matrix approximating the derivatives using a discrete 
			** differencing technique. Regardless of the scheme, the list of that must be 
			** initialized are: 
			- scaleFactors
			- transformation matrices
			- bases (xi & eta (tal))
			- normals (xi & eta (tal)) */
			virtual void initializeGridMetrics() = 0;

		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			StructuredGrid() {
				m_periodicBCs = m_subGrid = false;
				m_pBoundaryMarkers = NULL;
			}

			virtual ~StructuredGrid() {

			}

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			/** Grid Data*/

			/** Attempts a dynamic cast to the grid data pointer. If fails, returns null.*/
			FORCE_INLINE GridData<VectorT> * getGridData() {
				return m_pGridData;
			}

			/** Attempts a dynamic cast to the grid data pointer. If fails, returns null.*/
			FORCE_INLINE GridData2D * getGridData2D() const {
				return dynamic_cast<GridData2D *>(m_pGridData);
			}
			/** Attempts a dynamic cast to the grid data pointer. If fails, returns null.*/
			FORCE_INLINE GridData3D * getGridData3D() const {
				return dynamic_cast<GridData3D *>(m_pGridData);
			}

			/** Origin */
			VectorT getGridOrigin() const {
				return m_gridOrigin;
			}

			/** Properties */
			bool isPeriodic() const {
				return m_periodicBCs;
			}

			/** Name and type */
			void setGridName(const string &gridName) {
				m_gridName = gridName;
			}
			const string & getGridName() const {
				return m_gridName;
			}

			void setGridType(const string &gridType) {
				m_gridType = gridType;
			}
			const string & getGridType() const {
				return m_gridType;
			}

			/** Position and velocity */
			void setPosition(const VectorT &gridPos) {
				m_gridPosition = gridPos;
			}
			const VectorT & getPosition() const {
				return m_gridPosition;
			}

			void setVelocity(const VectorT &gridVel) {
				m_gridVelocity = gridVel;
			}
			const VectorT & getVelocity() const {
				return m_gridVelocity;
			}

			void rotate(Scalar angle);

			/** Dimensions */
			const dimensions_t & getDimensions() const {
				return m_dimensions;
			}

			/** Pointers */
			/*bool * getSolidMarkers() const {
				return m_pSolidMarkers;
			}

			bool * getBoundaryMarkers() const {
				return m_pBoundaryMarkers;
			}*/

			bool isSubGrid() const {
				return m_subGrid;
			}

			/** Utils */
			FORCE_INLINE bool isInsideGridRange(int i, int j) const { 
				if(i < 0 || i > m_dimensions.x -1 || j < 0 || j > m_dimensions.y - 1)
					return false;
				return true;
			}
			/************************************************************************/
			/* Solid cells functions                                                */
			/************************************************************************/
			bool isSolidCell(int i, int j) const { 
				if(i < 0 || i > m_dimensions.x -1 || j < 0 || j > m_dimensions.y - 1)
					return false;

				return m_pSolidMarkers[j*m_dimensions.x + i]; 
			}

			bool isSolidCell(int i, int j, int k) const {
				return m_pSolidMarkers[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i];
			}

			void setSolidCell(bool solid, int i, int j) {
				m_pSolidMarkers[j*m_dimensions.x + i] = solid;
			}

			void setSolidCell(bool solid, int i, int j, int k) {
				m_pSolidMarkers[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i] = solid;
			}

			//5 point adjacency
			inline bool isAdjacentToWall(int i, int j) const {
				return isSolidCell(i + 1, j) ||
					isSolidCell(i, j + 1) ||
					isSolidCell(i - 1, j) ||
					isSolidCell(i, j - 1);
			}
			//9 point adjacency
			inline bool isAdjacentToWallWide(int i, int j) const {
				return isSolidCell(i + 1, j) ||
					isSolidCell(i - 1, j - 1) ||
					isSolidCell(i + 1, j - 1) ||
					isSolidCell(i + 1, j + 1) ||
					isSolidCell(i - 1, j + 1) ;
			}


			/************************************************************************/
			/* Boundary cell functions			                                    */
			/************************************************************************/
			/** 2D Boundary cell verification: */
			bool isBoundaryCell(int i, int j) const {
				return m_pBoundaryMarkers[j*m_dimensions.x + i];
			}

			bool isBoundaryCell(int i, int j, int k) const {
				return m_pBoundaryMarkers[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i];
			}


			void setBoundaryCell(bool solid, int i, int j) {
				m_pBoundaryMarkers[j*m_dimensions.x + i] = solid;
			}

			void setBoundaryCell(bool solid, int i, int j, int k) {
				m_pBoundaryMarkers[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i] = solid;
			}

			bool * getBoundaryCells() {
				return m_pBoundaryMarkers;
			}

			//5 point adjacency
			inline bool isAdjacentToBoundaryCell(int i, int j) const {
				return isBoundaryCell(i + 1, j) ||
					isBoundaryCell(i, j + 1) ||
					isBoundaryCell(i - 1, j) ||
					isBoundaryCell(i, j - 1);
			}

			//5 point adjacency
			inline bool isAdjacentToSolidCell(int i, int j) const {
				return isSolidCell(i + 1, j) ||
					isSolidCell(i, j + 1) ||
					isSolidCell(i - 1, j) ||
					isSolidCell(i, j - 1);
			}

			inline int getAdjacentBoundaryCells(int i, int j) const {
				int numAdjacentCells = 0;
				numAdjacentCells = isBoundaryCell(i + 1, j) || isBoundaryCell(i - 1, j);
				numAdjacentCells += isBoundaryCell(i, j + 1) || isBoundaryCell(i, j - 1);
				return numAdjacentCells;
			}
			//9 point adjacency
			inline bool isAdjacentToBoundaryCellWide(int i, int j) const {
				return isAdjacentToBoundaryCell(i, j) ||
					isBoundaryCell(i - 1, j - 1) ||
					isBoundaryCell(i + 1, j - 1) ||
					isBoundaryCell(i + 1, j + 1) ||
					isBoundaryCell(i - 1, j + 1) ;
			}

			inline bool isAdjacentToSolidCellWide(int i, int j) const {
				return isAdjacentToSolidCell(i, j) ||
					isSolidCell(i - 1, j - 1) ||
					isSolidCell(i + 1, j - 1) ||
					isSolidCell(i + 1, j + 1) ||
					isSolidCell(i - 1, j + 1) ;
			}

			//Map markers
			int setPressureBoundaryCell(const dimensions_t &boundaryIndex) {
				int mapID = 0;

				if(boundaryIndex.z == 0)
					mapID = boundaryIndex.y*m_pGridData->getDimensions().x + boundaryIndex.x;
				else
					mapID = boundaryIndex.z*m_pGridData->getDimensions().x*m_pGridData->getDimensions().y + 
					boundaryIndex.y*m_pGridData->getDimensions().x + boundaryIndex.x;

				m_pressureBoundaryCellsIndices[mapID] = boundaryIndex;

				return mapID;
			}

			//Map markers
			int setVelocityBoundaryCell(const dimensions_t &boundaryIndex) {
				int mapID = 0;

				if(boundaryIndex.z == 0)
					mapID = boundaryIndex.y*m_pGridData->getDimensions().x + boundaryIndex.x;
				else
					mapID = boundaryIndex.z*m_pGridData->getDimensions().x*m_pGridData->getDimensions().y + 
					boundaryIndex.y*m_pGridData->getDimensions().x + boundaryIndex.x;

				m_velocityBoundaryCellsIndices[mapID] = boundaryIndex;

				return mapID;
			}

			map<int, dimensions_t> * getPressureBoundaryMap() {
				return &m_pressureBoundaryCellsIndices;
			}

			map<int, dimensions_t> * getVelocityBoundaryMap() {
				return &m_velocityBoundaryCellsIndices;
			}

			/************************************************************************/
			/* Grid I/O                                                             */
			/************************************************************************/
			virtual void loadGrid(const string &gridFilename) { gridFilename;};
			virtual void loadPeriodicGrid(const string &gridFilename) { gridFilename;};
			virtual void exportToFile(const string &gridExportFilename) { gridExportFilename;};

		};
	}
}
#endif