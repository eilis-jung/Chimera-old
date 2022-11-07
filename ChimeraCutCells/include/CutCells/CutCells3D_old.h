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

#ifndef __CHIMERA_CUT_CELLS_3D__
#define __CHIMERA_CUT_CELLS_3D__
#pragma once


#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "CutCells/CutCells.h"
#include "CutCells/CutCells2D.h"
#include "ChimeraMesh.h"

namespace Chimera {
	using namespace Core;
	
	namespace CutCells {
		
		template <class VectorType>
		class CutCells3D : public Mesh<VectorType, Volume>, public CutCellsBase<VectorType> {

		public:

			#pragma region Constructors
			CutCells3D(HexaGrid *pHexaGrid, Scalar thinObjectSize);
			#pragma endregion

			#pragma region AccessFunctions
			/** Faces access */
			vector<CutFace<Vector3D>*> & getFaceVector(const dimensions_t &index, faceLocation_t faceLocation);
			
			/** Gets current number of allocated cells */
			int getNumberOfCells() const {
				return static_cast<int>(m_cutVoxels.size());
			}
			int getMaxNumberOfCells() const {
				return m_maxNumberOfCells;
			}
			Scalar getGridSpacing () const {
				return m_gridSpacing;
			}
			void setSolidBoundaryType(solidBoundaryType_t solidType) {
				m_solidBoundaryType = solidType;
			}
			solidBoundaryType_t getSolidBoundaryType() const {
				return m_solidBoundaryType;
			}
			void setMixedNodeInterpolationType(mixedNodeInterpolationType_t mixedNodeType) {
				m_mixedNodeInterpolationType = mixedNodeType;
			}
			mixedNodeInterpolationType_t getMixedNodeInterpolationType() const {
				return m_mixedNodeInterpolationType;
			}
			void setThinObjectVelocities(const vector<Vector3> &thinObjectVelocities);

			/** Access the cell by its linear index */
			const CutVoxel & getCutVoxel(int index) const {
				return m_cutVoxels[index];
			}
			const CutVoxel  & getCutVoxel(int i, int j, int k) const {
				return m_cutVoxels[getCutVoxelIndex(i, j, k)];
			}
			CutVoxel  & getCutVoxel(int index) {
				return m_cutVoxels[index];
			}
			bool isSpecialCell(int i, int j, int k) const {
				bool ispecialCell = false;
				for(int index = 0; index < m_isSpecialCell.size(); index++) {
					ispecialCell = ispecialCell || m_isSpecialCell[index](i, j, k);
				}
				return ispecialCell;
			}

			bool isSpecialCell(const dimensions_t &dim) const {
				bool ispecialCell = false;
				for (int index = 0; index < m_isSpecialCell.size(); index++) {
					ispecialCell = ispecialCell || m_isSpecialCell[index](dim);
				}
				return ispecialCell;
			}

			bool isBoundaryCell(int i, int j, int k) const {
				bool ispecialCell = false;
				for(int index = 0; index < m_isBoundaryCell.size(); index++) {
					ispecialCell = ispecialCell || m_isBoundaryCell[index](i, j, k);
				}
				return ispecialCell;
			}

			bool isBoundaryCell(const dimensions_t &dim) const{
				bool ispecialCell = false;
				for (int index = 0; index < m_isBoundaryCell.size(); index++) {
					ispecialCell = ispecialCell || m_isBoundaryCell[index](dim);
				}
				return ispecialCell;
			}
			/** Returns the first index associated in the cell map */
			int getCutVoxelIndex(int i, int j, int k) const {
				return m_cellToSpecialMap(i, j, k);
			}
			Array3D<int> * getCellMapPtr() {
				return &m_cellToSpecialMap;
			}
			int getCutVoxelIndex(const Vector3D & position, vector<Mesh<Vector3D>> *pMeshes);
			/** 5 neighborhood adjacency test */
			bool isAdjacentToSpecialCell(int i, int j, int k) const {
				return m_isSpecialCell.back()(i + 1, j, k)  ||
						m_isSpecialCell.back()(i - 1, j, k) ||
						m_isSpecialCell.back()(i, j + 1, k) ||
						m_isSpecialCell.back()(i, j - 1, k) ||
						m_isSpecialCell.back()(i, j, k + 1) ||
						m_isSpecialCell.back()(i, j, k - 1);
			}
			/**Divergents and pressures */
			vector<Scalar> * getDivergentsPtr() {
				return &m_specialDivergents;
			}
			vector<Scalar> * getPressuresPtr() {
				return &m_specialPressures;
			}
			Scalar getPressure(int index) const {
				return m_specialPressures[index];
			}
			Scalar getDivergent(int index) const {
				return m_specialDivergents[index];
			}
			bool useSubthinObjectInformation() const {
				return m_useSubthinObjectInformation;
			}
			bool isAdjacentCrossingsInCell(const Crossing<Vector3D> &currCrossing, dimensions_t cellLocation);

			const vector<CutSlice3D *> & getCutCellsXZ() {
				return m_cutCellsXZVec;
			}
			const vector<CutSlice3D *> & getCutCellsXY() {
				return m_cutCellsXYVec;
			}
			const vector<CutSlice3D *> & getCutCellsYZ() {
				return m_cutCellsYZVec;
			}

			vector<CutFace<Vector3D> *> getLeftFaceVector(dimensions_t currIndex) {
				return m_leftSpecialFaces(currIndex);
			}
			vector<CutFace<Vector3D> *> getBottomFaceVector(dimensions_t currIndex) {
				return m_bottomSpecialFaces(currIndex);
			}
			vector<CutFace<Vector3D> *> getBackFaceVector(dimensions_t currIndex) {
				return m_backSpecialFaces(currIndex);
			}
			#pragma endregion

			#pragma region Functionalities
			void initializeThinBounds(Rendering::PolygonSurface *pPolySurface);
			void initializeThinBounds(const vector<Rendering::PolygonSurface *> &pPolySurfaces);

			

			void dumpCellInfo(int ithSelectedCell);
			/** Initialize auxiliary structures necessary to link the node velocity field with cutcells*/
			void linkNodeVelocityField(nodeVelocityField3D_t *pNodeVelocityField);
			void initializeMixedNodeVelocities(nodeVelocityField3D_t *pNodeVelocityField);
			void interpolateVelocityFaceNodes(nodeVelocityField3D_t *pNodeVelocityField);
			void spreadFreeSlipConservative(int ithCell, nodeVelocityField3D_t *pNodeVelocityField);
			void spreadFreeSlipAverage(int ithCell, nodeVelocityField3D_t *pNodeVelocityField);
			void preprocessVelocityDataFreeSlip(nodeVelocityField3D_t *pNodeVelocityField);
			void preprocessVelocityDataNoSlip(nodeVelocityField3D_t *pNodeVelocityField);
			void preprocessVelocityData(nodeVelocityField3D_t *pNodeVelocityField);
			CutFace<Vector3D> * createFullCutFace3D(const dimensions_t &regularGridIndex, faceLocation_t faceLocation);
			void initializeStreamFunction(nodeVelocityField3D_t *pNodeVelocityField);
			#pragma endregion

			#pragma region UpdateFunctions
			void updatePoissonMatrix(PoissonMatrix *pPoissonMatrix, const FlowSolverParameters::pressureSolverParameters_t & pressureSolverParams, const vector<BoundaryCondition<Vector3> *> &boundaryConditions);
			void updateDivergents(Scalar dt);
			#pragma endregion

			#pragma region ConvertionFunctions
			Vector3D convertPointTo3D(Vector2D point, const dimensions_t &regularGridIndex, faceLocation_t facelocation) const;
			CutFace<Vector3D> * convertTo3DFace(CutFace<Vector2D> *pFace, const dimensions_t &regularGridDimensions, faceLocation_t faceLocation);
			#pragma endregion


			#pragma region FlushingFunctions
			void flushThinBounds();
			void flushCutSlices();
			void flushCellVelocities();
			void initializeCutSlices(Rendering::PolygonSurface *pPolySurface, faceLocation_t facelocation);
			#pragma endregion

			protected:
			
			#pragma region ClassMembers
			/** Grid data */
			HexaGrid *m_pGrid;
			GridData3D *m_pGridData;

			/** Cut slices used for cut-faces initialization */
			vector<CutSlice3D *> m_cutCellsXYVec;
			vector<CutSlice3D *> m_cutCellsYZVec;
			vector<CutSlice3D *> m_cutCellsXZVec;

			/** Special cells dynamic vector. We need a map to map these cells to the grid space*/
			vector<CutVoxel> m_cutVoxels;

			/** Max number of special cells. It is very important to have an upper bound for this, since GPU working 
			 ** classes will have a difficult time handling dynamically allocated vectors. */
			int m_maxNumberOfCells;

			/** Solid boundary type: Free-slip or No-slip; */
			solidBoundaryType_t m_solidBoundaryType;
			mixedNodeInterpolationType_t m_mixedNodeInterpolationType;

			/** Faces arrays. Simplifies access directly with grid space coordinates */
			Array3D<vector<CutFace<Vector3D> *>> m_leftSpecialFaces;
			Array3D<vector<CutFace<Vector3D> *>> m_bottomSpecialFaces;
			Array3D<vector<CutFace<Vector3D> *>> m_backSpecialFaces;

			/** Grid mapping from special cells to grid space */
			Array3D<int> m_cellToSpecialMap;

			/** Polygon patch mapping to grid cells */
			Array3D<vector<Rendering::MeshPatchMap *>> m_multiMeshPatchMap;

			/** Identifies if the cells of a grid are special or not. Array of vectors to avoid double initialization*/
			vector<Array3D<bool>> m_isSpecialCell;
			vector<Array3D<bool>> m_isBoundaryCell;

			/** Special grid divergents and pressures in vector format. Useful for pressure solving */
			vector<Scalar> m_specialDivergents;
			vector<Scalar> m_specialPressures;

			/** Grid spacing*/
			Scalar m_gridSpacing;

			/** ThinObject overall velocity */
			vector<Vector3> m_thinObjectVelocities;

			/** Grid Tolerance for errors */
			Scalar m_gridTolerance;

			/** Poisson Matrix */
			PoissonMatrix *m_pPoissonMatrix;

			/** Options on CutCells3D initialization and modus operandi */
			bool m_useSubthinObjectInformation;

			nodeVelocityField3D_t *m_pNodeVelocityField;

			vector<vector<CutFace<Vector3D> *>> m_mixedNodesFacesMap;
			vector<Vector3D> m_allMixedNodes;
			#pragma endregion
	
			#pragma region InitializationFunctions
			
			void initializeCutSlices(const vector<Rendering::PolygonSurface *> &pPolySurfaces, faceLocation_t facelocation);

			void initializeCellFaces();
			void initializeMultiMeshPatchMap(const vector<Rendering::PolygonSurface *> &pPolySurfaces);
			void initializeSpecialCells();
			void initializeMixedNodesMap();

			void tagSpecialCells(Rendering::PolygonSurface *pPolySurface);
			#pragma endregion

			#pragma region PrivateFunctionalities
			int getRowIndex(PoissonMatrix *pPoissonMatrix, const dimensions_t &currDim, faceLocation_t faceLocation);
			Vector3D getFaceNormal(faceLocation_t faceLocation);
			Vector3 interpolateFaceVelocity(const vector<Vector3D> &points, const vector<Mesh<Vector3D>::nodeType_t> &nodeTypes, const vector<Vector3> &velocities, faceLocation_t faceLocation, const dimensions_t &currDimensions);
			Vector3 getFaceVelocity(int currVoxelIndex, faceLocation_t faceLocation);
			Vector3 interpolateMixedNodeVelocity(nodeVelocityField3D_t *pNodeVelocityField, int currVoxelIndex, int currPointIndex);
			Vector3 interpolateMixedNodeVelocityWeighted(nodeVelocityField3D_t *pNodeVelocityField, int currVoxelIndex, int currPointIndex, bool extraDimensions = true);
			Vector3 interpolateMixedNodeFaceVelocity(nodeVelocityField3D_t *pNodeVelocityField, int currVoxelIndex, int currPointIndex);
			void addFaceMixedNodePointsToMap(CutFace<Vector3D> *pFace);
			int addNodeToAllMixedNodes(const Vector3D &currMixedNode);
			int findMixedNode(const Vector3D &currMixedNode);
			void visitFacesAndAddToCutNodes(const Vector3D &mixedNodePoint, int mixedNodeID, int currCutVoxelID, map<int, bool> &visitedVoxels, map <int, vector<CutFace<Vector3D>*>> &nodeFaceMap, map<int, vector<Vector3D>> &nodeFaceNormalsMap);
			bool addFaceToMixedNodeMap(int mixedNodeID, CutFace<Vector3D> *pFace, faceLocation_t cutFaceLocation, map <int, vector<CutFace<Vector3D>*>> &nodeFaceMap, map<int, vector<Vector3D>> &nodeFaceNormalsMap);
			int findPatchMap(dimensions_t currDimensions, const simpleFace_t &currFace, const Array3D<vector<Rendering::MeshPatchMap *>> &meshMap, Rendering::PolygonSurface *pTargetPolySurface);

			vector<Rendering::MeshPatchMap *> getListOfPossibleTrianglesCollision(const Vector3 &initialPoint, const Vector3 &finalPoint, Scalar dt);


			#pragma endregion 
		};
	}
}


#endif
