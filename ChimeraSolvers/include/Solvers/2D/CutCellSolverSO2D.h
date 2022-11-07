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

#ifndef __CHIMERA_CUTCELLSOLVERSO_2D__
#define __CHIMERA_CUTCELLSOLVERSO_2D__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraAdvection.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraCutCells.h"
#include "ChimeraMesh.h"
#include "ChimeraSolids.h"
#include "ChimeraCGALWrapper.h"
#include "Solvers/2D/CutCellSolver2D.h"

namespace Chimera {

	using namespace Core;
	using namespace Advection;
	using namespace Meshes;
	using namespace Solids;

	/** Implementation of the classic Navier-Stokes solver, for unsteady incompressible flows It uses a CutCell 
	  * formulation for rigid body objects and a new approach for treatment of thin objects. 
			Following configurations:
			Dependent variables: Pressure, Cartesian velocity components;
			Variable arrangement: Nodal (only one supported);
			Pressure Coupling: Fractional Step; */

	namespace Solvers {
			class EdgePressureGradient {
			public:
				Vector2 node1, node2, direction;
				Scalar pressure1, pressure2, gradAlongEdge;
				Scalar distToRef;
				Scalar weight;
				EdgePressureGradient() {

				}
				void set(Vector2 start, Vector2 end, Scalar p1, Scalar p2, Vector2 ref) {
					/*
					start: starting HalfFace.
					end: ending HalfFace.
					p1: pressure at HalfFace start.
					p1: pressure at HalfFace end.
					ref: the grid point to be interpolated.
					*/
					this->node1 = start;
					this->node2 = end;
					direction = (node2 - node1).normalize();
					pressure1 = p1, pressure2 = p2;
					Scalar gradX = (node2.x != node1.x) ? (p2 - p1)*direction.x / (node2.x - node1.x) : 0;
					Scalar gradY = (node2.y != node1.y) ? (p2 - p1)*direction.y / (node2.y - node1.y) : 0;
					gradAlongEdge = (p2 - p1) / (node2 - node1).length();
					distToRef = fabs((node2.y - node1.y)*ref.x - (node2.x - node1.x)*ref.y + node2.x * node1.y - node2.y * node1.x) / (node2 - node1).length();
					weight = (1.0 / distToRef) * (1.0 / distToRef);
				}
			};

			class CutCellSolverSO2D : public CutCellSolver2D {

			public:
		
			#pragma region Constructors
				CutCellSolverSO2D(const params_t &params, StructuredGrid<Vector2> *pGrid,
							const vector<BoundaryCondition<Vector2> *> &boundaryConditions = vector<BoundaryCondition<Vector2> *>(), 
							const vector<RigidObject2D<Vector2> *> &rigidObjects = vector<RigidObject2D<Vector2> *>());
			#pragma endregion 


			protected:



			#pragma region PressureProjection

			uint m_toleranceIterations;

			/** Updates thin objects Poisson Matrix */
			virtual void updatePoissonThinSolidWalls() override;

			/** Updates internal cut-cells divergence */
			virtual void updateCutCellsDivergence(Scalar dt) override;

			/** Divergent calculation, based on the finite difference stencils.*/
			virtual Scalar calculateFluxDivergent(int i, int j) override;
			FORCE_INLINE virtual void solvePressure() {
				const Array<Scalar> *pRhs;
				Array<Scalar> *pPressures;

				if (m_dimensions.z == 0) {
					pRhs = &m_pGrid->getGridData2D()->getDivergentArray();
					pPressures = (Array<Scalar> *)(&m_pGrid->getGridData2D()->getPressureArray());
				}
				else {
					pRhs = &m_pGrid->getGridData3D()->getDivergentArray();
					pPressures = (Array<Scalar> *)(&m_pGrid->getGridData3D()->getPressureArray());
				}
				if (m_params.pPoissonSolverParams->platform == PlataformGPU)
					m_pPoissonSolver->solveGPU(pRhs, pPressures);
				else {
					if (m_params.pPoissonSolverParams->solverMethod == GaussSeidelMethod) {
						GaussSeidel *pGS = dynamic_cast<GaussSeidel *>(m_pPoissonSolver);
						uint numIter = pGS->getParams().maxIterations;
						Scalar dt = PhysicsCore<Vector2>::getInstance()->getParams()->timestep;
						for (int i = 0; i < numIter; i++) {
							pGS->serialIterationForCutCells((Scalar *)pRhs->getRawDataPointer(), (Scalar *)pPressures->getRawDataPointer());
							updateDivergents(dt);
							updateCutCellsDivergence(dt);
						}
					}
					else {
						m_pPoissonSolver->solveCPU(pRhs, pPressures);
					}
				}
					

				m_linearSolverIterations = m_pPoissonSolver->getNumberIterations();
			}

			#pragma endregion

			#pragma region InternalAuxFunctions

			int getInitialOffset()
			{
				int initialOffset = 0;
				if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
					initialOffset = (m_pGrid->getDimensions().x - 2)*(m_pGrid->getDimensions().y - 2);
				}
				else {
					initialOffset = m_pGrid->getDimensions().x*m_pGrid->getDimensions().y;
				}
				return initialOffset;
			}

			dimensions_t getDimensionsForMatrix()
			{
				dimensions_t dimForMatrix;
				if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
					dimForMatrix.x = m_dimensions.x - 2;
					dimForMatrix.y = m_dimensions.y - 2;
				}
				else {
					dimForMatrix = m_dimensions;
				}
				return dimForMatrix;
			}

			dimensions_t getCellDimensionsForMatrix(int ind)
			{
				/* ============================================================
				* This is only for getting matrix-related dims.
				* ============================================================ */
				int i, j;
				dimensions_t dimForMatrix = getDimensionsForMatrix();
				j = ind / dimForMatrix.x;
				i = ind % dimForMatrix.x;
				return dimensions_t(i + 1, j + 1);
			}

			dimensions_t getCellDimensions(int ind)
			{
				/* ============================================================
				* This is for getting non-matrix-related dims.
				* ============================================================ */
				int i, j;
				j = ind / m_dimensions.x;
				i = ind % m_dimensions.x;
				return dimensions_t(i, j);
			}

			HalfFace<Vector2> * getNeighborCutCell(const dimensions_t &currDim, halfEdgeLocation_t edgeLocation)
			{
				/* ============================================================
				* Only works for a regular cell with a cut-cell neighbour,
				* thus before you call this func, you have to check if its neighbour
				* is a cut-cell.
				* ============================================================ */
				int res = -1;
				dimensions_t nbDim = getNeighborCellDim(currDim, edgeLocation);

				if (nbDim.x >= m_pGridData->getDimensions().x || nbDim.x <= 0 || nbDim.y >= m_pGridData->getDimensions().y || nbDim.y <= 0)
					return nullptr;

				if (!m_pCutCells->isCutCell(nbDim))
					return nullptr;
				
				uint numFacesConn;
				vector<Edge<Vector2> *> pCurrEdges;
				Edge<Vector2> * pCurrEdge;
				switch (edgeLocation)
				{
				case rightHalfEdge:
					pCurrEdges = m_pCutCells->getEdgeVector(dimensions_t(currDim.x + 1, currDim.y), Meshes::yAlignedEdge);
					if (pCurrEdges.size() == 1) {
						pCurrEdge = pCurrEdges[0];
						uint nbCellInd = pCurrEdge->getConnectedHalfFaces()[0]->getID() == -1 ? pCurrEdge->getConnectedHalfFaces()[1]->getID() : pCurrEdge->getConnectedHalfFaces()[0]->getID();
						return &m_pCutCells->getCutCell(nbCellInd);
					}
					break;
				case bottomHalfEdge:
					pCurrEdges = m_pCutCells->getEdgeVector(currDim, Meshes::xAlignedEdge);
					if (pCurrEdges.size() == 1) {
						pCurrEdge = pCurrEdges[0];
						uint nbCellInd = pCurrEdge->getConnectedHalfFaces()[0]->getID() == -1 ? pCurrEdge->getConnectedHalfFaces()[1]->getID() : pCurrEdge->getConnectedHalfFaces()[0]->getID();
						return &m_pCutCells->getCutCell(nbCellInd);
					}
					break;
				case leftHalfEdge:
					pCurrEdges = m_pCutCells->getEdgeVector(currDim, Meshes::yAlignedEdge);
					if (pCurrEdges.size() == 1) {
						pCurrEdge = pCurrEdges[0];
						uint nbCellInd = pCurrEdge->getConnectedHalfFaces()[0]->getID() == -1 ? pCurrEdge->getConnectedHalfFaces()[1]->getID() : pCurrEdge->getConnectedHalfFaces()[0]->getID();
						return &m_pCutCells->getCutCell(nbCellInd);
					}
					break;
				case topHalfEdge:
					pCurrEdges = m_pCutCells->getEdgeVector(dimensions_t(currDim.x, currDim.y + 1), Meshes::xAlignedEdge);
					if (pCurrEdges.size() == 1) {
						pCurrEdge = pCurrEdges[0];
						uint nbCellInd = pCurrEdge->getConnectedHalfFaces()[0]->getID() == -1 ? pCurrEdge->getConnectedHalfFaces()[1]->getID() : pCurrEdge->getConnectedHalfFaces()[0]->getID();
						return &m_pCutCells->getCutCell(nbCellInd);
					}
					break;
				default:
					break;
				}
				return nullptr;
			}
			
			int getCellIndexForMatrix(const dimensions_t &currDim)
			{

				dimensions_t dimForMatrix = getDimensionsForMatrix();
				return  currDim.y*dimForMatrix.x + currDim.x;
			}
			
			int getCellIndex(const dimensions_t &currDim)
			{
				return  currDim.y*m_dimensions.x + currDim.x;
			}

			int getNeighborCellIndex(const dimensions_t &currDim, halfEdgeLocation_t edgeLocation)
			{
				switch (edgeLocation) {
				case rightHalfEdge:
					return getCellIndex(dimensions_t(currDim.x + 1, currDim.y));
					break;
				case bottomHalfEdge:
					return getCellIndex(dimensions_t(currDim.x, currDim.y - 1));
					break;
				case leftHalfEdge:
					return getCellIndex(dimensions_t(currDim.x - 1, currDim.y));
					break;
				case topHalfEdge:
					return getCellIndex(dimensions_t(currDim.x, currDim.y + 1));
					break;
				default:
					return -1;
					break;
				}
			}

			dimensions_t getNeighborCellDim(const dimensions_t &currDim, halfEdgeLocation_t edgeLocation)
			{
				switch (edgeLocation) {
				case rightHalfEdge:
					return dimensions_t(currDim.x + 1, currDim.y);
					break;
				case bottomHalfEdge:
					return dimensions_t(currDim.x, currDim.y - 1);
					break;
				case leftHalfEdge:
					return dimensions_t(currDim.x - 1, currDim.y);
					break;
				case topHalfEdge:
					return dimensions_t(currDim.x, currDim.y + 1);
					break;
				default:
					return dimensions_t(0, 0);
					break;
				}
			}

			int getNeighborCellIndexForMatrix(const dimensions_t &currDim, halfEdgeLocation_t edgeLocation)
			{
				switch (edgeLocation) {
				case rightHalfEdge:
					return getCellIndexForMatrix(dimensions_t(currDim.x + 1, currDim.y));
					break;
				case bottomHalfEdge:
					return getCellIndexForMatrix(dimensions_t(currDim.x, currDim.y - 1));
					break;
				case leftHalfEdge:
					return getCellIndexForMatrix(dimensions_t(currDim.x - 1, currDim.y));
					break;
				case topHalfEdge:
					return getCellIndexForMatrix(dimensions_t(currDim.x, currDim.y + 1));
					break;
				default:
					return -1;
					break;
				}
			}
			
			bool isInObject(dimensions_t cellLocation) {
				/* Shoot a ray from current location to 2 opposite directions
				* If the ray intersects with boundary on both sides, then
				* current point is inside the obj.
				*/
				if (m_pCutCells->isCutCellAt(cellLocation.x, cellLocation.y))
					return false;
				else
				{
					int i;
					vector<int> its;
					vector<int>::iterator iter;
					for (i = 1; i < m_pGrid->getDimensions().x - 1; i++)
						if (m_pCutCells->isCutCellAt(i, cellLocation.y))
							its.push_back(i);
					if (its.size() < 2) // This is impossible!!!
						return false;
					iter = its.begin();
					iter++;
					for (; iter < its.end(); iter++)
					{
						if ((*iter - cellLocation.x) * (its[0] - cellLocation.x) < 0)
							return true;
					}
				}
				return false;
			}

			bool isNeighboringCutCell(dimensions_t dim) {
				return (
					m_pCutCells->isCutCell(getNeighborCellDim(dim, rightHalfEdge)) ||
					m_pCutCells->isCutCell(getNeighborCellDim(dim, topHalfEdge)) ||
					m_pCutCells->isCutCell(getNeighborCellDim(dim, bottomHalfEdge)) ||
					m_pCutCells->isCutCell(getNeighborCellDim(dim, leftHalfEdge))
					);
			}

			vector<dimensions_t> getDimsOf4Neighbors(dimensions_t cellDim) {
				vector<dimensions_t> res;
				res.push_back(dimensions_t(cellDim.x - 1, cellDim.y + 1));
				res.push_back(dimensions_t(cellDim.x, cellDim.y + 1));
				res.push_back(dimensions_t(cellDim.x - 1, cellDim.y));
				res.push_back(cellDim);
				return res;
			}

			Vector2 getProjectionOfPointOntoLine(Vector2 pt, Vector2 v1, Vector2 v2) {
				Vector2 dir1 = v2 - v1;
				Vector2 dir2 = pt - v1;
				Scalar valDp = dir1.dot(dir2);
				Scalar lenLine1 = dir1.length();
				Scalar lenLine2 = dir2.length();
				Scalar cos = valDp / (lenLine1 * lenLine2);
				Scalar projLenOfLine = cos * lenLine2;
				Vector2 res =  v1 + Vector2(projLenOfLine * dir1.x / lenLine1, projLenOfLine * dir1.y / lenLine1);
				return res;
			}

			Vector2 leastSquareInterpolationFor2DVector(vector<Vector2> &values, vector<Vector2> &locations, Vector2 targetLocation) {
				int n = values.size();
				Vector2 res;
				Eigen::MatrixXd X(n, 2), W(n, n), YX(n, 1), YY(n, 1);
				Eigen::RowVectorXd target(2);
				target << targetLocation.x, targetLocation.y;

				X.setZero();
				W.setZero();
				for (int i = 0; i < n; i++) {
					W(i, i) = 1.0 / (locations[i] - targetLocation).length2();
					X(i, 0) = 1;
					X(i, 0) = locations[i].x;
					X(i, 1) = locations[i].y;
					YX(i, 0) = values[i].x;
					YY(i, 0) = values[i].y;
				}

				Eigen::MatrixXd LHSInv = (X.transpose() * W * X).inverse();
				Eigen::MatrixXd RHSX = X.transpose() * W * YX;
				Eigen::MatrixXd RHSY = X.transpose() * W * YY;
				Eigen::MatrixXd BX = LHSInv * RHSX;
				Eigen::MatrixXd BY = LHSInv * RHSY;

				Scalar gradX = (target * BX)(0, 0);
				Scalar gradY = (target * BY)(0, 0);
				res = (Vector2(gradX, gradY));
				return res;
			}

			Vector2 MVCInterpolationFor2DVector(vector<Vector2> &values, vector<Vector2> &locations, Vector2 targetLocation) {
				Vector2 res(0, 0);
				vector<std::pair<Vector2, unsigned> > t;
				for (int i = 0; i < locations.size(); i++) {
					t.push_back(std::make_pair(locations[i], i));
				}
				CGALWrapper::Triangulator<Vector2> tri(t);
				auto coordinates = tri.getBarycentricCoordinates(targetLocation);
				for (auto & coord : coordinates) {
					res += (values[coord.second]) * coord.first;
				}
				return res;
			}
			
			#pragma endregion



			#pragma region ExactPressureGradientInterpolation

			vector<Vector2> m_cutCellsPressureGradients;

			void initializePressureGradients();

			// set pressure gradient on regular & cut-cell nodes to zero.
			void clearPressureGradientOnNodes();

			// get pressure gradient for ONE grid cell
			Vector2 getPressureGradient(dimensions_t cellDim);

			// get pressure gradient for ONE cut-cell
			Vector2 getPressureGradient(const HalfFace<Vector2> *pCurrCell);

			// update pressure gradient on ALL grid nodes.
			void updatePressureGradientOnGridNodes();

			void updatePressureGradientOnGridNodes2();

			// update pressure gradient for ALL cut-cells.
			void updatePressureGradientsForCutCells();

			/** Calculates pressure gradient with a given cut-cell edge*/
			Vector2 calculateEdgePressureGradient(Edge<Vector2> *pEdge);

			FORCE_INLINE Vector2 getCutCellPressureGradient(uint cellID) {
				return m_cutCellsPressureGradients[cellID];
			}
			FORCE_INLINE vector<Vector2> * getPressuresGradientVectorPtr() {
				return &m_cutCellsPressureGradients;
			}




			#pragma endregion

			#pragma region Misc
			// Get gradient of pressure for cut-cells
			//Vector2 getPressureGradient(const HalfFace<Vector2> *pCurrCell);

			// Get gradient of pressure for regular grid cells
			Vector2 getPressureGradientOld(dimensions_t cellDimensions);
			
			// For cutcells
			void updateLeastSquare(const HalfFace<Vector2> *pNbCell, Eigen::Matrix2d & LSlhs, Eigen::Vector2d & LSrhs, Vector2 centroid, Scalar currPressure);

			// For regular grid cells
			void updateLeastSquare(dimensions_t dim, Eigen::Matrix2d & LSlhs, Eigen::Vector2d & LSrhs, Vector2 centroid, Scalar currPressure);
			#pragma endregion


			#pragma region Triangulation
			//CGALWrapper::Triangulator<Vector2> m_triangulator;

			//vector<pair<Vector2, dimensions_t> > m_referencePoints;

			//vector<dimensions_t> inexactRegularCells;

			//void initializeTriangulator();

			//vector<pair<Vector2, dimensions_t> > getExpandedBoundary();

			//vector<pair<Vector2, dimensions_t> > getShrinkedBoundary();
			#pragma endregion

			#pragma region ExactInterpolation
			//Scalar getCutCellPressureExact(int index);

			//void interpolateOnCutCellCenter();
			#pragma endregion
		};
	}
	

}

#endif