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

#include "Physics/GhostLiquidSolver2D.h"
#include "Physics/PhysicsCore.h"
#include "Rendering/GLRenderer2D.h"
#include <omp.h>

namespace Chimera {

	/************************************************************************/
	/* Public Functions                                                     */
	/************************************************************************/
	#pragma region UpdateFunctions
	bool GhostLiquidSolver::updatePoissonMatrix() {
		typedef LiquidRepresentation2D::levelSetCellType_t cellType;

		/** Update cell types first */
		updateCellTypes();

		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;

		/** First pass: setup GFM */
		for (int i = 1; i < m_pPoissonMatrix->getDimensions().x - 1; i++) {
			for (int j = 1; j < m_pPoissonMatrix->getDimensions().y - 1; j++) {
				Scalar pn, pw, ps, pe;
				/*If the cell is air our fluid, just initially set all its coefficients to the standard value. In a second
				 * pass the algorithm will fix up for boundary cells */
				if (m_cellTypes(i, j) == cellType::airCell) {
					Scalar invCoeff = 1 / m_airDensityCoeff;
					m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(i - 1, j - 1), -invCoeff, -invCoeff, invCoeff*4, -invCoeff, -invCoeff);
				}
				else if (m_cellTypes(i, j) == cellType::fluidCell) {
					Scalar invCoeff = 1 / m_liquidDensityCoeff;
					m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(i - 1, j - 1), -invCoeff, -invCoeff, invCoeff*4, -invCoeff, -invCoeff);
				}
			}
		}

		/** Second pass: updates boundary cells */
		for (int i = 1; i < m_pPoissonMatrix->getDimensions().x - 1; i++) {
			for (int j = 1; j < m_pPoissonMatrix->getDimensions().y - 1; j++) {
				if (m_pCutCells2D->isSpecialCell(i, j)) {
					updatePressureMatrixBoundaryCell(dimensions_t(i, j));
				}
			}
		}

		for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
			m_boundaryConditions[i]->updatePoissonMatrix(m_pPoissonMatrix);
		}

		if (m_pPoissonMatrix->isSingular() && m_params.getPressureSolverParams().getMethodCategory() == Krylov)
			m_pPoissonMatrix->applyCorrection(1e-3);

		m_pPoissonMatrix->updateCudaData();

		return false;

	}
	#pragma endregion UpdateFunctions

	#pragma region SimulationFunctions
	void GhostLiquidSolver::divergenceFree(Scalar dt) {
		int i, j;
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
		Scalar dy = m_pGridData->getScaleFactor(0, 0).y;

		typedef LiquidRepresentation2D::levelSetCellType_t cellType;

		#pragma omp parallel for
		for (i = 1; i < m_dimensions.x - 1; i++) {
			for (j = 1; j < m_dimensions.y - 1; j++) {
				Vector2 velocity;

				if (m_pCutCells2D->isSpecialCell(i, j)) {
					cellType currCellType = m_cellTypes(i, j);
					cellType nextCellType = m_cellTypes(i - 1, j);

					if (nextCellType == currCellType) { // Cell types match, use standard approach
						Scalar gradientCoeff = dt / (currCellType == cellType::airCell ? m_airDensityCoeff : m_liquidDensityCoeff);
						velocity.x = m_pGridData->getAuxiliaryVelocity(i, j).x - (gradientCoeff)*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i - 1, j)) / dx);
					}
					else { //Face types do not match, use GFM to properly set pressure gradient 
						Scalar distanceToMesh = distanceToLiquid(dimensions_t(i, j), dimensions_t(i - 1, j));
						//Normalize distanceToMesh relative to the grid spacing
						distanceToMesh /= dx;

						Scalar densityHat = 0;

						if (currCellType == cellType::airCell) //Other cell is fluid
							densityHat = m_airDensityCoeff*distanceToMesh + (1 - distanceToMesh)*m_liquidDensityCoeff;
						else //This cell is fluid, other cell is air
							densityHat = m_liquidDensityCoeff*distanceToMesh + (1 - distanceToMesh)*m_airDensityCoeff;

						Scalar gradientCoeff = dt / densityHat;
						velocity.x = m_pGridData->getAuxiliaryVelocity(i, j).x - (gradientCoeff)*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i - 1, j)) / dx);
						//velocity.x += gradientCoeff*m_surfaceTensionCoeff*calculateLiquidCurvatureLS(dimensions_t(i, j)) / dx;
					}

					/** Y component of velocity */
					nextCellType = m_cellTypes(i, j - 1);

					if (nextCellType == currCellType) { // Cell types match, use standard approach
						Scalar gradientCoeff = dt / (currCellType == cellType::airCell ? m_airDensityCoeff : m_liquidDensityCoeff);
						velocity.y = m_pGridData->getAuxiliaryVelocity(i, j).y - (gradientCoeff)*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i, j - 1)) / dx);
					}
					else { //Face types do not match, use GFM to properly set pressure gradient 
						Scalar distanceToMesh = distanceToLiquid(dimensions_t(i, j), dimensions_t(i, j - 1));
						//Normalize distanceToMesh relative to the grid spacing
						distanceToMesh /= dx;

						Scalar densityHat = 0;

						if (currCellType == cellType::airCell) //Other cell is fluid
							densityHat = m_airDensityCoeff*distanceToMesh + (1 - distanceToMesh)*m_liquidDensityCoeff;
						else //This cell is fluid, other cell is air
							densityHat = m_liquidDensityCoeff*distanceToMesh + (1 - distanceToMesh)*m_airDensityCoeff;

						Scalar gradientCoeff = dt / densityHat;
						velocity.y = m_pGridData->getAuxiliaryVelocity(i, j).y - (gradientCoeff)*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i, j - 1)) / dx);
						//velocity.y += gradientCoeff*m_surfaceTensionCoeff*calculateLiquidCurvatureLS(dimensions_t(i, j)) / dx;
					}
				} else if (m_cellTypes(i, j) == cellType::airCell) {
					velocity.x = m_pGridData->getAuxiliaryVelocity(i, j).x - (dt / m_airDensityCoeff)*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i - 1, j)) / dx);
					velocity.y = m_pGridData->getAuxiliaryVelocity(i, j).y - (dt / m_airDensityCoeff)*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i, j - 1)) / dy);
				}
				else if (m_cellTypes(i, j) == cellType::fluidCell) {
					velocity.x = m_pGridData->getAuxiliaryVelocity(i, j).x - (dt / m_liquidDensityCoeff)*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i - 1, j)) / dx);
					velocity.y = m_pGridData->getAuxiliaryVelocity(i, j).y - (dt / m_liquidDensityCoeff)*((m_pGridData->getPressure(i, j) - m_pGridData->getPressure(i, j - 1)) / dy);
				}
				m_pGridData->setVelocity(velocity, i, j);
			}
		}
	}

	Scalar GhostLiquidSolver::calculateFluxDivergent(int i, int j) {
		Scalar divergent = 0;

		int row = 0;
		if (m_params.getPressureSolverParams().getMethodCategory() == Krylov) {
			row = m_pPoissonMatrix->getRowIndex(i - 1, j - 1);
		}
		else {
			row = m_pPoissonMatrix->getRowIndex(i, j);
		}

		Scalar dx, dy = 0;

		dx = (m_pGridData->getAuxiliaryVelocity(i + 1, j).x - m_pGridData->getAuxiliaryVelocity(i, j).x) / m_pGridData->getScaleFactor(i, j).x;
		dy = (m_pGridData->getAuxiliaryVelocity(i, j + 1).y - m_pGridData->getAuxiliaryVelocity(i, j).y) / m_pGridData->getScaleFactor(i, j).y;
		
		divergent = dx + dy;

	/*	if (m_pCutCells2D->isSpecialCell(i, j)) {
			divergent += calculateLiquidCurvatureLS(dimensions_t(i, j))*0.01;
		}*/
		return divergent;
	}
	#pragma endregion SimulationFunctions

	#pragma region InternalFunctionalities
	void GhostLiquidSolver::updatePressureMatrixBoundaryCell(const dimensions_t &cellIndex) {
		typedef LiquidRepresentation2D::levelSetCellType_t cellType;

		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;

		/** Checking if the cell centroid is inside a liquid */
		Vector2 transformedCentroid = Vector2(cellIndex.x + 0.5, cellIndex.y + 0.5)*dx;
		
		/** Here cellType is used to identify whether the centroid of the cell is inside the fluid or not. This changes
		 ** the way that the cell is processed by GFM.*/
		cellType currCellType = m_cellTypes(cellIndex);
		
		Scalar pn = -calculatePoissonMatrixCoefficient(cellIndex, currCellType, topFace);
		Scalar pw = -calculatePoissonMatrixCoefficient(cellIndex, currCellType, leftFace);
		Scalar ps = -calculatePoissonMatrixCoefficient(cellIndex, currCellType, bottomFace);
		Scalar pe = -calculatePoissonMatrixCoefficient(cellIndex, currCellType, rightFace);

		Scalar pc = -(pn + pw + ps + pe);

		m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(cellIndex.x - 1, cellIndex.y - 1), pn, pw, pc, pe, ps);
	}

	Scalar GhostLiquidSolver::calculatePoissonMatrixCoefficient(const dimensions_t &cellIndex, 
																LiquidRepresentation2D::levelSetCellType_t currCellType, 
																faceLocation_t faceLocation) {
		Scalar pressureCoeff;
		typedef LiquidRepresentation2D::levelSetCellType_t cellType;
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;

		dimensions_t nextCellLocation = getFaceOffset(faceLocation) + cellIndex;
		if (m_pCutCells2D->isSpecialCell(nextCellLocation.x, nextCellLocation.y)) { //Next cell is also a boundary cell
			//A verification has to be done if the next cell centroid is inside/outside the liquid surface
			cellType nextCellType = m_cellTypes(nextCellLocation);
			if (nextCellType == currCellType) { // Cell types match but cell interface is a fractional face 
				Scalar faceFraction = calculateCutCellFraction(cellIndex, faceLocation);
				pressureCoeff = faceFraction * 1.0f / (currCellType == cellType::airCell ? m_airDensityCoeff : m_liquidDensityCoeff);
			}
			else { //Face types do not match, use GFM to properly set pressure gradient 
				Scalar distanceToMesh = distanceToLiquid(cellIndex, nextCellLocation);
				//Normalize distanceToMesh relative to the grid spacing
				distanceToMesh /= dx;

				if (distanceToMesh < 0) {
					//exit(1);
				}
				Scalar densityHat = 0;

				if (currCellType == cellType::airCell) //Other cell is fluid
					densityHat = m_airDensityCoeff*distanceToMesh + (1 - distanceToMesh)*m_liquidDensityCoeff;
				else //This cell is fluid, other cell is air
					densityHat = m_liquidDensityCoeff*distanceToMesh + (1 - distanceToMesh)*m_airDensityCoeff;

				pressureCoeff = 1 / densityHat;
			}
		} else if (m_cellTypes(nextCellLocation) == currCellType) { //Cell types match and the interface is a full face
			pressureCoeff = 1.0f / (currCellType == cellType::airCell ? m_airDensityCoeff : m_liquidDensityCoeff);
		} else { //Cell types do not match: use GFM to properly set pressure gradient 
			Scalar distanceToMesh = distanceToLiquid(cellIndex, nextCellLocation);

			//Normalize distanceToMesh relative to the grid spacing
			distanceToMesh /= dx;
			if (distanceToMesh < 0) {
				//exit(1);
			}
			Scalar densityHat = 0;

			if (currCellType == cellType::airCell) //Other cell is fluid
				densityHat = m_airDensityCoeff*distanceToMesh + (1 - distanceToMesh)*m_liquidDensityCoeff;
			else //This cell is fluid, other cell is air
				densityHat = m_liquidDensityCoeff*distanceToMesh + (1 - distanceToMesh)*m_airDensityCoeff;

			pressureCoeff = 1 / densityHat;
		}

		return pressureCoeff;
	}

	dimensions_t GhostLiquidSolver::getFaceOffset(faceLocation_t face) {
		switch (face) {
		case Chimera::Data::rightFace:
			return dimensions_t(1, 0);
			break;
		case Chimera::Data::bottomFace:
			return dimensions_t(0, -1);
			break;
		case Chimera::Data::leftFace:
			return dimensions_t(-1, 0);
			break;
		case Chimera::Data::topFace:
			return dimensions_t(0, 1);
			break;
		default:
			return dimensions_t(0, 0);
			break;
		}
	}

	Scalar GhostLiquidSolver::calculateCutCellFraction(const dimensions_t &cellIndex, faceLocation_t faceLocation) {
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
		Vector2 cellCentroid = Vector2(cellIndex.x + 0.5, cellIndex.y + 0.5);
		//getSpecialCellIndex works in grid coordinates
		CutFace<Vector2> currCell = m_pCutCells2D->getSpecialCell(m_pCutCells2D->getSpecialCellIndex(cellCentroid));
		Scalar faceFraction = 0.0f;
		for (int i = 0; i < currCell.m_cutEdges.size(); i++) {
			if (currCell.m_cutEdgesLocations[i] == faceLocation) {
				faceFraction += currCell.m_cutEdges[i]->getLengthFraction();
			}
		}
		return faceFraction;
	}

	Scalar GhostLiquidSolver::distanceToLiquid(const dimensions_t &cellIndex, const dimensions_t &nextCellIndex) {
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
		Vector2 cellCentroid = Vector2(cellIndex.x + 0.5, cellIndex.y + 0.5);
		Vector2 nextCellCentroid = Vector2(nextCellIndex.x + 0.5, nextCellIndex.y + 0.5);

		//getSpecialCellIndex works in grid coordinates
		int currCellIndex = m_pCutCells2D->getSpecialCellIndex(cellCentroid);
		if(currCellIndex != - 1) {
			CutFace<Vector2> currCell = m_pCutCells2D->getSpecialCell(currCellIndex);

			//Transform to world coordinates
			cellCentroid *= dx;
			nextCellCentroid *= dx;

			/* Checks if the ray traced from cellCentroid and nextCentroid intersects any geometric edges of the cut-cell/*/
			Scalar smallerDistance = FLT_MAX;
			for (int i = 0; i < currCell.m_cutEdges.size(); i++) {
				if (currCell.m_cutEdgesLocations[i] == geometryEdge) {
					Vector2 intersectionPoint;
					if (DoLinesIntersect(cellCentroid, nextCellCentroid, currCell.getEdgeInitialPoint(i), currCell.getEdgeFinalPoint(i), intersectionPoint, 1e-3)) {
						return (cellCentroid - intersectionPoint).length();
					}
				}
			}
		}
		else {
			//This is not supposed to happen
			cout << "Error: distance to liquid on wrong cut-cell index (" << cellIndex.x << ", " << cellIndex.y << ")" << endl;
		}
		
		/* If not, the ray may be intersecting the next cut-cell */
		int nextCutcellIndex = m_pCutCells2D->getSpecialCellIndex(nextCellCentroid / dx);
		if (nextCutcellIndex != -1) {
			CutFace<Vector2> currCell = m_pCutCells2D->getSpecialCell(nextCutcellIndex);
			for (int i = 0; i < currCell.m_cutEdges.size(); i++) {
				if (currCell.m_cutEdgesLocations[i] == geometryEdge) {
					Vector2 intersectionPoint;
					if (DoLinesIntersect(cellCentroid, nextCellCentroid, currCell.getEdgeInitialPoint(i), currCell.getEdgeFinalPoint(i), intersectionPoint, 1e-3)) {
						return (cellCentroid - intersectionPoint).length();
					}
				}
			}
		}
		

		/* Since numerical errors, the line that is crossing might not be detected by only verifying segments that are
		 * inside the cut-cell. The algorithm has to check all lines segments then */
		for (int k = 0; k < m_liquidMeshes.size(); k++) {
			Vector2 intersectionPoint;
			const vector<Vector2> &points = m_liquidMeshes[k]->getPoints();
			for (int i = 0; i < points.size(); i++) {
				int nextI = roundClamp<int>(i + 1, 0, points.size());
				if (DoLinesIntersect(cellCentroid, nextCellCentroid, points[i], points[nextI], intersectionPoint, 1e-2)) {
					return (cellCentroid - intersectionPoint).length();
				}
			}			
		}
		/* Error, no line intersects the geometry, user defined incorrect function arguments */
		return -1;
	}

	void GhostLiquidSolver::updateCellTypes() {
		typedef LiquidRepresentation2D::levelSetCellType_t cellType;
		m_cellTypes.assign(cellType::airCell);
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
		for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
			for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
				Vector2 transformedCellCentroid = Vector2(i + 0.5, j + 0.5)*dx;
				for (int k = 0; k < m_liquidMeshes.size(); k++) {
					if (isInsidePolygon(transformedCellCentroid, m_liquidMeshes[k]->getPoints())) {
						m_cellTypes(i, j) = cellType::fluidCell;
						break; //Not necessary to check liquid meshes anymore
					}
				}
			}
		}
	}

	Scalar GhostLiquidSolver::calculateLiquidCurvature(const dimensions_t &cellIndex) {
		return 0;/*
		for (int k = 0; k < m_liquidMeshes.size(); k++) {
			if (isInsidePolygon(nextCellCentroid, m_liquidMeshes[k]->getPoints())) {
				nextCellType = cellType::fluidCell;
			}
		}*/
	}

	Scalar GhostLiquidSolver::calculateLiquidCurvatureLS(const dimensions_t &cellIndex) {
		const LiquidRepresentation2D::params_t &params = m_pLiquidRepresentation->getParams();
		Array2D<Scalar> *pLevelSet = m_pLiquidRepresentation->getLevelSetGridPtr();
		
		int subdivisFactor = pow(2, params.levelSetGridSubdivisions);
		Scalar lsdx = m_pGridData->getScaleFactor(0, 0).x / subdivisFactor;


		int numCells = 0;
		Scalar currCurvature = 0;
		dimensions_t levelSetIndex(cellIndex.x*subdivisFactor, cellIndex.y*subdivisFactor);
		for(int i = levelSetIndex.x; i < levelSetIndex.x + subdivisFactor; i++) {
			for(int j = levelSetIndex.y; j < levelSetIndex.y + subdivisFactor; j++) {
				Scalar px = (*pLevelSet)(i + 1, j) - (*pLevelSet)(i - 1, j);
				Scalar py = (*pLevelSet)(i, j + 1) - (*pLevelSet)(i, j - 1);
				px /= 2 * lsdx;
				py /= 2 * lsdx;

				Scalar pxx = (*pLevelSet)(i + 1, j) - (*pLevelSet)(i, j) + (*pLevelSet)(i - 1, j);
				Scalar pyy = (*pLevelSet)(i, j + 1) - (*pLevelSet)(i, j) + (*pLevelSet)(i, j - 1);
				pxx /= lsdx*lsdx;
				pyy /= lsdx*lsdx;

				Scalar pxpy = (*pLevelSet)(i + 1, j + 1) - (*pLevelSet)(i + 1, j - 1) - (*pLevelSet)(i - 1, j + 1) + (*pLevelSet)(i - 1, j - 1);
				pxpy /= 4 * lsdx * lsdx;

				currCurvature += px*px*pyy - 2*px*py*pxpy + py*py*pxx;
				++numCells;
			}
		}
		currCurvature /= numCells;

		return currCurvature;
	}
	#pragma endregion InternalFunctionalities
}