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

#include "Solvers/2D/GhostLiquidSolver2D.h"
#include <omp.h>

namespace Chimera {

	/************************************************************************/
	/* Public Functions                                                     */
	/************************************************************************/
	#pragma region UpdateFunctions
	bool GhostLiquidSolver::updatePoissonMatrix() {
		typedef LiquidRepresentation2D<Vector2>::levelSetCellType_t cellType;

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
					//m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(i - 1, j - 1), -invCoeff, -invCoeff, invCoeff*4, -invCoeff, -invCoeff);
					m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(i - 1, j - 1), 0, 0, 1, 0, 0);
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
				if (m_boundaryCells(i, j)) {
					updatePressureMatrixBoundaryCell(dimensions_t(i, j));
				}
			}
		}

		for (unsigned int i = 0; i < m_boundaryConditions.size(); i++) {
			m_boundaryConditions[i]->updatePoissonMatrix(m_pPoissonMatrix);
		}

		if (m_pPoissonMatrix->isSingular() && m_params.pPoissonSolverParams->solverCategory == Krylov)
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

		typedef LiquidRepresentation2D<Vector2>::levelSetCellType_t cellType;

		#pragma omp parallel for
		for (i = 1; i < m_dimensions.x - 1; i++) {
			for (j = 1; j < m_dimensions.y - 1; j++) {
				Vector2 velocity;

				if (m_boundaryCells(i, j)) {
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

		typedef LiquidRepresentation2D<Vector2>::levelSetCellType_t cellType;
		if (m_cellTypes(i, j) == cellType::airCell) {
			return 0;
		}

		int row = 0;
		if (m_params.pPoissonSolverParams->solverCategory == Krylov) {
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
	void GhostLiquidSolver::applyForces(Scalar dt) {
		ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
		auto particlesTags = pParticleBasedAdv->getParticlesData()->getIntegerBasedAttribute("liquid");
		auto particlesPositions = pParticleBasedAdv->getParticlesData()->getPositions();
		for (int i = 0; i < particlesPositions.size(); i++) {
			if (particlesTags[i] == 1) {
				pParticleBasedAdv->getParticlesData()->getVelocities()[i].y -= 9.81*dt/m_pGridData->getGridSpacing();
			}
		}
	}
	void GhostLiquidSolver::update(Scalar dt) {
		m_numIterations++;
		m_totalSimulationTimer.start();
		updatePoissonMatrix();

		ParticleBasedAdvection<Vector2, Array2D> *pParticleBasedAdv = dynamic_cast<ParticleBasedAdvection<Vector2, Array2D> *>(m_pAdvection);
		if (pParticleBasedAdv == nullptr) {
			throw(exception("CutCellSolver2D: only particle based advection methods are supported now"));
		}

		if (PhysicsCore<Vector2>::getInstance()->getElapsedTime() < dt) {
			applyForces(dt);
			pParticleBasedAdv->updateGridAttributes();
			enforceBoundaryConditions();
			updateDivergents(dt);
			enforceBoundaryConditions();
			solvePressure();
			enforceBoundaryConditions();
			project(dt);
			enforceBoundaryConditions();

			/*if (m_pParticleBasedAdvection) {
			m_pParticleBasedAdvection->getParticlesSampler()->interpolateVelocities(m_pVelocityInterpolant, m_pParticleBasedAdvection->getParticlesData());
			}*/
		}

		applyForces(dt);
		enforceBoundaryConditions();

		m_advectionTimer.start();

		enforceBoundaryConditions();
		CubicStreamfunctionInterpolant2D<Vector2> *pCInterpolant = dynamic_cast<CubicStreamfunctionInterpolant2D<Vector2> *>(pParticleBasedAdv->getParticleBasedIntegrator()->getInterpolant());
		if (pCInterpolant)
			pCInterpolant->computeStreamfunctions();
		else {
			BilinearStreamfunctionInterpolant2D<Vector2> *pSInterpolant = dynamic_cast<BilinearStreamfunctionInterpolant2D<Vector2> *>(pParticleBasedAdv->getParticleBasedIntegrator()->getInterpolant());
			if (pSInterpolant)
				pSInterpolant->computeStreamfunctions();
		}

		//printInterpolationDivergence(10);

		enforceBoundaryConditions();
		pParticleBasedAdv->updatePositions(dt);
		pParticleBasedAdv->updateGridAttributes();
		if (m_pLiquidRepresentation) {
			m_pLiquidRepresentation->updateMeshes();
		}

		m_advectionTimer.stop();
		m_advectionTime = m_advectionTimer.secondsElapsed();
		enforceBoundaryConditions();

		/** Solve pressure */
		m_solvePressureTimer.start();
		updateDivergents(dt);
		solvePressure();

		enforceBoundaryConditions();
		m_solvePressureTimer.stop();
		m_solvePressureTime = m_solvePressureTimer.secondsElapsed();

		/** Project velocity */
		m_projectionTimer.start();
		project(dt);
		m_projectionTimer.stop();
		m_projectionTime = m_projectionTimer.secondsElapsed();
		enforceBoundaryConditions();

		enforceBoundaryConditions();

		if (pParticleBasedAdv) {
			pParticleBasedAdv->updateParticleAttributes();
			m_pGridData->getDensityBuffer().swapBuffers();
			m_pGridData->getTemperatureBuffer().swapBuffers();
		}


		enforceBoundaryConditions();

		updateKineticEnergy();

		m_totalSimulationTimer.stop();
		m_totalSimulationTime = m_totalSimulationTimer.secondsElapsed();

	}
	#pragma endregion SimulationFunctions

	#pragma region InternalFunctionalities
	void GhostLiquidSolver::updatePressureMatrixBoundaryCell(const dimensions_t &cellIndex) {
		typedef LiquidRepresentation2D<Vector2>::levelSetCellType_t cellType;

		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;

		/** Checking if the cell centroid is inside a liquid */
		Vector2 transformedCentroid = Vector2(cellIndex.x + 0.5, cellIndex.y + 0.5)*dx;
		
		/** Here cellType is used to identify whether the centroid of the cell is inside the fluid or not. This changes
		 ** the way that the cell is processed by GFM.*/
		cellType currCellType = m_cellTypes(cellIndex);
		
		Scalar pn = -calculatePoissonMatrixCoefficient(cellIndex, currCellType, topHalfEdge);
		Scalar pw = -calculatePoissonMatrixCoefficient(cellIndex, currCellType, leftHalfEdge);
		Scalar ps = -calculatePoissonMatrixCoefficient(cellIndex, currCellType, bottomHalfEdge);
		Scalar pe = -calculatePoissonMatrixCoefficient(cellIndex, currCellType, rightHalfEdge);

		Scalar pc = -(pn + pw + ps + pe);

		m_pPoissonMatrix->setRow(m_pPoissonMatrix->getRowIndex(cellIndex.x - 1, cellIndex.y - 1), pn, pw, pc, pe, ps);

		//Fixing neighboring cells for symmetry
		//North cell
		int rowIndex = m_pPoissonMatrix->getRowIndex(cellIndex.x - 1, cellIndex.y);
		if (pn != m_pPoissonMatrix->getSouthValue(rowIndex)) {
			Scalar psOld = m_pPoissonMatrix->getSouthValue(rowIndex); //Old value
			m_pPoissonMatrix->setSouthValue(rowIndex, pn);
			//To avoid recaculation, just sum the difference
			m_pPoissonMatrix->setCentralValue(rowIndex, (psOld - pn) + m_pPoissonMatrix->getCentralValue(rowIndex));
		}
		//South cell
		rowIndex = m_pPoissonMatrix->getRowIndex(cellIndex.x - 1, cellIndex.y - 2);
		if (ps != m_pPoissonMatrix->getNorthValue(rowIndex)) {
			Scalar pnOld = m_pPoissonMatrix->getNorthValue(rowIndex); //Old value
			m_pPoissonMatrix->setNorthValue(rowIndex, ps);
			//To avoid recaculation, just sum the difference
			m_pPoissonMatrix->setCentralValue(rowIndex, (pnOld - ps) + m_pPoissonMatrix->getCentralValue(rowIndex));
		}
		//West Cell
		rowIndex = m_pPoissonMatrix->getRowIndex(cellIndex.x - 2, cellIndex.y - 1);
		if (pw != m_pPoissonMatrix->getEastValue(rowIndex)) {
			Scalar peOld = m_pPoissonMatrix->getEastValue(rowIndex); //Old value
			m_pPoissonMatrix->setEastValue(rowIndex, pw);
			//To avoid recaculation, just sum the difference
			m_pPoissonMatrix->setCentralValue(rowIndex, (peOld - pw) + m_pPoissonMatrix->getCentralValue(rowIndex));
		}
		//East cell
		rowIndex = m_pPoissonMatrix->getRowIndex(cellIndex.x, cellIndex.y - 1);
		if (pe != m_pPoissonMatrix->getWestValue(rowIndex)) {
			Scalar pwOld = m_pPoissonMatrix->getWestValue(rowIndex); //Old value
			m_pPoissonMatrix->setWestValue(rowIndex, pe);
			//To avoid recaculation, just sum the difference
			m_pPoissonMatrix->setCentralValue(rowIndex, (pwOld - pe) + m_pPoissonMatrix->getCentralValue(rowIndex));
		}
	}

	Scalar GhostLiquidSolver::calculatePoissonMatrixCoefficient(const dimensions_t &cellIndex, 
																LiquidRepresentation2D<Vector2>::levelSetCellType_t currCellType, 
																halfEdgeLocation_t heLocation) {
		Scalar pressureCoeff;
		typedef LiquidRepresentation2D<Vector2>::levelSetCellType_t cellType;
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;

		dimensions_t nextCellLocation = getFaceOffset(heLocation) + cellIndex;
		if (m_boundaryCells(nextCellLocation.x, nextCellLocation.y)) { //Next cell is also a boundary cell
			//A verification has to be done if the next cell centroid is inside/outside the liquid surface
			cellType nextCellType = m_cellTypes(nextCellLocation);
			if (nextCellType == currCellType) { // Cell types match but cell interface is a fractional face 
				Scalar faceFraction = calculateCellFraction(cellIndex, heLocation);
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

	dimensions_t GhostLiquidSolver::getFaceOffset(halfEdgeLocation_t heLocation) {
		switch (heLocation) {
			case rightHalfEdge:
				return dimensions_t(1, 0);
			break;
			case bottomHalfEdge:
				return dimensions_t(0, -1);
			break;
			case leftHalfEdge:
				return dimensions_t(-1, 0);
			break;
			case topHalfEdge:
				return dimensions_t(0, 1);
			break;
			default:
				return dimensions_t(0, 0);
			break;
		}
	}

	Scalar GhostLiquidSolver::calculateCellFraction(const dimensions_t &cellIndex, halfEdgeLocation_t heLocation) {
		Scalar phiRight = 0, phiLeft = 0;
		Array2D<Scalar> scalarFieldValues = m_pLiquidRepresentation->getLevelSetArray();
		//Calculate subdivision factor: since each level-set point on the liquid representation is created as a subdivision
		//process from the coarse grid, here we are accessing the points that have a direct correspondence with the coarse
		int subdivisScalar = pow(2, m_pLiquidRepresentation->getParams().levelSetGridSubdivisions);

		switch (heLocation) {
			case leftHalfEdge:
				phiLeft = scalarFieldValues(cellIndex.x*subdivisScalar, cellIndex.y*subdivisScalar);
				phiRight = scalarFieldValues(cellIndex.x*subdivisScalar, (cellIndex.y + 1)*subdivisScalar);
			break;
			
			case rightHalfEdge:
				phiLeft = scalarFieldValues((cellIndex.x + 1)*subdivisScalar, cellIndex.y*subdivisScalar);
				phiRight = scalarFieldValues((cellIndex.x + 1)*subdivisScalar, (cellIndex.y + 1)*subdivisScalar);
			break;

			case bottomHalfEdge:
				phiLeft = scalarFieldValues(cellIndex.x*subdivisScalar, cellIndex.y*subdivisScalar);
				phiRight = scalarFieldValues((cellIndex.x + 1)*subdivisScalar, cellIndex.y*subdivisScalar);
			break;

			case topHalfEdge:
				phiLeft = scalarFieldValues(cellIndex.x*subdivisScalar, (cellIndex.y + 1)*subdivisScalar);
				phiRight = scalarFieldValues((cellIndex.x + 1)*subdivisScalar, (cellIndex.y + 1)*subdivisScalar);
			break;
			default:
			
			break;
		}

		if (phiLeft < 0 && phiRight < 0)
			return 1;
		if (phiLeft < 0 && phiRight >= 0)
			return phiLeft / (phiLeft - phiRight);
		if (phiLeft >= 0 && phiRight < 0)
			return phiRight / (phiRight - phiLeft);
		else
			return 0;
	}
	Scalar GhostLiquidSolver::distanceToLiquid(const dimensions_t &cellIndex, const dimensions_t &nextCellIndex) {
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
		Vector2 cellCentroid = Vector2(cellIndex.x + 0.5, cellIndex.y + 0.5)*dx;
		Vector2 nextCellCentroid = Vector2(nextCellIndex.x + 0.5, nextCellIndex.y + 0.5)*dx;

		/* The algorithm has to check all lines segments to check if the ray from one cell centroid crosses the liquid
		 * interface */
		for (int k = 0; k < m_pLiquidRepresentation->getLineMeshes().size(); k++) {
			Vector2 intersectionPoint;
			const vector<Vector2> &points = m_pLiquidRepresentation->getLineMeshes()[k]->getPoints();
			for (int i = 0; i < points.size(); i++) {
				int nextI = roundClamp<int>(i + 1, 0, points.size());
				if (DoLinesIntersect(cellCentroid, nextCellCentroid, points[i], points[nextI], intersectionPoint)) {
					return (cellCentroid - intersectionPoint).length();
				}
			}			
		}
		/* Error, no line intersects the geometry, user defined incorrect function arguments */
		return -1;
	}

	void GhostLiquidSolver::updateCellTypes() {
		typedef LiquidRepresentation2D<Vector2>::levelSetCellType_t cellType;
		m_cellTypes.assign(cellType::airCell);
		m_boundaryCells.assign(false);
		Scalar dx = m_pGridData->getScaleFactor(0, 0).x;
				
		for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
			for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
				Vector2 transformedCellCentroid = Vector2(i + 0.5, j + 0.5)*dx;
				for (int k = 0; k < m_pLiquidRepresentation->getLineMeshes().size(); k++) {
					if (isInsidePolygon(transformedCellCentroid, m_pLiquidRepresentation->getLineMeshes()[k]->getPoints())) {
						m_cellTypes(i, j) = cellType::fluidCell;
						break; //Not necessary to check liquid meshes anymore
					}
				}
			}
		}

		for (int k = 0; k < m_pLiquidRepresentation->getLineMeshes().size(); k++) {
			for (int i = 0; i < m_pLiquidRepresentation->getLineMeshes()[k]->getPoints().size(); i++) {
				Vertex<Vector2> *pVertex = m_pLiquidRepresentation->getLineMeshes()[k]->getVertices()[i];
				dimensions_t cellIndex(pVertex->getPosition().x / dx, pVertex->getPosition().y / dx);
				m_boundaryCells(cellIndex) = true;
			}
		}

		/*m_boundaryCells.assign(false);
		typedef LiquidRepresentation2D::levelSetCellType_t cellType;
		for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
			for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
				if (m_cellTypes(i, j) == cellType::fluidCell && (
					(i > 1 && m_cellTypes(i - 1, j) == cellType::airCell) ||
					(i < m_pGridData->getDimensions().x - 1 && m_cellTypes(i + 1, j) == cellType::airCell) ||
					(j > 1 && m_cellTypes(i, j - 1) == cellType::airCell) ||
					(j < m_pGridData->getDimensions().y - 1 && m_cellTypes(i, j + 1) == cellType::airCell))) {
						

					m_boundaryCells(i, j) = true;
				}
			}
		}*/
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
		const LiquidRepresentation2D<Vector2>::params_t &params = m_pLiquidRepresentation->getParams();
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