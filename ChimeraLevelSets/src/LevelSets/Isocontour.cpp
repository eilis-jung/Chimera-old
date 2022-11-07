#include "LevelSets/Isocontour.h"

namespace Chimera {
	namespace LevelSets {
		namespace Isocontour {

			///************************************************************************/
			///* Gradient stepping                                                    */
			///************************************************************************/
			//void gradientStepping(vector<Vector2> &isocontourPoints, GridData2D *pGridData, Scalar isoValue, Scalar timestep) {
			//	const Array2D<Scalar> & scalarField = pGridData->getLevelSetArray();
			//	Vector2 gridBoundaries(pGridData->getDimensions().x - 2.0f, pGridData->getDimensions().y - 2.0f);
			//	Scalar minValue = FLT_MAX;

			//	/**Finding the initial cell which has closest relation to the isoValue.
			//	 **Also, the initial velocity field for the stepping method is calculated*/ 
			//	dimensions_t initialIndexPoint;
			//	for(int i = 1; i < scalarField.getDimensions().x - 1; i++) {
			//		for(int j = 1; j < scalarField.getDimensions().y - 1; j++) {
			//			Vector2 gradientVelocity = interpolateGradient(Vector2(i + 0.5f, j + 0.5f), scalarField);
			//			gradientVelocity = gradientVelocity.normalized().perpendicular();
			//			pGridData->setVelocity(gradientVelocity, i, j);

			//			if(abs(scalarField(i, j) - isoValue) < minValue) {
			//				minValue = abs(scalarField(i, j) - isoValue);
			//				initialIndexPoint.x = i;
			//				initialIndexPoint.y = j;
			//			}
			//		}
			//	}

			//	/** Calculating the isoError  */
			//	Vector2 correction = correctIsoPosition(isoValue, Vector2(initialIndexPoint.x + 0.5f, initialIndexPoint.y + 0.5f), scalarField);
			//	isocontourPoints.push_back(pGridData->getCenterPoint(initialIndexPoint.x, initialIndexPoint.y) + correction);

			//	Scalar dx = pGridData->getScaleFactor(0,0).x;
			//	Vector2 gridOrigin = pGridData->getPoint(0, 0);

			//	trajectoryIntegratorParams_t<Vector2> integrationParams(&pGridData->getVelocityArray(), &pGridData->getAuxVelocityArray(), 
			//															&pGridData->getTransformationMatrices(), &pGridData->getScaleFactorsArray());
			//	integrationParams.dt = timestep;
			//	integrationParams.backwardsIntegration = false;
			//	integrationParams.periodicDomain = false;

			//	//for(int i = 0; i < maxIsoContourPoints; i++) {
			//	//	if(i > 10 && (isocontourPoints[i] - isocontourPoints[0]).length() < timestep) {
			//	//		break;
			//	//	}

			//	//	Vector2 transPos = (isocontourPoints[i] - gridOrigin)/dx; //Transforming from world space to grid space
			//	//	transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//	//	transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//	//	Vector2 g1 = interpolateGradient(transPos, scalarField);
			//	//	g1.normalize();
			//	//	g1 = g1.perpendicular();

			//	//	integrationParams.initialPosition = transPos;
			//	//	integrationParams.initialVelocity = g1;

			//	//	rungeKutta4(integrationParams);

			//	//	transPos = integrationParams.initialPosition;
			//	//	Vector2 finalPos = transPos*dx + gridOrigin; //Transforming back to world-space

			//	//	Vector2 correction = correctIsoPosition(isoValue, transPos, scalarField);
			//	//	isocontourPoints.push_back(finalPos);
			//	//}
			//	for(int i = 0; i < maxIsoContourPoints; i++) {
			//		if(i > 10 && (isocontourPoints[i] - isocontourPoints[0]).length() < timestep) {
			//			break;
			//		}

			//		Vector2 transPos = (isocontourPoints[i] - gridOrigin)/dx;
			//		transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//		transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//		Vector2 g1 = interpolateGradient(transPos, scalarField);
			//		g1.normalize();
			//		g1 = g1.perpendicular();

			//		Vector2 tempPos = isocontourPoints[i] + g1*timestep*0.5;
			//		transPos = (tempPos - gridOrigin)/dx;
			//		transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//		transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//		Vector2 g2 = interpolateGradient(transPos, scalarField);
			//		g2.normalize();
			//		g2 = g2.perpendicular();


			//		Vector2 finalPos = isocontourPoints[i] - g2*timestep;
			//		transPos = (finalPos - gridOrigin)/dx;
			//		transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//		transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//		//Correcting the point position
			//		Vector2 correction = correctIsoPosition(isoValue, transPos, scalarField);
			//		isocontourPoints.push_back(finalPos + correction);
			//	}

			//}

			//void gradientStepping(vector<Vector2> &isocontourPoints, const Array2D<Scalar> &scalarField, GridData2D *pGridData, Scalar isoValue, Scalar timestep) {
			//	Vector2 gridBoundaries(pGridData->getDimensions().x - 2.0f, pGridData->getDimensions().y - 2.0f);
			//	Scalar minValue = FLT_MAX;

			//	/**Finding the initial cell which has closest relation to the isoValue.
			//	 **Also, the initial velocity field for the stepping method is calculated*/ 
			//	dimensions_t initialIndexPoint;
			//	for(int i = 1; i < scalarField.getDimensions().x - 1; i++) {
			//		for(int j = 1; j < scalarField.getDimensions().y - 1; j++) {
			//			Vector2 gradientVelocity = interpolateGradient(Vector2(i + 0.5f, j + 0.5f), scalarField);
			//			gradientVelocity = gradientVelocity.normalized().perpendicular();
			//			pGridData->setVelocity(gradientVelocity, i, j);

			//			if(abs(scalarField(i, j) - isoValue) < minValue) {
			//				minValue = abs(scalarField(i, j) - isoValue);
			//				initialIndexPoint.x = i;
			//				initialIndexPoint.y = j;
			//			}
			//		}
			//	}

			//	/** Calculating the isoError  */
			//	Vector2 correction = correctIsoPosition(isoValue, Vector2(initialIndexPoint.x + 0.5f, initialIndexPoint.y + 0.5f), scalarField);
			//	isocontourPoints.push_back(pGridData->getCenterPoint(initialIndexPoint.x, initialIndexPoint.y) + correction);

			//	Scalar dx = pGridData->getScaleFactor(0,0).x;
			//	Vector2 gridOrigin = pGridData->getPoint(0, 0);

			//	trajectoryIntegratorParams_t<Vector2> integrationParams(&pGridData->getVelocityArray(), &pGridData->getAuxVelocityArray(), 
			//															&pGridData->getTransformationMatrices(), &pGridData->getScaleFactorsArray());
			//	integrationParams.dt = timestep;
			//	integrationParams.backwardsIntegration = false;
			//	integrationParams.periodicDomain = false;

			//	for(int i = 0; i < maxIsoContourPoints; i++) {
			//		if(i > 10 && (isocontourPoints[i] - isocontourPoints[0]).length() < timestep) {
			//			break;
			//		}

			//		Vector2 transPos = (isocontourPoints[i] - gridOrigin)/dx;
			//		transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//		transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//		Vector2 g1 = interpolateGradient(transPos, scalarField);
			//		g1.normalize();
			//		g1 = g1.perpendicular();

			//		Vector2 tempPos = isocontourPoints[i] + g1*timestep*0.5;
			//		transPos = (tempPos - gridOrigin)/dx;
			//		transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//		transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//		Vector2 g2 = interpolateGradient(transPos, scalarField);
			//		g2.normalize();
			//		g2 = g2.perpendicular();


			//		Vector2 finalPos = isocontourPoints[i] - g2*timestep;
			//		transPos = (finalPos - gridOrigin)/dx;
			//		transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//		transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//		//Correcting the point position
			//		Vector2 correction = correctIsoPosition(isoValue, transPos, scalarField);
			//		isocontourPoints.push_back(finalPos /*+ correction*/);
			//	}

			//}
			//void gradientStepping(vector<Vector2> &isocontourPoints, GridData2D *pGridData, Scalar timestep, const Vector2 &startingPoint, const Vector2 &endPoint) {
			//	
			//	/** Auxiliary vars*/
			//	Vector2 gridOrigin = pGridData->getPoint(0, 0);
			//	const Array2D<Scalar> & scalarField = pGridData->getLevelSetArray();
			//	Vector2 gridBoundaries(pGridData->getDimensions().x - 2.0f, pGridData->getDimensions().y - 2.0f);
			//	Scalar dx = pGridData->getScaleFactor(0,0).x;

			//	/**Calculating grid space starting point */
			//	Vector2 gsStartingPoint = (startingPoint - gridOrigin)/dx;
			//	/**Calculating the initial velocity field for the stepping method is calculated*/ 
			//	dimensions_t initialIndexPoint((int) floor(gsStartingPoint.x), (int) floor(gsStartingPoint.y), 0);
			//	for(int i = 2; i < scalarField.getDimensions().x - 2; i++) {
			//		for(int j = 2; j < scalarField.getDimensions().y - 2; j++) {
			//			Vector2 gradientVelocity;
			//			gradientVelocity.x = interpolateGradient(Vector2(i, j + 0.5f), scalarField).x;
			//			gradientVelocity.y = interpolateGradient(Vector2(i + 0.5f, j), scalarField).y;
			//			gradientVelocity = gradientVelocity.normalized().perpendicular();
			//			pGridData->setVelocity(gradientVelocity, i, j);
			//		}
			//	}

			//	/** Calculating isoValue based on startingPoint  */
			//	isocontourPoints.push_back(startingPoint);
			//	Scalar isoValue = interpolateScalar(gsStartingPoint, scalarField);

			//	
			//	/** Initializating trajectoryIntegrator params for RK-4*/
			//	trajectoryIntegratorParams_t<Vector2> integrationParams(&pGridData->getVelocityArray(), &pGridData->getAuxVelocityArray(), 
			//															&pGridData->getTransformationMatrices(), &pGridData->getScaleFactorsArray());
			//	integrationParams.dt = timestep;
			//	integrationParams.backwardsIntegration = false;
			//	integrationParams.periodicDomain = false;

			//	//for(int i = 0; i < maxIsoContourPoints; i++) {
			//	//	Scalar isoDiff = (isocontourPoints[i] - endPoint).length();
			//	//	if(i > 10 && (isocontourPoints[i] - endPoint).length() < timestep*2) {
			//	//		break;
			//	//	}

			//	//	Vector2 transPos = (isocontourPoints[i] - gridOrigin)/dx; //Transforming from world space to grid space
			//	//	transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//	//	transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//	//	Vector2 g1 = interpolateGradient(transPos, scalarField);
			//	//	g1.normalize();
			//	//	g1 = g1.perpendicular();

			//	//	integrationParams.initialPosition = transPos;
			//	//	integrationParams.initialVelocity = g1;

			//	//	rungeKutta4(integrationParams);

			//	//	transPos = integrationParams.initialPosition;
			//	//	Vector2 finalPos = transPos*dx + gridOrigin; //Transforming back to world-space

			//	//	Vector2 correction = correctIsoPosition(isoValue, transPos, scalarField);
			//	//	isocontourPoints.push_back(finalPos);//+ correction);
			//	//}
			//	for(int i = 0; i < maxIsoContourPoints; i++) {
			//		if(i > 10 && (isocontourPoints[i] - endPoint).length() < timestep) {
			//			break;
			//		}

			//		Vector2 transPos = (isocontourPoints[i] - gridOrigin)/dx;
			//		transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//		transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//		Vector2 g1 = interpolateGradient(transPos, scalarField);
			//		g1.normalize();
			//		g1 = g1.perpendicular();

			//		Vector2 tempPos = isocontourPoints[i] + g1*timestep*0.5;
			//		transPos = (tempPos - gridOrigin)/dx;
			//		transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//		transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//		Vector2 g2 = interpolateGradient(transPos, scalarField);
			//		g2.normalize();
			//		g2 = g2.perpendicular();


			//		Vector2 finalPos = isocontourPoints[i] - g2*timestep;
			//		transPos = (finalPos - gridOrigin)/dx;
			//		transPos.x = clamp(transPos.x, 0.0f, gridBoundaries.x);
			//		transPos.y = clamp(transPos.y, 0.0f, gridBoundaries.y);

			//		//Correcting the point position
			//		Vector2 correction = correctIsoPosition(isoValue, transPos, scalarField);
			//		isocontourPoints.push_back(finalPos + correction);
			//	}
			//}

			/************************************************************************/
			/* Marching Squares                                                     */
			/************************************************************************/
			Vector2 calculatePoint(const Array2D<int> &gridMask, const Array2D<Scalar> &scalarField, GridData2D *pGridData, const dimensions_t &currCell, Scalar isoValue) {
				int mask;
				mask = gridMask(currCell.x, currCell.y) == 1;// pGrid->isSolidCell(currCell.x, currCell.y);
				mask |= gridMask(currCell.x + 1, currCell.y) == 1 << 1; //pGrid->isSolidCell(currCell.x + 1, currCell.y) << 1;
				mask |= gridMask(currCell.x + 1, currCell.y + 1) == 1 << 2; //pGrid->isSolidCell(currCell.x + 1, currCell.y + 1) << 2;
				mask |= gridMask(currCell.x, currCell.y + 1) == 1 << 3; //pGrid->isSolidCell(currCell.x, currCell.y + 1) << 3;

				Scalar isoDistance, alfa;
				switch(mask) {
				case 1:
				case 3:
				case 7:
					isoDistance = abs(scalarField(currCell.x, currCell.y) - scalarField(currCell.x, currCell.y + 1));
					alfa = abs(scalarField(currCell.x, currCell.y) - isoValue)/isoDistance;
					return pGridData->getPoint(currCell.x, currCell.y)*(1-alfa) + pGridData->getPoint(currCell.x, currCell.y + 1)*alfa;
					break;

				case 2:
				case 6:
				case 14:
					isoDistance = abs(scalarField(currCell.x, currCell.y) - scalarField(currCell.x + 1, currCell.y));
					alfa = abs(scalarField(currCell.x, currCell.y) - isoValue)/isoDistance;
					return pGridData->getPoint(currCell.x, currCell.y)*(1-alfa) + pGridData->getPoint(currCell.x + 1, currCell.y)*alfa;
					break;

				case 4:
				case 12:
				case 13:
					isoDistance = abs(scalarField(currCell.x + 1, currCell.y) - scalarField(currCell.x + 1, currCell.y + 1));
					alfa = abs(scalarField(currCell.x + 1, currCell.y) - isoValue)/isoDistance;
					return pGridData->getPoint(currCell.x + 1, currCell.y)*(1-alfa) + pGridData->getPoint(currCell.x + 1, currCell.y + 1)*alfa;
					break;

				case 9:
				case 8:
				case 11:
					isoDistance = abs(scalarField(currCell.x, currCell.y + 1) - scalarField(currCell.x + 1, currCell.y + 1));
					alfa = abs(scalarField(currCell.x, currCell.y + 1) - isoValue)/isoDistance;
					return pGridData->getPoint(currCell.x, currCell.y + 1)*(1-alfa) + pGridData->getPoint(currCell.x + 1, currCell.y + 1)*alfa;
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
			dimensions_t goToNextCell(const Array2D<int> &gridMask, const dimensions_t &currCell) {
				int mask;
				mask = gridMask(currCell.x, currCell.y) == 1;// pGrid->isSolidCell(currCell.x, currCell.y);
				mask |= gridMask(currCell.x + 1, currCell.y) == 1 << 1; //pGrid->isSolidCell(currCell.x + 1, currCell.y) << 1;
				mask |= gridMask(currCell.x + 1, currCell.y + 1) == 1 << 2; //pGrid->isSolidCell(currCell.x + 1, currCell.y + 1) << 2;
				mask |= gridMask(currCell.x, currCell.y + 1) == 1 << 3; //pGrid->isSolidCell(currCell.x, currCell.y + 1) << 3;
				switch(mask) {
				case 0:
				case 4:
				case 12:
				case 13:
					return currCell + dimensions_t(1, 0, 0);
					break;

				case 1:
				case 3:
				case 5:
				case 7:
					return currCell + dimensions_t(-1, 0, 0);
					break;

				case 8:
				case 9:
				case 10:
				case 11:
					return currCell + dimensions_t(0, 1, 0);
					break;

				case 2:
				case 6:
				case 14:
					return currCell + dimensions_t(0, -1, 0);
					break;

				default:
					return currCell;
					break;
				}
			}
			void marchingSquares(vector<Vector2> *pIsocontourPoints, const Array2D<Scalar> &scalarField, GridData2D *pGridData, Scalar isoValue) {
				dimensions_t initialDim(-1, -1, -1);
				Scalar minDifference = FLT_MAX;
				Array2D<int> *pCellMarks = new Array2D<int>(pGridData->getDimensions());
				pCellMarks->assign(0); 
				//Initially mark all cells
				for(int i = 1; i < pGridData->getDimensions().x - 1; i++) {
					for(int j = 1; j < pGridData->getDimensions().y - 1; j++) {
						if(scalarField(i, j) < isoValue) {
							(*pCellMarks)(i, j) = 1; //->setSolidCell(true, i, j);
							if(initialDim.x == -1) {
								initialDim.x = i;
								initialDim.y = j;
							}
						} else {
							Scalar currDifference = pGridData->getPressure(i, j) - isoValue;
							if(currDifference < minDifference) {
								minDifference = currDifference;
								initialDim.x = i;
								initialDim.y = j;
							}
							(*pCellMarks)(i, j) = 0; //pGrid->setSolidCell(false, i, j);
						}
						(*pCellMarks)(i, j) = 0; // pGrid->setBoundaryCell(false, i, j);
					}
				}
				bool reachedEnd = false;
				int it = 0;
				while(!reachedEnd) {
					if((*pCellMarks)(initialDim.x, initialDim.y) == 2) { //pGrid->isBoundaryCell(initialDim.x, initialDim.y)) {
						reachedEnd = true;
						break;
					}
					(*pCellMarks)(initialDim.x, initialDim.y) = 2;//pGrid->setBoundaryCell(true, initialDim.x, initialDim.y);
					if(initialDim.x <=1 || initialDim.x >= pGridData->getDimensions().x - 2 
						|| initialDim.y <= 1 || initialDim.y >= pGridData->getDimensions().y - 1) {
						reachedEnd = true;
						break;
					}
					pIsocontourPoints->push_back(calculatePoint(*pCellMarks, scalarField, pGridData, initialDim, isoValue));
					initialDim = goToNextCell(*pCellMarks, initialDim);
					it++;
				}
			}


			
			/************************************************************************/
			/* Utils                                                                */
			/************************************************************************/
			Scalar calculateCurvature(int i, const vector<Vector2> & points) {
				if(i == 0) {
					Scalar dx = points[i + 1].x - points[i].x;
					Scalar curvature = (points[i + 1].y - points[i].y)/(dx);
					//Scalar dL = (points[i + 1] - points[i]).length();
					return curvature;
					//return curvature/dL;
				} else if(i == points.size() - 1) {
					Scalar dx = points[i].x - points[i - 1].x;
					Scalar curvature = (points[i].y - points[i - 1].y)/(dx);
					//Scalar dL = (points[i] - points[i - 1]).length();
					return curvature;
					//return curvature/dL;
				} else {
					Scalar dx = points[i + 1].x - points[i - 1].x;
					Scalar curvature = (points[i + 1].y - points[i - 1].y)/(2*dx);
					//Scalar dL = (points[i + 1]- points[i]).length() + (points[i] - points[i - 1]).length();
					return curvature;
					//return curvature/dL;
				}
			}

		}
	}
}