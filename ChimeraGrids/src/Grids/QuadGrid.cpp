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

#include "Grids/QuadGrid.h"

namespace Chimera {
	namespace Grids {
		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		QuadGrid::QuadGrid(const string &gridFilename, bool periodicBCs, bool subGrid) {
			m_periodicBCs = periodicBCs;
			m_subGrid = subGrid;

			size_t tempFound = gridFilename.rfind("/");
			if (tempFound != string::npos) {
				string tempStr = gridFilename.substr(tempFound, gridFilename.length() - tempFound);
				m_gridName = tempStr.substr(1, tempStr.rfind(".") - 1);
			} else {
				m_gridName = gridFilename.substr(0, gridFilename.rfind(".") - 1);
			}
			m_gridType = "QuadGrid";

			if(!periodicBCs)
				loadGrid(gridFilename);
			else
				loadPeriodicGrid(gridFilename);

			GridData2D *pGridData2D = new GridData2D(m_pGridPoints, m_dimensions);
			m_pGridData = pGridData2D;
			
			m_gridPosition = Vector2(0, 0);
			for(int i = 0; i < pGridData2D->getDimensions().x + 1; i++) {
				for(int j = 0; j < pGridData2D->getDimensions().y + 1; j++) {
					m_gridCentroid += pGridData2D->getPoint(i, j);
				}
			}
			m_gridOrigin = getPoint(0, 0);
			m_gridCentroid /= static_cast<Scalar>((pGridData2D->getDimensions().x + 1)*(pGridData2D->getDimensions().y + 1));
			m_gridVelocity = Vector2(0, 0);

			initializeGridMetrics();

			/** Initialize boundary and solid markers */
			m_pSolidMarkers = new bool[m_dimensions.x*m_dimensions.y];
			m_pBoundaryMarkers = new bool[m_dimensions.x*m_dimensions.y];
			for(int i = 0; i < m_dimensions.x; i++) {
				for(int j = 0; j < m_dimensions.y; j++) {
					m_pSolidMarkers[j*m_dimensions.x + i] = false;
					m_pBoundaryMarkers[j*m_dimensions.x + i] = false;
				}
			}
		}

		QuadGrid::QuadGrid(const Vector2 &initialPoint, const Vector2 &finalPoint, Scalar gridSpacing, bool subGrid /* = false */) {
			m_periodicBCs = false;
			m_subGrid = subGrid;

			/** Dimensions and boundaries calculation */
			Vector2 boundariesLenght(finalPoint.x - initialPoint.x, finalPoint.y - initialPoint.y);

			m_dimensions.x = static_cast<int>(boundariesLenght.x/gridSpacing);
			m_dimensions.y = static_cast<int>(boundariesLenght.y/gridSpacing);
			m_dimensions.z = 0;

			//Adjusting final boundaries
			Vector2 finalBoundary;
			finalBoundary.x = initialPoint.x + m_dimensions.x*gridSpacing;
			finalBoundary.y = initialPoint.y + m_dimensions.y*gridSpacing;

			//Setting up grid boundaries
			m_gridBoundingBox.lowerBounds = initialPoint;
			m_gridBoundingBox.upperBounds = finalBoundary;
			
			dimensions_t tempDimensions(m_dimensions);
			m_pGridPoints = new Array2D<Vector2>(tempDimensions);

			Scalar dx = (finalBoundary.x - initialPoint.x)/m_dimensions.x;
			Scalar dy = (finalBoundary.y - initialPoint.y) / m_dimensions.y;

			for(int i = 0; i < tempDimensions.x; i++) {
				for(int j = 0; j < tempDimensions.y; j++) {
					(*m_pGridPoints)(i,j).x = initialPoint.x + dx*i;
					(*m_pGridPoints)(i,j).y = initialPoint.y + dy*j;
				}
			}

			m_dimensions.x--;
			m_dimensions.y--; 

			GridData2D *pGridData2D = new GridData2D(m_pGridPoints, m_dimensions);
			pGridData2D->setMinBoundary(initialPoint);
			pGridData2D->setMaxBoundary(finalPoint);
			m_pGridData = pGridData2D;

			m_gridPosition = Vector2(0, 0);
			for(int i = 0; i < pGridData2D->getDimensions().x + 1; i++) {
				for(int j = 0; j < pGridData2D->getDimensions().y + 1; j++) {
					m_gridCentroid += pGridData2D->getPoint(i, j);
				}
			}

			m_gridOrigin = getPoint(0, 0);
			m_gridCentroid /= static_cast<Scalar>((pGridData2D->getDimensions().x + 1)*(pGridData2D->getDimensions().y + 1));
			m_gridVelocity = Vector2(0, 0);

			initializeGridMetrics();

			for(int i = 0; i < pGridData2D->getDimensions().x; i++) {
				for(int j = 0; j < pGridData2D->getDimensions().y; j++) {
 					pGridData2D->setScaleFactor(Vector2(dx, dy), i, j);
				}
			}
			
			if(dx == dy)
				pGridData2D->setGridSpacing(dx);

			/** Initialize boundary and solid markers */
			m_pSolidMarkers = new bool[m_dimensions.x*m_dimensions.y];
			m_pBoundaryMarkers = new bool[m_dimensions.x*m_dimensions.y];
			for(int i = 0; i < m_dimensions.x; i++) {
				for(int j = 0; j < m_dimensions.y; j++) {
					m_pSolidMarkers[j*m_dimensions.x + i] = false;
					m_pBoundaryMarkers[j*m_dimensions.x + i] = false;
				}
			}

			//Setting up grid name and type
			m_gridName = "customGrid " + intToStr(m_dimensions.x) + "x" + intToStr(m_dimensions.y);
			m_gridType = "QuadGrid";
			
			
		}

		QuadGrid::QuadGrid(Array2D<Vector2> *pGridPoints, bool subGrid /* = false */) {
			m_periodicBCs = false;
			m_subGrid = subGrid;

			int gridID = rand()*1000;
			m_gridName = "customGrid" + intToStr(gridID);// + gridDimensions.x + "x" + gridDimensions.y;
			m_gridType = "QuadGrid";
			m_dimensions = pGridPoints->getDimensions();
			m_dimensions.x -= 1; m_dimensions.y -= 1;

			m_pGridPoints = pGridPoints;

			GridData2D *pGridData2D = new GridData2D(m_pGridPoints, m_dimensions);
			m_pGridData = pGridData2D;

			m_gridPosition = Vector2(0, 0);
			for(int i = 0; i < pGridData2D->getDimensions().x + 1; i++) {
				for(int j = 0; j < pGridData2D->getDimensions().y + 1; j++) {
					m_gridCentroid += pGridData2D->getPoint(i, j);
				}
			}

			m_gridOrigin = getPoint(0, 0);
			m_gridCentroid /= static_cast<Scalar>((pGridData2D->getDimensions().x + 1)*(pGridData2D->getDimensions().y + 1));
			m_gridVelocity = Vector2(0, 0);

			initializeGridMetrics();


			/** Initialize boundary and solid markers */
			m_pSolidMarkers = new bool[m_dimensions.x*m_dimensions.y];
			m_pBoundaryMarkers = new bool[m_dimensions.x*m_dimensions.y];
			for(int i = 0; i < m_dimensions.x; i++) {
				for(int j = 0; j < m_dimensions.y; j++) {
					m_pSolidMarkers[j*m_dimensions.x + i] = false;
					m_pBoundaryMarkers[j*m_dimensions.x + i] = false;
				}
			}
		}

		/************************************************************************/
		/* Metrics & grid functionalities                                       */
		/************************************************************************/
		void QuadGrid::initializeGridMetrics() {
			Vector2 d_dXi, d_dEta, normalizedXi, normalizedEta;
			Vector2 scaleFactor;

			GridData2D *pGridData2D = getGridData2D();
			for(int j = 0; j < m_dimensions.y; j++) {
				for (int i = 0; i < m_dimensions.x; i++) {
					d_dXi.x =	getPoint(i + 1, j).x - getPoint(i, j).x;
					d_dXi.y =	getPoint(i + 1, j).y - getPoint(i, j).y;

					d_dEta.x =	getPoint(i, j + 1).x -	getPoint(i, j).x;
					d_dEta.y =	getPoint(i, j + 1).y -	getPoint(i, j).y;
	
					Vector2 centerPoint = (getPoint(i, j) + getPoint(i + 1, j) + getPoint(i, j + 1) + getPoint(i + 1, j + 1))*0.25; 
					pGridData2D->setCenterPoint(centerPoint, i, j);
					
					Scalar volume = abs((d_dXi.x*d_dEta.y - d_dEta.x*d_dXi.y));
					pGridData2D->setVolume(volume, i, j);

					Scalar dXiLength = d_dXi.length();
					Scalar dEtaLength = d_dEta.length();

					normalizedXi = d_dXi.normalized();
					normalizedEta = d_dEta.normalized();

					scaleFactor.x = dXiLength;
					scaleFactor.y = dEtaLength;

					pGridData2D->setScaleFactor(scaleFactor, i, j);

					Matrix2x2 transformationMatrix, inverseTransformationMatrix;
					transformationMatrix.column[0] = normalizedXi;
					transformationMatrix.column[1] = normalizedEta;
					inverseTransformationMatrix = transformationMatrix;
					transformationMatrix.invert();
					pGridData2D->setTransformationMatrix(transformationMatrix, i, j);
					pGridData2D->setInverseTransformationMatrix(inverseTransformationMatrix, i, j);

					pGridData2D->setXiBaseNormal((d_dXi.perpendicular()).normalized(), i, j);
					pGridData2D->setEtaBaseNormal((d_dEta.perpendicular()).normalized(), i, j);
				}
			}

			if (scaleFactor.x == scaleFactor.y)
				pGridData2D->setGridSpacing(scaleFactor.x);
		}
		
		/************************************************************************/
		/* Grid loading                                                         */
		/************************************************************************/
		void QuadGrid::loadGrid(const string &gridFilename) {
			shared_ptr<ifstream> fileStream(new ifstream(gridFilename.c_str()));
			if(fileStream->fail())
				throw("File not found: " + gridFilename);

			char fHeader[256];
			string fileStr;
			Scalar scaleFactor;

			fileStream->getline(fHeader, 256);						//#VRML V2.0 utf8
			fileStream->getline(fHeader, 256);						//#	
			fileStream->getline(fHeader, 256);						//# exported from Pointwise V16.00R2,  8-Dec-10   14:05:54
			fileStream->getline(fHeader, 256);						//#	dimX x dim Y

			/** Dimensionality*/
			(*fileStream) >> fHeader;								// Skipping #
			(*fileStream) >> m_dimensions.x; 
			(*fileStream) >> m_dimensions.y;
			m_dimensions.z = 0;
			fileStream->getline(fHeader, 256);						//Last \n

			/** Scale factor*/
			fileStream->getline(fHeader, 256);						//# scaleFactor
			(*fileStream) >> fHeader;								// Skipping #
			(*fileStream) >> scaleFactor;
			fileStream->getline(fHeader, 256);						//Last \n


			fileStream->getline(fHeader, 256);						//Shape {
			fileStream->getline(fHeader, 256);						//geometry IndexedFaceSet {
			fileStream->getline(fHeader, 256);						//coord Coordinate {
			fileStream->getline(fHeader, 256);						//point [

			/** Loading grid points*/
			m_pGridPoints = new Array2D<Vector2>(m_dimensions);
			Vector2 tempPoint;
			Scalar tempZ;
			int i = 0, j = 0;
			for(j = 0; j < m_dimensions.y; j++) {
				for (i = 0; i < m_dimensions.x; i++) {
					(*fileStream) >> tempPoint.x; (*fileStream) >> tempPoint.y; (*fileStream) >> tempZ;
					(*fileStream) >> fHeader;								// Skipping ,
					tempPoint *= scaleFactor;
					(*m_pGridPoints)(i, j) = tempPoint;
				}
			}

			m_dimensions.x--;
			m_dimensions.y--;

		}
		void QuadGrid::loadPeriodicGrid(const string &gridFilename) {
			shared_ptr<ifstream> fileStream(new ifstream(gridFilename.c_str()));
			if(fileStream->fail())
				throw("File not found: " + gridFilename);

			char fHeader[256];
			string fileStr;
			Scalar scaleFactor;

			fileStream->getline(fHeader, 256);						//#VRML V2.0 utf8
			fileStream->getline(fHeader, 256);						//#	
			fileStream->getline(fHeader, 256);						//# exported from Pointwise V16.00R2,  8-Dec-10   14:05:54
			fileStream->getline(fHeader, 256);						//#	dimX x dim Y

			/** Dimensionality*/
			(*fileStream) >> fHeader;								// Skipping #
			(*fileStream) >> m_dimensions.x; 
			(*fileStream) >> m_dimensions.y;
			m_dimensions.z = 0;
			fileStream->getline(fHeader, 256);						//Last \n

			/** Scale factor*/
			fileStream->getline(fHeader, 256);						//# scaleFactor
			(*fileStream) >> fHeader;								// Skipping #
			(*fileStream) >> scaleFactor;
			fileStream->getline(fHeader, 256);						//Last \n

			/** Skipping headers*/
			fileStream->getline(fHeader, 256);						// periodic
			fileStream->getline(fHeader, 256);						// Shape {
			fileStream->getline(fHeader, 256);						//geometry IndexedFaceSet {
			fileStream->getline(fHeader, 256);						//coord Coordinate {
			fileStream->getline(fHeader, 256);						//point [

			/** Loading grid points*/
			
			Vector2 tempPoint;
			Scalar tempZ;
			m_pGridPoints = new Array2D<Vector2>(m_dimensions);

			dimensions_t tempDimensions = m_dimensions;
			int i = 0, j = 0;
			//We have to load a band less of X cells, since its a periodic grid.
			tempDimensions.x -= 1;
			
			for(j = 0; j < tempDimensions.y; j++) {
				for (i = 0; i < tempDimensions.x; i++) {
					(*fileStream) >> tempPoint.x; (*fileStream) >> tempPoint.y; (*fileStream) >> tempZ;
					(*fileStream) >> fHeader;								// Skipping ,
					tempPoint *= scaleFactor;
					m_pGridPoints->getRawData()[j*(m_dimensions.x-1) + i] = tempPoint;
				}
			}

			/** Loading cell indices */
			fileStream->getline(fHeader, 256);						// \n
			fileStream->getline(fHeader, 256);						//]
			fileStream->getline(fHeader, 256);						//}
			fileStream->getline(fHeader, 256);						//coordIndex [
			int numCells = (m_dimensions.x - 1)*(m_dimensions.y - 1);
			int *pTempIndexes = new int[numCells*4];
			for(i = 0; i < numCells; i++) {
				(*fileStream) >> pTempIndexes[i*4];
				(*fileStream) >> fHeader;							// Skipping ,
				(*fileStream) >> pTempIndexes[i*4 + 1];
				(*fileStream) >> fHeader;							// Skipping ,
				(*fileStream) >> pTempIndexes[i*4 + 2];
				(*fileStream) >> fHeader;							// Skipping ,
				(*fileStream) >> pTempIndexes[i*4 + 3];
				fileStream->getline(fHeader, 256);					//Last of the line
			} 

			/** Re-organizing grid points*/
			Array2D<Vector2> *pTempPoints = new Array2D<Vector2>(m_dimensions);
			int ithCell = 0;
			for(j = 0; j < m_dimensions.y - 1; j++) {
				for(i = 0; i < m_dimensions.x - 1; i++) {
					int tempIndexes[4];
					tempIndexes[0] = pTempIndexes[ithCell*4];
					tempIndexes[1] = pTempIndexes[ithCell*4];
					tempIndexes[2] = pTempIndexes[ithCell*4];
					tempIndexes[3] = pTempIndexes[ithCell*4];

					pTempPoints->getRawData()[j*m_dimensions.x + i]			= m_pGridPoints->getRawData()[pTempIndexes[ithCell*4]];
					pTempPoints->getRawData()[j*m_dimensions.x + i + 1]		= m_pGridPoints->getRawData()[pTempIndexes[ithCell*4 + 1]];
					pTempPoints->getRawData()[(j + 1)*m_dimensions.x + i + 1]	= m_pGridPoints->getRawData()[pTempIndexes[ithCell*4 + 2]];
					pTempPoints->getRawData()[(j + 1)*m_dimensions.x + i]		= m_pGridPoints->getRawData()[pTempIndexes[ithCell*4 + 3]];
					ithCell++;
				}
			}

			delete m_pGridPoints;
			m_pGridPoints = pTempPoints;
			
			m_dimensions.x--;
			m_dimensions.y--;
		}

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		bool QuadGrid::isInsideCell(Vector2 position, int x, int y) const {
			Vector2 iniPos = getPoint(x, y);
			Vector2 triangle1Points[3];
			triangle1Points[0] = getPoint(x, y);
			triangle1Points[1] = getPoint(x + 1, y);
			triangle1Points[2] = getPoint(x, y + 1);

			Vector2 triangle2Points[3];
			triangle2Points[0] = getPoint(x + 1, y);
			triangle2Points[1] = getPoint(x, y + 1);
			triangle2Points[2] = getPoint(x + 1, y + 1);

			return isInsideTriangle(position - getPosition(), triangle1Points) || isInsideTriangle(position - getPosition(), triangle2Points);
		}

		/************************************************************************/
		/* Grid I/O			                                                    */
		/************************************************************************/
		void QuadGrid::loadSolidCircle(const Vector2 &centerPoint, Scalar circleSize) {
			GridData2D *pGridData2D = getGridData2D();
			for(int i = 0; i < m_dimensions.x; i++) {
				for(int j = 0; j < m_dimensions.y; j++) {
					if((centerPoint - pGridData2D->getCenterPoint(i, j)).length() < circleSize) {
						m_pSolidMarkers[j*m_dimensions.x + i] = true;
					}
				}
			}
		}
		void QuadGrid::loadSolidRectangle(const Vector2 &recPosition, const Vector2 &recSize) {
			GridData2D *pGridData2D = getGridData2D();
			Scalar dx = pGridData2D->getScaleFactor(0, 0).x;
			int lowerBoundX = static_cast<int>(floor(recPosition.x/dx)); 
			int lowerBoundY = static_cast<int>(floor(recPosition.y/dx));
			int upperBoundX = lowerBoundX + static_cast<int>(floor(recSize.x/dx)); 
			int upperBoundY = lowerBoundY + static_cast<int>(floor(recSize.y/dx)); 
			for(int i = lowerBoundX; i < upperBoundX; i++) {
				for(int j = lowerBoundY; j < upperBoundY; j++) {
					m_pSolidMarkers[j*m_dimensions.x + i] = true;
				}
			}
		}

		void QuadGrid::loadObject(const vector<Vector2> &objectPoints) {
			for (int i = 0; i < m_dimensions.x; i++) {
				for (int j = 0; j < m_dimensions.y; j++) {
					if (isInsidePolygon(getGridData2D()->getCenterPoint(i, j), objectPoints) ||
						isInsidePolygon(getGridData2D()->getPoint(i, j), objectPoints)) {
						setSolidCell(true, i, j);
					}
				}
			}
		}
		
		void QuadGrid::exportToFile(const string &gridExportname) {
			auto_ptr<ofstream> fileStream(new ofstream(gridExportname.c_str()));
			*fileStream << m_dimensions.x;
			*fileStream << " " << m_dimensions.y;
			*fileStream << endl;
			for(int j = 0; j < m_dimensions.x; j++) {
				for (int i = 0; i < m_dimensions.y; i++) {
					*fileStream <<			(*m_pGridPoints)(i, j).x;
					*fileStream << " " <<	(*m_pGridPoints)(i, j).y;
					*fileStream << endl;
				}
			}

		}

		
		
	}
}