#include "Grids/HexaGrid.h"
namespace Chimera {

	namespace Grids {

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		HexaGrid::HexaGrid(const string &gridFilename, bool periodicBCs) {
			m_periodicBCs = periodicBCs;
			size_t tempFound = gridFilename.rfind("/");
			if (tempFound != string::npos) {
				string tempStr = gridFilename.substr(tempFound, gridFilename.length() - tempFound);
				m_gridName = tempStr.substr(1, tempStr.rfind(".") - 1);
			}
			else {
				m_gridName = gridFilename.substr(0, gridFilename.rfind(".") - 1);
			}
			m_gridType = "HexaGrid";

			if (!periodicBCs)
				loadGrid(gridFilename);
			else
				loadPeriodicGrid(gridFilename);

			GridData3D *pGridData = new GridData3D(m_pGridPoints, m_dimensions);
			m_pGridData = pGridData;

			initializeGridMetrics();

			/**Initialize solid markers */
			m_pSolidMarkers = new bool[m_dimensions.x*m_dimensions.y*m_dimensions.z];
			for (int i = 0; i < m_dimensions.x; i++) {
				for (int j = 0; j < m_dimensions.y; j++) {
					for (int k = 0; k < m_dimensions.z; k++) {
						m_pSolidMarkers[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i] = false;
					}
				}
			}

			m_gridPosition = Vector3(0, 0, 0);
			for (int i = 0; i < pGridData->getDimensions().x + 1; i++) {
				for (int j = 0; j < pGridData->getDimensions().y + 1; j++) {
					for (int k = 0; k < pGridData->getDimensions().z + 1; k++) {
						m_gridCentroid += pGridData->getPoint(i, j, k);
					}
				}
			}

			m_gridOrigin = getPoint(0, 0, 0);
			m_gridCentroid /= static_cast<Scalar>((pGridData->getDimensions().x + 1)*(pGridData->getDimensions().y + 1)*(pGridData->getDimensions().z + 1));
			m_gridVelocity = Vector3(0, 0, 0);
		}

		HexaGrid::HexaGrid(const Vector3 &initialPoint, const Vector3 &finalPoint, Scalar gridSpacing) {
			m_periodicBCs = false;
			m_subGrid = false;

			/** Dimensions and boundaries calculation */
			Vector3 boundariesLenght(finalPoint - initialPoint);

			m_dimensions.x = static_cast<int>(boundariesLenght.x / gridSpacing);
			m_dimensions.y = static_cast<int>(boundariesLenght.y / gridSpacing);
			m_dimensions.z = static_cast<int>(boundariesLenght.z / gridSpacing);

			//Adjusting final boundaries
			Vector3 finalBoundary;
			finalBoundary.x = initialPoint.x + m_dimensions.x*gridSpacing;
			finalBoundary.y = initialPoint.y + m_dimensions.y*gridSpacing;
			finalBoundary.z = initialPoint.z + m_dimensions.z*gridSpacing;

			dimensions_t tempDimensions(m_dimensions);
			tempDimensions.x++;
			tempDimensions.y++;
			tempDimensions.z++;
			m_pGridPoints = new Array3D<Vector3>(tempDimensions);

			Scalar dx = (finalBoundary.x - initialPoint.x) / m_dimensions.x;
			Scalar dy = (finalBoundary.y - initialPoint.y) / m_dimensions.y;
			Scalar dz = (finalBoundary.z - initialPoint.z) / m_dimensions.z;

			for (int i = 0; i < tempDimensions.x; i++) {
				for (int j = 0; j < tempDimensions.y; j++) {
					for (int k = 0; k < tempDimensions.z; k++) {
						(*m_pGridPoints)(i, j, k).x = initialPoint.x + dx*i;
						(*m_pGridPoints)(i, j, k).y = initialPoint.y + dy*j;
						(*m_pGridPoints)(i, j, k).z = initialPoint.z + dz*k;
					}
				}
			}

			m_dimensions.x--;
			m_dimensions.y--;
			m_dimensions.z--;

			GridData3D *pGridData3D = new GridData3D(m_pGridPoints, m_dimensions);
			m_pGridData = pGridData3D;

			m_gridPosition = Vector3(0, 0, 0);
			for (int i = 0; i < pGridData3D->getDimensions().x + 1; i++) {
				for (int j = 0; j < pGridData3D->getDimensions().y + 1; j++) {
					for (int k = 0; k < pGridData3D->getDimensions().z + 1; k++) {
						m_gridCentroid += pGridData3D->getPoint(i, j, k);
					}
				}
			}

			m_gridOrigin = getPoint(0, 0, 0);
			m_gridCentroid /= static_cast<Scalar>((pGridData3D->getDimensions().x + 1)*(pGridData3D->getDimensions().y + 1)*(pGridData3D->getDimensions().z + 1));
			m_gridVelocity = Vector2(0, 0);

			initializeGridMetrics();

			for (int i = 0; i < pGridData3D->getDimensions().x; i++) {
				for (int j = 0; j < pGridData3D->getDimensions().y; j++) {
					for (int k = 0; k < pGridData3D->getDimensions().z; k++) {
						pGridData3D->setScaleFactor(Vector3(dx, dy, dz), i, j, k);
					}
				}
			}

			if ((dx == dy) && (dx == dz)) {
				pGridData3D->setGridSpacing(dx);
			}

			/** Initialize boundary and solid markers */
			m_pSolidMarkers = new bool[m_dimensions.x*m_dimensions.y*m_dimensions.z];
			m_pBoundaryMarkers = new bool[m_dimensions.x*m_dimensions.y*m_dimensions.z];
			for (int i = 0; i < m_dimensions.x; i++) {
				for (int j = 0; j < m_dimensions.y; j++) {
					for (int k = 0; k < pGridData3D->getDimensions().z; k++) {
						m_pSolidMarkers[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i] = false;
						m_pBoundaryMarkers[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i] = false;
					}
				}
			}

			//Setting up grid name and type
			m_gridName = "customGrid " + intToStr(m_dimensions.x) + "x" + intToStr(m_dimensions.y);
			m_gridType = "QuadGrid";
		}

		HexaGrid::~HexaGrid() {
		}

		/************************************************************************/
		/* Metrics & grid functionalities                                       */
		/************************************************************************/
		void HexaGrid::initializeGridMetrics() {
			Vector3 d_dXi, d_dEta, d_dTal, normalizedXi, normalizedEta, normalizedTal;
			Matrix3x3 transformationMatrix, inverseTransformationMatrix;

			GridData3D *pGridData3D = getGridData3D();

			Vector3 scaleFactor;
			Logger::get() << "Initializing grid metrics: Forward Space" << endl;
			for (int i = 0; i < m_dimensions.x; i++) {
				for (int j = 0; j < m_dimensions.y; j++) {
					for (int k = 0; k < m_dimensions.z; k++) {
						if (i == m_dimensions.x - 1) {
							d_dXi = pGridData3D->getPoint(i, j, k) - pGridData3D->getPoint(i - 1, j, k);
						}
						else {
							d_dXi = pGridData3D->getPoint(i + 1, j, k) - pGridData3D->getPoint(i, j, k);
						}

						if (j == m_dimensions.y - 1) {
							d_dEta = pGridData3D->getPoint(i, j, k) - pGridData3D->getPoint(i, j - 1, k);
						}
						else {
							d_dEta = pGridData3D->getPoint(i, j + 1, k) - pGridData3D->getPoint(i, j, k);
						}

						if (k == m_dimensions.z - 1) {
							d_dTal = pGridData3D->getPoint(i, j, k) - pGridData3D->getPoint(i, j, k - 1);
						}
						else {
							d_dTal = pGridData3D->getPoint(i, j, k + 1) - pGridData3D->getPoint(i, j, k);
						}

						Vector3 centerPoint = (pGridData3D->getPoint(i, j, k) +
							pGridData3D->getPoint(i + 1, j, k) +
							pGridData3D->getPoint(i + 1, j + 1, k) +
							pGridData3D->getPoint(i, j + 1, k) +
							pGridData3D->getPoint(i, j, k + 1) +
							pGridData3D->getPoint(i + 1, j, k + 1) +
							pGridData3D->getPoint(i + 1, j + 1, k + 1) +
							pGridData3D->getPoint(i, j + 1, k + 1))*0.125;

						pGridData3D->setCenterPoint(centerPoint, i, j, k);

						/** Face areas*/
						pGridData3D->setLeftFaceArea((d_dEta.cross(d_dTal)).length(), i, j, k);
						pGridData3D->setBottomFaceArea((d_dXi.cross(d_dTal)).length(), i, j, k);
						pGridData3D->setBackFaceArea((d_dXi.cross(d_dEta)).length(), i, j, k);

						/** Scale factors and volumes*/
						scaleFactor.x = d_dXi.length();
						scaleFactor.y = d_dEta.length();
						scaleFactor.z = d_dTal.length();
						pGridData3D->setScaleFactor(scaleFactor, i, j, k);
						pGridData3D->setVolume(fabs((d_dTal.cross(d_dEta)).dot(d_dXi)), i, j, k);

						//Normalization in order to setup transformations
						d_dXi.normalize();
						d_dEta.normalize();
						d_dTal.normalize();

						/** Transformation setup*/
						transformationMatrix[0] = d_dXi;
						transformationMatrix[1] = d_dEta;
						transformationMatrix[2] = d_dTal;
						inverseTransformationMatrix = transformationMatrix;
						transformationMatrix.invert();
						pGridData3D->setTransformationMatrix(transformationMatrix, i, j, k);
						pGridData3D->setInverseTransformationMatrix(inverseTransformationMatrix, i, j, k);

						/** Base normals */
						pGridData3D->setXiBaseNormal((d_dEta.cross(d_dTal)).normalized(), i, j, k);
						pGridData3D->setEtaBaseNormal((d_dTal.cross(d_dXi)).normalized(), i, j, k);
						pGridData3D->setTalBaseNormal((d_dXi.cross(d_dEta)).normalized(), i, j, k);

					}
				}
				if ((scaleFactor.x == scaleFactor.y) && (scaleFactor.x == scaleFactor.z)) {
					pGridData3D->setGridSpacing(scaleFactor.x);
				}
			}

			Logger::get() << "Grid metrics initialized." << endl;

		}

		/************************************************************************/
		/* Grid I/O		                                                        */
		/************************************************************************/
		void HexaGrid::loadPeriodicGrid(const string &gridFilename) {
			auto_ptr<ifstream> fileStream(new ifstream(gridFilename.c_str(), ios::in | ios::ate));
			char fHeader[256];
			string fileStr;
			int numPoints, numCells;
			int fileSize = fileStream->tellg();						//  get the file size (we started at the end)...
			Logger::get() << "Loading grid " << gridFilename << ", size: " << fileSize / 1024 << "Kb" << endl;
			fileStream->seekg(0, ios::beg);						//  ...then get back to start

			fileStream->getline(fHeader, 256);						//#    UCD geometry file from Pointwise V16.00R2
			fileStream->getline(fHeader, 256);						//#     7-Jan-11   13:27:32
			fileStream->getline(fHeader, 256);						//#

			*fileStream >> numPoints;
			*fileStream >> numCells;
			*fileStream >> m_dimensions.x;
			*fileStream >> m_dimensions.y;
			*fileStream >> m_dimensions.z;

			Logger::get() << "Grid size " << m_dimensions.x << "x" << m_dimensions.y << "x" << m_dimensions.z << endl;
			Logger::get() << "Total grid points: " << numPoints << endl;

			try {
				m_pGridPoints = new Array3D<Vector3>(m_dimensions);
			}
			catch (std::bad_alloc) {
				Logger::get() << "Grid size is too big! Allocation failed" << endl;
				exit(1);
			}

			Vector3 gridPoint, tempPoint;
			int temp;
			int percent = 10;	//	progress indicator
			int processedPoints = 0;

			//Temporary grid points
			Vector3 *pTempPoints = new Vector3[m_dimensions.x*m_dimensions.y*m_dimensions.z];
			for (int i = 0; i < numPoints; i++) {
				(*fileStream) >> temp;
				(*fileStream) >> gridPoint.x; (*fileStream) >> gridPoint.y; (*fileStream) >> gridPoint.z;
				pTempPoints[i] = gridPoint;
				processedPoints++;

				if ((fileSize > 1024 * 1024) && (100 * processedPoints / numPoints >= percent)) {
					percent = 100 * processedPoints / numPoints;
					percent = (percent / 10) * 10;
					Logger::get() << "  " << percent << " % done..." << endl;
					percent += 10;
				}
			}

			//Temporary grid indices
			int *pTempIndices = new int[numCells * 8]; //Estimating 8 vertices per cell
			string cellType;
			for (int i = 0; i < numCells; i++) {
				int localIndices[8];
				(*fileStream) >> temp; //Indicates the line number
				(*fileStream) >> temp; //0
				(*fileStream) >> cellType;

				if (cellType == "prism") {
					for (int k = 0; k < 6; k++) {
						(*fileStream) >> localIndices[k];
						localIndices[k] = localIndices[k] - 1; //Correct padding
					}
					pTempIndices[i * 8] = localIndices[4];
					pTempIndices[i * 8 + 1] = localIndices[3];
					pTempIndices[i * 8 + 2] = localIndices[0];
					pTempIndices[i * 8 + 3] = localIndices[1];

					pTempIndices[i * 8 + 4] = localIndices[4];
					pTempIndices[i * 8 + 5] = localIndices[5];
					pTempIndices[i * 8 + 6] = localIndices[2];
					pTempIndices[i * 8 + 7] = localIndices[1];

				}
				else /*if(cellType == " hex")*/ {
					for (int k = 0; k < 8; k++) {
						(*fileStream) >> localIndices[k];
						localIndices[k] = localIndices[k] - 1; //Correct padding
					}
					pTempIndices[i * 8] = localIndices[4];
					pTempIndices[i * 8 + 1] = localIndices[5];
					pTempIndices[i * 8 + 2] = localIndices[6];
					pTempIndices[i * 8 + 3] = localIndices[7];

					pTempIndices[i * 8 + 4] = localIndices[0];
					pTempIndices[i * 8 + 5] = localIndices[1];
					pTempIndices[i * 8 + 6] = localIndices[2];
					pTempIndices[i * 8 + 7] = localIndices[3];
				}
			}

			int ithCell = 0;
			for (int k = 0; k < m_dimensions.z - 1; k++) {
				for (int j = 0; j < m_dimensions.y - 1; j++) {
					for (int i = 0; i < m_dimensions.x - 1; i++) {
						(*m_pGridPoints)(i, j, k) = pTempPoints[pTempIndices[ithCell * 8]];
						(*m_pGridPoints)(i + 1, j, k) = pTempPoints[pTempIndices[ithCell * 8 + 1]];
						(*m_pGridPoints)(i + 1, j + 1, k) = pTempPoints[pTempIndices[ithCell * 8 + 2]];
						(*m_pGridPoints)(i, j + 1, k) = pTempPoints[pTempIndices[ithCell * 8 + 3]];

						(*m_pGridPoints)(i, j, k + 1) = pTempPoints[pTempIndices[ithCell * 8 + 4]];
						(*m_pGridPoints)(i + 1, j, k + 1) = pTempPoints[pTempIndices[ithCell * 8 + 5]];
						(*m_pGridPoints)(i + 1, j + 1, k + 1) = pTempPoints[pTempIndices[ithCell * 8 + 6]];
						(*m_pGridPoints)(i, j + 1, k + 1) = pTempPoints[pTempIndices[ithCell * 8 + 7]];
						ithCell++;
					}
				}
			}

			delete pTempPoints;
			delete pTempIndices;

			Logger::get() << "Grid sucessfully loaded" << endl;
			Logger::get().setDefaultLogLevel(Log_NormalPriority);

			--m_dimensions.x;
			--m_dimensions.y;
			--m_dimensions.z;
		}
		void HexaGrid::loadGrid(const string &gridFilename) {
			auto_ptr<ifstream> fileStream(new ifstream(gridFilename.c_str(), ios::in | ios::ate));
			char fHeader[256];
			string fileStr;
			int numPoints;
			int numFaces;
			int fileSize = fileStream->tellg();						//  get the file size (we started at the end)...
			Logger::get() << "Loading grid " << gridFilename << ", size: " << fileSize / 1024 << "Kb" << endl;
			fileStream->seekg(0, ios::beg);						//  ...then get back to start

			fileStream->getline(fHeader, 256);						//#    UCD geometry file from Pointwise V16.00R2
			fileStream->getline(fHeader, 256);						//#     7-Jan-11   13:27:32
			fileStream->getline(fHeader, 256);						//#

			*fileStream >> numPoints;
			*fileStream >> numFaces;
			*fileStream >> m_dimensions.x;
			*fileStream >> m_dimensions.y;
			*fileStream >> m_dimensions.z;

			numPoints = m_dimensions.y*m_dimensions.x*m_dimensions.z;

			Logger::get() << "Grid size " << m_dimensions.x << "x" << m_dimensions.y << "x" << m_dimensions.z << endl;
			Logger::get() << "Total grid points: " << numPoints << endl;


			try {
				m_pGridPoints = new Array3D<Vector3>(m_dimensions); //Too big perhaps?
			}
			catch (std::bad_alloc) {
				Logger::get() << "Grid size is too big! Allocation failed" << endl;
				exit(1);
			}

			Vector3 gridPoint, tempPoint;
			int temp;
			int percent = 10;	//	progress indicator
			int processedPoints = 0;

			/** Record where we are at the file parsing */
			int parsingPosition = fileStream->tellg();

			/** Get first two points */
			(*fileStream) >> temp;
			(*fileStream) >> gridPoint.x; (*fileStream) >> gridPoint.y; (*fileStream) >> gridPoint.z;
			(*fileStream) >> temp;
			(*fileStream) >> tempPoint.x; (*fileStream) >> tempPoint.y; (*fileStream) >> tempPoint.z;

			Logger::get().setDefaultLogLevel(Log_HighPriority);
			Scalar m_minLength = 1e-10f;
			if (fabs(tempPoint.x - gridPoint.x) > m_minLength) {
				for (int i = 0; i < m_dimensions.x - 3; i++) { // Iterate through the end of the line
					(*fileStream) >> temp;
					(*fileStream) >> gridPoint.x; (*fileStream) >> gridPoint.y; (*fileStream) >> gridPoint.z;
				}
				/** Get first two points of the next growing direction  */
				(*fileStream) >> temp;
				(*fileStream) >> gridPoint.x; (*fileStream) >> gridPoint.y; (*fileStream) >> gridPoint.z;
				(*fileStream) >> temp;
				(*fileStream) >> tempPoint.x; (*fileStream) >> tempPoint.y; (*fileStream) >> tempPoint.z;
				if (fabs(tempPoint.y - gridPoint.y) > m_minLength) { /** X Y Z */
					//Return to the grid parsing position:
					fileStream->seekg(parsingPosition, ios_base::beg);
					Logger::get() << "Grid leading dimensions: x, y, z" << endl;
					for (int k = 0; k < m_dimensions.z; k++) {
						for (int j = 0; j < m_dimensions.y; j++) {
							for (int i = 0; i < m_dimensions.x; i++) {
								(*fileStream) >> temp;
								(*fileStream) >> gridPoint.x; (*fileStream) >> gridPoint.y; (*fileStream) >> gridPoint.z;
								/*gridPoint *= 0.01;*/
								(*m_pGridPoints)(i, j, k) = gridPoint;
								if ((fileSize > 1024 * 1024) && (100 * processedPoints / numPoints >= percent)) {
									percent = 100 * processedPoints / numPoints;
									percent = (percent / 10) * 10;
									Logger::get() << "  " << percent << " % done..." << endl;
									percent += 10;
								}
								processedPoints++;
							}
						}
					}

				}
				else {
					exitProgram("Grid must be aligned on X, Y, Z");
				}
			}

			Logger::get() << "Grid successfully loaded" << endl;
			Logger::get().setDefaultLogLevel(Log_NormalPriority);

			--m_dimensions.x;
			--m_dimensions.y;
			--m_dimensions.z;
		}


		void HexaGrid::loadSolidCircle(const Vector3 &centerPoint, Scalar circleSize) {
			GridData3D *pGridData3D = getGridData3D();
			for (int i = 0; i < m_dimensions.x; i++) {
				for (int j = 0; j < m_dimensions.y; j++) {
					for (int k = 0; k < m_dimensions.z; k++) {
						if ((centerPoint - pGridData3D->getCenterPoint(i, j, k)).length() < circleSize) {
							m_pSolidMarkers[k*m_dimensions.x*m_dimensions.y + j*m_dimensions.x + i] = true;
						}
					}
				}
			}
		}

		void HexaGrid::exportToFile(const string &gridExportname) {
			gridExportname;
		}


		bool HexaGrid::isInsideCell(Vector3 position, int x, int y, int z) const {
			GridData3D *pGridData3D = getGridData3D();
			Vector3 transformedVector = (position - pGridData3D->getPoint(x, y, z))/pGridData3D->getGridSpacing();
			Vector3 oldVec = pGridData3D->getPoint(x, y, z);
			if (transformedVector.x > 1 || transformedVector.x < 0 ||
				transformedVector.y > 1 || transformedVector.y < 0 ||
				transformedVector.z > 1 || transformedVector.z < 0)
				return false;

			return true;

		}
	}




}