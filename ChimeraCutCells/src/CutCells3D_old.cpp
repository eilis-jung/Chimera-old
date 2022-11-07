#include "Grids/CutCells3D.h"
#include "Mesh/TriangleMesh3D.h"
#include "CGAL/PolygonSurface.h"
#include "CGAL/MeshPatchMapSplitter.h"
#include "Mesh/NonManifoldMesh3D.h"
#include "Mesh/MeshUtils.h"
#include "Physics/PhysicsCore.h"


namespace Chimera {
	namespace CutCells {
		Scalar const CutCells3D::g_smallCellThreshold = 1e-2;


		#pragma region InitializationFunctions
		void CutCells3D::initializeCutSlices(Rendering::PolygonSurface *pPolySurface, faceLocation_t facelocation) {
			vector<vector<Vector3D>> cutLines;
			switch (facelocation) {
				case Chimera::Data::bottomFace:
					cutLines = CGALWrapper::MeshCutter::getInstance()->polygonSlicer(pPolySurface->getCGALPolyehedron(), Vector3(0, 1, 0),
						Vector3(0, m_gridSpacing, 0), Vector3(0, m_gridSpacing, 0), m_pGridData->getDimensions().y - 1);
				break;
				case Chimera::Data::leftFace:
					cutLines = CGALWrapper::MeshCutter::getInstance()->polygonSlicer(pPolySurface->getCGALPolyehedron(), Vector3(1, 0, 0),
						Vector3(m_gridSpacing, 0, 0), Vector3(m_gridSpacing, 0, 0), m_pGridData->getDimensions().x - 1);
				break;
				case Chimera::Data::backFace:
					cutLines = CGALWrapper::MeshCutter::getInstance()->polygonSlicer(pPolySurface->getCGALPolyehedron(), Vector3(0, 0, 1),
						Vector3(0, 0, m_gridSpacing), Vector3(0, 0, m_gridSpacing), m_pGridData->getDimensions().z - 1);
				break;
			}

			//Check cut-slices normal calculation orientation
			for (int i = 0; i < cutLines.size(); i++) {
				Vector3D pointsVecCentroid = Data::MeshUtils::calculateCentroid(cutLines[i]);
				if (Data::MeshUtils::signedDistanceFunction(pointsVecCentroid, cutLines[i], facelocation) > 0) {
					reverse(cutLines[i].begin(), cutLines[i].end());
				}
			}
						
			for (int i = 0; i < cutLines.size(); i++) {
				vector<LineMesh<Vector3D>::params_t> lineMeshParamsVec;
				int k;
				switch (facelocation) {
					case Chimera::Data::bottomFace:
						k = floor(cutLines[i].front().y / m_gridSpacing);
					break;
					case Chimera::Data::leftFace:
						k = floor(cutLines[i].front().x / m_gridSpacing);
					break;
					case Chimera::Data::backFace:
						k = floor(cutLines[i].front().z / m_gridSpacing);
					break;
				}
				int lastK = k;
				Scalar totalLineSize = 0;
				do  {
					LineMesh<Vector3D>::params_t lineMeshParams;
					for (int j = 0; j < cutLines[i].size(); j++) {
						int nextJ = roundClamp<int>(j + 1, 0, cutLines[i].size());
						totalLineSize += (cutLines[i][nextJ] - cutLines[i][j]).length();
						lineMeshParams.initialPoints.push_back(cutLines[i][j]);
					}
					lineMeshParams.closedMesh = true;
					lineMeshParamsVec.push_back(lineMeshParams);

					i++;
					lastK = k;
					if (i == cutLines.size())
						break;

					switch (facelocation) {
						case Chimera::Data::bottomFace:
							k = floor(cutLines[i].front().y / m_gridSpacing);
						break;
						case Chimera::Data::leftFace:
							k = floor(cutLines[i].front().x / m_gridSpacing);
						break;
						case Chimera::Data::backFace:
							k = floor(cutLines[i].front().z / m_gridSpacing);
						break;
					}
				} while (i < cutLines.size() && k == lastK);
				i--;

				vector<LineMesh<Vector3D> *> lineMeshVec;
				for (int j = 0; j < lineMeshParamsVec.size(); j++) {
					if (lineMeshParamsVec[j].initialPoints.size() > 0) 
						lineMeshVec.push_back(new LineMesh<Vector3D>(lineMeshParamsVec[j]));
				}

				if (lineMeshVec.size() > 0) {
					dimensions_t sliceLocation = m_pGridData->getDimensions();
					CutSlice3D *pCutSlice;
					switch (facelocation) {
						case Chimera::Data::bottomFace:
							sliceLocation.y = lastK;
							pCutSlice = new CutSlice3D(dimensions_t(sliceLocation.x, sliceLocation.z), bottomFace, lastK, m_gridSpacing, lineMeshVec);
							m_cutCellsXZVec.push_back(pCutSlice);
						break;
						case Chimera::Data::leftFace:
							sliceLocation.x = lastK;
							pCutSlice = new CutSlice3D(dimensions_t(sliceLocation.z, sliceLocation.y), leftFace, lastK, m_gridSpacing, lineMeshVec);
							m_cutCellsYZVec.push_back(pCutSlice);
						break;
						case Chimera::Data::backFace:
							sliceLocation.z = lastK;
							pCutSlice = new CutSlice3D(dimensions_t(sliceLocation.x, sliceLocation.y), backFace, lastK, m_gridSpacing, lineMeshVec);
							m_cutCellsXYVec.push_back(pCutSlice);
						break;
					}
				}
			}
		}

		void CutCells3D::initializeCutSlices(const vector<Rendering::PolygonSurface *> &pPolySurfaces, faceLocation_t facelocation) {
			vector<vector<Vector3D>> cutLines;
			vector<CGALWrapper::CgalPolyhedron *> pCgalPolyhedrons;
			for (int i = 0; i < pPolySurfaces.size(); i++) {
				pCgalPolyhedrons.push_back(pPolySurfaces[i]->getCGALPolyehedron());
			}
			switch (facelocation) {
			case Chimera::Data::bottomFace:
				cutLines = CGALWrapper::MeshCutter::getInstance()->polygonSlicer(pCgalPolyhedrons, Vector3(0, 1, 0),
					Vector3(0, m_gridSpacing, 0), Vector3(0, m_gridSpacing, 0), m_pGridData->getDimensions().y - 1);
				break;
			case Chimera::Data::leftFace:
				cutLines = CGALWrapper::MeshCutter::getInstance()->polygonSlicer(pCgalPolyhedrons, Vector3(1, 0, 0),
					Vector3(m_gridSpacing, 0, 0), Vector3(m_gridSpacing, 0, 0), m_pGridData->getDimensions().x - 1);
				break;
			case Chimera::Data::backFace:
				cutLines = CGALWrapper::MeshCutter::getInstance()->polygonSlicer(pCgalPolyhedrons, Vector3(0, 0, 1),
					Vector3(0, 0, m_gridSpacing), Vector3(0, 0, m_gridSpacing), m_pGridData->getDimensions().z - 1);
				break;
			}

			//Check cut-slices normal calculation orientation
			for (int i = 0; i < cutLines.size(); i++) {
				Vector3D pointsVecCentroid = Data::MeshUtils::calculateCentroid(cutLines[i]);
				if (Data::MeshUtils::signedDistanceFunction(pointsVecCentroid, cutLines[i], facelocation) > 0) {
					reverse(cutLines[i].begin(), cutLines[i].end());
				}
			}

			for (int i = 0; i < cutLines.size(); i++) {
				vector<LineMesh<Vector3D>::params_t> lineMeshParamsVec;
				int k;
				switch (facelocation) {
				case Chimera::Data::bottomFace:
					k = floor(cutLines[i].front().y / m_gridSpacing);
					break;
				case Chimera::Data::leftFace:
					k = floor(cutLines[i].front().x / m_gridSpacing);
					break;
				case Chimera::Data::backFace:
					k = floor(cutLines[i].front().z / m_gridSpacing);
					break;
				}
				int lastK = k;
				Scalar totalLineSize = 0;
				do  {
					LineMesh<Vector3D>::params_t lineMeshParams;
					for (int j = 0; j < cutLines[i].size(); j++) {
						int nextJ = roundClamp<int>(j + 1, 0, cutLines[i].size());
						totalLineSize += (cutLines[i][nextJ] - cutLines[i][j]).length();
						lineMeshParams.initialPoints.push_back(cutLines[i][j]);
					}
					if ((lineMeshParams.initialPoints.front() - lineMeshParams.initialPoints.back()).length() < singlePrecisionThreshold/m_gridSpacing) {
						lineMeshParams.closedMesh = true;
					}
					else {
						lineMeshParams.closedMesh = false;
					}
					
					lineMeshParamsVec.push_back(lineMeshParams);

					i++;
					lastK = k;
					if (i == cutLines.size())
						break;

					switch (facelocation) {
					case Chimera::Data::bottomFace:
						k = floor(cutLines[i].front().y / m_gridSpacing);
						break;
					case Chimera::Data::leftFace:
						k = floor(cutLines[i].front().x / m_gridSpacing);
						break;
					case Chimera::Data::backFace:
						k = floor(cutLines[i].front().z / m_gridSpacing);
						break;
					}
				} while (i < cutLines.size() && k == lastK);
				i--;

				vector<LineMesh<Vector3D> *> lineMeshVec;
				for (int j = 0; j < lineMeshParamsVec.size(); j++) {
					if (lineMeshParamsVec[j].initialPoints.size() > 0)
						lineMeshVec.push_back(new LineMesh<Vector3D>(lineMeshParamsVec[j]));
				}

				if (lineMeshVec.size() > 0) {
					dimensions_t sliceLocation = m_pGridData->getDimensions();
					CutSlice3D *pCutSlice;
					switch (facelocation) {
					case Chimera::Data::bottomFace:
						sliceLocation.y = lastK;
						pCutSlice = new CutSlice3D(dimensions_t(sliceLocation.x, sliceLocation.z), bottomFace, lastK, m_gridSpacing, lineMeshVec);
						m_cutCellsXZVec.push_back(pCutSlice);
						break;
					case Chimera::Data::leftFace:
						sliceLocation.x = lastK;
						pCutSlice = new CutSlice3D(dimensions_t(sliceLocation.z, sliceLocation.y), leftFace, lastK, m_gridSpacing, lineMeshVec);
						m_cutCellsYZVec.push_back(pCutSlice);
						break;
					case Chimera::Data::backFace:
						sliceLocation.z = lastK;
						pCutSlice = new CutSlice3D(dimensions_t(sliceLocation.x, sliceLocation.y), backFace, lastK, m_gridSpacing, lineMeshVec);
						m_cutCellsXYVec.push_back(pCutSlice);
						break;
					}
				}
			}
		}

		void CutCells3D::tagSpecialCells(Rendering::PolygonSurface *pPolySurface) {
			m_isSpecialCell.push_back(Array3D<bool>(m_pGridData->getDimensions()));
			m_isSpecialCell.back().assign(false);
			m_isBoundaryCell.push_back(Array3D<bool>(m_pGridData->getDimensions()));
			m_isBoundaryCell.back().assign(false);

			/*for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
				for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
					for (int k = 0; k < m_pGridData->getDimensions().z; k++) {
						if (m_leftFaceCrossings(i, j, k).size() > 0 ||
							m_bottomFaceCrossings(i, j, k).size() > 0 ||
							m_backFaceCrossings(i, j, k).size() > 0) {
							m_isSpecialCell.back()(i, j, k) = true;
						}
					}
				}
			}*/

			Scalar dx = m_pGridData->getScaleFactor(0, 0, 0).x;
			for (int i = 0; i < pPolySurface->getVertices().size(); i++) {
				Vector3D transformedPosition = pPolySurface->getVertices()[i] / dx;
				dimensions_t currDim(floor(transformedPosition.x), floor(transformedPosition.y), floor(transformedPosition.z));
				m_isSpecialCell.back()(currDim) = true;
			}
			
			for (int i = 0; i < pPolySurface->getFacesCentroids().size(); i++) {
				Vector3D transformedPosition = pPolySurface->getFacesCentroids()[i] / dx;
				dimensions_t currDim(floor(transformedPosition.x), floor(transformedPosition.y), floor(transformedPosition.z));
				m_isSpecialCell.back()(currDim) = true;
				if (pPolySurface->getFaces()[i].borderFace) {
					m_isBoundaryCell.back()(currDim) = true;
				}
			}
			//Do not tag boundary cells, as we assume that the boundary triangle meshes will not be degenerate 
			//Do not tag boundary cells, as we assume that the boundary triangle meshes will not be degenerate 
			//Do not tag boundary cells, as we assume that the boundary triangle meshes will not be degenerate 

		}
		#pragma endregion InitializationFunctions

		#pragma region PrivateFunctionalities
		int CutCells3D::getRowIndex(PoissonMatrix *pPoissonMatrix, const dimensions_t &currDim, faceLocation_t faceLocation) {
			if (currDim.x == 1 || currDim.x == m_pGridData->getDimensions().x - 1 ||
				currDim.y == 1 || currDim.y == m_pGridData->getDimensions().y - 1 ||
				currDim.z == 1 || currDim.z == m_pGridData->getDimensions().z - 1) {
				//We don't set neighbors if the row is adjacent here
				return -1;
			}
			switch (faceLocation) {
			case rightFace:
				return pPoissonMatrix->getRowIndex(currDim.x, currDim.y - 1, currDim.z - 1);
				break;
			case bottomFace:
				return pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y - 2, currDim.z - 1);
				break;
			case leftFace:
				return pPoissonMatrix->getRowIndex(currDim.x - 2, currDim.y - 1, currDim.z - 1);
				break;
			case topFace:
				return pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y, currDim.z - 1);
				break;
			case frontFace:
				return pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y - 1, currDim.z);
				break;
			case backFace:
				return pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y - 1, currDim.z - 2);
				break;
			default:
				return -1;
				break;
			}
		}
		Vector3D CutCells3D::getFaceNormal(faceLocation_t faceLocation) {
			switch (faceLocation) {
			case rightFace:
				return Vector3D(1, 0, 0);
				break;
			case bottomFace:
				return Vector3D(0, -1, 0);
				break;
			case leftFace:
				return Vector3D(-1, 0, 0);
				break;
			case topFace:
				return Vector3D(0, 1, 0);
				break;
			case frontFace:
				return Vector3D(0, 0, 1);
				break;
			case backFace:
				return Vector3D(0, 0, -1);
				break;
			default:
				return Vector3D(0, 0, 0);
				break;
			}
		}
		Vector3 CutCells3D::interpolateFaceVelocity(const vector<Vector3D> &points, const vector<Mesh<Vector3D>::nodeType_t> &nodeTypes, const vector<Vector3> &velocities, faceLocation_t faceLocation, const dimensions_t &currDimensions) {
			Vector3 faceVelocity;
			int numVelocitiesAdded = 0;
			for (int i = 0; i < points.size(); i++) {
				dimensions_t gridFaceLocation(floor(points[i].x), floor(points[i].y), floor(points[i].z));
				if (nodeTypes[i] == Mesh<Vector3D>::gridNode) {
					//int gridFaceTemp = isOnGridFace(points[i], m_gridSpacing, gridFaceLocation);
					if (/*gridFaceTemp == 1 && */(faceLocation == backFace || faceLocation == frontFace)) {
						if (currDimensions.z == gridFaceLocation.z) {
							faceVelocity += velocities[i];
							numVelocitiesAdded++;
						}
					}
					else if (/*gridFaceTemp == 2 && */(faceLocation == bottomFace || faceLocation == topFace)) {
						if (currDimensions.y == gridFaceLocation.y) {
							faceVelocity += velocities[i];
							numVelocitiesAdded++;
						}
					}
					else if (/*gridFaceTemp == 3 && */(faceLocation == leftFace || faceLocation == rightFace)) {
						if (currDimensions.x == gridFaceLocation.x) {
							faceVelocity += velocities[i];
							numVelocitiesAdded++;
						}
					}
				}
			}
			if (numVelocitiesAdded == 0) {
				throw exception("CutCells3D: Invalid cell face interpolation");
			}
			faceVelocity /= numVelocitiesAdded;
			return faceVelocity;
		}

		Vector3 CutCells3D::getFaceVelocity(int currVoxelIndex, faceLocation_t faceLocation) {
			Vector3 faceVelocity;
			int numFaces = 0;
			for (int i = 0; i < m_cutVoxels[currVoxelIndex].cutFaces.size(); i++) {
				if (m_cutVoxels[currVoxelIndex].cutFacesLocations[i] == faceLocation) {
					faceVelocity += convertToVector3F(getFaceNormal(faceLocation)*m_cutVoxels[currVoxelIndex].cutFaces[i]->m_velocity);
					++numFaces;
				}
			}
			if (numFaces == 0) {
				throw exception("CutCells3D: Invalid face location on getFaceVelocity");
			}
			faceVelocity /= numFaces;
			return faceVelocity;
		}

		Vector3 CutCells3D::interpolateMixedNodeVelocity(nodeVelocityField3D_t *pNodeVelocityField, int currVoxelIndex, int currPointIndex) {
			const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(currVoxelIndex).getPoints();
			const vector<typename Mesh<Vector3D>::meshPolygon_t> &polygons = pNodeVelocityField->pMeshes->at(currVoxelIndex).getMeshPolygons();
			const vector<Vector3> &velocities = pNodeVelocityField->nodesVelocities[currVoxelIndex];
			const vector<typename Mesh<Vector3D>::nodeType_t> &nodeTypes = pNodeVelocityField->pMeshes->at(currVoxelIndex).getNodeTypes();
			map <int, vector<CutFace<Vector3D> *>> &faceMap = pNodeVelocityField->pMeshes->at(currVoxelIndex).getCutFaceToMixedNodeMap();
			map <int, vector<Vector3D>> &faceNormalMap = pNodeVelocityField->pMeshes->at(currVoxelIndex).getCutFaceNormalsToMixedNodeMap();
			
			MatrixNxND leastSquaresMat(faceNormalMap[currPointIndex].size(), 3);
			for (int i = 0; i < faceNormalMap[currPointIndex].size(); i++) {
				for (int j = 0; j < 3; j++) {
					leastSquaresMat(i, j) = faceNormalMap[currPointIndex][i][j];
				}
			}
			MatrixNxND leastSquaresMatTranspose(leastSquaresMat);
			leastSquaresMatTranspose.transpose();

			MatrixNxND fluxMat(faceNormalMap[currPointIndex].size(), 1);
			for (int i = 0; i < faceNormalMap[currPointIndex].size(); i++) {
				fluxMat(i, 0) = faceNormalMap[currPointIndex][i].dot(faceMap[currPointIndex][i]->m_velocity);
			}

			MatrixNxND finalFluxMat(leastSquaresMatTranspose*fluxMat); //3x1 matrix
			Matrix3x3D normalMatrix(leastSquaresMatTranspose*leastSquaresMat);
			normalMatrix.invert();

			Vector3 mixedNodeVel = convertToVector3F(normalMatrix*Vector3D(finalFluxMat(0, 0), finalFluxMat(1, 0), finalFluxMat(2, 0)));
			//Vector3 mixedNodeVel = normalMatrix*Vector3(geometryNormal.dot(geometryVelocity), v1Normal.dot(v1Interp), v2Normal.dot(v2Interp));
			return mixedNodeVel;
		}

		Vector3 CutCells3D::interpolateMixedNodeVelocityWeighted(nodeVelocityField3D_t *pNodeVelocityField, int currVoxelIndex, int currPointIndex, bool extraDimensions) {
			const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(currVoxelIndex).getPoints();
			const vector<typename Mesh<Vector3D>::meshPolygon_t> & polygons = pNodeVelocityField->pMeshes->at(currVoxelIndex).getMeshPolygons();
			const vector<Vector3> &velocities = pNodeVelocityField->nodesVelocities[currVoxelIndex];
			const vector<typename Mesh<Vector3D>::nodeType_t> &nodeTypes = pNodeVelocityField->pMeshes->at(currVoxelIndex).getNodeTypes();
			map <int, vector<CutFace<Vector3D> *>> &faceMap = pNodeVelocityField->pMeshes->at(currVoxelIndex).getCutFaceToMixedNodeMap();
			map <int, vector<Vector3D>> &faceNormalMap = pNodeVelocityField->pMeshes->at(currVoxelIndex).getCutFaceNormalsToMixedNodeMap();

			//Find the missing face location
			bool leftFaces = false, bottomFaces = false, backFaces = false;
			for (int i = 0; i < faceMap[currPointIndex].size(); i++) {
				if (abs(abs(faceNormalMap[currPointIndex][i].x) - 1) < singlePrecisionThreshold) { //Left or right face
					leftFaces = true;
				}
				if (abs(abs(faceNormalMap[currPointIndex][i].y) - 1) < singlePrecisionThreshold) { //Bottom or top face
					bottomFaces = true;
				}
				if (abs(abs(faceNormalMap[currPointIndex][i].z) - 1) < singlePrecisionThreshold) { //Front or back face
					backFaces = true;
				}
			}
			vector<Vector3D> extraNormals;
			vector<DoubleScalar> extraFluxes;
			vector<Vector3D> extraCentroids;
			if (extraDimensions) {
				for (int i = 0; i < m_cutVoxels[currVoxelIndex].cutFaces.size(); i++) {
					if (m_cutVoxels[currVoxelIndex].cutFacesLocations[i] == leftFace || m_cutVoxels[currVoxelIndex].cutFacesLocations[i] == rightFace) {
						if (!leftFaces) {
							extraNormals.push_back(getFaceNormal(m_cutVoxels[currVoxelIndex].cutFacesLocations[i]));
							extraFluxes.push_back(extraNormals.back().dot(m_cutVoxels[currVoxelIndex].cutFaces[i]->m_velocity));
							extraCentroids.push_back(m_cutVoxels[currVoxelIndex].cutFaces[i]->m_centroid);
						}
					}
					else if (m_cutVoxels[currVoxelIndex].cutFacesLocations[i] == bottomFace || m_cutVoxels[currVoxelIndex].cutFacesLocations[i] == topFace) {
						if (!bottomFaces) {
							extraNormals.push_back(getFaceNormal(m_cutVoxels[currVoxelIndex].cutFacesLocations[i]));
							extraFluxes.push_back(extraNormals.back().dot(m_cutVoxels[currVoxelIndex].cutFaces[i]->m_velocity));
							extraCentroids.push_back(m_cutVoxels[currVoxelIndex].cutFaces[i]->m_centroid);
						}
					}
					else if (m_cutVoxels[currVoxelIndex].cutFacesLocations[i] == backFace || m_cutVoxels[currVoxelIndex].cutFacesLocations[i] == frontFace) {
						if (!backFaces) {
							extraNormals.push_back(getFaceNormal(m_cutVoxels[currVoxelIndex].cutFacesLocations[i]));
							extraFluxes.push_back(extraNormals.back().dot(m_cutVoxels[currVoxelIndex].cutFaces[i]->m_velocity));
							extraCentroids.push_back(m_cutVoxels[currVoxelIndex].cutFaces[i]->m_centroid);
						}
					}
				}
			}
			
			
			/** The size of the weighted least squares matrix is the number of faces connected directly to the point,
				given by faceNormalMap[currPointIndex].size() plus the faces that are not directly connected to the point,
				which are stored on the extranormals vector: */
			MatrixNxND leastSquaresMat(faceNormalMap[currPointIndex].size() + extraNormals.size(), 3);
			/** The lestSquaresMat is firstly initialized with the faceNormalMap, then its initialized with the information
				inside the extraNormals vector.*/
			for (int i = 0; i < faceNormalMap[currPointIndex].size(); i++) {
				for (int j = 0; j < 3; j++) {
					leastSquaresMat(i, j) = faceNormalMap[currPointIndex][i][j];
				}
			}
			for (int i = faceNormalMap[currPointIndex].size(); i < faceNormalMap[currPointIndex].size() + extraNormals.size(); i++) {
				for (int j = 0; j < 3; j++) {
					leastSquaresMat(i, j) = extraNormals[i - faceNormalMap[currPointIndex].size()][j];
				}
			}

			MatrixNxND leastSquaresMatTranspose(leastSquaresMat);
			leastSquaresMatTranspose.transpose();

			/** Flux mat is initialized similarly as leastSquaresMat*/
			MatrixNxND fluxMat(faceNormalMap[currPointIndex].size() + extraNormals.size(), 1);
			for (int i = 0; i < faceNormalMap[currPointIndex].size(); i++) {
				fluxMat(i, 0) = faceNormalMap[currPointIndex][i].dot(faceMap[currPointIndex][i]->m_velocity);
			}
			for (int i = faceNormalMap[currPointIndex].size(); i < faceNormalMap[currPointIndex].size() + extraNormals.size(); i++) {
				fluxMat(i, 0) = extraFluxes[i - faceNormalMap[currPointIndex].size()];
			}

			MatrixNxND weightsMat(0.0f, faceNormalMap[currPointIndex].size() + extraNormals.size(), faceNormalMap[currPointIndex].size() + extraNormals.size());
			DoubleScalar totalWeight = 0;
			/** Weights are the inverse square distances of the center of the faces to the node location + a small
				number to avoid division by zero. */
			for (int i = 0; i < faceNormalMap[currPointIndex].size(); i++) {
				weightsMat(i, i) = (faceMap[currPointIndex][i]->m_centroid - points[currPointIndex]).lengthSqr() + singlePrecisionThreshold;
				weightsMat(i, i) = 1 / (weightsMat(i, i));
				//weightsMat(i, i) *= faceMap[currPointIndex][i]->m_areaFraction;
				totalWeight += weightsMat(i, i);
			}
			vector<DoubleScalar> tempWeights;
			for (int i = faceNormalMap[currPointIndex].size(); i < faceNormalMap[currPointIndex].size() + extraNormals.size(); i++) {
				weightsMat(i, i) = (extraCentroids[i - faceNormalMap[currPointIndex].size()] - points[currPointIndex]).lengthSqr() + singlePrecisionThreshold;
				weightsMat(i, i) = 1 / (weightsMat(i, i));
				//weightsMat(i, i) *= faceMap[currPointIndex][i]->m_areaFraction;
				totalWeight += weightsMat(i, i);
			}
			/** The weights are normalized by the sum of the total weights, to increase numerical accuracy:*/
			for (int i = 0; i < faceNormalMap[currPointIndex].size() + extraNormals.size(); i++) {
				weightsMat(i, i) /= totalWeight;
				tempWeights.push_back(weightsMat(i, i));
			}

			/** I use the equation (6) to get the final expression for calculation: */
			MatrixNxND finalFluxMat(leastSquaresMatTranspose*weightsMat*fluxMat); //3x1 matrix
			Matrix3x3D normalMatrix(leastSquaresMatTranspose*weightsMat*leastSquaresMat);
			normalMatrix.invert();

			/** Then u is found by mulitplying the final flux mat and the inverted normal matrix. The
				convertToVector3F converts the vector to single point-precision*/
			Vector3 mixedNodeVel = convertToVector3F(normalMatrix*Vector3D(finalFluxMat(0, 0), finalFluxMat(1, 0), finalFluxMat(2, 0)));
			return mixedNodeVel;
		}

		Vector3 CutCells3D::interpolateMixedNodeFaceVelocity(nodeVelocityField3D_t *pNodeVelocityField, int currVoxelIndex, int currPointIndex) {
			const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(currVoxelIndex).getPoints();
			const vector<typename Mesh<Vector3D>::meshPolygon_t> & polygons = pNodeVelocityField->pMeshes->at(currVoxelIndex).getMeshPolygons();
			const vector<Vector3> &velocities = pNodeVelocityField->nodesVelocities[currVoxelIndex];
			const vector<typename Mesh<Vector3D>::nodeType_t> &nodeTypes = pNodeVelocityField->pMeshes->at(currVoxelIndex).getNodeTypes();
			map <int, vector<CutFace<Vector3D> *>> &faceMap = pNodeVelocityField->pMeshes->at(currVoxelIndex).getCutFaceToMixedNodeMap();
			
			for (int i = 0; i < faceMap[currPointIndex].size(); i++) {
				if (faceMap[currPointIndex][i]->m_faceLocation != geometryFace) {
					return convertToVector3F(faceMap[currPointIndex][i]->m_velocity);
				}
			}
			return Vector3(0, 0, 0);
		}


		void CutCells3D::addFaceMixedNodePointsToMap(CutFace<Vector3D> *pFace) {
			for (int i = 0; i < pFace->m_cutEdges.size(); i++) {
				if (isOnGridEdge(pFace->m_cutEdges[i]->m_initialPoint, m_gridSpacing) > 0) {
					int allNodesIndex = addNodeToAllMixedNodes(pFace->m_cutEdges[i]->m_initialPoint);
					if (allNodesIndex < m_mixedNodesFacesMap.size()) {
						m_mixedNodesFacesMap[allNodesIndex].push_back(pFace);
					}
					else {
						m_mixedNodesFacesMap.push_back(vector<CutFace<Vector3D> *>());
						m_mixedNodesFacesMap.back().push_back(pFace);
					}
				}
			}
		}

		int CutCells3D::addNodeToAllMixedNodes(const Vector3D &currMixedNode) {
			int mixedNodeIndex = findMixedNode(currMixedNode);
			if (mixedNodeIndex == -1) {
				m_allMixedNodes.push_back(currMixedNode);
				return m_allMixedNodes.size() - 1;
			}
			else {
				return mixedNodeIndex;
			}		
		}

		int CutCells3D::findMixedNode(const Vector3D &currMixedNode) {
			DoubleScalar tempPrecision = singlePrecisionThreshold / m_gridSpacing;
			for (int i = 0; i < m_allMixedNodes.size(); i++) {
				if ((m_allMixedNodes[i] - currMixedNode).length() < tempPrecision) {
					return i;
				}
			}
			return -1;
		}

		bool CutCells3D::addFaceToMixedNodeMap(int mixedNodeID, CutFace<Vector3D> *pFace, faceLocation_t cutFaceLocation,  map <int, vector<CutFace<Vector3D>*>> &nodeFaceMap, map<int, vector<Vector3D>> &nodeFaceNormalsMap) {
			if (nodeFaceMap.find(mixedNodeID) != nodeFaceMap.end()) {
				for (int i = 0; i < nodeFaceMap[mixedNodeID].size(); i++) {
					if (nodeFaceMap[mixedNodeID][i] == pFace) {
						return false;
					}
				}
			}
			if (cutFaceLocation != geometryFace) {
				Vector3D faceNormal;
				switch (cutFaceLocation) {
					case leftFace:
					case rightFace:
						faceNormal = Vector3D(1, 0, 0);
					break;
					case bottomFace:
					case topFace:
						faceNormal = Vector3D(0, 1, 0);
					break;
					case backFace:
					case frontFace:
						faceNormal = Vector3D(0, 0, 1);
					break;
				}
				nodeFaceNormalsMap[mixedNodeID].push_back(faceNormal);
			} else {
				nodeFaceNormalsMap[mixedNodeID].push_back(pFace->m_normal);
			}
			nodeFaceMap[mixedNodeID].push_back(pFace);
			return true;
		}

		int CutCells3D::findPatchMap(dimensions_t currDimensions, const simpleFace_t &currFace, const Array3D<vector<Rendering::MeshPatchMap *>> &meshMap, Rendering::PolygonSurface *pTargetPolySurface) {
			for (int i = 0; i < meshMap(currDimensions).size(); i++) {
				if (meshMap(currDimensions)[i]->pPolySurface == pTargetPolySurface) {
					////We have to check if the current mesh patch map is connected
					////to this new face being added
					//MeshPatchMap *pPatchMap = m_multiMeshPatchMap(currDimensions)[i];
					//simpleFace_t lastFace = pTargetPolySurface->getFaces()[pPatchMap->faces.back()];
					////I think its enough to look only on the last added face for a connected edge
					//if (!lastFace.isConnectedTo(currFace)) {
					//	//We have to create a new patch map, return -1
					//	return -1;
					//}
					return i;
				}	
			}
			return -1;
		}

		vector<Rendering::MeshPatchMap *> CutCells3D::getListOfPossibleTrianglesCollision(const Vector3 &initialPoint, const Vector3 &finalPoint, Scalar dt) {
			vector<Rendering::MeshPatchMap*> possibleCollisions;
			Vector3 effectiveVelocity = (finalPoint - initialPoint) / dt;
			Scalar cflLength = (finalPoint - initialPoint).length();
			int cfl = ceil(effectiveVelocity.length() / m_gridSpacing);

			for (int i = 0; i < m_cutVoxels.size(); i++) {
				Vector3 centroidVec = convertToVector3F(m_cutVoxels[i].centroid);
				if ((initialPoint - centroidVec).length() < cflLength) {
					dimensions_t currDimensions(centroidVec.x / m_gridSpacing, centroidVec.y / m_gridSpacing, centroidVec.z / m_gridSpacing);
					for (int i = 0; i < m_multiMeshPatchMap(currDimensions).size(); i++) {
						possibleCollisions.push_back(m_multiMeshPatchMap(currDimensions)[i]);
					}
				}
				else if ((finalPoint - centroidVec).length() < cflLength) {
					dimensions_t currDimensions(centroidVec.x / m_gridSpacing, centroidVec.y / m_gridSpacing, centroidVec.z / m_gridSpacing);
					for (int i = 0; i < m_multiMeshPatchMap(currDimensions).size(); i++) {
						possibleCollisions.push_back(m_multiMeshPatchMap(currDimensions)[i]);
					}
				}
			}

			return possibleCollisions;
		}

		void CutCells3D::visitFacesAndAddToCutNodes(const Vector3D &mixedNodePoint, int mixedNodeID, int currCutVoxelID, map<int, bool> &visitedVoxels, map <int, vector<CutFace<Vector3D>*>> &nodeFaceMap, map<int, vector<Vector3D>> &nodeFaceNormalsMap) {
			CutVoxel &cutVoxel = m_cutVoxels[currCutVoxelID];
			Mesh<Vector3D> *pMesh = &m_pNodeVelocityField->pMeshes->at(currCutVoxelID);
			visitedVoxels[cutVoxel.ID] = true;
			DoubleScalar tempPrecision = doublePrecisionThreshold / m_gridSpacing;
			for (int i = 0; i < cutVoxel.cutFaces.size(); i++) {
				CutFace<Vector3D> *pCutFace = cutVoxel.cutFaces[i];
				for (int j = 0; j < pCutFace->m_cutEdges.size(); j++) {
					if ((pCutFace->m_cutEdges[j]->m_initialPoint - mixedNodePoint).length() < tempPrecision) { //Point is the mixed node that we are searching for
						if (addFaceToMixedNodeMap(mixedNodeID, pCutFace, cutVoxel.cutFacesLocations[i], nodeFaceMap, nodeFaceNormalsMap) && cutVoxel.cutFacesLocations[i] != geometryFace) {
							int n1, n2;
							pCutFace->getNeighbors(n1, n2);
							int nextCutVoxelID = n1 == cutVoxel.ID? n2 : n1;
							if (visitedVoxels.find(nextCutVoxelID) == visitedVoxels.end()) {
								visitFacesAndAddToCutNodes(mixedNodePoint, mixedNodeID, nextCutVoxelID, visitedVoxels, nodeFaceMap, nodeFaceNormalsMap);
							}
						}
					}
				}
			}
		}

		
		
		#pragma endregion

		/************************************************************************/
		/* Public interface                                                     */
		/************************************************************************/

		#pragma region Constructors
		CutCells3D::CutCells3D(HexaGrid *pHexaGrid, Scalar thinObjectSize) :
			m_cellToSpecialMap(pHexaGrid->getDimensions()), m_leftSpecialFaces(pHexaGrid->getDimensions()),
			m_bottomSpecialFaces(pHexaGrid->getDimensions()), m_backSpecialFaces(pHexaGrid->getDimensions()), 
			m_multiMeshPatchMap(pHexaGrid->getDimensions()){
			m_useSubthinObjectInformation = true;
			m_pGrid = pHexaGrid;
			m_pGridData = pHexaGrid->getGridData3D();
			m_gridSpacing = m_pGridData->getScaleFactor(0, 0, 0).x;

			m_cellToSpecialMap.assign(-1);
			m_solidBoundaryType = Solid_NoSlip;
			m_mixedNodeInterpolationType = WeightedNoExtraDimensions;
			m_maxNumberOfCells = thinObjectSize * 40 / m_gridSpacing;

			//Tolerance is set to be about 5% of gridSpacing
			m_gridTolerance = 0.005*0.01; //Magical number

			//m_specialCells.reserve(m_maxNumberOfCells);
			m_specialDivergents.reserve(m_maxNumberOfCells);
			m_specialPressures.reserve(m_maxNumberOfCells);
		}
		#pragma endregion

		#pragma region AccessFunctions
		vector<CutFace<Vector3D> *> & CutCells3D::getFaceVector(const dimensions_t &index, faceLocation_t faceLocation) {
			switch (faceLocation) {
			case rightFace:
				return m_leftSpecialFaces(index.x + 1, index.y, index.z);
				break;
			case leftFace:
				return m_leftSpecialFaces(index.x, index.y, index.z);
				break;

			case topFace:
				return m_bottomSpecialFaces(index.x, index.y + 1, index.z);
				break;
			case bottomFace:
				return m_bottomSpecialFaces(index.x, index.y, index.z);
				break;

			case frontFace:
				return m_backSpecialFaces(index.x, index.y, index.z + 1);
				break;
			case backFace:
				return m_backSpecialFaces(index.x, index.y, index.z);
				break;

			default:
				return m_bottomSpecialFaces(index.x, index.y, index.z);
				break;
			}
		}

		int CutCells3D::getCutVoxelIndex(const Vector3D & position, vector<Mesh<Vector3D>> *pMeshes) {
			dimensions_t regularGridIndex = dimensions_t(position.x, position.y, position.z);
			if (isSpecialCell(regularGridIndex.x, regularGridIndex.y, regularGridIndex.z)) {
				if (isBoundaryCell(regularGridIndex.x, regularGridIndex.y, regularGridIndex.z)) {
					return m_cellToSpecialMap(regularGridIndex.x, regularGridIndex.y, regularGridIndex.z);
				}
				int specialCellIndex = m_cellToSpecialMap(regularGridIndex.x, regularGridIndex.y, regularGridIndex.z);

				CutVoxel currCell = m_cutVoxels[specialCellIndex];
				while (specialCellIndex < getNumberOfCells() && currCell.regularGridIndex == regularGridIndex) {
					if (!currCell.getMesh()->hasTriangleMesh())
						return specialCellIndex;

					if (pMeshes->at(specialCellIndex).isInsideMesh(position*m_gridSpacing)) {
						return specialCellIndex;
					}
					if (specialCellIndex < getNumberOfCells() - 1)
						currCell = m_cutVoxels[++specialCellIndex];
					else
						break;
				}
				return specialCellIndex; // = -1, closest index (probably the particle is on top of a geometry face)
			}
			Logger::getInstance()->get() << "Special cell not found on cell (" << regularGridIndex.x << ", " << regularGridIndex.y << ", " << regularGridIndex.z << "), ";
			Logger::getInstance()->get() << "position: (" << position.x*m_gridSpacing << ", " << position.y*m_gridSpacing << ", " << position.z*m_gridSpacing << ")" << endl;
			//Probably an error on insideMeshFunction (point on face). Return any special cell
			return m_cellToSpecialMap(regularGridIndex.x, regularGridIndex.y, regularGridIndex.z);

		}

		void CutCells3D::setThinObjectVelocities(const vector<Vector3> &velocities) {
			m_thinObjectVelocities = velocities;
			for (int ithVel = 0; ithVel < velocities.size(); ithVel++) {
				for (int i = 0; i < getNumberOfCells(); i++) {
					CutVoxel cutVoxel = getCutVoxel(i);
					for (int j = 0; j < cutVoxel.cutFaces.size(); j++) {
						CutFace<Vector3D> *pFace = cutVoxel.cutFaces[j];
						Vector3D rotatationalVelocity;
						if (cutVoxel.cutFacesLocations[j] == geometryFace && pFace->m_pPolygonSurface->getPolygonID() == ithVel) {
							Vector3D pointToCentroid = pFace->getCentroid() - convertToVector3D(pFace->m_pPolygonSurface->getCentroid());
							Vector3D rotationAxis(0, 0, 1);
							DoubleScalar elapsedTime = Rendering::PhysicsCore::getInstance()->getElapsedTime();
							if (elapsedTime >= pFace->m_pPolygonSurface->getRotationFunction().startingTime && elapsedTime < pFace->m_pPolygonSurface->getRotationFunction().endingTime &&
								abs(pointToCentroid.dot(rotationAxis)) > singlePrecisionThreshold) {
								rotatationalVelocity = pointToCentroid.cross(rotationAxis)*pFace->m_pPolygonSurface->getRotationFunction().speed;
							}
							pFace->m_velocity = convertToVector3D(velocities[ithVel]) - rotatationalVelocity;
						}
					}
				}
			}
		}
		#pragma endregion

		#pragma region Functionalities
		void CutCells3D::initializeCellFaces() {
			Logger::getInstance()->get() << "	Initializing cell faces" << endl;
			Scalar dx = m_pGridData->getScaleFactor(0, 0, 0).x;

			/** Left faces: plane YZ */
			for (int i = 0; i < m_cutCellsYZVec.size(); i++) {
				const vector<NonManifoldMesh2D *> & nonManifoldMeshes = m_cutCellsYZVec[i]->getNonManifoldMeshes();
				CutSlice3D *pCutSlice = m_cutCellsYZVec[i];
				
				for (int j = 0; j < nonManifoldMeshes.size(); j++) {
					dimensions_t cellIndex = nonManifoldMeshes[j]->getFaceDimensions();

					vector<CutFace<Vector3D> *> leftCutFaces = nonManifoldMeshes[j]->split();
					for (int k = 0; k < leftCutFaces.size(); k++) {
						CutFace<Vector3D> *pFace = leftCutFaces[k];
						vector<Vector3D> tempFacePoints;
						for (int l = 0; l < pFace->m_cutEdges.size(); l++) {
							tempFacePoints.push_back(pFace->getEdgeInitialPoint(l));
						}
						pFace->m_areaFraction = calculatePolygonArea(tempFacePoints, getFaceNormal(leftFace)) / (dx*dx);
					}
					m_leftSpecialFaces(cellIndex) = leftCutFaces;
				}
			}

			/** Bottom faces: plane XZ */
			for (int i = 0; i < m_cutCellsXZVec.size(); i++) {
				const vector<NonManifoldMesh2D *> & nonManifoldMeshes = m_cutCellsXZVec[i]->getNonManifoldMeshes();
				CutSlice3D *pCutSlice = m_cutCellsXZVec[i];

				for (int j = 0; j < nonManifoldMeshes.size(); j++) {
					dimensions_t cellIndex = nonManifoldMeshes[j]->getFaceDimensions();

					vector<CutFace<Vector3D> *> bottomCutFaces = nonManifoldMeshes[j]->split();
					for (int k = 0; k < bottomCutFaces.size(); k++) {
						CutFace<Vector3D> *pFace = bottomCutFaces[k];
						vector<Vector3D> tempFacePoints;
						for (int l = 0; l < pFace->m_cutEdges.size(); l++) {
							tempFacePoints.push_back(pFace->getEdgeInitialPoint(l));
						}
						pFace->m_areaFraction = calculatePolygonArea(tempFacePoints, getFaceNormal(bottomFace)) / (dx*dx);
					}
					m_bottomSpecialFaces(cellIndex) = bottomCutFaces;
				}
			}

			/** Back faces: plane XY */
			for (int i = 0; i < m_cutCellsXYVec.size(); i++) {
				const vector<NonManifoldMesh2D *> & nonManifoldMeshes = m_cutCellsXYVec[i]->getNonManifoldMeshes();
				CutSlice3D *pCutSlice = m_cutCellsXYVec[i];

				for (int j = 0; j < nonManifoldMeshes.size(); j++) {
					dimensions_t cellIndex = nonManifoldMeshes[j]->getFaceDimensions();

					vector<CutFace<Vector3D> *> backCutFaces = nonManifoldMeshes[j]->split();
					for (int k = 0; k < backCutFaces.size(); k++) {
						CutFace<Vector3D> *pFace = backCutFaces[k];
						vector<Vector3D> tempFacePoints;
						for (int l = 0; l < pFace->m_cutEdges.size(); l++) {
							tempFacePoints.push_back(pFace->getEdgeInitialPoint(l));
						}
						pFace->m_areaFraction = calculatePolygonArea(tempFacePoints, getFaceNormal(backFace)) / (dx*dx);
					}
					m_backSpecialFaces(cellIndex) = backCutFaces;
				}
			}
			
			/** Initializing full (area fraction = 1) 3-D cut faces that weren't initialized by cutCells2D slices */
			for (int ipass = 0; ipass < 2; ipass++) {
				for (int k = 0; k < m_pGridData->getDimensions().z - 1; k++) {
					int iniJ = (k + ipass) % 2;
					for (int i = 0; i < m_pGridData->getDimensions().x - 1; i++, iniJ = 2 - (iniJ + 1)) {
						for (int j = iniJ; j < m_pGridData->getDimensions().y - 1; j += 2) {
							if (isSpecialCell(i, j, k)) {
								if (m_leftSpecialFaces(i, j, k).size() == 0) 
									m_leftSpecialFaces(i, j, k).push_back(createFullCutFace3D(dimensions_t(i, j, k), leftFace));
								
								if (m_leftSpecialFaces(i + 1, j, k).size() == 0) 
									m_leftSpecialFaces(i + 1, j, k).push_back(createFullCutFace3D(dimensions_t(i + 1, j, k), rightFace));
								
								if (m_bottomSpecialFaces(i, j, k).size() == 0) 
									m_bottomSpecialFaces(i, j, k).push_back(createFullCutFace3D(dimensions_t(i, j, k), bottomFace));
				
								if (m_bottomSpecialFaces(i, j + 1, k).size() == 0)
									m_bottomSpecialFaces(i, j + 1, k).push_back(createFullCutFace3D(dimensions_t(i, j + 1, k), topFace));

								if (m_backSpecialFaces(i, j, k).size() == 0)
									m_backSpecialFaces(i, j, k).push_back(createFullCutFace3D(dimensions_t(i, j, k), backFace));

								if (m_backSpecialFaces(i, j, k + 1).size() == 0)
									m_backSpecialFaces(i, j, k + 1).push_back(createFullCutFace3D(dimensions_t(i, j, k + 1), frontFace));
							}
						}
					}
				}
			}

		}

		void CutCells3D::initializeSpecialCells() {
			Scalar dx = m_pGridData->getScaleFactor(0, 0, 0).x;
			bool onEdgeMixedNodes = m_mixedNodeInterpolationType != Unweighted;
			cout << "On edge mixed nodes: " << onEdgeMixedNodes << endl;
			for (int i = 0; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 0; j < m_pGridData->getDimensions().y - 1; j++) {
					for (int k = 0; k < m_pGridData->getDimensions().z - 1; k++) {
						if (isSpecialCell(i, j, k)) {

							dimensions_t currDim(i, j, k);
							
							if (m_multiMeshPatchMap(currDim).size() == 0) {
								throw("CutCells3D: Error while accessing specials cells meshpatchMaps");
							}

							NonManifoldMesh3D nonManifoldMesh(currDim, m_multiMeshPatchMap(currDim), this);
							vector<CutVoxel> cutVoxels = nonManifoldMesh.split(m_cutVoxels.size(), onEdgeMixedNodes);
							m_cellToSpecialMap(i, j, k) = m_cutVoxels.size();
							for (int l = 0; l < cutVoxels.size(); l++) {
								cutVoxels[l].ID = m_cutVoxels.size();
								m_cutVoxels.push_back(cutVoxels[l]);
								m_specialPressures.push_back(0);
								m_specialDivergents.push_back(0);
							}
						}
					}
				}
			}


			Timer timerTriangulation;
			
			timerTriangulation.start();
			for (int i = 0; i < m_cutVoxels.size(); i++) {
				vector<Vector3D> normals;
				for (int j = 0; j < m_cutVoxels[i].cutFaces.size(); j++) {
					if (m_cutVoxels[i].cutFacesLocations[j] != geometryFace) {
						normals.push_back(getFaceNormal(m_cutVoxels[i].cutFacesLocations[j]));
					}
					else {
						normals.push_back(m_cutVoxels[i].cutFaces[j]->m_normal);
					}
				}
				m_cutVoxels[i].initializeMesh(getGridSpacing(), normals, onEdgeMixedNodes);
			}
			timerTriangulation.stop();
			Logger::getInstance()->get() << "Time spent triangulating cells " << timerTriangulation.secondsElapsed() << endl;
		}

		void CutCells3D::initializeMixedNodesMap() {
			for (int i = 1; i < m_pGridData->getDimensions().x - 1; i++) {
				for (int j = 1; j < m_pGridData->getDimensions().y - 1; j++) {
					for (int k = 1; k < m_pGridData->getDimensions().z - 1; k++) {
						if (getFaceVector(dimensions_t(i, j, k), leftFace).size() > 1) { // If theres only 1 face, its a regular face
							vector<CutFace<Vector3D> *> &currFaceVector = getFaceVector(dimensions_t(i, j, k), leftFace);
							for (int sf = 0; sf < currFaceVector.size(); sf++) {
								CutFace<Vector3D> *pFace = currFaceVector[sf];
								addFaceMixedNodePointsToMap(pFace);
							}
						}
						if (getFaceVector(dimensions_t(i, j, k), bottomFace).size() > 1) { // If theres only 1 face, its a regular face
							vector<CutFace<Vector3D> *> &currFaceVector = getFaceVector(dimensions_t(i, j, k), bottomFace);
							for (int sf = 0; sf < currFaceVector.size(); sf++) {
								CutFace<Vector3D> *pFace = currFaceVector[sf];
								addFaceMixedNodePointsToMap(pFace);
							}
						}
						if (getFaceVector(dimensions_t(i, j, k), backFace).size() > 1) { // If theres only 1 face, its a regular face
							vector<CutFace<Vector3D> *> &currFaceVector = getFaceVector(dimensions_t(i, j, k), backFace);
							for (int sf = 0; sf < currFaceVector.size(); sf++) {
								CutFace<Vector3D> *pFace = currFaceVector[sf];
								addFaceMixedNodePointsToMap(pFace);
							}
						}
					}
				}
			}
		}

		void CutCells3D::initializeMultiMeshPatchMap(const vector<Rendering::PolygonSurface *> &pPolySurfaces) {
			for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
				for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
					for (int k = 0; k < m_pGridData->getDimensions().z; k++) {
						m_multiMeshPatchMap(i, j, k).clear();
					}
				}
			}

			for (int j = 0; j < pPolySurfaces.size(); j++) {
				const vector<Vector3D> & facesCentroids = pPolySurfaces[j]->getFacesCentroids();
				const vector<simpleFace_t> & simpleFaces = pPolySurfaces[j]->getFaces();
				for (unsigned int i = 0; i < facesCentroids.size(); i++) {
					dimensions_t currCellDim;
					currCellDim.x = floor(facesCentroids[i].x / m_gridSpacing);
					currCellDim.y = floor(facesCentroids[i].y / m_gridSpacing);
					currCellDim.z = floor(facesCentroids[i].z / m_gridSpacing);
					
					int patchMapIndex = findPatchMap(currCellDim, simpleFaces[i], m_multiMeshPatchMap,pPolySurfaces[j]);
					MeshPatchMap *pNewPatchMap = NULL;
					if (patchMapIndex == -1) {
						pNewPatchMap = new MeshPatchMap();
						pNewPatchMap->pPolySurface = pPolySurfaces[j];
						m_multiMeshPatchMap(currCellDim).push_back(pNewPatchMap);
					}
					else {
						pNewPatchMap = m_multiMeshPatchMap(currCellDim)[patchMapIndex];
					}
					pNewPatchMap->faces.push_back(i);
					if (simpleFaces[i].borderFace) {
						pNewPatchMap->danglingPatch = true;
					}
				}
			}

			for (int i = 0; i < m_multiMeshPatchMap.getDimensions().x; i++) {
				for (int j = 0; j < m_multiMeshPatchMap.getDimensions().y; j++) {
					for (int k = 0; k < m_multiMeshPatchMap.getDimensions().z; k++) {
						if (m_multiMeshPatchMap(i, j, k).size() > 0) {
							vector <MeshPatchMap *> currMeshPatchMaps = m_multiMeshPatchMap(i, j, k);
							m_multiMeshPatchMap(i, j, k).clear();
							for (int l = 0; l < currMeshPatchMaps.size(); l++) {
								MeshPatchMapSplitter meshPatchSplitter(currMeshPatchMaps[l]);
								vector<MeshPatchMap *> splittedMeshPatchMaps = meshPatchSplitter.split();
								for (int o = 0; o < splittedMeshPatchMaps.size(); o++) {
									m_multiMeshPatchMap(i, j, k).push_back(splittedMeshPatchMaps[o]);
								}
							}
						}
					}
				}
			}

		}
		void CutCells3D::initializeThinBounds(const vector<Rendering::PolygonSurface *> &pPolySurfaces) {

			Timer timingCgal;

			timingCgal.start();

			initializeCutSlices(pPolySurfaces, backFace);
			initializeCutSlices(pPolySurfaces, leftFace);
			initializeCutSlices(pPolySurfaces, bottomFace);


			for (int i = 0; i < pPolySurfaces.size(); i++) {
				//pPolySurfaces[i]->simplifyMesh();
				pPolySurfaces[i]->getCGALPolyehedron()->normalize_border();
				CGALWrapper::triangulatePolyhedron(pPolySurfaces[i]->getCGALPolyehedron());
				//pPolySurfaces[i]->simplifyMesh();
				pPolySurfaces[i]->updateLocalDataStructures();
				pPolySurfaces[i]->fixDuplicatedVertices();
				pPolySurfaces[i]->treatVerticesOnGridPoints();
				pPolySurfaces[i]->reinitializeVBOBuffers();
				timingCgal.stop();
				Logger::getInstance()->get() << "Time spent on CGAL functions " << timingCgal.secondsElapsed() << endl;
				Logger::getInstance()->get() << "Number of polygons on the polygon surface " << pPolySurfaces[i]->getFaces().size() << endl;
			}

			initializeMultiMeshPatchMap(pPolySurfaces);
			for (int i = 0; i < pPolySurfaces.size(); i++) {
				tagSpecialCells(pPolySurfaces[i]);
			}

			initializeCellFaces();
			initializeSpecialCells();
			Timer mixedNodesTimes;
			mixedNodesTimes.start();
			initializeMixedNodesMap();
			mixedNodesTimes.stop();
			Logger::getInstance()->get() << "Time spent generating mixed node maps " << mixedNodesTimes.secondsElapsed() << endl;

			Logger::getInstance()->get() << "The number of cut-cells generated is " << m_cutVoxels.size() << endl;
		}

		void CutCells3D::dumpCellInfo(int ithSelectedCell) {
			CutVoxel selectedVoxel = m_cutVoxels[ithSelectedCell];
			int initialOffset = (m_pGrid->getDimensions().x - 2)*(m_pGrid->getDimensions().y - 2)*(m_pGrid->getDimensions().z - 2);
			int pressureID = initialOffset + selectedVoxel.ID;
			cout << "Dumping cell " << ithSelectedCell << " info:" << endl;
			cout << "\t Pressure info:" << endl;
			cout << "\t \t Center pressure value " << m_pPoissonMatrix->getValue(pressureID, pressureID) << endl;
			cout << "\t Faces info:" << endl;
			for (int i = 0; i < selectedVoxel.cutFaces.size(); i++) {
				CutFace<Vector3D> *pFace = selectedVoxel.cutFaces[i];
				cout << "\t" << i << "th face" << endl;
				cout << "\t Location:" << selectedVoxel.cutFacesLocations[i] << endl;
				cout << "\t Neighbors: " << pFace->m_faceNeighbors[0] << " " << pFace->m_faceNeighbors[1] << endl;
				cout << "\t Area fraction: " << pFace->m_areaFraction << endl;
			}
		}

		void CutCells3D::linkNodeVelocityField(nodeVelocityField3D_t *pNodeVelocityField) {
			m_pNodeVelocityField = pNodeVelocityField;
			for (int i = 0; i < getNumberOfCells(); i++) {
				//if (m_pNodeVelocityField->pMeshes->at(i).hasTriangleMesh()) {
					m_cutVoxels[i].initializePointsToMeshMap(m_pNodeVelocityField->pMeshes->at(i));
					m_cutVoxels[i].initializeFacesEdgeMap(m_pNodeVelocityField->pMeshes->at(i));
					if (m_mixedNodeInterpolationType != Unweighted)
						m_pNodeVelocityField->pMeshes->at(i).initializeFaceLines(m_cutVoxels[i]);
				//}
			}
			/** Nodes velocities and weights that are auxiliary to free-slip boundary conditions */
			if (m_pNodeVelocityField->nodesVelocities.size() == 0) {
				for (int i = 0; i < getNumberOfCells(); i++) {
					const vector<Vector3D> & points = m_pNodeVelocityField->pMeshes->at(i).getPoints();
					m_pNodeVelocityField->nodesVelocities.push_back(vector<Vector3>(points.size(), Vector3(0, 0, 0)));
					m_pNodeVelocityField->nodesWeights.push_back(vector<Scalar>(points.size(), 0));
				}
			}
			/** Linking local mixed nodes to global (CutCells3D) mixed nodes structures */
			for (int i = 0; i < getNumberOfCells(); i++) {
				//if (m_pNodeVelocityField->pMeshes->at(i).hasTriangleMesh()) {
					const vector<typename Mesh<Vector3D>::nodeType_t> & nodeTypes = m_pNodeVelocityField->pMeshes->at(i).getNodeTypes();
					const vector<Vector3D> & points = m_pNodeVelocityField->pMeshes->at(i).getPoints();
					for (int j = 0; j < nodeTypes.size(); j++) {
						if (nodeTypes[j] == Mesh<Vector3D>::mixedNode) {
							map<int, bool> visitedVoxels;
							visitFacesAndAddToCutNodes(points[j], j, i, visitedVoxels, m_pNodeVelocityField->pMeshes->at(i).getCutFaceToMixedNodeMap(), m_pNodeVelocityField->pMeshes->at(i).getCutFaceNormalsToMixedNodeMap());
						}
					}
				//}
			}
		}
		void CutCells3D::initializeMixedNodeVelocities(nodeVelocityField3D_t *pNodeVelocityField) {
			for (int i = 0; i < getNumberOfCells(); i++) {
				const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(i).getPoints();
				const vector<Mesh<Vector3D>::nodeType_t> & nodeTypes = pNodeVelocityField->pMeshes->at(i).getNodeTypes();
				const vector<typename Mesh<Vector3D>::meshPolygon_t> & polygons = pNodeVelocityField->pMeshes->at(i).getMeshPolygons();
				for (int k = 0; k < polygons.size(); k++) {
					if (polygons[k].polygonType == Mesh<Vector3D>::geometryPolygon) {
						for (int j = 0; j < 3; j++) {
							int pointIndex = polygons[k].edges[j].first;
							if (nodeTypes[pointIndex] == Mesh<Vector3D>::mixedNode) {
								if (m_mixedNodeInterpolationType == Unweighted) {
									pNodeVelocityField->nodesVelocities[i][pointIndex] = interpolateMixedNodeVelocity(pNodeVelocityField, i, pointIndex);
								}
								else if (m_mixedNodeInterpolationType == WeightedExtraDimensions) {
									pNodeVelocityField->nodesVelocities[i][pointIndex] = interpolateMixedNodeVelocityWeighted(pNodeVelocityField, i, pointIndex);
								}
								else if (m_mixedNodeInterpolationType == WeightedNoExtraDimensions) {
									pNodeVelocityField->nodesVelocities[i][pointIndex] = interpolateMixedNodeVelocityWeighted(pNodeVelocityField, i, pointIndex, false);
								}
								else if (m_mixedNodeInterpolationType == FaceVelocity) {
									pNodeVelocityField->nodesVelocities[i][pointIndex] = interpolateMixedNodeVelocityWeighted(pNodeVelocityField, i, pointIndex);
								}
							}
						}
					}
				}	
			}
		}

		void CutCells3D::interpolateVelocityFaceNodes(nodeVelocityField3D_t *pNodeVelocityField) {
			for (int i = 0; i < getNumberOfCells(); i++) {
				const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(i).getPoints();
				const vector<vector<int>> & faceLines = pNodeVelocityField->pMeshes->at(i).getFaceLineIndices();
				const vector<Scalar> &faceLinesTotalLenghts = pNodeVelocityField->pMeshes->at(i).getFaceLinesTotalLengths();
				for (int k = 0; k < faceLines.size(); k++) {
					Scalar currLenght = 0;
					Scalar totalLenght = faceLinesTotalLenghts[k];
					Vector3 iniVelocity = pNodeVelocityField->nodesVelocities[i][faceLines[k].front()];
					Vector3 finalVelocity = pNodeVelocityField->nodesVelocities[i][faceLines[k].back()];
					for (int j = 1; j < faceLines[k].size() - 1; j++) { //Discard first and last points, which are mixed nodes
						currLenght += (points[faceLines[k][j]] - points[faceLines[k][j - 1]]).length();
						Vector3 velocity = iniVelocity*(1 - currLenght / totalLenght) + finalVelocity*(currLenght / totalLenght);
						pNodeVelocityField->nodesVelocities[i][faceLines[k][j]] = velocity;
					}
				}
			}
		}

		void CutCells3D::preprocessVelocityDataNoSlip(nodeVelocityField3D_t *pNodeVelocityField) {
			for (int i = 0; i < getNumberOfCells(); i++) {
				const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(i).getPoints();
				const vector<typename Mesh<Vector3D>::nodeType_t> & nodeTypes = pNodeVelocityField->pMeshes->at(i).getNodeTypes();
				const vector<typename Mesh<Vector3D>::meshPolygon_t> & polygons = pNodeVelocityField->pMeshes->at(i).getMeshPolygons();
				PolygonSurface *pAPolygonSurface = NULL;
				//FIX THIS
				for (int j = 0; j < polygons.size(); j++) {
					if (polygons[j].pPolygonSurface != NULL) {
						pAPolygonSurface = polygons[j].pPolygonSurface;
						break;
					}
				}

				for (int j = 0; j < points.size(); j++) {
					dimensions_t currDimension;
					if (nodeTypes[j] == Mesh<Vector3D>::gridNode) {
						isOnGridPoint(points[j], m_gridSpacing, currDimension);
						pNodeVelocityField->nodesVelocities[i][j] = (*pNodeVelocityField->pGridNodesVelocities)(currDimension);
					}
					else if (nodeTypes[j] == Mesh<Vector3D>::geometryNode || nodeTypes[j] == Mesh<Vector3D>::mixedNode) { //geometry bode
						if (pNodeVelocityField->pMeshes->at(i).hasTriangleMesh()) {
							//Velocity at thinObject
							vector<int> geometryIndices = m_cutVoxels[i].pointsToTriangleMeshMap[j];
							if (m_thinObjectVelocities.size() > 0) {
								int thinObjectID = polygons[geometryIndices.front()].pPolygonSurface->getPolygonID();
								pNodeVelocityField->nodesVelocities[i][j] = m_thinObjectVelocities[thinObjectID];
							}

							Vector3D rotationalVelocity;
							Vector3D pointToPosition = points[j] - convertToVector3D(polygons[geometryIndices.front()].pPolygonSurface->getCentroid());
							Vector3D rotationAxis(0, 0, 1);

							pAPolygonSurface = polygons[geometryIndices.front()].pPolygonSurface;
							DoubleScalar elapsedTime = Rendering::PhysicsCore::getInstance()->getElapsedTime();

							if (elapsedTime >= pAPolygonSurface->getRotationFunction().startingTime && elapsedTime < pAPolygonSurface->getRotationFunction().endingTime
								&& abs(pointToPosition.dot(rotationAxis)) > singlePrecisionThreshold) {
								rotationalVelocity = pointToPosition.cross(rotationAxis)*polygons[geometryIndices.front()].pPolygonSurface->getRotationFunction().speed;
							}

							pNodeVelocityField->nodesVelocities[i][j] -= convertToVector3F(rotationalVelocity);

						}
						else {
							if (m_thinObjectVelocities.size() > 0)
								pNodeVelocityField->nodesVelocities[i][j] = m_thinObjectVelocities[0];

							Vector3D rotationalVelocity;
							Vector3D pointToCentroid = points[j] - convertToVector3D(pAPolygonSurface->getCentroid());
							Vector3D rotationAxis(0, 0, 1);

							DoubleScalar elapsedTime = Rendering::PhysicsCore::getInstance()->getElapsedTime();
							if (elapsedTime >= pAPolygonSurface->getRotationFunction().startingTime && elapsedTime < pAPolygonSurface->getRotationFunction().endingTime
								&& abs(pointToCentroid.dot(rotationAxis)) > singlePrecisionThreshold) {
								rotationalVelocity = pointToCentroid.cross(rotationAxis)*pAPolygonSurface->getRotationFunction().speed;
							}

							pNodeVelocityField->nodesVelocities[i][j] -= convertToVector3F(rotationalVelocity);
						}
					}
				}
			}
		}

		void CutCells3D::spreadFreeSlipAverage(int ithCell, nodeVelocityField3D_t *pNodeVelocityField) {
			const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(ithCell).getPoints();
			const vector<typename Mesh<Vector3D>::nodeType_t> & nodeTypes = pNodeVelocityField->pMeshes->at(ithCell).getNodeTypes();
			const vector<typename Mesh<Vector3D>::meshPolygon_t> & polygons = pNodeVelocityField->pMeshes->at(ithCell).getMeshPolygons();

			int numIterations = 50;
			for (int l = 0; l < numIterations; l++) {
				for (int j = 0; j < polygons.size(); j++) {
					if (polygons[j].polygonType == Mesh<Vector3D>::geometryPolygon) {
						for (int k = 0; k < 3; k++) {
							int pointIndex = polygons[j].edges[k].first;
							if (isOnGridFace(points[pointIndex], m_gridSpacing) == 0 && nodeTypes[pointIndex] != Mesh<Vector3D>::mixedNode) {
								int kNext = roundClamp<int>(k + 1, 0, 3);
								int kNextNext = roundClamp<int>(k + 1, 0, 3);
								Vector3 newVelocity = pNodeVelocityField->nodesVelocities[ithCell][pointIndex];
								newVelocity += pNodeVelocityField->nodesVelocities[ithCell][polygons[j].edges[kNext].first];
								newVelocity += pNodeVelocityField->nodesVelocities[ithCell][polygons[j].edges[kNextNext].first];
								newVelocity /= 3;
								pNodeVelocityField->nodesVelocities[ithCell][pointIndex] = newVelocity;
							}
						}
					}

				}
			}

		}
		void CutCells3D::spreadFreeSlipConservative(int ithCell, nodeVelocityField3D_t *pNodeVelocityField) {
			const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(ithCell).getPoints();
			const vector<typename Mesh<Vector3D>::nodeType_t> & nodeTypes = pNodeVelocityField->pMeshes->at(ithCell).getNodeTypes();
			const vector<typename Mesh<Vector3D>::meshPolygon_t> & polygons = pNodeVelocityField->pMeshes->at(ithCell).getMeshPolygons();

			int numIterations = 50;

			for (int j = 0; j < points.size(); j++) {
				pNodeVelocityField->nodesWeights[ithCell][j] = 0;
			}


			for (int j = 0; j < polygons.size(); j++) {
				if (polygons[j].polygonType == Mesh<Vector3D>::geometryPolygon) {
					for (int k = 0; k < 3; k++) {
						int pointIndex = polygons[j].edges[k].first;
						if (nodeTypes[pointIndex] == Mesh<Vector3D>::mixedNode || isOnGridFace(points[pointIndex], m_gridSpacing)){
							pNodeVelocityField->nodesWeights[ithCell][pointIndex] = pNodeVelocityField->nodesVelocities[ithCell][pointIndex].length();
						}
					}
				}
			}

			for (int j = 0; j < points.size(); j++) {
				if (nodeTypes[j] == Mesh<Vector3D>::geometryNode) {
					pNodeVelocityField->nodesVelocities[ithCell][j] *= pNodeVelocityField->nodesWeights[ithCell][j];
				}
			}

			for (int l = 0; l < numIterations; l++) {
				for (int j = 0; j < polygons.size(); j++) {
					if (polygons[j].polygonType == Mesh<Vector3D>::geometryPolygon) {
						for (int k = 0; k < 3; k++) {
							int pointIndex = polygons[j].edges[k].first;
							if (isOnGridFace(points[pointIndex], m_gridSpacing) == 0 && nodeTypes[pointIndex] != Mesh<Vector3D>::mixedNode) {
								int kNext = roundClamp<int>(k + 1, 0, 3);
								int kNextNext = roundClamp<int>(k + 1, 0, 3);

								DoubleScalar averageWeight = pNodeVelocityField->nodesWeights[ithCell][pointIndex];
								averageWeight += pNodeVelocityField->nodesWeights[ithCell][polygons[j].edges[kNext].first];
								averageWeight += pNodeVelocityField->nodesWeights[ithCell][polygons[j].edges[kNextNext].first];
								averageWeight /= 3;
								pNodeVelocityField->nodesWeights[ithCell][pointIndex] = averageWeight;

								Vector3 newVelocity = pNodeVelocityField->nodesVelocities[ithCell][pointIndex];
								newVelocity += pNodeVelocityField->nodesVelocities[ithCell][polygons[j].edges[kNext].first];
								newVelocity += pNodeVelocityField->nodesVelocities[ithCell][polygons[j].edges[kNextNext].first];
								newVelocity /= 3;

								/*Vector3 geometryNormal = convertToVector3F(pNodeVelocityField->pMeshes->at(i).getPointsNormals()[j]);
								Vector3 vRelative = newVelocity;
								if (m_thinObjectVelocities.size() > 0)
								vRelative -= m_thinObjectVelocities[0];
								DoubleScalar normalProj = vRelative.dot(geometryNormal);
								newVelocity = newVelocity - geometryNormal*normalProj;*/
								pNodeVelocityField->nodesVelocities[ithCell][pointIndex] = newVelocity.normalized();
								//pNodeVelocityField->nodesVelocities[ithCell][pointIndex] = newVelocity;									
							}
						}
					}

				}
			}
		}

		void CutCells3D::preprocessVelocityDataFreeSlip(nodeVelocityField3D_t *pNodeVelocityField) {
			initializeMixedNodeVelocities(pNodeVelocityField);

			if (m_mixedNodeInterpolationType != Unweighted)
				interpolateVelocityFaceNodes(pNodeVelocityField);
			
			for (int i = 0; i < getNumberOfCells(); i++) {
				//Propagating mixed node velocities to the geometry points
				const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(i).getPoints();
				const vector<typename Mesh<Vector3D>::nodeType_t> & nodeTypes = pNodeVelocityField->pMeshes->at(i).getNodeTypes();
				const vector<typename Mesh<Vector3D>::meshPolygon_t> & polygons = pNodeVelocityField->pMeshes->at(i).getMeshPolygons();

				for (int j = 0; j < pNodeVelocityField->pMeshes->at(i).getPoints().size(); j++) {
					pNodeVelocityField->nodesWeights[i][j] = 0;
					if (nodeTypes[j] == Mesh<Vector3D>::gridNode) {
						dimensions_t currDimension;
						isOnGridPoint(points[j], m_gridSpacing, currDimension);
						pNodeVelocityField->nodesVelocities[i][j] = (*pNodeVelocityField->pGridNodesVelocities)(currDimension);
					}
				}

				spreadFreeSlipAverage(i, pNodeVelocityField);
			}

			//Projecting the velocities to respect the normal condition
			for (int i = 0; i < getNumberOfCells(); i++) {
				const vector<Vector3D> & points = pNodeVelocityField->pMeshes->at(i).getPoints();
				const vector<typename Mesh<Vector3D>::nodeType_t> & nodeTypes = pNodeVelocityField->pMeshes->at(i).getNodeTypes();
				const vector<typename Mesh<Vector3D>::meshPolygon_t> & polygons = pNodeVelocityField->pMeshes->at(i).getMeshPolygons();
				PolygonSurface *pAPolygonSurface = NULL;

				//FIX THIS
				for (int j = 0; j < polygons.size(); j++) {
					if (polygons[j].pPolygonSurface != NULL) {
						pAPolygonSurface = polygons[j].pPolygonSurface;
						break;
					}
				}

				for (int j = 0; j < points.size(); j++) {
					if (nodeTypes[j] == Mesh<Vector3D>::geometryNode || nodeTypes[j] == Mesh<Vector3D>::mixedNode) {
						Vector3 geometryNormal = convertToVector3F(pNodeVelocityField->pMeshes->at(i).getPointsNormals()[j]);
						//pNodeVelocityField->nodesVelocities[i][j] -= geometryNormal*pNodeVelocityField->nodesVelocities[i][j].dot(geometryNormal);

						Vector3 rotationalVelocity;
						Vector3 pointToCentroid = convertToVector3F(points[j] - pAPolygonSurface->getCentroid());
						Vector3 rotationAxis(0, 0, 1);

						DoubleScalar elapsedTime = Rendering::PhysicsCore::getInstance()->getElapsedTime();
						
						if (elapsedTime >= pAPolygonSurface->getRotationFunction().startingTime && elapsedTime < pAPolygonSurface->getRotationFunction().endingTime
							&& abs(pointToCentroid.dot(rotationAxis)) > singlePrecisionThreshold) {
							rotationalVelocity = pointToCentroid.cross(rotationAxis)*pAPolygonSurface->getRotationFunction().speed;
						}

						Vector3 solidVelocity = m_thinObjectVelocities[0] - rotationalVelocity;

						Vector3 vRelative = pNodeVelocityField->nodesVelocities[i][j] - solidVelocity;
						DoubleScalar normalProj = vRelative.dot(geometryNormal);
						pNodeVelocityField->nodesVelocities[i][j] = pNodeVelocityField->nodesVelocities[i][j] - geometryNormal*normalProj;
					}
				}
			}
		}
		
		void CutCells3D::preprocessVelocityData(nodeVelocityField3D_t *pNodeVelocityField) {
			if (m_solidBoundaryType == Solid_FreeSlip) {
				preprocessVelocityDataFreeSlip(pNodeVelocityField);
			}
			else if (m_solidBoundaryType == Solid_NoSlip) {
				preprocessVelocityDataNoSlip(pNodeVelocityField);
			}
			
		}

		CutFace<Vector3D> * CutCells3D::createFullCutFace3D(const dimensions_t &regularGridDimensions, faceLocation_t facelocation) {
			CutFace<Vector3D> *p3DFace = new CutFace<Vector3D>(facelocation);
			p3DFace->m_areaFraction = 1.0f;
			Scalar dx = m_pGridData->getScaleFactor(0, 0, 0).x;
			switch (facelocation) {
				case Chimera::Data::rightFace:
				case Chimera::Data::leftFace:
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z)*dx,
											Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z + 1)*dx,
											dx, edge3D);
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z + 1)*dx,
											Vector3D(regularGridDimensions.x, regularGridDimensions.y + 1, regularGridDimensions.z + 1)*dx,
											dx, edge3D);
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x, regularGridDimensions.y + 1, regularGridDimensions.z + 1)*dx,
											Vector3D(regularGridDimensions.x, regularGridDimensions.y + 1, regularGridDimensions.z)*dx,
											dx, edge3D);
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x, regularGridDimensions.y + 1, regularGridDimensions.z)*dx,
											Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z)*dx,
											dx, edge3D);
					p3DFace->m_centroid = Vector3D(regularGridDimensions.x*dx, (regularGridDimensions.y + 0.5)*dx, (regularGridDimensions.z + 0.5)*dx);
				break;
				case Chimera::Data::bottomFace:
				case Chimera::Data::topFace:
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z)*dx,
											Vector3D(regularGridDimensions.x + 1, regularGridDimensions.y, regularGridDimensions.z)*dx,
											dx, edge3D);
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x + 1, regularGridDimensions.y, regularGridDimensions.z)*dx,
											Vector3D(regularGridDimensions.x + 1, regularGridDimensions.y, regularGridDimensions.z + 1)*dx,
											dx, edge3D);
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x + 1, regularGridDimensions.y, regularGridDimensions.z + 1)*dx,
											Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z + 1)*dx,
											dx, edge3D);
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z + 1)*dx,
											Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z)*dx,
											dx, edge3D);
					p3DFace->m_centroid = Vector3D((regularGridDimensions.x + 0.5)*dx, regularGridDimensions.y*dx, (regularGridDimensions.z + 0.5)*dx);
				break;
				case Chimera::Data::frontFace:
				case Chimera::Data::backFace:
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z)*dx,
											Vector3D(regularGridDimensions.x + 1, regularGridDimensions.y, regularGridDimensions.z)*dx,
											dx, edge3D);
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x + 1, regularGridDimensions.y, regularGridDimensions.z)*dx,
											Vector3D(regularGridDimensions.x + 1, regularGridDimensions.y + 1, regularGridDimensions.z)*dx,
											dx, edge3D);
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x + 1, regularGridDimensions.y + 1, regularGridDimensions.z)*dx,
											Vector3D(regularGridDimensions.x, regularGridDimensions.y + 1, regularGridDimensions.z)*dx,
											dx, edge3D);
					p3DFace->insertCutEdge(Vector3D(regularGridDimensions.x, regularGridDimensions.y + 1, regularGridDimensions.z)*dx,
											Vector3D(regularGridDimensions.x, regularGridDimensions.y, regularGridDimensions.z)*dx,
											dx, edge3D);
					p3DFace->m_centroid = Vector3D((regularGridDimensions.x + 0.5)*dx, (regularGridDimensions.y + 0.5)*dx, regularGridDimensions.z*dx);
				break;
				case Chimera::Data::geometryFace:
					//We cant have full geometry faces inserted like this right?
				break;
			}
			Vector3D p1 = p3DFace->m_cutEdges[0]->m_initialPoint;
			Vector3D p2 = p3DFace->m_cutEdges[1]->m_initialPoint;
			Vector3D p3 = p3DFace->m_cutEdges[2]->m_initialPoint;
			Vector3D v1 = p2 - p1;
			Vector3D v2 = p3 - p1;
			v1.normalize();
			v2.normalize();
			p3DFace->m_normal = (v1).cross(v2);
			p3DFace->m_normal.normalize();
			p3DFace->m_interiorPoint = p3DFace->m_centroid;
			return p3DFace;
		}

		void CutCells3D::initializeStreamFunction(nodeVelocityField3D_t *pNodeVelocityField) {
			for (int i = 0; i < m_cutVoxels.size(); i++) {
				////Initial streamfunction value is initialized to 0
				//pNodeVelocityField->nodeStreamfunctions[i][0] = 0;
				////Starting from the first streamfunction node, we access the flux on the "previous" egde to initilize the current streamfunction
				//for (int j = 1; j < pNodeVelocityField->nodePositions[i].size(); j++) {
				//	DoubleScalar currFlux;
				//	Vector2 currFaceNormal;
				//	//Setting up normals. Geometry edges have "correct" normals, since they don't share faces. 
				//	//Grid edges are initialized correctly (pointing outwards) using the getEdgeNormal function
				//	if (m_cutFaces[i].m_cutEdgesLocations[j - 1] != geometryEdge) {
				//		currFaceNormal = getEdgeNormal(m_cutFaces[i], j - 1);
				//	}
				//	else {
				//		currFaceNormal = m_cutFaces[i].m_cutEdges[j - 1]->getNormal();
				//	}
				//	//Since the poisson matrix is normalized on regular cells, we multiply the fluxes by m_gridSpacing everywhere. 
				//	//LengthFraction gives us the un-obstructed cell fraction
				//	currFlux = currFaceNormal.dot(m_cutFaces[i].m_cutEdges[j - 1]->m_velocity)*m_cutFaces[i].m_cutEdges[j - 1]->getLengthFraction()*m_gridSpacing;
				//	//The streamfunction is updated relative to the previous value
				//	pNodeVelocityField->nodeStreamfunctions[i][j] = currFlux + pNodeVelocityField->nodeStreamfunctions[i][j - 1];
				//}
			}
			for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
				for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
					for (int k = 0; k < m_pGridData->getDimensions().z; k++) {
						//(*pNodeVelocityField->pGridNodesStreamfunctions)(i, j)[0] = 0;


						////if (!isSpecialCell(i, j)) {
						////First node is zero
						//(*pNodeVelocityField->pGridNodesStreamfunctions)(i, j)[0] = 0;

						////Second node is the first flux
						//Scalar currFlux = -m_pGridData->getVelocity(i, j).y*m_gridSpacing;
						//(*pNodeVelocityField->pGridNodesStreamfunctions)(i, j)[1] = (*pNodeVelocityField->pGridNodesStreamfunctions)(i, j)[0] + currFlux;

						////Third node
						//if (i == m_pGridData->getDimensions().x - 1) {
						//	currFlux = m_pGridData->getVelocity(i, j).x*m_gridSpacing;
						//}
						//else if (m_leftSpecialFaces(i + 1, j).size() == 0) {
						//	currFlux = m_pGridData->getVelocity(i + 1, j).x*m_gridSpacing;
						//}
						//else {
						//	currFlux = m_leftSpecialFaces(i + 1, j)[0].m_velocity.x*m_gridSpacing;
						//}
						//(*pNodeVelocityField->pGridNodesStreamfunctions)(i, j)[2] = (*pNodeVelocityField->pGridNodesStreamfunctions)(i, j)[1] + currFlux;

						////Fourth node
						//if (j == m_pGridData->getDimensions().y - 1) {
						//	currFlux = m_pGridData->getVelocity(i, j).y*m_gridSpacing;
						//}
						//else if (m_bottomSpecialFaces(i, j + 1).size() == 0) {
						//	currFlux = m_pGridData->getVelocity(i, j + 1).y*m_gridSpacing;
						//}
						//else {
						//	currFlux = m_bottomSpecialFaces(i, j + 1)[0].m_velocity.y*m_gridSpacing;
						//}
						//(*pNodeVelocityField->pGridNodesStreamfunctions)(i, j)[3] = (*pNodeVelocityField->pGridNodesStreamfunctions)(i, j)[2] + currFlux;
						//}
					}
				}
			}
		}
		#pragma endregion

		#pragma region UpdateFunctions
		void CutCells3D::updatePoissonMatrix(PoissonMatrix *pPoissonMatrix, const FlowSolverParameters::pressureSolverParameters_t & pressureSolverParams, const vector<Data::BoundaryCondition<Vector3> *> &boundaryConditions) {
			m_pPoissonMatrix = pPoissonMatrix;
			//Double check the initial offset number
			int initialOffset = 0;
			if(pressureSolverParams.getMethodCategory() == Krylov) {
				initialOffset = (m_pGrid->getDimensions().x - 2)*(m_pGrid->getDimensions().y - 2)*(m_pGrid->getDimensions().z - 2); 
			} else {
				initialOffset = m_pGrid->getDimensions().x*m_pGrid->getDimensions().y*m_pGrid->getDimensions().z;
			}

			//Regular cells that are now special cells are treated as solid cells 
			if(pressureSolverParams.getMethodCategory() == Krylov) {
				for(int k = 1; k < m_pGrid->getDimensions().z - 1; k++) {
					for(int j = 1; j < m_pGrid->getDimensions().y - 1; ++j) {
						for(int i = 1; i < m_pGrid->getDimensions().x - 1; ++i) {
							int currRowIndex = pPoissonMatrix->getRowIndex(i - 1, j - 1, k - 1);
							if(isSpecialCell(i, j, k)) {
								pPoissonMatrix->setRow(currRowIndex, 0, 0, 0, 1, 0, 0, 0);
							} else {
								if(isSpecialCell(i - 1, j, k)) {
									pPoissonMatrix->setEastValue(currRowIndex, 0);
								}
								if(isSpecialCell(i + 1,j, k)) {
									pPoissonMatrix->setWestValue(currRowIndex, 0);
								}

								if(isSpecialCell(i,j + 1, k)) {
									pPoissonMatrix->setNorthValue(currRowIndex, 0);
								}
								if(isSpecialCell(i,j - 1, k)) {
									pPoissonMatrix->setSouthValue(currRowIndex, 0);
								}

								if(isSpecialCell(i,j, k - 1)) {
									pPoissonMatrix->setBackValue(currRowIndex, 0);
								}
								if(isSpecialCell(i,j, k + 1)) {
									pPoissonMatrix->setFrontValue(currRowIndex, 0);
								}
							}
						}
					}
				}
			}
				

			pPoissonMatrix->copyDIAtoCOO();
			int numEntries = pPoissonMatrix->getNumberOfEntriesCOO();
			pPoissonMatrix->resizeToFitCOO();

			int matrixInternalId = numEntries;
			DoubleScalar biggerPressureValue = 0;
			DoubleScalar biggerCentreValue = 0;
			DoubleScalar smallerPressureValue = FLT_MAX;
			DoubleScalar smallerCentreValue = FLT_MAX;
			DoubleScalar minimumThreshold = singlePrecisionThreshold;
			//Build the linear system for pressure
			for(int k = 1; k < m_pGrid->getDimensions().z - 1; k++) {
				for(int j = 1; j < m_pGrid->getDimensions().y - 1; ++j) {
					for(int i = 1; i < m_pGrid->getDimensions().x - 1; ++i) {

						if(isSpecialCell(i,j,k)) {
							//iterate over all sub-cells in this cell
							int pressureID = m_cellToSpecialMap(i, j, k); //where do the special cells for this location begin?
							CutVoxel curCell = m_cutVoxels[pressureID];
							//keep going while we're still in the same cell, and haven't run out of sub-cells altogether
							while(curCell.regularGridIndex == dimensions_t(i, j, k) && pressureID < m_cutVoxels.size()) {

								int row = initialOffset + pressureID; //compute the matrix index after the last regular cell
								Scalar poissonCoefficients[6];
								int poissonRows[6];
								for(int sf = 0; sf < 6; sf++) {
									poissonRows[sf] = 0;
									poissonCoefficients[sf] = 0.0f;
								}

								if (curCell.cutFaces.size() >= 3) {
									for (int sf = 0; sf < curCell.cutFaces.size(); sf++) {
										CutFace<Vector3D> *pFace = curCell.cutFaces[sf];
										if (curCell.cutFacesLocations[sf] != geometryFace) {
											int otherPressure = pFace->m_faceNeighbors[0] == pressureID ? pFace->m_faceNeighbors[1] : pFace->m_faceNeighbors[0];
											int thisPressure = pFace->m_faceNeighbors[0] == pressureID ? pFace->m_faceNeighbors[0] : pFace->m_faceNeighbors[1];
											if (thisPressure == -1) {
												poissonRows[curCell.cutFacesLocations[sf]] = 0;
												continue;
											}
											else if (otherPressure == -1) {
												poissonRows[curCell.cutFacesLocations[sf]] = getRowIndex(pPoissonMatrix, dimensions_t(i, j, k), curCell.cutFacesLocations[sf]);
											}
											else {
												poissonRows[curCell.cutFacesLocations[sf]] = initialOffset + otherPressure;
											}
											Scalar pressureCoefficient = pFace->m_areaFraction;
											if (pressureCoefficient < smallerPressureValue)
												smallerPressureValue = pressureCoefficient;

											if (pressureCoefficient > biggerPressureValue)
												biggerPressureValue = pressureCoefficient;

											/*if (pressureCoefficient < minimumThreshold)
											pressureCoefficient = minimumThreshold;*/

											poissonCoefficients[curCell.cutFacesLocations[sf]] += pressureCoefficient;


											pPoissonMatrix->setValue(matrixInternalId++, row, poissonRows[curCell.cutFacesLocations[sf]], -pressureCoefficient);
											if (otherPressure == -1 && poissonRows[curCell.cutFacesLocations[sf]] != -1) //If we have a regular cell neighbor
												pPoissonMatrix->setValue(matrixInternalId++, poissonRows[curCell.cutFacesLocations[sf]], row, -pressureCoefficient); //Guarantees matrix symmetry
										}
									}
									Scalar pc = 0;

									int numNeighs = 0;
									for (int sf = 0; sf < 6; sf++) {
										if (poissonRows[sf] > 0) {
											//pPoissonMatrix->setValue(matrixInternalId++, row, poissonRows[sf], -poissonCoefficients[sf]);
											pc += poissonCoefficients[sf];
											numNeighs++;
										}
									}

									pPoissonMatrix->setValue(matrixInternalId++, row, row, pc);


									if (pc < smallerCentreValue)
										smallerCentreValue = pc;

									if (pc > biggerCentreValue)
										biggerCentreValue = pc;

									//pPoissonMatrix->setValue(matrixInternalId++, row, row, pc);

									//move to the next sub-cell
									++pressureID;
									if (pressureID < m_cutVoxels.size())
										curCell = m_cutVoxels[pressureID];
								}
								else {
									cout << "Treating defective cut-cell" << endl;
									for (int sf = 0; sf < curCell.cutFaces.size(); sf++) {
										CutFace<Vector3D> *pFace = curCell.cutFaces[sf];
										if (curCell.cutFacesLocations[sf] != geometryFace) {
											int otherPressure = pFace->m_faceNeighbors[0] == pressureID ? pFace->m_faceNeighbors[1] : pFace->m_faceNeighbors[0];
											int thisPressure = pFace->m_faceNeighbors[0] == pressureID ? pFace->m_faceNeighbors[0] : pFace->m_faceNeighbors[1];
											if (thisPressure == -1) {
												poissonRows[curCell.cutFacesLocations[sf]] = 0;
												continue;
											}
											else if (otherPressure == -1) {
												poissonRows[curCell.cutFacesLocations[sf]] = getRowIndex(pPoissonMatrix, dimensions_t(i, j, k), curCell.cutFacesLocations[sf]);
											}
											else {
												poissonRows[curCell.cutFacesLocations[sf]] = initialOffset + otherPressure;
											}
											
											pPoissonMatrix->setValue(matrixInternalId++, row, poissonRows[curCell.cutFacesLocations[sf]], 0);
											if (otherPressure == -1 && poissonRows[curCell.cutFacesLocations[sf]] != -1) //If we have a regular cell neighbor
												pPoissonMatrix->setValue(matrixInternalId++, poissonRows[curCell.cutFacesLocations[sf]], row, 0); //Guarantees matrix symmetry
										}
									}
									
									pPoissonMatrix->setValue(matrixInternalId++, row, row, 1.0f);
								}
								
								
							}
						}
					}
				}
			}
			if(pPoissonMatrix->isSingular())
				cout << "Singular Poisson Matrix!" << endl;

			pPoissonMatrix->copyCOOtoHyb();
			cout << "Smaller value " << smallerPressureValue << endl;

			cout << "Smaller centre value " << smallerCentreValue << endl;

			cout << "Bigger value " << biggerPressureValue << endl;

			cout << "Bigger centre value " << biggerCentreValue << endl;
		}

		std::ostream& operator<<(std::ostream& out, const dimensions_t& dim){
			return out << "(" << dim.x << ", " << dim.y << ", " << dim.z << ")";
		}

		void CutCells3D::updateDivergents(Scalar dt) {
			Scalar dx = m_pGridData->getScaleFactor(0, 0, 0).x;
			Scalar sumOfgeometryFacesDivergents = 0;
			dimensions_t lastSumDim = m_cutVoxels.front().regularGridIndex;

			for(int i = 0; i < m_cutVoxels.size(); i++) {
				Scalar divergent = 0;
				bool isLeftCell = false;
				
				vector<Vector3D> geometryFacesFluxes;
				for(unsigned int sf = 0; sf < m_cutVoxels[i].cutFaces.size(); ++sf) {
					CutFace<Vector3D> *pFace;
					pFace = m_cutVoxels[i].cutFaces[sf];
					if(m_cutVoxels[i].cutFacesLocations[sf] != geometryFace) {
						divergent += getFaceNormal(m_cutVoxels[i].cutFacesLocations[sf]).dot(pFace->m_velocity)*pFace->m_areaFraction*dx;
					}
					else {
						geometryFacesFluxes.push_back(pFace->m_velocity*pFace->m_areaFraction);
						//divergent += pFace->m_normal.dot(pFace->m_velocity)*pFace->m_areaFraction;
					}
				}

				for (unsigned int j = 0; j < m_cutVoxels[i].geometryFacesToMesh.size(); j++) {
					int polygonMeshIndex = m_cutVoxels[i].geometryFacesToMesh[j]; 
					Vector3D correctedNormal = m_pNodeVelocityField->pMeshes->at(i).getMeshPolygons()[polygonMeshIndex].normal;
					correctedNormal.normalize();
					divergent -= correctedNormal.dot(geometryFacesFluxes[j])*dx;
					//geometryFacesFluxes[j].x = correctedNormal.dot(geometryFacesFluxes[j])*dx;
				}
				m_specialDivergents[i] = -divergent/dt;
				sumOfgeometryFacesDivergents += m_specialDivergents[i];
			}
			/*
			for (int i = 0; i < m_specialDivergents.size(); i++) {
				cout << m_specialDivergents[i] << " ";
				if (i % 15 == 0)
					cout << endl;
			}*/

			Logger::getInstance()->get() << "Sum of all geometry divergents  " << sumOfgeometryFacesDivergents << endl;
		}
		#pragma endregion

		#pragma region ConvertionFunctions
		Vector3D CutCells3D::convertPointTo3D(Vector2D point, const dimensions_t &regularGridIndex,faceLocation_t facelocation) const {
			Scalar dx = m_pGridData->getScaleFactor(0, 0, 0).x;
			switch (facelocation) {
				case rightFace:
					return Vector3D((regularGridIndex.x + 1)*dx, point.y, point.x);
				break;
				case leftFace:
					return Vector3D((regularGridIndex.x)*dx, point.y, point.x);
				break;
				case topFace:
					return Vector3D(point.x, (regularGridIndex.y + 1)*dx, point.y);
				break;
				case bottomFace:
					return Vector3D(point.x, (regularGridIndex.y)*dx, point.y);
				break;
				case frontFace:
					return Vector3D(point.x, point.y, (regularGridIndex.z + 1)*dx);
				break;
				case backFace:
					return Vector3D(point.x, point.y, (regularGridIndex.z)*dx);
				break;

				default:
					return Vector3D(0, 0, 0);
				break;
			}
		}
		CutFace<Vector3D> * CutCells3D::convertTo3DFace(CutFace<Vector2D> *p2DFace, const dimensions_t &regularGridIndex, faceLocation_t faceLocation) {
			CutFace<Vector3D> * p3DFace = new CutFace<Vector3D>(faceLocation);
			p3DFace->m_regularGridIndex = regularGridIndex;
			vector<Vector3D> tempCellPoints;
			Scalar dx = m_pGridData->getScaleFactor(0, 0, 0).x;
			//Converting edges
			for (int i = 0; i < p2DFace->m_cutEdges.size(); i++) {
				Vector3D edgeInitialPoint = convertPointTo3D(p2DFace->getEdgeInitialPoint(i), regularGridIndex, faceLocation);
				p3DFace->insertCutEdge(edgeInitialPoint, convertPointTo3D(p2DFace->getEdgeFinalPoint(i), regularGridIndex, faceLocation), dx, p2DFace->m_cutEdgesLocations[i], p2DFace->m_cutEdges[i]->m_thinObjectID);
				tempCellPoints.push_back(edgeInitialPoint);
				p3DFace->m_centroid += edgeInitialPoint;
			}
			p3DFace->m_centroid /= tempCellPoints.size();
			p3DFace->updateCentroid();
			//Calculating area fraction
			p3DFace->m_areaFraction = calculatePolygonArea(tempCellPoints, getFaceNormal(faceLocation))/(dx*dx);
			return p3DFace;
		}
		#pragma endregion


		#pragma region FlushinFunctions
		void CutCells3D::flushCellVelocities() {
			for(int i = 0; i < getNumberOfCells(); i++) {
				for(int j = 0; j < m_cutVoxels[i].cutFaces.size(); j++) {
					CutFace<Vector3D> *pFace = m_cutVoxels[i].cutFaces[j];
					pFace->m_velocity.x = pFace->m_velocity.y = pFace->m_velocity.z = 0.0f;
					//pFace->intermediateVelocity.x = pFace->intermediateVelocity.y = 0.0f;
					//pFace->velocityWeight = 0.0f;
				}
			}
		}
		void CutCells3D::flushThinBounds() {
			m_cutVoxels.clear();
			m_isSpecialCell.clear();
			m_isBoundaryCell.clear();

			flushCutSlices();

			/** Arrays conformed with grid space. Simplifies access directly with grid space coordinates */
			for(int i = 0; i < m_leftSpecialFaces.getDimensions().x; i++) {
				for(int j = 0; j < m_leftSpecialFaces.getDimensions().y; j++) {
					for(int k = 0; k < m_leftSpecialFaces.getDimensions().z; k++) {
						m_leftSpecialFaces(i, j, k).clear();
					}
				}
			}
			for(int i = 0; i < m_bottomSpecialFaces.getDimensions().x; i++) {
				for(int j = 0; j < m_bottomSpecialFaces.getDimensions().y; j++) {
					for(int k = 0; k < m_bottomSpecialFaces.getDimensions().z; k++) {
						m_bottomSpecialFaces(i, j, k).clear();
					}
				}
			}

			for(int i = 0; i < m_backSpecialFaces.getDimensions().x; i++) {
				for(int j = 0; j < m_backSpecialFaces.getDimensions().y; j++) {
					for(int k = 0; k < m_backSpecialFaces.getDimensions().z; k++) {
						m_backSpecialFaces(i, j, k).clear();
					}
				}
			}

			m_cellToSpecialMap.assign(-1);

			m_specialDivergents.clear();
			m_specialPressures.clear();
		}
		#pragma endregion
	}

	void CutCells3D::flushCutSlices() {
		for (int i = 0; i < m_cutCellsXYVec.size(); i++) {
			delete m_cutCellsXYVec[i];
		}
		m_cutCellsXYVec.clear();

		for (int i = 0; i < m_cutCellsXZVec.size(); i++) {
			delete m_cutCellsXZVec[i];
		}
		m_cutCellsXZVec.clear();

		for (int i = 0; i < m_cutCellsYZVec.size(); i++) {
			delete m_cutCellsYZVec[i];
		}
		m_cutCellsYZVec.clear();
	}
}