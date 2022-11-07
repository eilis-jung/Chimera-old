#include "CutCells/CutSlice3D.h"
namespace Chimera {

	namespace CutCells {

		#pragma region Constructors
		CutSlice3D::CutSlice3D(dimensions_t sliceDimensions, faceLocation_t sliceLocation, unsigned int orthogonalDimension, 
								Scalar gridSpacing, vector<LineMesh<Vector3D> *> lineMeshes) : m_lineSegments(sliceDimensions), 
			m_lineMeshes(lineMeshes), m_dx(gridSpacing), m_sliceDimensions(sliceDimensions), m_sliceLocation(sliceLocation){
			m_orthogonalDimensionalIndex = orthogonalDimension;
			m_orthogonalDimensionalPosition = m_dx*m_orthogonalDimensionalIndex;
			m_proximityThreshold = 1e-5;

			remeshLines();
			initializeLineSegments();
			initializeManifoldMeshes();
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		void CutSlice3D::remeshLines() {
			for (int i = 0; i < m_lineMeshes.size(); i++) {
				//Transforming to 2-D vectors to facilitate remeshing 
				vector<Vector2D> remeshPoints2D;
				for (int j = 0; j < m_lineMeshes[i]->getPoints().size(); j++) {
					remeshPoints2D.push_back(convertVectorTo2D(m_lineMeshes[i]->getPoints()[j]));
				}
				bool remeshedLine;
				remeshPoints2D = remeshLinePoints(remeshPoints2D, m_dx, remeshedLine,  m_lineMeshes[i]->isClosedMesh());
				m_lineMeshesCrossGrid.push_back(remeshedLine);
				vector<Vector3D> remeshPoints3D;
				for (int j = 0; j < remeshPoints2D.size(); j++) {
					remeshPoints3D.push_back(convertVectorTo3D(remeshPoints2D[j]));
				}
				m_lineMeshes[i]->getPoints() = remeshPoints3D;
			}
		}
		#pragma endregion
		#pragma region InitializationFunctions
		void CutSlice3D::initializeLineSegments() {
			for (int i = 0; i < m_lineMeshes.size(); i++) {
				LineMesh<Vector3D> *pLine = m_lineMeshes[i];
				if (pLine->isClosedMesh()) {
					for (int j = 0; j < pLine->getPoints().size(); j++) {
						vector<pair<unsigned int, unsigned int>> *pEdges = new vector<pair<unsigned int, unsigned int>>();
						int nextJ = roundClamp<int>(j + 1, 0, pLine->getPoints().size());
						Vector3D centroid = (pLine->getPoints()[j] + pLine->getPoints()[nextJ])*0.5;
						dimensions_t lineSegmentIndex2D = convertVectorToDimensions2D(centroid);
						dimensions_t currDim(centroid.x/m_dx, centroid.y/m_dx, centroid.z/m_dx);
						dimensions_t lastDim(currDim);
						do {
							pEdges->push_back(pair<unsigned int, unsigned int>(j, nextJ));

							j++;
							currDim = lastDim;
							if (j == pLine->getPoints().size())
								break;

							nextJ = roundClamp<int>(j + 1, 0, pLine->getPoints().size());
							centroid = (pLine->getPoints()[j] + pLine->getPoints()[nextJ])*0.5;
							currDim = dimensions_t(centroid.x / m_dx, centroid.y / m_dx, centroid.z / m_dx);
						} while (j < pLine->getPoints().size() && lastDim == currDim);
						j--;
						
						if (m_lineSegments(lineSegmentIndex2D).size() > 0) {
							bool foundArray = false;
							for (int k = 0; k < m_lineSegments(lineSegmentIndex2D).size(); k++) {
								if (m_lineSegments(lineSegmentIndex2D)[k].pLine == pLine) {
									foundArray = true;
									for (int l = 0; l < pEdges->size(); l++) {
										m_lineSegments(lineSegmentIndex2D)[k].pEdges->push_back(pEdges->at(l));
									}
								}
							}
							if (!foundArray) {
								lineSegment_t<Vector3D> lineSegment;
								lineSegment.pEdges = pEdges;
								lineSegment.pLine = pLine;
								lineSegment.cellGridIndex = lastDim;
								lineSegment.crossesGrid = m_lineMeshesCrossGrid[i];
								m_lineSegments(lineSegmentIndex2D).push_back(lineSegment);
							}
						}
						else {
							lineSegment_t<Vector3D> lineSegment;
							lineSegment.pEdges = pEdges;
							lineSegment.pLine = pLine;
							lineSegment.cellGridIndex = lastDim;
							lineSegment.crossesGrid = m_lineMeshesCrossGrid[i];
							m_lineSegments(lineSegmentIndex2D).push_back(lineSegment);
						}

					}
				}
				else {
					for (int j = 0; j < pLine->getPoints().size() - 1; j++) {
						vector<pair<unsigned int, unsigned int>> *pEdges = new vector<pair<unsigned int, unsigned int>>();
						Vector3D centroid = (pLine->getPoints()[j] + pLine->getPoints()[j + 1])*0.5;
						dimensions_t currDim(centroid.x / m_dx, centroid.y / m_dx, centroid.z / m_dx);
						dimensions_t lineSegmentIndex2D = convertVectorToDimensions2D(centroid);
						dimensions_t lastDim(currDim);
						do {
							pEdges->push_back(pair<unsigned int, unsigned int>(j, j + 1));

							j++;
							currDim = lastDim;
							if (j == pLine->getPoints().size() - 1)
								break;

							centroid = (pLine->getPoints()[j] + pLine->getPoints()[j + 1])*0.5;
							currDim = dimensions_t(centroid.x / m_dx, centroid.y / m_dx, centroid.z / m_dx);
						} while (j < pLine->getPoints().size() - 1 && lastDim == currDim);
						j--;

						if (m_lineSegments(lineSegmentIndex2D).size() > 0) {
							bool foundArray = false;
							for (int k = 0; k < m_lineSegments(lineSegmentIndex2D).size(); k++) {
								if (m_lineSegments(lineSegmentIndex2D)[k].pLine == pLine) {
									foundArray = true;
									for (int l = 0; l < pEdges->size(); l++) {
										m_lineSegments(lineSegmentIndex2D)[k].pEdges->push_back(pEdges->at(l));
									}
								}
							}
							if (!foundArray) {
								lineSegment_t<Vector3D> lineSegment;
								lineSegment.pEdges = pEdges;
								lineSegment.pLine = pLine;
								lineSegment.cellGridIndex = lastDim;
								lineSegment.crossesGrid = m_lineMeshesCrossGrid[i];
								m_lineSegments(lineSegmentIndex2D).push_back(lineSegment);
							}
						}
						else {
							lineSegment_t<Vector3D> lineSegment;
							lineSegment.pEdges = pEdges;
							lineSegment.pLine = pLine;
							lineSegment.cellGridIndex = lastDim;
							lineSegment.crossesGrid = m_lineMeshesCrossGrid[i];
							m_lineSegments(lineSegmentIndex2D).push_back(lineSegment);
						}
					}
				}
				
			}
		}
		void CutSlice3D::initializeManifoldMeshes() {
			for (int i = 0; i < m_lineSegments.getDimensions().x; i++) {
				for (int j = 0; j < m_lineSegments.getDimensions().y; j++) {
					if (m_lineSegments(i, j).size() > 0) {
						m_manifoldMeshes.push_back(new NonManifoldMesh2D(m_lineSegments(i, j).front().cellGridIndex, m_sliceLocation, m_lineSegments(i, j), m_dx));
					}
				}
			}
		}
		#pragma endregion
		
		#pragma region ConversionFunctions
		Vector2D CutSlice3D::convertVectorTo2D(const Vector3D &vec) {
			switch (m_sliceLocation) {
				case Chimera::Data::leftFace:
					return Vector2D(vec.z, vec.y);
				break;

				case Chimera::Data::bottomFace:
					return Vector2D(vec.x, vec.z);
				break;
				
				case Chimera::Data::backFace:
					return Vector2D(vec.x, vec.y);
				break;
			}
		}

		Vector3D CutSlice3D::convertVectorTo3D(const Vector2D &vec) {
			switch (m_sliceLocation) {
				case Chimera::Data::leftFace:
					return Vector3D(m_orthogonalDimensionalPosition, vec.y, vec.x);
				break;

				case Chimera::Data::bottomFace:
					return Vector3D(vec.x, m_orthogonalDimensionalPosition, vec.y);
				break;

				case Chimera::Data::backFace:
					return Vector3D(vec.x, vec.y, m_orthogonalDimensionalPosition);
				break;
			}
		}

		dimensions_t CutSlice3D::convertVectorToDimensions2D(const Vector3D &position) {
			Vector2D convertedPosition = convertVectorTo2D(position);
			return dimensions_t(convertedPosition.x / m_dx, convertedPosition.y / m_dx);
		}
		#pragma endregion
	}
}