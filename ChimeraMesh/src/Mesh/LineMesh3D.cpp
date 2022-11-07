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
//	

#include "Mesh/LineMesh.h"

namespace Chimera {

	namespace Meshes {

		#pragma region PrivateFunctionalities
		template<class VectorType>
		void LineMeshT<VectorType, false>::initializeVertices(const vector<VectorType> &points) {
			uint pointsSize = points.size();
			if (m_isClosedMesh) {
				pointsSize = points.size() - 1;
			}
			if (m_planeLocation == XYFace) { 
				for (int i = 0; i < pointsSize; i++) {
					VectorType gridSpaceVertex = points[i] / m_gridDx;
					VectorType gridSpaceDim(	gridSpaceVertex.x - floor(gridSpaceVertex.x),
												gridSpaceVertex.y - floor(gridSpaceVertex.y), 
												gridSpaceVertex.z - floor(gridSpaceVertex.z));
					if(gridSpaceDim.x == 0 || gridSpaceDim.y == 0) { //On edge vertex
						m_vertices.push_back(new Vertex<VectorType>(points[i], edgeVertex));
					} else {
						m_vertices.push_back(new Vertex<VectorType>(points[i], geometryVertex));
					}						
				}
			}
			else if (m_planeLocation == YZFace) {
				for (int i = 0; i < pointsSize; i++) {
					VectorType gridSpaceVertex = points[i] / m_gridDx;
					VectorType gridSpaceDim(	gridSpaceVertex.x - floor(gridSpaceVertex.x),
												gridSpaceVertex.y - floor(gridSpaceVertex.y),
												gridSpaceVertex.z - floor(gridSpaceVertex.z));
					if (gridSpaceDim.y == 0 || gridSpaceDim.z == 0) { //On edge vertex
						m_vertices.push_back(new Vertex<VectorType>(points[i], edgeVertex));
					}
					else {
						m_vertices.push_back(new Vertex<VectorType>(points[i], geometryVertex));
					}
				}
			}
			else if (m_planeLocation == XZFace) { 
				for (int i = 0; i < pointsSize; i++) {
					VectorType gridSpaceVertex = points[i] / m_gridDx;
					VectorType gridSpaceDim(	gridSpaceVertex.x - floor(gridSpaceVertex.x),
												gridSpaceVertex.y - floor(gridSpaceVertex.y),
												gridSpaceVertex.z - floor(gridSpaceVertex.z));
					if (gridSpaceDim.x == 0 || gridSpaceDim.z == 0) { //On edge vertex
						m_vertices.push_back(new Vertex<VectorType>(points[i], edgeVertex));
					}
					else {
						m_vertices.push_back(new Vertex<VectorType>(points[i], geometryVertex));
					}
				}
			}
		}

		
		template<class VectorType>
		void LineMeshT<VectorType, false>::initializeRegularGridPatches() {
			if (m_planeLocation == XYFace) { // x = i, y = j
				m_regularGridPatches.resize(dimensions_t(m_gridDimensions.x, m_gridDimensions.y, 0));
				for (uint j = 0; j < m_elements.size(); j++) {
					Edge<VectorType> *pCurrEdge = m_elements[j];
					VectorType gridSpacePosition = pCurrEdge->getCentroid() / m_gridDx;
					dimensions_t dimTemp(floor(gridSpacePosition.x), floor(gridSpacePosition.y));
					m_regularGridPatches(floor(gridSpacePosition.x), floor(gridSpacePosition.y)).push_back(j);
				}
			}
			else if (m_planeLocation == YZFace) { //z = i, y = j
				m_regularGridPatches.resize(dimensions_t(m_gridDimensions.z, m_gridDimensions.y, 0));
				for (uint j = 0; j < m_elements.size(); j++) {
					Edge<VectorType> *pCurrEdge = m_elements[j];
					VectorType gridSpacePosition = pCurrEdge->getCentroid() / m_gridDx;
					dimensions_t dimTemp(floor(gridSpacePosition.z), floor(gridSpacePosition.y));
					m_regularGridPatches(floor(gridSpacePosition.z), floor(gridSpacePosition.y)).push_back(j);
				}
			}
			else if (m_planeLocation == XZFace) { //x = j, z = j
				m_regularGridPatches.resize(dimensions_t(m_gridDimensions.x, m_gridDimensions.z, 0));
				for (uint j = 0; j < m_elements.size(); j++) {
					Edge<VectorType> *pCurrEdge = m_elements[j];
					VectorType gridSpacePosition = pCurrEdge->getCentroid() / m_gridDx;
					dimensions_t dimTemp(floor(gridSpacePosition.x), floor(gridSpacePosition.z));
					m_regularGridPatches(floor(gridSpacePosition.x), floor(gridSpacePosition.z)).push_back(j);
				}
			}
		}

		template<class VectorType>
		bool crossingsSortFunction(pair<DoubleScalar, VectorType> c1, pair<DoubleScalar, VectorType> c2) {
			return c1.first < c2.first;
		}

		template<class VectorType>
		void LineMeshT<VectorType, false>::computeGridCrossingsXY(Scalar dx, bool perturbPoints /* = false */) {
			const vector<VectorType> &initialPoints = m_params.initialPoints;
			m_vertices.clear();

			//Z point is always fixed
			VectorType iniHorizPoint, initVerticalPoint;
			iniHorizPoint.z = initVerticalPoint.z = floor(initialPoints[0].z);
			for (int i = 0; i < initialPoints.size() - 1; i++) {
				int nextI = roundClamp<int>(i + 1, 0, initialPoints.size());
				m_vertices.push_back(new Vertex<VectorType>(m_params.initialPoints[i], geometryVertex));

				iniHorizPoint.x = floor(initialPoints[i].x / dx)*dx;
				iniHorizPoint.y = floor(initialPoints[i].y / dx)*dx;

				initVerticalPoint.y = floor(initialPoints[i].y / dx)*dx;
				initVerticalPoint.x = floor(initialPoints[i].x / dx)*dx;

				vector<pair<DoubleScalar, VectorType>> allCrossings;
				DoubleScalar currLength = (initialPoints[nextI] - initialPoints[i]).length();

				/** Compute horizontal crossings */
				int numCells = floor(initialPoints[nextI].y / dx) - floor(iniHorizPoint.y / dx);
				int numCellsSign = numCells == 0 ? 1 : numCells / abs(numCells);
				for (int currIndex = 0; currIndex <= abs(numCells); currIndex++) {
					VectorType horizontalLine = iniHorizPoint + VectorType(0, numCellsSign*currIndex*dx, 0);
					VectorType horizontalCrossing;
					if (segmentLineIntersection(initialPoints[i], initialPoints[nextI], horizontalLine, horizontalLine + VectorType(dx, 0, 0), horizontalCrossing)) {
						DoubleScalar alpha = (horizontalCrossing - initialPoints[i]).length() / currLength;
						allCrossings.push_back(pair<DoubleScalar, VectorType>(alpha, horizontalCrossing));
					}
				}

				/** Compute vertical crossings */
				numCells = floor(initialPoints[nextI].x / dx) - floor(initVerticalPoint.x / dx);
				numCellsSign = numCells == 0 ? 1 : numCells / abs(numCells);
				for (int currIndex = 0; currIndex <= abs(numCells); currIndex++) {
					VectorType verticalCrossing;
					VectorType verticalLine = initVerticalPoint + VectorType(numCellsSign*currIndex*dx, 0, 0);
					if (segmentLineIntersection(initialPoints[i], initialPoints[nextI], verticalLine, verticalLine + VectorType(0, dx, 0), verticalCrossing)) {
						DoubleScalar alpha = (verticalCrossing - initialPoints[i]).length() / currLength;
						allCrossings.push_back(pair<DoubleScalar, VectorType>(alpha, verticalCrossing));
					}
				}

				std::sort(allCrossings.begin(), allCrossings.end(), crossingsSortFunction<VectorType>);

				Scalar alphaPrecisionThreshold = Grids::singlePrecisionThreshold*PI;
				for (int j = 0; j < allCrossings.size(); j++) {
					if (j > 0 && (allCrossings[j].first - allCrossings[j - 1].first) < alphaPrecisionThreshold) {
						//Dont push this vertex and mark previous vertex as hybrid
						//Warning
						cout << "Merging vertical and horizontal crossings due proximity" << endl;

						//Snapping the vertex position to be exaclty on the top of a grid node
						dimensions_t snappedDim(round(m_vertices.back()->getPosition().x / dx), round(m_vertices.back()->getPosition().y / dx), round(m_vertices.back()->getPosition().z / dx));
						VectorType snappedPosition(snappedDim.x*dx, snappedDim.y*dx, snappedDim.z*dx);
						m_vertices.back()->setPosition(snappedPosition);
						m_vertices.back()->setOnGridNode(true);
					}
					else {
						m_vertices.push_back(new Vertex<VectorType>(allCrossings[j].second, edgeVertex));
					}
				}
			}

			if (m_vertices.front()->getPosition() != m_params.initialPoints.back())
				m_vertices.push_back(new Vertex<VectorType>(m_params.initialPoints.back(), geometryVertex));
		}

		template<class VectorType>
		void LineMeshT<VectorType, false>::computeGridCrossingsXZ(Scalar dx, bool perturbPoints /* = false */) {
			const vector<VectorType> &initialPoints = m_params.initialPoints;
			m_vertices.clear();

			//Z point is always fixed
			VectorType iniHorizPoint;
			iniHorizPoint.y = floor(initialPoints[0].y);
			for (int i = 0; i < initialPoints.size() - 1; i++) {
				int nextI = roundClamp<int>(i + 1, 0, initialPoints.size());
				m_vertices.push_back(new Vertex<VectorType>(m_params.initialPoints[i], geometryVertex));

				iniHorizPoint.x = floor(initialPoints[i].x / dx)*dx;
				iniHorizPoint.z = floor(initialPoints[i].z / dx)*dx;

				vector<pair<DoubleScalar, VectorType>> allCrossings;
				DoubleScalar currLength = (initialPoints[nextI] - initialPoints[i]).length();

				/** Compute transversal crossings */
				int numCells = floor(initialPoints[nextI].x / dx) - floor(iniHorizPoint.x / dx);
				int numCellsSign = numCells == 0 ? 1 : numCells / abs(numCells);
				for (int currIndex = 0; currIndex <= abs(numCells); currIndex++) {
					VectorType horizontalLine = iniHorizPoint + VectorType(numCellsSign*currIndex*dx, 0, 0);
					VectorType horizontalCrossing;
					if (segmentLineIntersection(initialPoints[i], initialPoints[nextI], horizontalLine, horizontalLine + VectorType(0, 0, dx), horizontalCrossing)) {
						DoubleScalar alpha = (horizontalCrossing - initialPoints[i]).length() / currLength;
						allCrossings.push_back(pair<DoubleScalar, VectorType>(alpha, horizontalCrossing));
					}
				}

				std::sort(allCrossings.begin(), allCrossings.end(), crossingsSortFunction<VectorType>);

				Scalar alphaPrecisionThreshold = Grids::singlePrecisionThreshold*PI;
				for (int j = 0; j < allCrossings.size(); j++) {
					if (j > 0 && (allCrossings[j].first - allCrossings[j - 1].first) < alphaPrecisionThreshold) {
						//Dont push this vertex and mark previous vertex as hybrid
						//Warning
						cout << "Merging vertical and horizontal crossings due proximity" << endl;

						//Snapping the vertex position to be exaclty on the top of a grid node
						dimensions_t snappedDim(round(m_vertices.back()->getPosition().x / dx), round(m_vertices.back()->getPosition().y / dx), round(m_vertices.back()->getPosition().z / dx));
						VectorType snappedPosition(snappedDim.x*dx, snappedDim.y*dx, snappedDim.z*dx);
						m_vertices.back()->setPosition(snappedPosition);
						m_vertices.back()->setOnGridNode(true);
					}
					else {
						m_vertices.push_back(new Vertex<VectorType>(allCrossings[j].second, edgeVertex));
					}
				}
			}

			for (int i = 0; i < m_vertices.size(); i++) {
				if (m_vertices[i]->getPosition().z / dx - floor(m_vertices[i]->getPosition().z / dx) == 0) {
					m_vertices[i]->setVertexType(edgeVertex);
				}
			}

			if (m_vertices.front()->getPosition() != m_params.initialPoints.back())
				m_vertices.push_back(new Vertex<VectorType>(m_params.initialPoints.back(), geometryVertex));
		}

		template<class VectorType>
		void LineMeshT<VectorType, false>::computeGridCrossingsYZ(Scalar dx, bool perturbPoints /* = false */) {
			for (int i = 0; i < m_vertices.size(); i++) {
				if ( (m_vertices[i]->getPosition().y / dx - floor(m_vertices[i]->getPosition().y / dx) == 0) || 
					 (m_vertices[i]->getPosition().z / dx - floor(m_vertices[i]->getPosition().z / dx) == 0)) {
						m_vertices[i]->setVertexType(edgeVertex);
				}
			}
		}
		#pragma endregion
		template class LineMeshT<Vector3, isVector2<Vector3>::value>;
		template class LineMeshT<Vector3D, isVector2<Vector3D>::value>;
	}
}