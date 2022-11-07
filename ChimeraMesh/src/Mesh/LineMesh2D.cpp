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
		bool crossingsSortFunction(pair<DoubleScalar, VectorType> c1, pair<DoubleScalar, VectorType> c2) {
			return c1.first < c2.first;
		}

		template<class VectorType>
		void LineMeshT<VectorType, true>::computeGridCrossings(Scalar dx, bool perturbPoints) {
			if (perturbPoints) {
				for (int i = 0; i < m_params.initialPoints.size() - 1; i++) {
					while (Grids::isOnGridEdge(m_params.initialPoints[i], dx)) {
						VectorType perturb(((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)));
						perturb *= 1e-5;
						m_params.initialPoints[i] += perturb;
						cout << "Perturbing points" << endl;
						if (i == 0) { //Changed duplicated point, 
							m_params.initialPoints.back() = m_params.initialPoints[0];
						}
					}
				}
			}
			const vector<VectorType> &initialPoints = m_params.initialPoints;
			m_vertices.clear();
			for (int i = 0; i < initialPoints.size() - 1; i++) {
				int nextI = roundClamp<int>(i + 1, 0, initialPoints.size());
				m_vertices.push_back(new Vertex<VectorType>(m_params.initialPoints[i], geometryVertex));
			
				VectorType iniHorizPoint;
				iniHorizPoint.x = floor(initialPoints[i].x / dx)*dx;
				iniHorizPoint.y = floor(initialPoints[i].y / dx)*dx;

				VectorType initVerticalPoint;
				initVerticalPoint.y = floor(initialPoints[i].y / dx)*dx;
				initVerticalPoint.x = floor(initialPoints[i].x / dx)*dx;

				vector<pair<DoubleScalar, VectorType>> allCrossings;
				DoubleScalar currLength = (initialPoints[nextI] - initialPoints[i]).length();

				/** Compute horizontal crossings */
				int numCells = floor(initialPoints[nextI].y / dx) - floor(iniHorizPoint.y / dx);
				int numCellsSign = numCells == 0 ? 1 : numCells / abs(numCells);
				for (int currIndex = 0; currIndex <= abs(numCells); currIndex++) {
					VectorType horizontalLine = iniHorizPoint + VectorType(0, numCellsSign*currIndex*dx);
					VectorType horizontalCrossing;
					if (segmentLineIntersection(initialPoints[i], initialPoints[nextI], horizontalLine, horizontalLine + VectorType(dx, 0), horizontalCrossing)) {
						DoubleScalar alpha = (horizontalCrossing - initialPoints[i]).length() / currLength;
						allCrossings.push_back(pair<DoubleScalar, VectorType>(alpha, horizontalCrossing));
					}
				}

				/** Compute vertical crossings */
				numCells = floor(initialPoints[nextI].x / dx) - floor(initVerticalPoint.x / dx);
				numCellsSign = numCells == 0 ? 1 : numCells / abs(numCells);
				for (int currIndex = 0; currIndex <= abs(numCells); currIndex++) {
					VectorType verticalCrossing;
					VectorType verticalLine = initVerticalPoint + VectorType(numCellsSign*currIndex*dx, 0);
					if (segmentLineIntersection(initialPoints[i], initialPoints[nextI], verticalLine, verticalLine + VectorType(0, dx), verticalCrossing)) {
						DoubleScalar alpha = (verticalCrossing - initialPoints[i]).length() / currLength;
						allCrossings.push_back(pair<DoubleScalar, VectorType>(alpha, verticalCrossing));
					}
				}

				std::sort(allCrossings.begin(), allCrossings.end(), crossingsSortFunction<VectorType>);

				Scalar alphaPrecisionThreshold = Grids::singlePrecisionThreshold*PI * 10;
				for (int j = 0; j < allCrossings.size(); j++) {
					if (j > 0 && (allCrossings[j].first - allCrossings[j - 1].first) < alphaPrecisionThreshold) {
						//Dont push this vertex and mark previous vertex as hybrid
						//Warning
						cout << "Merging vertical and horizontal crossings due proximity" << endl;
						
						//Snapping the vertex position to be exaclty on the top of a grid node
						dimensions_t snappedDim(round(m_vertices.back()->getPosition().x / dx), round(m_vertices.back()->getPosition().y / dx));
						VectorType snappedPosition(snappedDim.x*dx, snappedDim.y*dx);
						m_vertices.back()->setPosition(snappedPosition);
						m_vertices.back()->setOnGridNode(true);
					}
					else {
						m_vertices.push_back(new Vertex<VectorType>(allCrossings[j].second, edgeVertex));
					}
				}
			}
			if(m_vertices.front()->getPosition() != m_params.initialPoints.back()) 
				m_vertices.push_back(new Vertex<VectorType>(m_params.initialPoints.back(), geometryVertex)); 
		}

		template<class VectorType>
		void LineMeshT<VectorType, true>::extrudeAlongNormals() {
			vector<VectorType> tempPoints(m_params.initialPoints.size());
			for (int i = 0; i < m_params.initialPoints.size(); i++) {
				tempPoints.push_back(m_params.initialPoints[i]);
				m_vertices.push_back(new Vertex<VectorType>(m_params.initialPoints[i], geometryVertex));
			}
			for (int i = 0; i < m_params.initialPoints.size(); i++) {
				VectorType normal;
				if (i == m_params.initialPoints.size() - 1) {
					normal = (m_params.initialPoints.back() - m_params.initialPoints[m_params.initialPoints.size() - 2]);
				}
				else if (i == 0) {
					normal = m_params.initialPoints[i + 1] - m_params.initialPoints[i];
				}
				else {
					normal = (m_params.initialPoints[i + 1] - m_params.initialPoints[i])*0.5 +
						(m_params.initialPoints[i] - m_params.initialPoints[i - 1])*0.5;
				}
				normal = normal.perpendicular().normalized();

				VectorType tempNode = m_params.initialPoints[i] + normal*m_params.extrudeAlongNormalWidth;
				if (i > 0 && segmentIntersection(tempNode, tempPoints[(m_params.initialPoints.size() - 1) - (i - 1)])) {
					normal = -normal;
				}
				tempPoints[(m_params.initialPoints.size() - 1) - i] = m_params.initialPoints[i] + normal*m_params.extrudeAlongNormalWidth;

				while (Grids::isOnGridEdge(tempPoints[(m_params.initialPoints.size() - 1) - i], m_gridDx)) {
					VectorType perturb(((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)));
					perturb *= 1e-5;
					tempPoints[(m_params.initialPoints.size() - 1) - i] += perturb;
					cout << "Perturbing points" << endl;
				}
			}

			m_vertices.clear();
			for (int i = 0; i < tempPoints.size(); i++) {
				m_vertices.push_back(new Vertex<VectorType>(tempPoints[i], geometryVertex));
			}

			m_isClosedMesh = true;
			m_params.initialPoints.resize(m_vertices.size());
			for (int i = 0; i < m_vertices.size(); i++) {
				m_params.initialPoints[i] = m_vertices[i]->getPosition();
			}
			m_params.initialPoints.push_back(m_params.initialPoints.front());

			//initializeEdges();
			updateCentroid();
		}

		template<class VectorType>
		void LineMeshT<VectorType, true>::perturbOnGridLineEdges(Scalar dx) {
			/*for (int i = 0; i < m_vertices.size() - 1; i++) {
				VectorType edgeCentroid = (m_vertices[i]->getPosition() + m_vertices[i + 1]->getPosition())*0.5;
				if (Grids::isOnGridEdge(edgeCentroid, dx)) {
					cout << "HOUSTON WE NEED TO CHECK THIS" << endl;
					VectorType edgeNormal = (m_vertices[i + 1]->getPosition() - m_vertices[i]->getPosition()).perpendicular();
					edgeNormal.normalize();
					m_vertices[i + 1]->setPosition(m_vertices[i + 1]->getPosition() + edgeNormal*1e-5);
					m_vertices[i]->setPosition(m_vertices[i]->getPosition() - edgeNormal*1e-5);
				}
			}*/
	
		}


		#pragma endregion
		template class LineMeshT<Vector2, isVector2<Vector2>::value>;
		template class LineMeshT<Vector2D, isVector2<Vector2D>::value>;
	}
}