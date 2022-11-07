#include "Mesh/NonManifoldMesh3D.h"
#include "CGAL/PolygonSurface.h"

namespace Chimera {
	namespace Meshes {

		NonManifoldMesh3D::NonManifoldMesh3D(dimensions_t cellDim, const vector<Rendering::MeshPatchMap *> &patchMaps, CutCells3D *pCutCells) : m_patchMaps(patchMaps) {
			m_pCutCells = pCutCells;
			m_voxelDim = cellDim;
			m_distanceProximity = doublePrecisionThreshold;;

			Scalar dx = m_pCutCells->getGridSpacing();

			//Adding all grid point vertices from the cut-cells into the vertices structure
			for (int k = 0; k < 2; k++) {
				for (int j = 0; j < 2; j++) {
					for (int i = 0; i < 2; i++) {
						Vector3D gridPoint((cellDim.x + i)*dx, (cellDim.y + j)*dx, (cellDim.z + k)*dx);
						m_vertices.push_back(gridPoint);
					}
				}
			}
			//Saving the initial index offset. All grid points after this offset are the points generated
			//by the intersection of the object with the cut-voxel
			m_initialVertexOffset = m_vertices.size();

			//Adding all cell faces structures into the non-manifold mesh
			vector<CutFace<Vector3D> *> leftFaces = m_pCutCells->getLeftFaceVector(cellDim);
			for (int i = 0; i < leftFaces.size(); i++) {
				addFace(leftFaces[i], leftFace);
			}
			vector<CutFace<Vector3D> *> rightFaces = m_pCutCells->getLeftFaceVector(dimensions_t(cellDim.x + 1, cellDim.y, cellDim.z));
			for (int i = 0; i < rightFaces.size(); i++) {
				addFace(rightFaces[i], rightFace);
			}
			vector<CutFace<Vector3D> *> bottomFaces = m_pCutCells->getBottomFaceVector(cellDim);
			for (int i = 0; i < bottomFaces.size(); i++) {
				addFace(bottomFaces[i], bottomFace);
			}
			vector<CutFace<Vector3D> *> topFaces = m_pCutCells->getBottomFaceVector(dimensions_t(cellDim.x, cellDim.y + 1, cellDim.z));
			for (int i = 0; i < topFaces.size(); i++) {
				addFace(topFaces[i], topFace);
			}
			vector<CutFace<Vector3D> *> backFaces = m_pCutCells->getBackFaceVector(cellDim);
			for (int i = 0; i < backFaces.size(); i++) {
				addFace(backFaces[i], backFace);
			}
			vector<CutFace<Vector3D> *> frontFaces = m_pCutCells->getBackFaceVector(dimensions_t(cellDim.x, cellDim.y, cellDim.z + 1));
			for (int i = 0; i < frontFaces.size(); i++) {
				addFace(frontFaces[i], frontFace);
			}

			//Adding geometric faces
			for (int k = 0; k < m_patchMaps.size(); k++) {
				for (int i = 0; i < m_patchMaps[k]->faces.size(); i++) {
					CutFace<Vector3D> *pNewFace = new CutFace<Vector3D>(geometryFace);
					vector<pair<unsigned int, unsigned int>> currFace = m_patchMaps[k]->pPolySurface->getFaces()[m_patchMaps[k]->faces[i]].edges;
					Vector3D trianglePoints[3];
					if (currFace.size() > 3) {
						throw("NonManifoldMesh3D: invalid number of points on geometry face");
					}
					for (int j = 0; j < currFace.size(); j++) {
						int nextJ = roundClamp<int>(j + 1, 0, currFace.size());
						pNewFace->insertCutEdge(m_patchMaps[k]->pPolySurface->getVertices()[currFace[j].first], m_patchMaps[k]->pPolySurface->getVertices()[currFace[j].second], dx, geometryEdge, 0);
						trianglePoints[j] = m_patchMaps[k]->pPolySurface->getVertices()[currFace[j].first];
						pNewFace->m_originalPolyIndices.push_back(currFace[j].first);
					}
					pNewFace->m_normal = m_patchMaps[k]->pPolySurface->getNormals()[m_patchMaps[k]->faces[i]];
					pNewFace->m_centroid = m_patchMaps[k]->pPolySurface->getFacesCentroids()[m_patchMaps[k]->faces[i]];
					pNewFace->m_pPolygonSurface = m_patchMaps[k]->pPolySurface;
					pNewFace->m_areaFraction = abs(calculateTriangleArea(trianglePoints[0], trianglePoints[1], trianglePoints[2]) / (dx*dx));
					addFace(pNewFace, geometryFace);
					m_faces.back().pPolygonSurface = m_patchMaps[k]->pPolySurface;
					m_faces.back().meshPatchID = k;
				}
			}

			//Verify edges sizes for projected points inconsistencies
			//for (int i = 0; i < m_faces.size(); i++) {
			//	for (int j = 0; j < m_faces[i].edgeIndices.size(); j++) {
			//		DoubleScalar edgeDistance = (m_vertices[m_faces[i].edgeIndices[j].second] - m_vertices[m_faces[i].edgeIndices[j].first]).length();
			//		if (edgeDistance < m_distanceProximity) { //Beyond numerical error for float representation
			//			Logger::getInstance()->get() << "Warning: Tagging geometry face as small edge. Edge distance: " << edgeDistance << endl;
			//			m_faces[i].m_smalledgeFace = true;
			//			break;
			//		}
			//	}

			//}

			//Adding all faces references to the edge map
			for (int i = 0; i < m_faces.size(); i++) {
				for (int j = 0; j < m_faces[i].edgeIndices.size(); j++) {
					DoubleScalar edgeDistance = (m_vertices[m_faces[i].edgeIndices[j].second] - m_vertices[m_faces[i].edgeIndices[j].first]).length();
					if (edgeDistance < m_distanceProximity || m_faces[i].edgeIndices[j].second == m_faces[i].edgeIndices[j].first) { //Beyond numerical error for float representation
						Logger::getInstance()->get() << "Warning: Tagging geometry face as small edge. Edge distance: " << edgeDistance << endl;
						Logger::getInstance()->get() << "Edge indices " << m_faces[i].edgeIndices[j].first << " " << m_faces[i].edgeIndices[j].second << endl;
						Logger::getInstance()->get() << "Number of edges of this face: " << m_faces[i].edgeIndices.size() << endl;
						Logger::getInstance()->get() << "Dimensions " << cellDim.x << ", " << cellDim.y << ", " << cellDim.z << endl;
						m_faces[i].m_smalledgeFace = true;
						//Removing all edges from the small edge face face
						for (int k = j - 1; k >= 0; k--) {
							removeFaceFromEdgeMap(m_faces[i].edgeIndices[k].first, m_faces[i].edgeIndices[k].second, m_faces[i].ID);
						}
						break;
					}
					else {
						m_edgeMap[edgeHash(m_faces[i].edgeIndices[j].first, m_faces[i].edgeIndices[j].second)].push_back(&m_faces[i]);
					}
				}
			}


			//correctInteriorPointsForRobustness();
		}
		DoubleScalar NonManifoldMesh3D::computeSignedDistanceFunction(const Vector3D & point, Rendering::PolygonSurface *pPolySurface) {
			Vector3D pseudoNormal, closestPointOnMesh;
			pPolySurface->getClosestPoint(point, closestPointOnMesh, pseudoNormal);

			Vector3D directionVec = point - closestPointOnMesh;
			DoubleScalar dotDir = pseudoNormal.dot(directionVec);
			if (abs(dotDir) < singlePrecisionThreshold) {
				Logger::getInstance()->get() << "NonManifoldMesh3D::computeSignedDistanceFunction() warning: small dot product at " << vector3ToStr(point) << endl;
			}
			if (dotDir < 0) {
				return -directionVec.length();
			}
			return directionVec.length();
		}

		int NonManifoldMesh3D::computeSignedDistanceFunctionTag(const Vector3D & point, Rendering::PolygonSurface *pPolySurface) {
			return computeSignedDistanceFunction(point, pPolySurface) > 0 ? 1 : 0;
		}

		vector<CutVoxel> NonManifoldMesh3D::split(unsigned int totalCutVoxels, bool onEdgeMixedNodes) {
			int ithNonVisitedFace;

			vector<CutVoxel> cutVoxels;
			while ((ithNonVisitedFace = findNonVisitedFace()) != -1) {
				CutVoxel nCutVoxel;
				nCutVoxel.ID = totalCutVoxels + cutVoxels.size();
				nCutVoxel.regularGridIndex = m_voxelDim;
				nCutVoxel.centroid.x = (m_voxelDim.x + 0.5)*m_pCutCells->getGridSpacing();
				nCutVoxel.centroid.y = (m_voxelDim.y + 0.5)*m_pCutCells->getGridSpacing();
				nCutVoxel.centroid.z = (m_voxelDim.z + 0.5)*m_pCutCells->getGridSpacing();
				/** Setting cut-voxels vertices to be the same as the non-manifold mesh.
				Some vertices will be unreferenced inside the cut-voxel, but this can be treated
				later when passing it to CGAL.*/
				nCutVoxel.m_vertices = m_vertices;
				//Tag all geometry faces as not visited
				for (int i = 0; i < m_faces.size(); i++) {
					if (m_faces[i].faceLocation == geometryFace) {
						m_visitedFaces[i] = false;
					}
				}
				//Tag all dangling patch map as non visited 
				for (int i = 0; i < m_patchMaps.size(); i++) {
					if (m_patchMaps[i]->danglingPatch) {
						m_patchMaps[i]->visited = false;
					}
				}
				int regionTag = -1; //Uninitialized region tags

				breadthFirstSearch(m_faces[ithNonVisitedFace], nCutVoxel, regionTag);

				/** Checking for patch maps with dangling faces*/
				for (int i = 0; i < m_patchMaps.size(); i++) {
					if (m_patchMaps[i]->danglingPatch && m_patchMaps[i]->visited) {
						addDoubleSidedGeometryFaces(nCutVoxel, m_patchMaps[i], i, m_pCutCells->getGridSpacing());
						nCutVoxel.danglingVoxel = true;
					}
				}
				cutVoxels.push_back(nCutVoxel);
			}

			return cutVoxels;
		}

		void NonManifoldMesh3D::addDoubleSidedGeometryFaces(CutVoxel &cutVoxel, Rendering::MeshPatchMap *pPatchMap, int patchMapID, Scalar dx) {
			for (int i = 0; i <pPatchMap->faces.size(); i++) {
				CutFace<Vector3D> *pNewFace1 = new CutFace<Vector3D>(geometryFace);
				CutFace<Vector3D> *pNewFace2 = new CutFace<Vector3D>(geometryFace);
				vector<pair<unsigned int, unsigned int>> currFace = pPatchMap->pPolySurface->getFaces()[pPatchMap->faces[i]].edges;
				Vector3D trianglePoints[3];
				vector<pair<unsigned int, unsigned int>> meshFaceIndices;
				for (int j = 0; j < currFace.size(); j++) {
					int nextJ = roundClamp<int>(j + 1, 0, currFace.size());
					Vector3D edgeInitialPoint = pPatchMap->pPolySurface->getVertices()[currFace[j].first];
					Vector3D edgeFinalPoint = pPatchMap->pPolySurface->getVertices()[currFace[j].second];
					pNewFace1->insertCutEdge(edgeInitialPoint, edgeFinalPoint, dx, geometryEdge, 0);
					trianglePoints[j] = pPatchMap->pPolySurface->getVertices()[currFace[j].first];
					pNewFace1->m_originalPolyIndices.push_back(currFace[j].first);
					int initialPointIndex, finalPointIndex;
					dimensions_t gridPointDim;
					if (isOnGridPoint(pPatchMap->pPolySurface->getVertices()[currFace[j].first], dx, gridPointDim)) {
						initialPointIndex = gridPointHash(gridPointDim);
					}
					else {
						initialPointIndex = addVertex(edgeInitialPoint);
					}
					if (isOnGridPoint(pPatchMap->pPolySurface->getVertices()[currFace[j].second], dx, gridPointDim)) {
						finalPointIndex = gridPointHash(gridPointDim);
					}
					else {
						finalPointIndex = addVertex(edgeFinalPoint);
					}
					meshFaceIndices.push_back(pair<unsigned int, unsigned int>(initialPointIndex, finalPointIndex));
				}
				pNewFace1->m_normal = pPatchMap->pPolySurface->getNormals()[pPatchMap->faces[i]];
				pNewFace1->m_centroid = pPatchMap->pPolySurface->getFacesCentroids()[pPatchMap->faces[i]];
				pNewFace1->m_pPolygonSurface = pPatchMap->pPolySurface;
				pNewFace1->m_areaFraction = abs(calculateTriangleArea(trianglePoints[0], trianglePoints[1], trianglePoints[2]) / (dx*dx));
				*pNewFace2 = *pNewFace1;
				pNewFace2->m_normal = -pNewFace2->m_normal;
				cutVoxel.cutFaces.push_back(pNewFace1);
				cutVoxel.m_edgeIndices.push_back(meshFaceIndices);
				cutVoxel.cutFaces.push_back(pNewFace2);

				reverse(meshFaceIndices.begin(), meshFaceIndices.end());
				for (int j = 0; j < meshFaceIndices.size(); j++) {
					swap(meshFaceIndices[j].first, meshFaceIndices[j].second);
				}
				cutVoxel.m_edgeIndices.push_back(meshFaceIndices);
				cutVoxel.cutFacesLocations.push_back(geometryFace);
				cutVoxel.cutFacesLocations.push_back(geometryFace);
			}
		}


		Vector3D NonManifoldMesh3D::getFaceNormal(faceLocation_t faceLocation) {
			switch (faceLocation) {
			case leftFace:
				return Vector3D(1, 0, 0);
				break;
			case rightFace:
				return Vector3D(-1, 0, 0);
				break;
			case bottomFace:
				return Vector3D(0, 1, 0);
				break;
			case topFace:
				return Vector3D(0, -1, 0);
				break;
			case backFace:
				return Vector3D(0, 0, 1);
				break;
			case frontFace:
				return Vector3D(0, 0, -1);
				break;
			}
			return Vector3D(0, 0, 0);
		}

		Vector2D NonManifoldMesh3D::convertToVec2(const Vector3D &vec, faceLocation_t faceLocation) {
			switch (faceLocation) {
			case leftFace:
			case rightFace:
				return Vector2D(vec.z, vec.y);
				break;

			case Chimera::Data::bottomFace:
			case Chimera::Data::topFace:
				return Vector2D(vec.x, vec.z);
				break;

			case Chimera::Data::backFace:
			case Chimera::Data::frontFace:
				return Vector2D(vec.x, vec.y);
				break;
			}
			return Vector2D(0, 0);
		}

		Vector2D NonManifoldMesh3D::projectIntoPlane(const Vector3D &vec, faceLocation_t faceLocation) {
			switch (faceLocation) {
			case leftFace:
			case rightFace:
				return Vector2D(vec.z, vec.y);
				break;

			case Chimera::Data::bottomFace:
			case Chimera::Data::topFace:
				return Vector2D(vec.x, vec.z);
				break;

			case Chimera::Data::backFace:
			case Chimera::Data::frontFace:
				return Vector2D(vec.x, vec.y);
				break;
			}
			return Vector2D(0, 0);
		}

		void NonManifoldMesh3D::breadthFirstSearch(face_t &face, CutVoxel &nCutVoxel, int regionTag) {
			if (m_visitedFaces[face.ID] || face.m_smalledgeFace)
				return;

			m_visitedFaces[face.ID] = true;
			nCutVoxel.cutFaces.push_back(face.pCutFace);
			nCutVoxel.cutFacesLocations.push_back(face.faceLocation);
			if (face.pCutFace->m_faceNeighbors[0] == -1) {
				face.pCutFace->m_faceNeighbors[0] = nCutVoxel.ID;
			}
			else if(face.pCutFace->m_faceNeighbors[1] == -1) {
				face.pCutFace->m_faceNeighbors[1] = nCutVoxel.ID;
			}
			else {
				throw exception("Invalid face neightbors");
			}
			nCutVoxel.m_edgeIndices.push_back(face.edgeIndices);

			for (int j = 0; j < face.edgeIndices.size(); j++) {
				vector<face_t *> currEdgeFaces = m_edgeMap[edgeHash(face.edgeIndices[j].first, face.edgeIndices[j].second)];
				//If still we have more than expected number of neighbars
				if (currEdgeFaces.size() > 3) {
					string exceptionStr("NonManifoldMesh: high frequency feature on top of an edge.\n");
					exceptionStr += "***COMMENCING INFORMATION DUMPING***\n";
					exceptionStr += "Dimensions: " + intToStr(nCutVoxel.regularGridIndex.x) + ", ";
					exceptionStr += intToStr(nCutVoxel.regularGridIndex.y) + ", " + intToStr(nCutVoxel.regularGridIndex.z) + '\n';
					exceptionStr += "Faces location, currFace: " + intToStr(face.faceLocation) + ", ";
					exceptionStr += "Edge points, initial point: " + vector3ToStr(m_vertices[face.edgeIndices[j].first]) + '\n';
					exceptionStr += "Edge points, final point: " + vector3ToStr(m_vertices[face.edgeIndices[j].second]) + '\n';
					exceptionStr += "Number of face neighbors " + intToStr(currEdgeFaces.size()) + '\n';
					exceptionStr += "Face ID: " + intToStr(face.ID) + '\n';
					exceptionStr += "Face location: " + intToStr(face.faceLocation) + '\n';
					exceptionStr += "Face interior point: " + vector3ToStr(face.pCutFace->m_interiorPoint) + '\n';
					throw exception(exceptionStr.c_str());
				}
				else if (currEdgeFaces.size() > 2) {
					face_t *pF1 = currEdgeFaces[0]->ID != face.ID ? currEdgeFaces[0] : currEdgeFaces[1];
					face_t *pF2 = currEdgeFaces[2]->ID != face.ID ? currEdgeFaces[2] : currEdgeFaces[1];
					if (face.faceLocation != geometryFace) {
						if (pF1->faceLocation == geometryFace) {
							if (m_patchMaps[pF1->meshPatchID]->danglingPatch) {
								//Mark this patch map as visited, to add to this cut-voxel later
								m_patchMaps[pF1->meshPatchID]->visited = true;
								//This whole patch is a dangling face, ignore it and continue to the next grid face
								breadthFirstSearch(*pF2, nCutVoxel, regionTag);
							}
							else { //Initialize region tag, if not initialized yet
								if (regionTag == -1) { //Every time we access a geometry face from a grid face, we initialize region tag
									bool reverseOrientation = false;
									for (int i = 0; i < pF1->edgeIndices.size(); i++) {
										if (pF1->edgeIndices[i].second == face.edgeIndices[j].first && pF1->edgeIndices[i].first == face.edgeIndices[j].second) {
											reverseOrientation = true;
											break;
										}
									}
									if (reverseOrientation) {
										regionTag = 1;
									}
									else {
										regionTag = 0;
									}
									//regionTag = computeSignedDistanceFunctionTag(face.interiorPoint, pF1->pPolygonSurface);
								}
								breadthFirstSearch(*pF1, nCutVoxel, regionTag);
							}
						}
						else if (pF2->faceLocation == geometryFace) {
							if (m_patchMaps[pF2->meshPatchID]->danglingPatch) {
								//Mark this patch map as visited, to add to this cut-voxel later
								m_patchMaps[pF2->meshPatchID]->visited = true;
								//This whole patch is a dangling face, ignore it and continue to the next grid face
								breadthFirstSearch(*pF1, nCutVoxel, regionTag);
							}
							else { //Initialize region tag, if not initialized yet
								if (regionTag == -1) { //Every time we access a geometry face from a grid face, we initialize region tag
									//Find geometry edge that is shared with regular face
									bool reverseOrientation = false;
									for (int i = 0; i < pF2->edgeIndices.size(); i++) {
										if (pF2->edgeIndices[i].second == face.edgeIndices[j].first && pF2->edgeIndices[i].first == face.edgeIndices[j].second) {
											reverseOrientation = true;
											break;
										}
									}
									if (reverseOrientation) {
										regionTag = 1;
									}
									else {
										regionTag = 0;
									}
									//regionTag = computeSignedDistanceFunctionTag(face.interiorPoint, pF2->pPolygonSurface);
								}
								breadthFirstSearch(*pF2, nCutVoxel, regionTag);
							}
						}
						else {
							throw exception("NonManifoldMesh: invalid edge neighbor configuration");
						}
					}
					else { //This face is a geometry face and the edge is on a T junction
						if (regionTag == -1) {
							throw ("NonManifoldMesh: invalid region tag.\n");
						}
						if (pF1->m_smalledgeFace || pF2->m_smalledgeFace) { //We cant know
							return;
						}

						int regionTagF1, regionTagF2;
						//Find geometry edge that is shared with regular face
						bool reverseOrientation = false;
						for (int i = 0; i < pF1->edgeIndices.size(); i++) {
							if (pF1->edgeIndices[i].second == face.edgeIndices[j].first && pF1->edgeIndices[i].first == face.edgeIndices[j].second) {
								reverseOrientation = true;
								break;
							}
						}
						if (reverseOrientation) {
							regionTagF1 = 1;
						}
						else {
							regionTagF1 = 0;
						}

						reverseOrientation = false;
						for (int i = 0; i < pF2->edgeIndices.size(); i++) {
							if (pF2->edgeIndices[i].second == face.edgeIndices[j].first && pF2->edgeIndices[i].first == face.edgeIndices[j].second) {
								reverseOrientation = true;
								break;
							}
						}
						if (reverseOrientation) {
							regionTagF2 = 1;
						}
						else {
							regionTagF2 = 0;
						}

						//DoubleScalar s1DistanceFunction = computeSignedDistanceFunction(pF1->interiorPoint, face.pPolygonSurface);
						//DoubleScalar s2DistanceFunction = computeSignedDistanceFunction(pF2->interiorPoint, face.pPolygonSurface);

						DoubleScalar s1DistanceFunction = 0;
						DoubleScalar s2DistanceFunction = 0;
						if (regionTagF1 == regionTag) {
							breadthFirstSearch(*pF1, nCutVoxel, regionTag);
						}
						else if (regionTagF2 == regionTag) {
							breadthFirstSearch(*pF2, nCutVoxel, regionTag);
						}
						else {
							string exceptionStr("NonManifoldMesh: invalid plane orientation check.\n");
							exceptionStr += "***COMMENCING INFORMATION DUMPING***\n";
							if (regionTag) {
								exceptionStr += "Outside iteration loop" + '\n';
							}
							else {
								exceptionStr += "Inside iteration loop" + '\n';
							}
							exceptionStr += "Dimensions: " + intToStr(nCutVoxel.regularGridIndex.x) + ", ";
							exceptionStr += intToStr(nCutVoxel.regularGridIndex.y) + ", " + intToStr(nCutVoxel.regularGridIndex.z) + '\n';
							exceptionStr += "Edge points, initial point: " + vector3ToStr(m_vertices[face.edgeIndices[j].first]) + '\n';
							exceptionStr += "Edge points, final point: " + vector3ToStr(m_vertices[face.edgeIndices[j].first]) + '\n';
							exceptionStr += "Faces location, currFace: " + intToStr(face.faceLocation) + ", ";
							exceptionStr += "pF1: " + intToStr(pF1->faceLocation) + ", ";
							exceptionStr += "pF2: " + intToStr(pF2->faceLocation) + '\n';
							exceptionStr += "Faces interior points, pF1: " + vector3ToStr(pF1->interiorPoint) + '\n';
							exceptionStr += "pF2: " + vector3ToStr(pF2->interiorPoint) + '\n';
							Vector3D tempPseudoNormal, tempClosestPointOnMesh, tempDirectionVector;
							exceptionStr += "***COMMENCING SIGNED DISTANCE DUMPING***\n";

							exceptionStr += "pF1 distance value" + scalarToStr(s1DistanceFunction) + '\n';
							m_patchMaps.front()->pPolySurface->getClosestPoint(pF1->interiorPoint, tempClosestPointOnMesh, tempPseudoNormal);
							exceptionStr += "pF1 normal " + vector3ToStr(tempPseudoNormal) + '\n';
							exceptionStr += "pF1 closest Point " + vector3ToStr(tempClosestPointOnMesh) + '\n';
							tempDirectionVector = pF1->interiorPoint - tempClosestPointOnMesh;
							exceptionStr += "pF1 direction vector" + vector3ToStr(tempDirectionVector) + '\n';
							exceptionStr += "pF1 dot" + scalarToStr(tempDirectionVector.normalized().dot(tempPseudoNormal)) + '\n';

							exceptionStr += "pF2 distance value" + scalarToStr(s2DistanceFunction) + '\n';
							m_patchMaps.front()->pPolySurface->getClosestPoint(pF2->interiorPoint, tempClosestPointOnMesh, tempPseudoNormal);
							exceptionStr += "pF2 normal " + vector3ToStr(tempPseudoNormal) + '\n';
							exceptionStr += "pF2 closest Point " + vector3ToStr(tempClosestPointOnMesh) + '\n';
							tempDirectionVector = pF2->interiorPoint - tempClosestPointOnMesh;
							exceptionStr += "pF2 direction vector" + vector3ToStr(tempDirectionVector) + '\n';
							throw exception(exceptionStr.c_str());
						}
					}
				}
				else if (currEdgeFaces.size() > 1) {
					face_t *pNextFace = currEdgeFaces[0]->ID != face.ID ? currEdgeFaces[0] : currEdgeFaces[1];
					if (pNextFace->faceLocation == geometryFace && m_patchMaps[pNextFace->meshPatchID]->danglingPatch) {
						//Just ignore it, we are not following through dangling faces
					}
					else {
						breadthFirstSearch(*pNextFace, nCutVoxel, regionTag);
					}
				}
				else if (currEdgeFaces.size() == 1) { //Probably a hole left from removing small faces, just continue

				}
				else {
					string exceptionStr("NonManifoldMesh: invalid number of edge neighbors\n");
					exceptionStr += "***COMMENCING INFORMATION DUMPING***\n";
					exceptionStr += "Dimensions: " + intToStr(nCutVoxel.regularGridIndex.x) + ", ";
					exceptionStr += intToStr(nCutVoxel.regularGridIndex.y) + ", " + intToStr(nCutVoxel.regularGridIndex.z) + '\n';
					exceptionStr += "Edge points, initial point: " + vector3ToStr(m_vertices[face.edgeIndices[j].first]) + '\n';
					exceptionStr += "Edge points, final point: " + vector3ToStr(m_vertices[face.edgeIndices[j].second]) + '\n';
					exceptionStr += "Number of edge neighbors " + intToStr(currEdgeFaces.size()) + '\n';
					exceptionStr += "Face ID: " + intToStr(face.ID) + '\n';
					exceptionStr += "Face location: " + intToStr(face.faceLocation) + '\n';
					exceptionStr += "Face interior point: " + vector3ToStr(face.pCutFace->m_interiorPoint) + '\n';


					throw exception(exceptionStr.c_str());
				}
			}
		}

		unsigned int NonManifoldMesh3D::addVertex(const Vector3D &newVertex) {
			for (int i = m_initialVertexOffset; i < m_vertices.size(); i++) {
				DoubleScalar vertexDistance = (m_vertices[i] - newVertex).length();
				if (vertexDistance < m_distanceProximity) {
					return i;
				}
			}
			m_vertices.push_back(newVertex);
			return m_vertices.size() - 1;
		}

		void NonManifoldMesh3D::addFace(CutFace<Vector3D> *pFace, faceLocation_t faceLocation) {
			Scalar dx = m_pCutCells->getGridSpacing();
			face_t currFace(m_faces.size());
			currFace.pCutFace = pFace;
			bool reverseEdgeIndices = false;
			if ((pFace->m_normal - getFaceNormal(faceLocation)).length() < m_distanceProximity) {
				reverseEdgeIndices = true;
			}
			//Adding faces indices
			for (int j = 0; j < pFace->m_cutEdges.size(); j++) {
				dimensions_t gridPointDim;
				Vector3D edgeInitialPoint = pFace->getEdgeInitialPoint(j);
				Vector3D edgeFinalPoint = pFace->getEdgeFinalPoint(j);
				int initialPointIndex, finalPointIndex;
				if (isOnGridPoint(edgeInitialPoint, dx, gridPointDim)) {
					initialPointIndex = gridPointHash(gridPointDim);
				}
				else {
					initialPointIndex = addVertex(edgeInitialPoint);
				}
				if (isOnGridPoint(edgeFinalPoint, dx, gridPointDim)) {
					finalPointIndex = gridPointHash(gridPointDim);
				}
				else {
					finalPointIndex = addVertex(edgeFinalPoint);
				}
				currFace.edgeIndices.push_back(pair<unsigned int, unsigned int>(initialPointIndex, finalPointIndex));
				if (initialPointIndex == finalPointIndex) {
					std::cout.precision(15);
					Logger::getInstance()->get() << "Invalid edge detected with index " << initialPointIndex << endl;
					Logger::getInstance()->get() << "Corresponding edge points are " << vector3ToStr(edgeInitialPoint) << "; " << vector3ToStr(edgeFinalPoint) << endl;
				}
				currFace.faceLocation = faceLocation;
			}

			if (currFace.edgeIndices.back() == currFace.edgeIndices.front())
				currFace.edgeIndices.pop_back();

			if (reverseEdgeIndices) {
				for (int i = 0; i < currFace.edgeIndices.size(); i++) {
					swap(currFace.edgeIndices[i].first, currFace.edgeIndices[i].second);
				}
				reverse(currFace.edgeIndices.begin(), currFace.edgeIndices.end()); 
			}
			//If the face is a geometryFace it is a triangle
			//if (currFace.faceLocation == geometryFace) {
			//	Vector3D v1 = m_vertices[currFace.edgeIndices[0].first];
			//	Vector3D v2 = m_vertices[currFace.edgeIndices[1].first];
			//	Vector3D v3 = m_vertices[currFace.edgeIndices[2].first ];

			//	currFace.interiorPoint = (v1 + v2 + v3)/ 3;
			//} else {
			//	//Finding a valid point inside a face. Find an arbitrary valid triangle and use its centroid
			//	for (int j = 0; j < currFace.edgeIndices.size(); j++) {
			//		int nextJ = roundClamp<int>(j + 1, 0, currFace.edgeIndices.size());
			//		int nextNextJ = roundClamp<int>(j + 2, 0, currFace.edgeIndices.size());

			//		Vector3D v1 = m_vertices[currFace.edgeIndices[j].first];
			//		Vector3D v2 = m_vertices[currFace.edgeIndices[nextJ].first];
			//		Vector3D v3 = m_vertices[currFace.edgeIndices[nextNextJ].first];

			//		Vector3D centroid = (v1 + v2 + v3) / 3;

			//		if (isInsideFace(centroid, currFace)) {
			//			currFace.interiorPoint = centroid;
			//			break;
			//		}
			//		else if (j == currFace.edgeIndices.size() - 1) {
			//			throw exception("NonManifoldMesh: valid point inside cell face not found");
			//		}
			//	}
			//}
			currFace.interiorPoint = currFace.pCutFace->m_interiorPoint;
			m_faces.push_back(currFace);
			m_visitedFaces.push_back(false);
		}

		bool NonManifoldMesh3D::isInsideFace(const Vector3D & point, const face_t &face) {
			vector<Vector2D> polygonTempPoints;
			Vector2D projectedPoint;
			switch (face.faceLocation) {
			case leftFace:
			case rightFace:
				for (int i = 0; i < face.edgeIndices.size(); i++) {
					Vector3D originalPoint = m_vertices[face.edgeIndices[i].first];
					//Projecting onto the YZ plane
					Vector2D convertedPoint(originalPoint.y, originalPoint.z);
					polygonTempPoints.push_back(convertedPoint);
				}
				projectedPoint.x = point.y;
				projectedPoint.y = point.z;
				break;

			case bottomFace:
			case topFace:
				for (int i = 0; i < face.edgeIndices.size(); i++) {
					Vector3D originalPoint = m_vertices[face.edgeIndices[i].first];
					//Projecting onto the XZ plane
					Vector2D convertedPoint(originalPoint.x, originalPoint.z);
					polygonTempPoints.push_back(convertedPoint);
				}
				projectedPoint.x = point.x;
				projectedPoint.y = point.z;
				break;

			case backFace:
			case frontFace:
				for (int i = 0; i < face.edgeIndices.size(); i++) {
					Vector3D originalPoint = m_vertices[face.edgeIndices[i].first];
					//Projecting onto the XY plane
					Vector2D convertedPoint(originalPoint.x, originalPoint.y);
					polygonTempPoints.push_back(convertedPoint);
				}
				projectedPoint.x = point.x;
				projectedPoint.y = point.y;
				break;

			default:
				throw exception("NonManifoldMesh: invalid call to isInsideFace");
				break;
			}

			return isInsidePolygon(projectedPoint, polygonTempPoints);
		}

		void NonManifoldMesh3D::removeFaceFromEdgeMap(unsigned int i, unsigned int j, int faceID) {
			for (int index = 0; index < m_edgeMap[edgeHash(i, j)].size();) {
				if (m_edgeMap[edgeHash(i, j)][index]->ID == faceID) {
					m_edgeMap[edgeHash(i, j)].erase(m_edgeMap[edgeHash(i, j)].begin() + index);
				}
				else {
					index++;
				}
			}
		}
	}
}
