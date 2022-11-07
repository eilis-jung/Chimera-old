#include "CutCells/CutCellsVelocities2D.h"
#include "ChimeraEigenWrapper.h"

namespace Chimera {
	namespace CutCells {
		#pragma region Constructors
		CutCellsVelocities2D::CutCellsVelocities2D(CutCells2D<Vector2> *pCutCells, solidBoundaryType_t solidBoundaryType) 
			: CutCellsVelocities(pCutCells, solidBoundaryType) {
			m_pCutCells2D = pCutCells;
		}
		#pragma endregion 

		#pragma region Functionalities

		void CutCellsVelocities2D::update(const Array2D<Vector2> &nodalVelocities, bool useAuxiliaryVelocities) {
			
			if (m_solidBoundary == Solid_NoSlip) {
				processNoSlipVelocities(nodalVelocities, useAuxiliaryVelocities);
			}
			else if (m_solidBoundary == Solid_FreeSlip) {
				processNoSlipVelocities(nodalVelocities, useAuxiliaryVelocities);
				processFreeSlipVelocities(nodalVelocities, useAuxiliaryVelocities);
			}
		}

		void CutCellsVelocities2D::projectMixedNodeVelocities() {
			for (int i = 0; i < m_pMesh->getVertices().size(); i++) {
				auto edges = m_pMesh->getVertices()[i]->getConnectedEdges();
				MatrixNxN leastSquaresSys(edges.size(), 2);
				//Setup least squares matrix system 
				/*for (int j = 0; j < edges.size(); j++) {
					Scalar edgeFlux = edges[j].getVelocity()
				}*/
			}
			
			/*for (int i = 0; i < m_nodeVelocities.size(); i++) {
				for (int j = 0; j < m_nodeVelocities[i].size(); j++) {
					int nextJ = roundClamp<int>(j + 1, 0, m_nodeVelocities[i].size());
					int prevJ = roundClamp<int>(j - 1, 0, m_nodeVelocities[i].size());
					auto cutCell = m_pCutCells2D->getSpecialCell(i);

					CutEdge<Vector2> prevEdge = *cutCell.m_cutEdges[prevJ];
					CutEdge<Vector2> currEdge = *cutCell.m_cutEdges[j];

					if (cutCell.m_cutEdgesLocations[j] == geometryEdge || cutCell.m_cutEdgesLocations[prevJ] == geometryEdge) {
						Vector2 normal = cutCell.m_cutEdgesLocations[j] == geometryEdge ? cutCell.m_cutEdges[j]->getNormal() :
																							cutCell.m_cutEdges[prevJ]->getNormal();
						Vector2 velocity = cutCell.m_cutEdgesLocations[j] == geometryEdge ? cutCell.m_cutEdges[j]->getVelocity() :
																							cutCell.m_cutEdges[prevJ]->getVelocity();
						m_nodeVelocities[i][j] = projectFreeSlipMixedNodes(m_nodeVelocities[i][j], velocity, normal);
					}
				}
			}*/
		}

		#pragma endregion 

		#pragma region PrivateFunctionalities
		Vector2 CutCellsVelocities2D::interpolateMixedNodeVelocity(Scalar fluidFlux, Scalar thinObjectFlux, const Vector2 &faceNormal, const Vector2 &thinObjectNormal) {
			Matrix2x2 normalMatrix;
			normalMatrix.column[0].x = thinObjectNormal.x;
			normalMatrix.column[1].x = thinObjectNormal.y;
			normalMatrix.column[0].y = faceNormal.x;
			normalMatrix.column[1].y = faceNormal.y;
			normalMatrix.invert();
			return normalMatrix*Vector2(thinObjectFlux, fluidFlux);
		}

		Vector2 CutCellsVelocities2D::projectFreeSlipMixedNodes(const Vector2 &nodalVelocity, const Vector2 &faceVelocity, const Vector2 &faceNormal) {
			Vector2 vRelative = nodalVelocity - faceVelocity;
			Scalar normalProj = vRelative.dot(faceNormal);
			Vector2 projectedVelocity = nodalVelocity - faceNormal*normalProj;
			return projectedVelocity;
		}

		Vector2 CutCellsVelocities2D::getNextMixedNodeVelocity(const CutFace<Vector2> &cutFace, int ithThinObjectPoint) {
			/*int i = 0;
			if (cutFace.m_cutEdgesLocations[ithThinObjectPoint] == geometryEdge) {
				for (i = ithThinObjectPoint; i < cutFace.m_cutEdges.size(); i++) {
					if (cutFace.m_cutEdgesLocations[i] != geometryEdge)
						break;
				}
			}
			else {
				for (i = ithThinObjectPoint; i < cutFace.m_cutEdges.size(); i++) {
					if (cutFace.m_cutEdgesLocations[i] == geometryEdge)
						break;
				}
			}
			if (i == cutFace.m_cutEdges.size()) {
				i = 0;
			}
			int prevI = roundClamp<int>(i - 1, 0, cutFace.m_cutEdges.size());
			Vector2 prevNormal = cutFace.getEdgeNormal(prevI);
			Vector2 currNormal = cutFace.getEdgeNormal(i);
			return interpolateMixedNodeVelocity(currNormal.dot(cutFace.m_cutEdges[i]->getVelocity()), prevNormal.dot(cutFace.m_cutEdges[prevI]->getVelocity()), currNormal, prevNormal);*/
			return Vector2(0, 0);
		}


		void CutCellsVelocities2D::processNoSlipVelocities(const Array2D<Vector2> &nodalVelocities, bool useAuxiliaryVelocities) {
			if (m_pCutCells2D) {
				Scalar dx = m_pCutCells2D->getGridSpacing();
				for (int i = 0; i < m_pCutCells2D->getNumberCutCells(); i++) {
					auto cutCell = m_pCutCells2D->getCutCell(i);
					Vector2 nodalVelocity;
					for (int j = 0; j < cutCell.getHalfEdges().size(); j++) {
						int prevIndex = roundClamp<int>(j - 1, 0, cutCell.getHalfEdges().size());
						auto currVertex = cutCell.getHalfEdges()[j]->getVertices().first;
						if (currVertex->getVertexType() == gridVertex) {
							Vector2 tempPoint = currVertex->getPosition() / dx;
							if (useAuxiliaryVelocities)
								currVertex->setAuxiliaryVelocity(nodalVelocities(floor(tempPoint.x), floor(tempPoint.y)));
							else
								currVertex->setVelocity(nodalVelocities(floor(tempPoint.x), floor(tempPoint.y)));
						}
						/*else {
						if (useAuxiliaryVelocities)
						currVertex->setAuxiliaryVelocity(Vector2(0, 0));
						else
						currVertex->setVelocity(Vector2(0, 0));
						}*/
					}
				}
			}
			
		}
		void CutCellsVelocities2D::processFreeSlipVelocities(const Array2D<Vector2> &nodalVelocities, bool useAuxiliaryVelocities) {
			/** Using has updated boolean tag to update each mixed node velocity only once. Initially all nodes have to be updated */
			for (int i = 0; i < m_pCutCells2D->getVertices().size(); i++) {
				m_pCutCells2D->getVertices()[i]->setUpdated(false);
			}


			/** First we find the velocities at mixed nodes */
			for (int i = 0; i < m_pCutCells2D->getVertices().size(); i++) {
				auto connectedEdges = m_pCutCells2D->getVertices()[i]->getConnectedEdges();

			}
			for (int i = 0; i < m_pCutCells2D->getNumberCutCells(); i++) {
				auto cutCell = m_pCutCells2D->getCutCell(i);
				for (int j = 0; j < cutCell.getHalfEdges().size(); j++) {
					auto halfEdge = cutCell.getHalfEdges()[j];
					if (halfEdge->getVertices().first->getVertexType() == edgeVertex && !halfEdge->getVertices().first->hasUpdated()) { //We have to solve a system for each edgeVertex
						vector<uint> edgesMatIndices;
						auto connectedEdges = halfEdge->getVertices().first->getConnectedEdges();
						//Getting the edges that are connected to this vertex but belong to this cell - shortcut in 2-D to check if they belong to the same side of the mesh 
						//Also we have consider in edges that are geometry
						for (int k = 0; k < connectedEdges.size(); k++) {
							if (connectedEdges[k]->getType() == Meshes::geometricEdge || cutCell.hasEdge(connectedEdges[k])) {
								edgesMatIndices.push_back(k);
							}
						}

						if (edgesMatIndices.size() != connectedEdges.size()) {
							throw(exception("CutCellsVelocities2D: Invalid adjacency configuration"));
						}
						MatrixNxN leastSquaresSys(edgesMatIndices.size(), 2);
						vector<Scalar> rhs(edgesMatIndices.size(), 1);
						for (int k = 0; k < edgesMatIndices.size(); k++) {
							auto currEdge = connectedEdges[edgesMatIndices[k]];
							auto normal = currEdge->getHalfEdges().first->getNormal();
							leastSquaresSys(k, 0) = normal.x;
							leastSquaresSys(k, 1) = normal.y;
							if(useAuxiliaryVelocities)
								rhs[k] = normal.dot(currEdge->getAuxiliaryVelocity());
							else
								rhs[k] = normal.dot(currEdge->getVelocity());
						}

						EigenWrapper::LeastSquaresSolver<Scalar> leastSquaresSolver(&leastSquaresSys);
						vector<Scalar> result = leastSquaresSolver.solve(rhs);
						Vector2 velocity(result[0], result[1]);
						if (useAuxiliaryVelocities)
							halfEdge->getVertices().first->setAuxiliaryVelocity(velocity);
						else
							halfEdge->getVertices().first->setVelocity(velocity);
						halfEdge->getVertices().first->setUpdated(true);
					}
				}
			}
			
			for (int i = 0; i < m_pCutCells2D->getNumberCutCells(); i++) {
				auto cutCell = m_pCutCells2D->getCutCell(i);
				for (int j = 0; j < cutCell.getHalfEdges().size(); j++) {
					vector<HalfEdge<Vector2> *> &halfedges = cutCell.getHalfEdges();
					auto halfEdge = halfedges[j];
					if (halfEdge->getVertices().first->getVertexType() == edgeVertex && halfEdge->getVertices().second->getVertexType() == geometryVertex) { 
						Vertex<Vector2> *pFirstMixedNode = halfEdge->getVertices().first, *pSecondMixedNode = nullptr;
						//Run through the halfedges until next mix node is found
						//Lets use these iterations to calculate the total line size that will be used on interpolation
						DoubleScalar totalLineSize = 0;
						int k; 
						for (k = j + 1; k < halfedges.size() && pSecondMixedNode == nullptr; k++) {
							totalLineSize += halfedges[k]->getEdge()->getLength();
							if (halfedges[k]->getVertices().second->getVertexType() == edgeVertex) {
								pSecondMixedNode = halfedges[k]->getVertices().second;
							}
						}
						if (pSecondMixedNode == nullptr) {
							throw ("Invalid mixed node configuration on Free-slip velocity fields");
						}

						DoubleScalar accLineSize = 0;
						for (k = j + 1; k < halfedges.size(); k++) {
							accLineSize += halfedges[k]->getEdge()->getLength();
							Vertex<Vector2> *pCurrVertex = halfedges[k]->getVertices().first;
							DoubleScalar alpha = accLineSize / totalLineSize;
							if (useAuxiliaryVelocities)
								pCurrVertex->setAuxiliaryVelocity(pFirstMixedNode->getAuxiliaryVelocity()*(1 - alpha) + pSecondMixedNode->getAuxiliaryVelocity()*alpha);
							else
								pCurrVertex->setVelocity(pFirstMixedNode->getVelocity()*(1 - alpha) + pSecondMixedNode->getVelocity()*alpha);
							if (halfedges[k]->getVertices().second->getVertexType() == edgeVertex) {
								break;
							}
						}
						j = k;
					}
				}
			}

			for (int i = 0; i < m_pCutCells2D->getNumberCutCells(); i++) {
				auto cutCell = m_pCutCells2D->getCutCell(i);
				for (int j = 0; j < cutCell.getHalfEdges().size(); j++) {
					auto halfEdge = cutCell.getHalfEdges()[j];
					if (halfEdge->getVertices().first->getVertexType() == geometryVertex || halfEdge->getVertices().first->getVertexType() == edgeVertex) {
						Scalar normalProj;
						if(useAuxiliaryVelocities)
							normalProj = halfEdge->getVertices().first->getNormal().dot(halfEdge->getVertices().first->getAuxiliaryVelocity());
						else
							normalProj = halfEdge->getVertices().first->getNormal().dot(halfEdge->getVertices().first->getVelocity());
						
						Vector2 projectedVelocity;
						if(useAuxiliaryVelocities)
							projectedVelocity = halfEdge->getVertices().first->getAuxiliaryVelocity() - halfEdge->getVertices().first->getNormal()*normalProj;
						else
							projectedVelocity = halfEdge->getVertices().first->getVelocity() - halfEdge->getVertices().first->getNormal()*normalProj;

						if (useAuxiliaryVelocities)
							halfEdge->getVertices().first->setAuxiliaryVelocity(projectedVelocity);
						else
							halfEdge->getVertices().first->setVelocity(projectedVelocity);
					}
					
				}
			}
			

			//Scalar dx = m_pCutCells2D->getGridSpacing();
			//for (int i = 0; i < m_pCutCells2D->getNumberOfCells(); i++) {
			//	const CutFace<Vector2> &cutCell = m_pCutCells2D->getSpecialCell(i);
			//	vector<edgeLocation_t> edgeLocations = cutCell.m_cutEdgesLocations;
			//	Vector2 nodalVelocity;
			//	for (int j = 0; j < cutCell.m_cutEdges.size(); j++) {
			//		int prevIndex = roundClamp<int>(j - 1, 0, cutCell.m_cutEdges.size());
			//		CutEdge<Vector2> prevEdge = *cutCell.m_cutEdges[prevIndex];
			//		CutEdge<Vector2> currEdge = *cutCell.m_cutEdges[j];
			//		
			//		Scalar fluidFlux, geometryFlux;
			//		if (edgeLocations[j] != geometryEdge && edgeLocations[prevIndex] != geometryEdge) { //this is on a grid point
			//			Vector2 tempPoint = currEdge.getInitialPoint(edgeLocations[j]) / dx;
			//			m_nodeVelocities[i][j] = nodalVelocities(floor(tempPoint.x), floor(tempPoint.y));
			//		} else if (edgeLocations[j] != geometryEdge && edgeLocations[prevIndex] == geometryEdge) {
			//			fluidFlux = currEdge.getVelocity().dot(currEdge.getNormal());
			//			geometryFlux = prevEdge.getVelocity().dot(prevEdge.getNormal());
			//			nodalVelocity = interpolateMixedNodeVelocity(fluidFlux, geometryFlux, currEdge.getNormal(), prevEdge.getNormal());
			//			nodalVelocity = projectFreeSlipMixedNodes(nodalVelocity, prevEdge.getVelocity(), prevEdge.getNormal());
			//			m_nodeVelocities[i][j] = nodalVelocity; // .push_back(nodalVelocity);
			//		}
			//		else if (edgeLocations[prevIndex] != geometryEdge && edgeLocations[j] == geometryEdge) {
			//			fluidFlux = prevEdge.getVelocity().dot(prevEdge.getNormal());
			//			geometryFlux = currEdge.getVelocity().dot(currEdge.getNormal());
			//			nodalVelocity = interpolateMixedNodeVelocity(fluidFlux, geometryFlux, prevEdge.getNormal(), currEdge.getNormal());
			//			nodalVelocity = projectFreeSlipMixedNodes(nodalVelocity, currEdge.getVelocity(), currEdge.getNormal());
			//			Vector2 auxVelocity = getNextMixedNodeVelocity(cutCell, j);
			//			int auxEdgeIndex = roundClamp<int>(cutCell.getNextFluidEdgeIndex(j) - 1, 0, cutCell.m_cutEdges.size());
			//			auxVelocity = projectFreeSlipMixedNodes(auxVelocity, cutCell.m_cutEdges[auxEdgeIndex]->getVelocity(), cutCell.m_cutEdges[auxEdgeIndex]->getNormal());
			//			Vector2 initialThinObjectPoint = currEdge.getInitialPoint(edgeLocations[j]);
			//			Vector2 finalThinObjectPoint = cutCell.getNextMixedNodePoint(j);
			//			m_nodeVelocities[i][j] = nodalVelocity;
			//			if (j < cutCell.m_cutEdges.size() - 1 && cutCell.m_cutEdgesLocations[j + 1] == geometryEdge) {
			//				j++;
			//			}
			//			Scalar initialFinalLenght = (finalThinObjectPoint - initialThinObjectPoint).length();
			//			if (initialFinalLenght < 1e-7) {
			//				while (j < cutCell.m_cutEdges.size() && cutCell.m_cutEdgesLocations[j] == geometryEdge) {
			//					m_nodeVelocities[i][j] = (nodalVelocity + auxVelocity)*0.5;
			//					if (j < cutCell.m_cutEdges.size() - 1 && cutCell.m_cutEdgesLocations[j + 1] == geometryEdge) {
			//						j++;
			//					}
			//					else {
			//						break;
			//					}
			//				}
			//			}
			//			else {
			//				Scalar totalAlfa = 0;
			//				Scalar totalEdgeLenght = cutCell.getTotalGeometryEdgeSize(j - 1);
			//				if (totalEdgeLenght == 0)
			//					totalEdgeLenght = cutCell.getTotalGeometryEdgeSize(j);
			//				while (j < cutCell.m_cutEdges.size() && cutCell.m_cutEdgesLocations[j] == geometryEdge) {
			//					Vector2 initialEdgeLocation = cutCell.m_cutEdges[j]->getInitialPoint(edgeLocations[j]);
			//					totalAlfa += (cutCell.m_cutEdges[j]->m_finalPoint - cutCell.m_cutEdges[j]->m_initialPoint).length();
			//					Scalar alfa = totalAlfa / totalEdgeLenght;
			//					Vector2 nodalNormal;
			//					Scalar freeSlipParam = 1.0f;
			//					int prevJ = roundClamp<int>(j - 1, 0, cutCell.m_cutEdges.size());
			//					if (cutCell.m_cutEdgesLocations[prevJ] == geometryEdge) {
			//						nodalNormal = (cutCell.m_cutEdges[j]->getNormal() + cutCell.m_cutEdges[prevJ]->getNormal())*0.5;
			//						nodalNormal.normalize();
			//						freeSlipParam = clamp<Scalar>(cutCell.m_cutEdges[j]->getNormal().dot(cutCell.m_cutEdges[prevJ]->getNormal()), 0.0f, 1.0f);
			//					}
			//					else {
			//						nodalNormal = cutCell.m_cutEdges[j]->getNormal();
			//						freeSlipParam = 1;
			//					}
			//					nodalVelocity = nodalVelocity*(1 - alfa) + auxVelocity*alfa;
			//					nodalVelocity = nodalVelocity*freeSlipParam + cutCell.m_cutEdges[j]->getVelocity()*(1 - freeSlipParam);
			//					nodalVelocity = projectFreeSlipMixedNodes(nodalVelocity, cutCell.m_cutEdges[j]->getVelocity(), nodalNormal);
			//					cutCell.m_cutEdges[j]->setVelocity(nodalVelocity);
			//					m_nodeVelocities[i][j] = nodalVelocity;
			//					if (j < cutCell.m_cutEdges.size() - 1 && cutCell.m_cutEdgesLocations[j + 1] == geometryEdge) {
			//						j++;
			//					}
			//					else {
			//						break;
			//					}
			//				}
			//			}
			//		}
			//		else { //edgeLocations[prevIndex] == geometryEdge && edgeLocations[j] == geometryEdge
			//			nodalVelocity = (prevEdge.getVelocity() + currEdge.getVelocity())*0.5;
			//			m_nodeVelocities[i][j] = nodalVelocity;
			//		}
			//	}
			//}
		}
		#pragma endregion
	}
}