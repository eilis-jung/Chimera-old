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

#ifndef __CHIMERA_CUT_FACE__
#define __CHIMERA_CUT_FACE__
#pragma once

#include "ChimeraCore.h"
#include "CutCells/CutEdge.h"

namespace Chimera {
	using namespace Core;
	
	namespace Rendering {
		class PolygonSurface;
	}

	namespace CutCells {
		
		#pragma region InternalStructures
		#pragma endregion

		template <class VectorT>
		class CutFace {

		public:
			//TO-DO Create access functions and protect class members
			#pragma region ClassMembers
			/** Cell ID: Usually it is the position on the specialCells vector; however this can be used in order to 
			 ** re-organize special cells*/
			int m_ID;

			/** CutFaces point list: facilitator, since uses duplicated cutEdges information*/
			vector<VectorT> m_points;

			/** Edges references by cellPoints. Each cell point is connected to two special edges. Edges are initialized
			 ** in a counterclockwise fashion. */
			vector<CutEdge<VectorT> *> m_cutEdges;

			/** Edges locations. These can't be stored directly on edges because we will have ambiguity when two faces share
			 ** a same edge pointer. So each CutFace has to store edge locations separately.*/
			vector<edgeLocation_t> m_cutEdgesLocations;

			/** Cell centroid */
			VectorT m_centroid;

			VectorT m_interiorPoint;

			/** If the cell has an open ended face this cell is openEnded */
			bool m_openEnded;

			/** Regular grid index */
			dimensions_t m_regularGridIndex;


			/************************************************************************/
			/* 3D IMPLEMENTATION                                                    */
			/************************************************************************/
			/**  A face can have 2 neighbors. */
			int m_faceNeighbors[2];

			/** Face location relative to the CutVoxel*/
			faceLocation_t m_faceLocation;

			/** Area fraction relative to the area of the regular cell of the grid */
			Scalar m_areaFraction;

			/** Normals and velocities */
			VectorT m_normal;
			VectorT m_velocity;

			/** If this face is associated with thinObject geometry, this is the geometry polygon surface pointer*/
			Rendering::PolygonSurface *m_pPolygonSurface;

			/** If this face has a hole (disconnected region) inside of it*/
			bool m_discontinuousEdges;

			/** If this face is a geometry face, store the original indices in the PolygonSurface that it belongs */
			vector<unsigned int> m_originalPolyIndices;
			#pragma endregion
		
		public:
			#pragma region Constructors
			
			//Usually used in 2-D version
			CutFace(int ID) {
				m_ID = ID;
				m_faceLocation = geometryFace; //It doesn't matter in 2-D;
				m_openEnded = false;
				m_faceNeighbors[0] = m_faceNeighbors[1] = -1;
				m_discontinuousEdges = false;
			}

			//Usually used in 3-D version, where the ID is not important and the face location is
			CutFace(faceLocation_t faceLocation) {
				m_ID = -1;
				m_faceLocation = faceLocation;
				m_openEnded = false;
				m_faceNeighbors[0] = m_faceNeighbors[1] = -1;
				m_discontinuousEdges = false;
			}
			#pragma endregion


			#pragma	region Functionalities
			void updateCentroid();
			#pragma endregion Functionalities

			#pragma region AccessFunctions
			int getID() const {
				return m_ID;
			}
			
			const vector<Vector2> & getPoints() const {
				return m_points;
			}

			void getNeighbors(int &n1, int &n2) const {
				n1 = m_faceNeighbors[0];
				n2 = m_faceNeighbors[1];
			}
			void setNeighbors(int n1, int n2) {
				m_faceNeighbors[0] = n1;
				m_faceNeighbors[1] = n2;
			}

			VectorT getCentroid() const {
				return m_centroid;
			}

			Scalar getAreaFraction() const {
				return m_areaFraction;
			}
			bool isOpenEnded() const {
				return m_openEnded;
			}
			Vector2 getEdgeNormal(int ithEdge) const {
				switch (m_cutEdgesLocations[ithEdge]) {
				case rightEdge:
					return Vector2(1, 0);
					break;
				case bottomEdge:
					return Vector2(0, -1);
					break;
				case leftEdge:
					return Vector2(-1, 0);
					break;
				case topEdge:
					return Vector2(0, 1);
					break;
				default:
					return -(getEdgeFinalPoint(ithEdge) 
							- getEdgeInitialPoint(ithEdge)).perpendicular().normalized();
					break;
				}
			}
			VectorT getPreviousMixedNodePoint(int ithThinObjectPoint) const {
				int i = 0;
				VectorT lastPoint;

				if(m_cutEdges.size() == 0)
					return lastPoint;

				if(m_cutEdgesLocations[ithThinObjectPoint] == geometryEdge) {
					for(i = ithThinObjectPoint; i >= 0; i--) {
						if(m_cutEdgesLocations[i] != geometryEdge)
							return m_cutEdges[i]->getFinalPoint(m_cutEdgesLocations[i]);
					}
					return m_cutEdges.back()->getFinalPoint(m_cutEdgesLocations.back());
				} else {
					for(i = ithThinObjectPoint; i >= 0; i--) {
						if(m_cutEdgesLocations[i] == geometryEdge)
							return m_cutEdges[i]->getFinalPoint(m_cutEdgesLocations[i]);
					}
					return m_cutEdges.back()->getFinalPoint(m_cutEdgesLocations.back());
				}
			}
			VectorT getNextMixedNodePoint(int ithThinObjectPoint) const {
				int i = 0;
				VectorT lastPoint;

				if(m_cutEdges.size() == 0)
					return lastPoint;

				if(m_cutEdgesLocations[ithThinObjectPoint] == geometryEdge) {
					for(i = ithThinObjectPoint; i <  m_cutEdges.size(); i++) {
						if(m_cutEdgesLocations[i] != geometryEdge)
							return m_cutEdges[i]->getInitialPoint(m_cutEdgesLocations[i]);
					}
					return m_cutEdges.front()->getInitialPoint(m_cutEdgesLocations.back());
				} else {
					for(i = ithThinObjectPoint; i <  m_cutEdges.size(); i++) {
						if(m_cutEdgesLocations[i] == geometryEdge)
							return m_cutEdges[i]->getInitialPoint(m_cutEdgesLocations[i]);
					}
					return m_cutEdges.front()->getInitialPoint(m_cutEdgesLocations.back());
				}

			}

			VectorT getPreviousMixedNodeVelocity(int ithThinObjectPoint) const {
				int i = 0;
				if(m_cutEdges[ithThinObjectPoint] == geometryEdge) {
					for(i = ithThinObjectPoint; i >= 0; i--) {
						if(m_cutEdgesLocations[i] != geometryEdge)
							break;
					}
				} else {
					for(i = ithThinObjectPoint; i >= 0; i--) {
						if(m_cutEdgesLocations[i] == geometryEdge)
							break;
					}
				}
				if(i == -1) {
					i = m_cutEdges.size() - 1;
				}

			}

		
			int getNextFluidEdgeIndex(int currEdge) const {
				int i = 0;
				if (m_cutEdgesLocations[currEdge] == geometryEdge) {
					for (i = currEdge; i < m_cutEdges.size(); i++) {
						if (m_cutEdgesLocations[i] != geometryEdge)
							return i;
					}
				}
				else {
					for (i = currEdge; i < m_cutEdges.size(); i++) {
						if (m_cutEdgesLocations[i] == geometryEdge)
							return i;
					}
				}
				if (i == m_cutEdges.size()) {
					return 0;
				}
				return -1;
			}


			CutEdge<VectorT> * getPreviousFluidEdge(int currEdge) const {
				for(int i = currEdge; i >= 0; i--) {
					if(m_cutEdgesLocations[i] != geometryEdge)
						return m_cutEdges[i];
				}
				return m_cutEdges.back();
			}
			CutEdge<VectorT> * getNextFluidEdge(int currEdge) const {
				for(int i = currEdge; i <  m_cutEdges.size(); i++) {
					if(m_cutEdgesLocations[i] != geometryEdge)
						return m_cutEdges[i];
				}
				return m_cutEdges.front();
			}

			/** Gets the size of the geometry edges starting from the index argument */
			Scalar getTotalGeometryEdgeSize(int currEdge) const {
				Scalar totalGeometryEdgeSize = 0;
				for (int i = currEdge; i < m_cutEdges.size(); i++) {
					if (m_cutEdgesLocations[i] != geometryEdge)
						break;
					totalGeometryEdgeSize += (m_cutEdges[i]->m_finalPoint - m_cutEdges[i]->m_initialPoint).length();
				}
				return totalGeometryEdgeSize;
			}

			void setVelocityThinObjectEdges(const VectorT &velocity) {
				for(int i = 0; i < m_cutEdges.size(); i++) {
					if(m_cutEdgesLocations[i] == Edge<VectorT>::geometry) {
						m_cutEdges[i].setVelocity(velocity);
					}
				}
			}

			CutEdge<Vector2> * getMixedNodeByInitialPoint(const VectorT &point) {
				for (int i = 0; i < m_cutEdges.size(); i++) {
					int prevIndex = roundClamp<int>(i - 1, 0, m_cutEdges.size());
					if (m_cutEdgesLocations[i] == geometryEdge && m_cutEdgesLocations[prevIndex] != geometryEdge) {
						if (getEdgeInitialPoint(i) == point)
							return m_cutEdges[i];
					}
					else if (m_cutEdgesLocations[i] != geometryEdge && m_cutEdgesLocations[prevIndex] == geometryEdge) {
						if (getEdgeInitialPoint(i) == point)
							return m_cutEdges[i];
					}
				}
				return NULL;
			}

			void insertCutEdge(const VectorT &initialPoint, const VectorT &finalPoint, Scalar dx, edgeLocation_t edgeLocation, int thinObjectID = -1) {
				CutEdge<VectorT> *pCurrEdge = new CutEdge<VectorT>(0, initialPoint, finalPoint, dx, edgeLocation, thinObjectID);
				m_cutEdges.push_back(pCurrEdge);
				m_cutEdgesLocations.push_back(edgeLocation);
				m_points.push_back(initialPoint);
			}

			VectorT getEdgeInitialPoint(int i) const {
				return m_cutEdges[i]->getInitialPoint(m_cutEdgesLocations[i]);
			}

			VectorT getEdgeFinalPoint(int i) const {
				return m_cutEdges[i]->getFinalPoint(m_cutEdgesLocations[i]);
			}

			bool crossedThroughGeometry(const Vector2 &p1, const Vector2 &p2) const {
				bool crossedThroughGeometry = false;
				for (int i = 0; i < m_cutEdges.size(); i++) {
					if (m_cutEdgesLocations[i] == geometryEdge) {
						if (DoLinesIntersect(p1, p2, getEdgeInitialPoint(i), getEdgeFinalPoint(i))) {
							crossedThroughGeometry = true;
							break;
						}
					}
				}
				return crossedThroughGeometry;
			}

			Vector2 crossedThroughGeometry(const Vector2 &p1, const Vector2 &p2, bool &crossed) const {
				crossed = false;
				for (int i = 0; i < m_cutEdges.size(); i++) {
					if (m_cutEdgesLocations[i] == geometryEdge) {
						Vector2 intersectionPoint;
						if (DoLinesIntersect(p1, p2, getEdgeInitialPoint(i), getEdgeFinalPoint(i), intersectionPoint, 1e-3)) {
							crossed = true;
							return intersectionPoint;
						}
					}
				}
				return Vector2(0, 0);
			}

			int findEdgeIndex(const VectorT &edgeInitialPoint, const VectorT &edgeFinalPoint) {
				for (int i = 0; i < m_cutEdges.size(); i++) {
					if (((m_cutEdges[i]->m_initialPoint - edgeInitialPoint).length() < singlePrecisionThreshold && 
						(m_cutEdges[i]->m_finalPoint - edgeFinalPoint).length() < singlePrecisionThreshold) ||
						((m_cutEdges[i]->m_initialPoint - edgeFinalPoint).length() < singlePrecisionThreshold &&
						(m_cutEdges[i]->m_finalPoint - edgeInitialPoint).length() < singlePrecisionThreshold)) {
						return i;
					}
				}
				return -1;
			}

			Scalar getDistanceToBoundary(const VectorT &v) {
				Scalar minDistance = FLT_MAX;
				for (int i = 0; i < m_cutEdges.size(); i++) {
					if (m_cutEdgesLocations[i] == geometryEdge) {
						int nextI = roundClamp(i + 1, 0, (int) m_cutEdges.size());
						Scalar currDistance = distanceToLineSegment(v, getEdgeInitialPoint(i), getEdgeFinalPoint(nextI));
						if (currDistance < minDistance) {
							minDistance = currDistance;
						}
					}
				}
				return minDistance;
			}

			//Interpolates velocity on points that are exactly on cut face edges
			bool intersectsGeometryEdges(const VectorT &p1, const VectorT &p2) const;
			#pragma endregion
			

			
		};
	}
}

#endif
