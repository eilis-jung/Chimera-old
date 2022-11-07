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

#ifndef __CHIMERA_CUT_EDGE__
#define __CHIMERA_CUT_EDGE__
#pragma once

#include "ChimeraCore.h"

namespace Chimera {

	using namespace Core;

	namespace CutCells {

		template <class VectorT>
		class CutEdge {
		public:
			
		
			/** Edge initial and final points */
			VectorT m_initialPoint;
			VectorT m_finalPoint;

		public:
				//TO-DO Create access functions and protect class members
				#pragma region ClassMembers
			
				/** Unique edge identifier */
				int m_ID;

				/** An edge can have 2 neighbors */
				int m_edgeNeighbors[2];

				Scalar m_lengthFraction; //the length of this edge relative to regular grid 

				/** Still useful for identifying edges by accessing left and bottom edge arrays. */
				edgeLocation_t m_edgeLocation;

				/** Flux and velocity*/
				VectorT m_velocity;
				VectorT m_intermediateVelocity;

				/** Used for PIC/FLIP advection */
				Scalar m_velocityWeight;

				/** Used for PIC/FLIP advection for merging node contributions */
				bool m_mergedNode;

				/** Edge normal: only makes sense in 2-D. */
				VectorT m_normal;

				/** Edge centroid */
				VectorT m_centroid;

				/** The edge can be invalid if it its too small. Therefore the flow solver must disregard this edge and treat 
				 ** the flow as a solid no slip condition.*/ 
				bool m_isValid;

				/** If this edge is associated with thinObject geometry, which of the thinObjects ID it belongs to?*/
				int m_thinObjectID;

				#pragma endregion

		public:
				#pragma region Constructors
				//Initializes a thinObject-edge or fluid-edge
				CutEdge(int ID, const VectorT &initialPoint, const VectorT &finalPoint, Scalar dx, edgeLocation_t edgeLocation, int thinObjectID = -1);
				#pragma endregion

				#pragma region AcessFunctions
				int getID() const {
					return m_ID;
				}
				void getNeighbors(int &n1, int &n2) const {
					n1 = m_edgeNeighbors[0];
					n2 = m_edgeNeighbors[1];
				}
				void setNeighbors(int n1, int n2) {
					m_edgeNeighbors[0] = n1;
					m_edgeNeighbors[1] = n2;
				}
				Scalar getLengthFraction() const {
					return m_lengthFraction;
				}
				const VectorT & getInitialPoint(edgeLocation_t edgeLocation) const {
					switch(edgeLocation) {
						case bottomEdge:
							if(m_initialPoint.x > m_finalPoint.x) {
								return m_finalPoint;
							}
							return m_initialPoint;
						break;
						case topEdge:
							if(m_initialPoint.x < m_finalPoint.x) {
								return m_finalPoint;
							}
							return m_initialPoint;
						break;
						case rightEdge:
							if(m_initialPoint.y > m_finalPoint.y) {
								return m_finalPoint;
							}
							return m_initialPoint;
						break;

						case leftEdge:
							if(m_finalPoint.y > m_initialPoint.y) {
								return m_finalPoint;
							}
							return m_initialPoint;
						break;
					}
					return m_initialPoint;
				}
				const VectorT & getFinalPoint(edgeLocation_t edgeLocation) const {
					switch(edgeLocation) {
						case bottomEdge:
							if(m_initialPoint.x > m_finalPoint.x) {
								return m_initialPoint;
							}
							return m_finalPoint;
						break;
						case topEdge:
							if(m_initialPoint.x < m_finalPoint.x) {
								return m_initialPoint;
							}
							return m_finalPoint;
							break;
						case rightEdge:
							if(m_initialPoint.y > m_finalPoint.y) {
								return m_initialPoint;
							}
							return m_finalPoint;
						break;

						case leftEdge:
							if(m_finalPoint.y > m_initialPoint.y) {
								return m_initialPoint;
							}
							return m_finalPoint;
						break;
					}
					return m_finalPoint;
				}
				const VectorT & getVelocity() const {
					return m_velocity;
				}
				void setVelocity(const VectorT &velocity) {
					m_velocity = velocity;
				}
				void addVelocity(const VectorT &velocity) {
					m_velocity += velocity;
				}
				const VectorT & getNormal() const {
					return m_normal;
				}
				const void setNormal(const VectorT & normal) {
					m_normal = normal;
				}
				const VectorT & getCentroid() const {
					return m_centroid;
				}
				const VectorT getIntermediaryVelocity() const {
					return m_intermediateVelocity;
				}
				void setIntermediaryVelocity(const VectorT & velocity) {
					m_intermediateVelocity = velocity;
				}
				const Scalar getWeight() const {
					return m_velocityWeight;
				}
				void addWeight(Scalar weight) {
					m_velocityWeight += weight;
				}
				void setWeight(Scalar weight) {
					m_velocityWeight = weight;
				}
				int getThinObjectID() const {
					return m_thinObjectID;
				}
				edgeLocation_t getLocation() const {
					return m_edgeLocation;
				}
				#pragma endregion
		};
	}
}
#endif