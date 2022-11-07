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

#ifndef __CHIMERA_EDGE_H_
#define __CHIMERA_EDGE_H_

#include "ChimeraCore.h"
#include "MeshesCoreDefs.h"
#include "Mesh/Vertex.h"

namespace Chimera {
	namespace Meshes {

		/** Edge Class Forward Declaration */
		template <class VectorType>
		class Edge;

		/** Halfedge structure: does not stores centroids or vertices relative to edges. It only indicates
		* direction by the use of vertices indices. If one wants to access concrete attributes, it should do it
		* through Edge class. */
		template <class VectorType>
		class HalfEdge {
		public:
			#pragma region Constructors
			HalfEdge(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2, Edge<VectorType> *pEdge) {
				m_vertices.first = pV1;
				m_vertices.second = pV2;
				m_normal = -(pV2->getPosition() - pV1->getPosition()).perpendicular().normalized();
				m_pEdge = pEdge;
				m_ID = m_currID++;
			}

			HalfEdge(const pair<Vertex<VectorType> *, Vertex<VectorType> *> &pointIndices, Edge<VectorType> *pEdge) :
				m_vertices(pointIndices) {
				m_normal = -(m_vertices.second->getPosition() - m_vertices.first->getPosition()).perpendicular().normalized();
				m_pEdge = pEdge;
				m_ID = m_currID++;
			}
			
			static void resetIDs() {
				m_currID = 0;
			}

			#pragma endregion

			#pragma region AccessFunctions
			uint getID() const {
				return m_ID;
			}
			Edge<VectorType> * getEdge() {
				return m_pEdge;
			}

			const pair<Vertex<VectorType> *, Vertex<VectorType> *> & getVertices() const {
				return m_vertices;
			}

			pair<Vertex<VectorType> *, Vertex<VectorType> *> & getVertices() {
				return m_vertices;
			}

			void setFirstVertex(Vertex<VectorType> *pVertex) {
				m_vertices.first = pVertex;
			}

			void setSecondVertex(Vertex<VectorType> *pVertex) {
				m_vertices.second = pVertex;
			}

			void setLocation(halfEdgeLocation_t halfEdgeLocation) {
				m_location = halfEdgeLocation;
			}

			halfEdgeLocation_t getLocation() const {
				return m_location;
			}

			void setNormal(const VectorType &edgeNormal) {
				m_normal = edgeNormal;
			}

			const VectorType & getNormal() const {
				return m_normal;
			}
			#pragma endregion

			#pragma region Operators
			FORCE_INLINE bool operator==(const HalfEdge<VectorType> &rhs) const {
				return m_ID == rhs.getID();
			}
			#pragma endregion

		protected:
			pair<Vertex<VectorType> *, Vertex<VectorType> *> m_vertices;
			Edge<VectorType> *m_pEdge;

			VectorType m_normal;

			// This is an accelerator structure that facilitates 2-D cut-cells creation
			halfEdgeLocation_t m_location;

			/** ID vars */
			uint m_ID;
			static uint m_currID;
		};


		/** Face Class Forward Declaration */
		template <class VectorType>
		class Face;

		/** Face Class Forward Declaration */
		template <class VectorType>
		class HalfFace;

		/** Edge structure: stores centroids and attributes relative to edges. It might have the ownership of 
		  * two half-edges, which are created on the construction of this class. A edge could be created with 
		  * only one side, this is configured by the last parameter. In this way, the edge has only one true side. */
		template <class VectorType>
		class Edge {
		public:

			#pragma region Constructors
			Edge(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2, edgeType_t edgeType, bool borderEdge = false) {
				m_ID = m_currID++;
				m_vertices.first = pV1;
				m_vertices.second = pV2;
				m_edgeType = edgeType;
				m_borderEdge = borderEdge;
				m_halfEdges.first = new HalfEdge<VectorType>(pV1, pV2, this);
				m_visited = false;
				m_edgeFraction = 0.0f;

				m_vertices.first->addConnectedEdge(this);
				m_vertices.second->addConnectedEdge(this);

				if (borderEdge)
					m_halfEdges.second = nullptr;
				else
					m_halfEdges.second = new HalfEdge<VectorType>(pV2, pV1, this);

				m_centroid = (pV1->getPosition() + pV2->getPosition())*0.5;
				
			}

			//Edge(const pair<Vertex<VectorType> *, Vertex<VectorType> *> &vertexPair, edgeType_t edgeType, bool borderEdge = false)
			//	: m_vertices(vertexPair) {
			//	m_ID = m_currID++;
			//	m_edgeType = edgeType;
			//	m_borderEdge = borderEdge;
			//	m_halfEdges.first = new HalfEdge<VectorType>(vertexPair.first, vertexPair.second, this);
			//	m_visited = false;
			//	m_edgeFraction = 0.0f;

			//	//m_vertices.first->addConnectedEdge(this);
			//	//m_vertices.second->addConnectedEdge(this);

			//	if (borderEdge)
			//		m_halfEdges.second = nullptr;
			//	else
			//		m_halfEdges.second = new HalfEdge<VectorType>(vertexPair.second, vertexPair.first, this);

			//	m_centroid = (v1 + v2)*0.5;
			//	m_borderEdge = borderEdge;
			//}

			static void resetIDs() {
				m_currID = 0;
			}
			#pragma endregion

			#pragma region AccessFunctions
			const pair<HalfEdge<VectorType> *, HalfEdge<VectorType> *> & getHalfEdges() const {
				return m_halfEdges;
			}

			uint getID() const {
				return m_ID;
			}

			const VectorType & getCentroid() const {
				return m_centroid;
			}

			/** Returns the first vertex of the first halfedge */
			Vertex<VectorType> * getVertex1() const {
				return m_halfEdges.first->getVertices().first;
			}

			/** Returns the second vertex of the first halfedge */
			Vertex<VectorType> * getVertex2() const {
				return m_halfEdges.first->getVertices().second;
			}

			bool isVisited() const {
				return m_visited;
			}

			void setVisited(bool visited) {
				m_visited = visited;
			}

			edgeType_t getType() const {
				return m_edgeType;
			}

			DoubleScalar getLength() const {
				return (m_vertices.second->getPosition() - m_vertices.first->getPosition()).length();
			}

			void setVelocity(const VectorType &velocity) {
				m_velocity = velocity;
			}

			const VectorType & getVelocity() const {
				return m_velocity;
			}

			void setAuxiliaryVelocity(const VectorType &velocity) {
				m_auxiliaryVelocity = velocity;
			}

			const VectorType & getAuxiliaryVelocity() const {
				return m_auxiliaryVelocity;
			}

			void setRelativeFraction(DoubleScalar fraction) {
				m_edgeFraction = fraction;
			}

			DoubleScalar getRelativeFraction() const {
				return m_edgeFraction;
			}

			/** Appends an face that is connected to this edge. Does not verify duplicates for performance. Makes sense
				only in 2-D, where one can access a half-face through an normal edge (e.g., pressure projection) */
			void addConnectedHalfFace(HalfFace<VectorType> *pFace) {
				m_connectedFaces.push_back(pFace);
			}

			const vector<HalfFace<VectorType> *> getConnectedHalfFaces() {
				return m_connectedFaces;
			}
			#pragma endregion

			#pragma region Operators
			FORCE_INLINE bool operator==(const Edge<VectorType> &rhs) const {
				return m_ID == rhs.getID();
			}
			#pragma endregion

			#pragma region Functionalities
			static halfEdgeLocation_t classifyEdge(Edge<VectorType> *pEdge, const dimensions_t &gridDimensions, DoubleScalar gridDx, faceLocation_t faceLocation);

			static halfEdgeLocation_t classifyEdgeXY(Edge<VectorType> *pEdge, const dimensions_t &gridDimensions, DoubleScalar gridDx);
			static halfEdgeLocation_t classifyEdgeXZ(Edge<VectorType> *pEdge, const dimensions_t &gridDimensions, DoubleScalar gridDx);
			static halfEdgeLocation_t classifyEdgeYZ(Edge<VectorType> *pEdge, const dimensions_t &gridDimensions, DoubleScalar gridDx);
			#pragma endregion


		protected:
			#pragma region ClassMembers
			/** Halfedges and vertices */
			pair<HalfEdge<VectorType> *, HalfEdge<VectorType> *> m_halfEdges;
			pair<Vertex<VectorType> *, Vertex<VectorType> *> m_vertices;
			
			/** VectorType properties */
			VectorType m_centroid;
			VectorType m_normal;

			/** Border edge marker and edge type*/
			bool m_borderEdge;
			edgeType_t m_edgeType;

			/** Velocity is stored per edge, so its shared between halfedges */
			VectorType m_velocity;

			/** Auxiliary (intermediary) velocity is also stored per edge */
			VectorType m_auxiliaryVelocity;

			/** Relative Edge fraction compared to regular grid edge size. */
			DoubleScalar m_edgeFraction;

			/** Halffaces connected to this edge */
			vector<HalfFace<VectorType> *> m_connectedFaces;

			/** Auxiliary structure used in decompositions */
			bool m_visited;

			/** ID vars */
			uint m_ID;
			static uint m_currID;
			#pragma endregion
		};


		/** Given a vertex ID, gets all the edges that are connected to this vertex */
		template <class VectorType>
		class VertexToEdgeMap {
		
		public:
			//Empty constructor
			VertexToEdgeMap() { } ;
			VertexToEdgeMap(const vector<Edge<VectorType> *> &edges) {
				initializeMap(edges);
			};
			
			const vector<Edge<VectorType> *> & getEdges(Vertex<VectorType> *pVertex)  {
				return m_vertexToEdgeMap[pVertex->getID()];
			}

			void initializeMap(const vector<Edge<VectorType> *> &edges);

		protected:
			map<uint, vector<Edge<VectorType> *>> m_vertexToEdgeMap;
		};

		template <class VectorType>
		unsigned int HalfEdge<VectorType>::m_currID = 0;

		template <class VectorType>
		unsigned int Edge<VectorType>::m_currID = 0;
	}
}

#endif